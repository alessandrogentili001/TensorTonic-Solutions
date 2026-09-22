# <span style="font-size: 20px;">Matrix Addition</span>

<span style="font-size: 14px;">Matrix addition computes a pointwise sum of two matrices, $C[i,j] = A[i,j] + B[i,j]$. It is the two-dimensional analogue of vector addition: still an **embarrassingly parallel map**, with every output depending on exactly one input from each operand and zero communication between threads. The only thing that changes is the indexing - a 2-D coordinate that must be flattened onto a row-major buffer - which makes this the cleanest place to learn how 2-D grids map onto linear device memory.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For a row $i$ in $[0, M)$ and a column $j$ in $[0, N)$, the kernel evaluates:</span>

$$
C[i,j] = A[i,j] + B[i,j]
$$

<span style="font-size: 14px;">All three matrices are $M \times N$ and stored **row-major**, so the logical element $(i,j)$ lives at the linear offset `i * N + j`. The 2-D shape is purely a labeling convenience: the data is one contiguous buffer, and every byte of $A$ and $B$ is read once, every byte of $C$ written once. Nothing is shared, reused, or reordered.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $M \times N$ outputs are mutually independent, the natural decomposition is **one thread per output element**. The launch covers the matrix with a two-dimensional grid of two-dimensional blocks, and each thread reconstructs its $(i,j)$ coordinate from its block and lane indices on both axes:</span>

$$
j = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}, \quad i = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}
$$

<span style="font-size: 14px;">A `16x16` block is the conventional choice for 2-D maps. It holds 256 threads - a multiple of the 32-lane **warp** so no lanes are wasted - and is small enough that many blocks fit on one **SM (Streaming Multiprocessor)**, keeping **occupancy** high. The grid needs $\lceil N / 16 \rceil$ blocks across and $\lceil M / 16 \rceil$ blocks down to tile the whole matrix.</span>

<span style="font-size: 14px;">Mapping the fast-moving `threadIdx.x` to the column $j$ is deliberate, not arbitrary. Adjacent threads in a warp then differ by adjacent columns, and adjacent columns in a row-major layout are adjacent addresses. The flatten step `i * N + j` turns the 2-D coordinate back into the linear index the hardware actually addresses.</span>

<span style="font-size: 14px;">Rounding the grid up on both axes means the right and bottom edge blocks usually overhang the matrix. Those surplus threads must do nothing, which is why the body is guarded by `if (i < M && j < N)`. Without that bounds check the edge threads would read and write past the buffers. There is no `__syncthreads()` and no shared state anywhere; the computation is one flat wave of independent work.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The kernel touches three global arrays per element: it loads `A[i*N+j]`, loads `B[i*N+j]`, and stores `C[i*N+j]`. There is no reuse - each datum is used exactly once - so `__shared__` memory and extra register caching would only add overhead. Shared memory exists to enable reuse; a map has none.</span>

<span style="font-size: 14px;">The access pattern is, however, ideal for the one thing that matters: **coalescing**. Because `threadIdx.x` maps to $j$, the 32 threads of a warp hold consecutive column indices in the same row, so they read 32 consecutive addresses of $A$ (and of $B$) and write 32 consecutive addresses of $C$. The memory controller serves each warp-wide request in the minimum number of transactions, delivering near-peak effective bandwidth.</span>

<span style="font-size: 14px;">Reversing the mapping - putting the row $i$ on `threadIdx.x` - would make a warp stride by $N$ elements between lanes, shattering each request into up to 32 separate transactions. The matrix is just a 1-D buffer; coalescing depends entirely on which index the contiguous thread axis lands on.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per output element the kernel moves 12 bytes of global memory - two 4-byte loads and one 4-byte store - and performs exactly one floating-point addition. Its **arithmetic intensity** is therefore about:</span>

$$
\frac{1 \text{ FLOP}}{12 \text{ bytes}} \approx 0.083 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline** model a kernel is compute-bound only when its intensity exceeds the GPU's ridge point, which sits in the range of tens of FLOPs per byte on modern hardware. At $0.083$, matrix addition is two to three orders of magnitude below that line: it is **deeply memory-bound**. The single adder is idle almost the entire time, waiting for operands to arrive from DRAM.</span>

<span style="font-size: 14px;">This dictates the whole optimization story. Cleverer arithmetic is pointless - there is one add. The only levers that change runtime raise effective bandwidth: coalesced access (already optimal), enough warps in flight to hide DRAM latency, and wider transactions. A correct kernel runs at essentially $12MN$ bytes divided by achievable bandwidth.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global-memory load costs hundreds of cycles of latency. The GPU hides that not with large caches but with **massive multithreading**: when a warp issues its loads of $A$ and $B$ and stalls, the SM scheduler switches to another resident warp that is ready to run. With high occupancy there is always other work to issue, and the memory pipeline stays saturated.</span>

<span style="font-size: 14px;">For a map the launch configuration, not the kernel logic, determines performance. There are no divergent branches (every active thread takes the same path), no synchronization, and no shared-memory pressure, so occupancy is limited only by having launched enough blocks. For a large matrix this is automatic; the 2-D grid is huge and the SMs are flooded with warps.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The one-thread-per-element kernel is already near-optimal because the problem is bandwidth-bound and the access is coalesced. There is little headroom, but two refinements squeeze out the remainder:</span>

<span style="font-size: 14px;">1. **Flatten to a 1-D launch**: since the matrix is one contiguous buffer of $MN$ floats, a kernel can ignore the 2-D shape entirely and treat it as a flat vector add over $MN$ elements with a grid-stride loop. This decouples the launch from $M$ and $N$ and lets one configuration handle any shape while keeping access perfectly contiguous.</span>

<span style="font-size: 14px;">2. **Vectorized loads**: reinterpreting the rows as `float4` lets each thread load and store 16 bytes per instruction instead of 4. Fewer, wider transactions use the bus more efficiently, nudging the kernel closer to the bandwidth ceiling. This is safe only when each row length $N$ is a multiple of 4 so alignment holds.</span>

<span style="font-size: 14px;">Both are micro-optimizations on an already-saturated memory pipeline. The headline stands: for a map you cannot beat the bandwidth roofline, you can only approach it.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a $2 \times 3$ matrix ($M=2$, $N=3$) with `16x16` blocks. One block more than covers it, and most of that block's 256 threads overhang the matrix.</span>

* <span style="font-size: 14px;">**Active threads**: the six lanes with $(i,j) \in \{(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)\}$ pass `i < 2 && j < 3`. Each flattens to `i*3+j`, giving linear offsets $0,1,2,3,4,5$ - the buffer in order.</span>
* <span style="font-size: 14px;">**Overhang threads**: every lane with $j \ge 3$ or $i \ge 2$ fails the bounds check and exits without touching memory.</span>

<span style="font-size: 14px;">With $A = \begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$ and $B = \begin{bmatrix}10&20&30\\40&50&60\end{bmatrix}$, the six active threads independently produce $C = \begin{bmatrix}11&22&33\\44&55&66\end{bmatrix}$. Within row 0 the warp's lanes touched offsets $0,1,2$ - consecutive addresses, the coalesced ideal.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Mapping the row to `threadIdx.x`.** This strides a warp by $N$ between lanes and fragments each coalesced request into up to 32 transactions, collapsing bandwidth. The fast thread axis must drive the contiguous column index $j$.</span>
* <span style="font-size: 14px;">**Omitting the 2-D bounds check.** When $M$ or $N$ is not a multiple of 16, the edge blocks overhang; without `if (i < M && j < N)` those threads read and write out of bounds.</span>
* <span style="font-size: 14px;">**Integer overflow in `i * N + j`.** For very large matrices the row-major flatten can exceed 32-bit `int`; compute the offset in `size_t` to avoid wraparound and corrupt addressing.</span>
* <span style="font-size: 14px;">**Reading results before `cudaDeviceSynchronize()`.** The launch is asynchronous; reading $C$ too early observes stale, partially written data.</span>

---