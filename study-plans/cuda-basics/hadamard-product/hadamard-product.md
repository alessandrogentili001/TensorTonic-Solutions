# <span style="font-size: 20px;">Hadamard Product</span>

<span style="font-size: 14px;">The Hadamard product is the elementwise multiplication of two equally shaped matrices, $C[i,j] = A[i,j] \cdot B[i,j]$. Despite living on 2-D operands, it is still the canonical **embarrassingly parallel map**: every output entry depends on exactly one element from each input, with no communication between threads. The only thing that changes versus a 1-D elementwise kernel is the index arithmetic; the systems story - coalesced, no reuse, memory-bound - is identical, and the kernel is again a pure test of how fast the GPU can move data through global memory.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For each row $i$ in $[0, M)$ and column $j$ in $[0, N)$, the kernel computes:</span>

$$
C[i,j] = A[i,j] \cdot B[i,j]
$$

<span style="font-size: 14px;">All three matrices are $M \times N$, stored in row-major order as flat contiguous buffers of $M N$ 32-bit floats in device (global) memory. Row-major means element $(i, j)$ lives at the flat offset $i \cdot N + j$, so an entire row is contiguous and consecutive columns are adjacent in memory. Output $(i, j)$ reads only the matching entries of `A` and `B`; nothing is shared or reused.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $M N$ products are mutually independent, the decomposition is **one thread per output element**, laid out on a two-dimensional grid that mirrors the matrix shape. Each thread recovers its row and column from its 2-D block and thread coordinates, then flattens them into the linear offset:</span>

$$
i = \text{blockIdx.y} \cdot \text{blockDim.y} + \text{threadIdx.y}, \quad j = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">The flat index is then `idx = i * N + j`, and the kernel reads `A[idx]`, `B[idx]`, and writes `C[idx]`. A 16x16 block - 256 threads - is the conventional 2-D choice: it is a whole number of 32-lane **warps**, it supplies enough warps per **SM (Streaming Multiprocessor)** for latency hiding, and its square shape tiles a matrix cleanly. The grid then needs $\lceil N / 16 \rceil$ blocks across and $\lceil M / 16 \rceil$ blocks down.</span>

<span style="font-size: 14px;">Rounding both grid dimensions up means blocks along the right and bottom edges overhang the matrix, so the kernel guards every thread with `if (i < M && j < N)`. The two-dimensional bounds check matters more here than in the 1-D case, because overhang can occur on either axis independently. Without it, edge threads compute an offset past the buffer and read or write out of bounds, which is undefined behavior.</span>

<span style="font-size: 14px;">There is no `__syncthreads()` and no shared state. Threads never observe each other's results, so the kernel is one flat sheet of independent multiplications: the 2-D grid is purely an indexing convenience that maps naturally onto a matrix, not a sign of any cooperation.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Per output element the kernel issues three global accesses: load `A[idx]`, load `B[idx]`, store `C[idx]`. Each value is used once and dropped, so there is no reuse and therefore no role for `__shared__` memory or for caching values in registers. Shared memory pays off only when a loaded datum serves many threads; a map loads each datum for exactly one thread, so staging it would be pure overhead.</span>

<span style="font-size: 14px;">The crucial property is that the row-major layout keeps the access **coalesced**. Within a 16x16 block the lanes of a warp share the same `threadIdx.y` and walk consecutive `threadIdx.x`, which means consecutive `j` values within one row. Consecutive columns of a row-major matrix are consecutive in memory, so a warp touches 32 consecutive words of `A` (and `B`, and `C`). The controller serves each warp-wide request in the minimum number of transactions, delivering near-peak bandwidth.</span>

<span style="font-size: 14px;">This is why threads walk columns rather than rows. If the index assignment were transposed - consecutive lanes stepping down a column - each warp would touch addresses $N$ words apart, fragmenting one transaction into many and gutting effective bandwidth. The same elementwise math with the wrong axis mapping can run many times slower purely from the access pattern. The host copies both inputs across the PCIe bus before launch; the kernel only ever touches the device-resident buffers it is given.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Each output element moves 12 bytes of global memory - two 4-byte loads and one 4-byte store - for a single floating-point multiply. The **arithmetic intensity** is:</span>

$$
\frac{1 \text{ FLOP}}{12 \text{ bytes}} \approx 0.083 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline** model a kernel turns compute-bound only above the ridge point, which sits in the tens of FLOPs per byte on modern hardware. At $0.083$ the Hadamard product falls two to three orders of magnitude below that line: it is **deeply memory-bound**. The multiplier idles almost continuously, waiting on operands from DRAM. This contrasts sharply with true matrix multiplication, which shares the same operands but reuses each loaded value $O(N)$ times and can become compute-bound; the Hadamard product has no such reuse to climb the roofline with.</span>

<span style="font-size: 14px;">The classification fixes the optimization story. Faster arithmetic is meaningless with one multiply per element; only bandwidth-side levers change the runtime: coalesced access (already in place via the column-walking layout), enough warps to hide DRAM latency, and wider transactions. Runtime is essentially $12 M N$ bytes divided by achievable bandwidth.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global load costs hundreds of cycles. The GPU hides that not with large caches but with **massive multithreading**: when a warp issues its loads of `A` and `B` and stalls, the SM scheduler switches to another resident warp that is ready. With enough warps per SM - high **occupancy** - the issue slots stay full and the memory pipeline never starves.</span>

<span style="font-size: 14px;">As with any map, the launch configuration rather than the kernel logic governs throughput. There are no divergent branches beyond the uniform edge guard, no synchronization, and no shared-memory pressure, so occupancy is limited only by launching enough blocks. For large matrices the 2-D grid is huge and the SMs are flooded with warps automatically.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The one-thread-per-element 2-D kernel is already near-optimal because the work is bandwidth-bound and the column-walking layout keeps it coalesced. Two refinements squeeze the last few percent:</span>

<span style="font-size: 14px;">1. **Flattened grid-stride loop**: treat the matrix as one length-$M N$ array, launch a fixed device-sized 1-D grid, and let each thread stride `blockDim.x * gridDim.x` through the flat buffer. Because row-major storage is already contiguous, the flattened view is both legal and fully coalesced, and one configuration handles any shape while amortizing launch overhead.</span>

<span style="font-size: 14px;">2. **Vectorized loads**: reinterpreting each row as `float4` lets a thread move 16 bytes per memory instruction instead of 4, provided $N$ is a multiple of four and the rows are aligned. Fewer, wider transactions use the bus more efficiently and nudge the kernel toward the bandwidth ceiling.</span>

<span style="font-size: 14px;">Both sit atop an already-saturated pipeline. The lesson is unchanged: for an elementwise map you approach the bandwidth roofline, you never beat it.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a $2 \times 3$ matrix ($M = 2$, $N = 3$) with 16x16 blocks - a single block covers it, with most of its 256 threads guarded off. Consider the six in-bounds threads:</span>

* <span style="font-size: 14px;">**Row 0** (`threadIdx.y = 0`): columns $j = 0, 1, 2$ give flat offsets $0, 1, 2$. These three lanes are consecutive in the warp and touch consecutive addresses - coalesced.</span>
* <span style="font-size: 14px;">**Row 1** (`threadIdx.y = 1`): columns $j = 0, 1, 2$ give flat offsets $3, 4, 5$, contiguous with row 0 because storage is row-major.</span>

<span style="font-size: 14px;">With $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$ and $B = \begin{bmatrix} 10 & 10 & 10 \\ 2 & 2 & 2 \end{bmatrix}$, the six active threads independently produce $C = \begin{bmatrix} 10 & 20 & 30 \\ 8 & 10 & 12 \end{bmatrix}$. Threads at $j \ge 3$ or $i \ge 2$ fail the guard and exit. No thread waits on any other.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Transposing the index-to-axis mapping.** Letting consecutive lanes walk rows instead of columns makes each warp stride $N$ words apart, shattering coalescing and slashing bandwidth; consecutive `threadIdx.x` must map to consecutive `j`.</span>
* <span style="font-size: 14px;">**Omitting the 2-D bounds check.** Overhang can occur on either axis, so both `i < M` and `j < N` are required; dropping either lets edge threads read or write past the buffer.</span>
* <span style="font-size: 14px;">**Integer overflow in `i * N + j`.** For large matrices the flat offset can exceed 32-bit `int` range; compute the index in `size_t` to avoid wraparound and corrupt addressing.</span>
* <span style="font-size: 14px;">**Expecting arithmetic optimizations to help.** At $\approx 0.083$ FLOP/byte the kernel is memory-bound, so only bandwidth-side changes matter; there is just one multiply to optimize.</span>

---