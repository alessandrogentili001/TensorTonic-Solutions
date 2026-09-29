# <span style="font-size: 20px;">Outer Product</span>

<span style="font-size: 14px;">The outer product of two vectors builds a matrix from every pairwise product, $C[i,j] = a[i] \cdot b[j]$. It is still an **embarrassingly parallel map** over the output - each entry is one independent multiply with no thread-to-thread communication - but it carries a twist absent from a plain elementwise kernel: heavy **broadcast reuse**. Every thread in a row reads the same $a[i]$, and every thread in a column reads the same $b[j]$, so a tiny amount of input feeds a large output. It is also exactly the $K = 1$ special case of matrix multiplication, which makes it a clean lens on where the GEMM reuse story begins.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given a column vector $a$ of length $M$ and a row vector $b$ of length $N$, the kernel computes, for each $i$ in $[0, M)$ and $j$ in $[0, N)$:</span>

$$
C[i,j] = a[i] \cdot b[j]
$$

<span style="font-size: 14px;">The output `C` is an $M \times N$ matrix stored row-major as a flat $M N$-float buffer in device (global) memory, so entry $(i, j)$ sits at offset $i \cdot N + j$. Inputs `a` and `b` are small relative to the output: together they are $M + N$ floats, while the output is $M N$ floats. That ratio is the whole story.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $M N$ outputs are independent, the decomposition is **one thread per output element** on a two-dimensional grid shaped like the result matrix. Each thread maps its 2-D coordinates to a row and column:</span>

$$
i = \text{blockIdx.y} \cdot \text{blockDim.y} + \text{threadIdx.y}, \quad j = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">Each thread reads exactly one element of `a` (using its row $i$) and one element of `b` (using its column $j$), multiplies them, and writes the single output `C[i * N + j]`. A 16x16 block - 256 threads, a whole number of 32-lane **warps** - is the conventional 2-D choice, giving enough warps per **SM (Streaming Multiprocessor)** to hide latency while tiling the matrix cleanly. The grid spans $\lceil N / 16 \rceil$ blocks across and $\lceil M / 16 \rceil$ down, and an `if (i < M && j < N)` guard silences the overhanging edge threads so they never address past the output buffer.</span>

<span style="font-size: 14px;">There is no `__syncthreads()` and no shared state in the simplest kernel. Each thread fetches its own scalars and writes its own output, so the computation is a flat sheet of independent multiplies; the 2-D grid is just an indexing scheme that matches the matrix shape.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The reuse pattern is the defining feature. All 16 threads in a row of a block load the identical $a[i]$, and all 16 threads in a column load the identical $b[j]$. That is enormous redundancy: across the full output, $a[i]$ is read $N$ times and $b[j]$ is read $M$ times. The hardware softens this for free - a warp of lanes reading the same `b[j]` benefits from the broadcast path, and repeated reads of the small `a` and `b` vectors hit the L2 cache rather than DRAM - but the structure invites explicit caching.</span>

<span style="font-size: 14px;">The output stores are what govern bandwidth, and they are **coalesced**. Within a 16x16 block the lanes of a warp share `threadIdx.y` and walk consecutive `threadIdx.x`, hence consecutive `j`, hence consecutive row-major addresses of `C`. So a warp writes 32 contiguous words and the controller serves the store in the minimum number of transactions. The load of `b[j]` across those same lanes is also consecutive and coalesced; the load of `a[i]` is a single broadcast value shared by the whole warp.</span>

<span style="font-size: 14px;">It is worth contrasting this with the inputs' total footprint. The two vectors together occupy $M + N$ floats, while the output occupies $M N$ - for a $1024 \times 1024$ result that is a few kilobytes of input feeding four megabytes of output. The inputs comfortably resident in cache is exactly why their repeated reads do not dominate, and why the output write, touched exactly once per element and never cached for reuse, is the traffic that sets the runtime.</span>

<span style="font-size: 14px;">This is where `__shared__` memory earns its keep, unlike in a plain map. A block can cooperatively stage its 16-element slice of `a` and 16-element slice of `b` into shared memory once, then have all 256 threads read those scalars from shared memory instead of re-fetching from global. That converts $256$ global reads of the inputs per block into $32$, raising arithmetic intensity slightly - the first hint of the tiling idea that makes full GEMM compute-bound. The host copies `a` and `b` across the PCIe bus before launch; the kernel touches only device-resident buffers.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Count traffic against work. Each output element performs one multiply and, dominantly, one 4-byte store; the input reads amortize away thanks to reuse and caching, leaving roughly 4 bytes written per element. The **arithmetic intensity** is therefore on the order of:</span>

$$
\frac{1 \text{ FLOP}}{4 \text{ bytes}} \approx 0.25 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">Even with the inputs fully cached, that sits far below the **roofline** ridge point of tens of FLOPs per byte, so the outer product is **memory-bound on the $M N$ output write**. The kernel is, in effect, a structured way to fill an $M \times N$ buffer; its runtime tracks how fast the GPU can stream $4 M N$ bytes out to DRAM. As the $K = 1$ case of GEMM it shows why matmul is interesting: there, each output accumulates over $K$ products, so the same loaded inputs do $K$ times more arithmetic and intensity climbs with the tile until the kernel crosses into compute-bound territory. With $K = 1$ there is nothing to amortize over, so it stays bandwidth-limited.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global access costs hundreds of cycles, hidden through **massive multithreading**: when a warp stalls on its store or its `b` load, the SM scheduler runs another ready warp. With high **occupancy** the memory pipeline stays saturated. The kernel has no divergent branches beyond the uniform edge guard and, in the shared-memory variant, one `__syncthreads()` after staging the input slices so all threads see them before reading. Occupancy is otherwise capped only by launching enough blocks, which a large output guarantees.</span>

<span style="font-size: 14px;">The shared-memory variant's single barrier is cheap because it happens once per block, not once per output, and the staged slices are tiny - 32 floats for a 16x16 tile, far below the tens of kilobytes of shared memory an SM provides. That small footprint means the staging never throttles occupancy, so the optimization is pure upside on the input side. The broadcast read of `a[i]` by a whole warp also avoids shared-memory **bank conflicts** entirely, because a uniform address across lanes is served by the broadcast path rather than 32 separate bank accesses.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel has every thread independently load `a[i]` and `b[j]` from global memory, issuing $M N$ loads of each input across the grid - a vast redundancy, even if the L2 cache absorbs most of it.</span>

<span style="font-size: 14px;">1. **Shared-memory staging**: each block loads its 16 `a` values and 16 `b` values into `__shared__` memory once, syncs, then all 256 threads read scalars from shared memory. This cuts per-block global input reads from $256$ to $32$, an $8\times$ reduction in input traffic for a 16x16 tile.</span>

<span style="font-size: 14px;">2. **Vectorized stores**: writing the output as `float4` moves 16 bytes per instruction instead of 4, using the bus more efficiently on the dominant store traffic.</span>

<span style="font-size: 14px;">The payoff is real but bounded: because the kernel is dominated by the $M N$ output write, no amount of input caching escapes the bandwidth roofline. Optimization approaches the write ceiling, it does not lift it.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $a = [1, 2, 3]$ ($M = 3$) and $b = [10, 20]$ ($N = 2$), producing a $3 \times 2$ output. With a 16x16 block one block covers it, most threads guarded off. The six in-bounds threads compute:</span>

* <span style="font-size: 14px;">**Row 0** ($a[0] = 1$): $C[0,0] = 1 \cdot 10 = 10$, $C[0,1] = 1 \cdot 20 = 20$ at flat offsets $0, 1$.</span>
* <span style="font-size: 14px;">**Row 1** ($a[1] = 2$): $C[1,0] = 20$, $C[1,1] = 40$ at offsets $2, 3$.</span>
* <span style="font-size: 14px;">**Row 2** ($a[2] = 3$): $C[2,0] = 30$, $C[2,1] = 60$ at offsets $4, 5$.</span>

<span style="font-size: 14px;">Note that both threads in row 0 read the same $a[0] = 1$, and both threads in column 0 (across rows) read the same $b[0] = 10$ - the broadcast reuse made concrete. The full result is $C = \begin{bmatrix} 10 & 20 \\ 20 & 40 \\ 30 & 60 \end{bmatrix}$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Transposing the axis mapping.** Mapping consecutive lanes to rows instead of columns makes the output stores stride $N$ apart, breaking coalescing on the dominant traffic and collapsing bandwidth; consecutive `threadIdx.x` must map to consecutive `j`.</span>
* <span style="font-size: 14px;">**Missing `__syncthreads()` in the shared variant.** Reading the staged `a` or `b` slices before every lane has finished writing them is a race that yields nondeterministic wrong results.</span>
* <span style="font-size: 14px;">**Omitting the 2-D bounds check.** Edge overhang on either axis lets a thread compute an offset past the output; both `i < M` and `j < N` are required.</span>
* <span style="font-size: 14px;">**Expecting input caching to beat the roofline.** The kernel is bounded by the $M N$ output write, so reducing input reads helps only up to the bandwidth ceiling of the store traffic.</span>

---