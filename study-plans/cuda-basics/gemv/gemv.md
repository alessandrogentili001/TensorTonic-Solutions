# <span style="font-size: 20px;">Matrix-Vector Multiply (GEMV)</span>

<span style="font-size: 14px;">GEMV computes $y[i] = \sum_j A[i,j]\, x[j]$ - a matrix times a vector. Structurally it is a grid of independent **dot products**, one per output row, with no communication between rows. Unlike full matmul there is almost no data reuse on the large operand: every element of $A$ is read exactly once, so the matrix traffic alone pins the kernel to the memory roofline. The only reuse available is on the small vector $x$, which every row rereads, and that reuse is what the cache and the access pattern are built around.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For each output index $i$ in $[0, M)$:</span>

$$
y[i] = \sum_{j=0}^{N-1} A[i,j]\, x[j]
$$

<span style="font-size: 14px;">$A$ is $M \times N$ row-major, so row $i$ occupies the contiguous span `i*N + 0 .. i*N + (N-1)`. The vector $x$ has length $N$ and $y$ has length $M$. Each output is a length-$N$ inner product of one row of $A$ against the shared $x$.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">The natural decomposition is **one thread per output row**. Thread $i$ walks the whole length-$N$ row, multiplying each `A[i,j]` by `x[j]` and accumulating in a register, then writes the single scalar `y[i]`. The launch is a 1-D grid over the $M$ rows:</span>

$$
i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block of 256 threads (a multiple of the 32-lane **warp**) is conventional, with $\lceil M / 256 \rceil$ blocks and the usual `if (i < M)` bounds check for the rounded-up tail. The accumulator lives in a register, so there is no `__syncthreads()` and no inter-thread communication: each row is its own private reduction.</span>

<span style="font-size: 14px;">This row-per-thread mapping is the obvious decomposition because it keeps each output's accumulation entirely local to one thread, but it pays for that simplicity at the memory controller, as the access section shows. The alternative - splitting each row's dot product across many threads - trades the private register accumulator for a cross-thread reduction, and that tradeoff is exactly what the optimized form makes.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The two operands behave completely differently, and that asymmetry is the heart of GEMV.</span>

<span style="font-size: 14px;">**The matrix $A$** is read once and never reused: $MN$ elements stream through, each touched by exactly one thread. This single pass dominates total traffic. But the access is **uncoalesced** in the one-thread-per-row layout: at loop step $j$, the 32 lanes of a warp are on 32 different rows but the same column, so they read addresses `0*N+j, 1*N+j, ...` - a stride of $N$. Each warp-wide load fragments into up to 32 transactions, the classic GEMV bandwidth tax.</span>

<span style="font-size: 14px;">**The vector $x$** is tiny and **reused by every one of the $M$ rows**. All lanes at loop step $j$ read the same `x[j]`, a broadcast the hardware and L2/L1 cache serve cheaply. Because $x$ is read $M$ times but lives in a small footprint, it stays hot in cache and contributes negligible DRAM traffic after the first pass. The whole memory story is therefore: $A$ is the cost, $x$ is nearly free.</span>

<span style="font-size: 14px;">Concretely, the DRAM traffic is about $4MN$ bytes for $A$ plus $4N$ bytes for one cold pass of $x$ plus $4M$ bytes for $y$. For any non-trivial matrix the $4MN$ term swamps the rest, so the kernel's runtime is essentially $4MN$ divided by achievable bandwidth. This is why $A$'s coalescing is the only memory lever that moves the needle - reducing $x$ or $y$ traffic changes a term that was already negligible.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Each inner-product step does one multiply and one add - 2 FLOPs - while loading 4 bytes of $A$ (the $x$ load is amortized away by reuse). The **arithmetic intensity** is therefore about:</span>

$$
\frac{2 \text{ FLOP}}{4 \text{ bytes}} = 0.5 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">That is far below the **roofline** ridge point of tens of FLOPs per byte, so GEMV is firmly **memory-bound** - the multiply-add units sit idle waiting on $A$. Crucially, unlike matmul there is no tiling trick that rescues it: each $A$ element is genuinely used only once, so no amount of shared-memory staging raises its reuse. The intensity ceiling of $0.5$ is intrinsic to the operation. Optimization means maximizing bandwidth, not raising reuse.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">Hundreds-of-cycle global latency on each $A$ load is hidden by **massive multithreading**: while one warp waits on its strided row reads, the SM scheduler runs other resident warps. High **occupancy** therefore matters even though the kernel is simple, because there is a lot of latency to cover and each thread's long row loop issues many dependent loads.</span>

<span style="font-size: 14px;">There is no warp divergence - every active thread runs the same length-$N$ loop - and no synchronization. The one structural weakness is the uncoalesced $A$ access, which the optimized form attacks directly.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive one-thread-per-row kernel is correct but reads $A$ with a stride of $N$ across a warp, so effective bandwidth on the dominant operand is a fraction of peak. Two refinements recover it:</span>

<span style="font-size: 14px;">1. **One warp per row**: assign a full warp to each output row instead of a single thread. The 32 lanes now read 32 consecutive elements of the same row - a **coalesced** load - and stride together along the row. Each lane accumulates a partial dot product, then a warp-shuffle reduction collapses the 32 partials into `y[i]` with no shared memory and no `__syncthreads()`.</span>

<span style="font-size: 14px;">2. **Cache $x$ in `__shared__`**: cooperatively stage the reused vector into shared memory once per block so every row's loop reads it from on-chip memory rather than relying on L2, removing the last bit of $x$ latency.</span>

<span style="font-size: 14px;">The payoff is bandwidth on $A$: the warp-per-row layout turns the strided GEMV load into a coalesced stream, which is the dominant cost. Arithmetic is untouched because the kernel is memory-bound and there is nothing to compute faster.</span>

<span style="font-size: 14px;">There is a tradeoff. One warp per row means only $M$ warps of useful work where the naive kernel had $M$ threads, so for small $M$ the warp-per-row form can under-fill the SMs and lose the latency hiding it needs. The usual remedy is to assign several rows per block and several warps per row only when $N$ is large enough to keep each warp busy, balancing coalesced access against having enough warps resident.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $A$ as $2 \times 3$ and $x$ of length $3$, with one thread per row.</span>

* <span style="font-size: 14px;">**Thread 0** walks row 0: offsets `0,1,2`, computing $A[0,0]x[0] + A[0,1]x[1] + A[0,2]x[2]$.</span>
* <span style="font-size: 14px;">**Thread 1** walks row 1: offsets `3,4,5`, computing the same dot product against the same $x$.</span>

<span style="font-size: 14px;">With $A = \begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$ and $x = [1,1,1]$, the two threads produce $y = [6, 15]$. At loop step $j=0$ both threads read $A$ offsets $0$ and $3$ - three apart, the strided pattern - while both read the same `x[0]`, the broadcast reuse. That contrast is GEMV in miniature.</span>

<span style="font-size: 14px;">In the warp-per-row form the same row 0 is instead split across the lanes of one warp: lane $j$ holds the partial `A[0,j]*x[j]`, the lanes read $A$ offsets $0,1,2$ as one coalesced burst, and a warp-shuffle sums the three partials into `y[0] = 6`. Same arithmetic, but the dominant $A$ load is now contiguous instead of strided.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Uncoalesced $A$ access in the one-thread-per-row layout.** Lanes stride by $N$ down a column, fragmenting each load; assign a warp per row so lanes read consecutive row elements.</span>
* <span style="font-size: 14px;">**Treating $x$ traffic as the bottleneck.** $x$ is reused $M$ times and stays cache-resident; the real cost is the single uncoalesced pass over $A$, so optimize that.</span>
* <span style="font-size: 14px;">**Integer overflow in `i * N + j`.** For tall or wide matrices the row-major offset can exceed 32-bit `int`; use `size_t` for the index math.</span>
* <span style="font-size: 14px;">**Reading $y$ before `cudaDeviceSynchronize()`.** The launch is asynchronous; the host must synchronize before trusting the output vector.</span>

---