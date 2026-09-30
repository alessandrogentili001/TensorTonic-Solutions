# <span style="font-size: 20px;">Sum of Array</span>

<span style="font-size: 14px;">Summing an array reduces $N$ values to a single scalar. It is the canonical **reduction**, and unlike a map it forces threads to cooperate: no single thread sees all the data, so the work cannot be done in one flat pass and must instead collapse in stages across the memory hierarchy. Addition is associative, which is the only property that makes parallel reduction legal at all.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For a contiguous, row-major buffer of $N$ 32-bit floats, the kernel produces one output:</span>

$$
\text{result} = \sum_{i=0}^{N-1} \text{input}[i]
$$

<span style="font-size: 14px;">There is exactly one output address and $N$ input addresses. The structure to exploit is not in the data layout but in the combining operator: because $+$ is associative, the $N$ additions can be reassociated into any order, including a balanced tree.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">A serial loop is inherently sequential and takes $N$ steps. The parallel form reorganizes it into a **tree reduction**: pairs of values are combined simultaneously, then pairs of partial results, and so on, finishing in $\log_2 N$ steps. Each block claims a contiguous chunk of the array, reduces it to one **block partial**, and the partials are then combined across blocks. This two-level structure - reduce within a block, then combine across blocks - is the defining shape of every CUDA reduction.</span>

<span style="font-size: 14px;">Within a block, each thread first loads one (or several) elements with its global index `blockIdx.x * blockDim.x + threadIdx.x` into `__shared__` memory, then the block walks the tree. At each step the active half of the threads add an element a fixed stride away into their own slot, the stride halves, and a `__syncthreads()` separates the steps so every write of one level is visible before the next level reads it. After $\log_2(\text{blockDim.x})$ steps, slot $0$ holds the block partial.</span>

<span style="font-size: 14px;">A block size of 256 is conventional: a multiple of the 32-lane **warp** so no lanes are wasted, large enough to give the scheduler many warps for latency hiding, and small enough that several blocks fit per **SM (Streaming Multiprocessor)** to keep **occupancy** high. A common refinement is to have each thread sum two or more elements during the initial load, so half the threads are not already idle on the first tree step - this folds the array down before the tree even begins and improves the work-per-thread ratio.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Each input element is read from global memory exactly once, and consecutive threads in a warp read consecutive addresses, so the initial loads are fully **coalesced** and hit near-peak bandwidth. After that, all of the tree-combining traffic stays in `__shared__` memory, which is roughly an order of magnitude lower latency than global memory and avoids ever re-reading DRAM.</span>

<span style="font-size: 14px;">The shared-memory indexing matters. The naive scheme where thread $t$ adds the element at $t + \text{stride}$ keeps the active lanes contiguous and is **bank-conflict-free**: shared memory is split into 32 **banks**, and as long as the active lanes touch distinct banks their accesses are served in one cycle. An interleaved scheme where active threads are spread by a growing stride makes multiple lanes hit the same bank, serializing those accesses. The contiguous, stride-halving layout is the one to use.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">The kernel performs roughly one addition per 4-byte element loaded, an **arithmetic intensity** of about:</span>

$$
\frac{1 \text{ FLOP}}{4 \text{ bytes}} = 0.25 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline** model the ridge point sits in the range of tens of FLOPs per byte, so at $0.25$ the reduction is two orders of magnitude under the line: it is firmly **memory-bound**. The shared-memory tree and the single adder are never the bottleneck; the runtime is essentially $4N$ bytes divided by achievable bandwidth. Every optimization that matters is about moving those bytes efficiently - coalescing and enough warps in flight - not about the arithmetic.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Cross-Block Combine</span>

<span style="font-size: 14px;">The tree reduction handles one block, but the block partials still have to become one number, and partials in different blocks cannot see each other through shared memory. Two standard strategies bridge that gap.</span>

<span style="font-size: 14px;">The first is **`atomicAdd`**: each block adds its single partial into one global accumulator. A read-modify-write atomic serializes concurrent writers to the same address, so if every thread atomically added its own value the contention would be catastrophic. The mitigation is exactly the two-level structure: reduce the whole block in shared memory first, then issue one atomic per block. That turns $N$ contending atomics into $\lceil N / \text{blockDim.x} \rceil$ of them, making contention negligible.</span>

<span style="font-size: 14px;">The second is a **two-kernel partials approach**: the first kernel writes one partial per block into a global buffer with no atomics at all, and a second launch reduces that small buffer. The kernel-launch boundary acts as a global barrier, which CUDA otherwise does not provide across blocks. It trades an extra launch for zero atomic contention and a deterministic combine order.</span>

<span style="font-size: 14px;">For `sum` the atomic version is attractive because `atomicAdd` on `float` is natively supported and one atomic per block costs almost nothing. The choice between the two comes down to whether bit-reproducibility matters: the atomic combine interleaves block partials in nondeterministic order, while the two-kernel buffer reduces them in a fixed order each run.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive block reduction syncs after every one of the $\log_2(\text{blockDim.x})$ tree levels, even the last five where only a single warp is still active. Once the stride drops to 32 or below, all surviving lanes are inside one warp and execute in lockstep, so the `__syncthreads()` calls are pure overhead.</span>

<span style="font-size: 14px;">The optimized form replaces those final levels with a **warp-shuffle reduction**: the `__shfl_down_sync` intrinsic lets a lane read another lane's register directly, with no shared memory and no barrier. The block reduces down to one partial per warp in shared memory, then a single warp finishes the combine entirely in registers via shuffles. This removes the tail of `__syncthreads()` calls and the shared-memory round-trips for the last $\log_2 32 = 5$ steps, the dominant cost once the data is small.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a block of 8 threads reducing $[3, 1, 7, 0, 4, 1, 6, 2]$ in shared memory. The tree finishes in $\log_2 8 = 3$ steps, each followed by a `__syncthreads()`:</span>

* <span style="font-size: 14px;">**Step 1** (stride 4): threads $0..3$ add the element 4 slots away. Slots become $[7, 2, 13, 2, \dots]$ - the upper half is now dead.</span>
* <span style="font-size: 14px;">**Step 2** (stride 2): threads $0,1$ add the element 2 slots away. Slots become $[20, 4, \dots]$.</span>
* <span style="font-size: 14px;">**Step 3** (stride 1): thread $0$ adds slot $1$. Slot $0$ holds $24$, the block partial.</span>

<span style="font-size: 14px;">Eight values collapsed to one in three parallel steps instead of seven serial adds, and the number of active lanes halved each step. This single partial is then combined across blocks by one `atomicAdd` or written to the partials buffer.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Missing or misplaced `__syncthreads()`.** Reading a shared slot before the previous tree level finished writing it is a race that yields nondeterministic wrong sums; placing the barrier inside divergent control flow can hang the block.</span>
* <span style="font-size: 14px;">**Atomic contention.** Letting every thread `atomicAdd` into one global address serializes them; always reduce in shared memory first and issue one atomic per block.</span>
* <span style="font-size: 14px;">**Shared-memory bank conflicts.** An interleaved active-lane stride makes multiple lanes hit the same bank and serializes; the contiguous stride-halving pattern keeps accesses conflict-free.</span>
* <span style="font-size: 14px;">**Float accumulation order.** Parallel reduction sums in a different order than a serial reference, so results differ in the low bits; this is a correctness concern that is why floating-point tolerances exist, not a bug.</span>

---