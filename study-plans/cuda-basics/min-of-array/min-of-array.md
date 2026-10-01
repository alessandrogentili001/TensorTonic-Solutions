# <span style="font-size: 20px;">Min of Array</span>

<span style="font-size: 14px;">Finding the minimum of an array reduces $N$ values to a single scalar. It is a **reduction**, structurally identical to summation but with `fminf` as the combining operator instead of `+`. Like every reduction it forces threads to cooperate: no single thread sees all the data, so the work cannot be done in one flat pass and must collapse in stages across the memory hierarchy. The operator is associative and commutative, which is what makes the parallel tree legal.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For a contiguous, row-major buffer of $N$ 32-bit floats, the kernel produces one output:</span>

$$
\text{result} = \min_{0 \le i < N} \text{input}[i]
$$

<span style="font-size: 14px;">The combining operator has an **identity** of $+\text{FLT\_MAX}$: any thread without a real element to contribute initializes its slot to that value so it can never win the comparison. The structure to exploit is the operator, not the layout - because `min` is associative, the comparisons can be reassociated into a balanced tree.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">A serial scan takes $N$ steps. The parallel form reorganizes it into a **tree reduction**: pairs of values are compared simultaneously with `fminf`, then pairs of partial minima, and so on, finishing in $\log_2 N$ steps. Each block claims a contiguous chunk, reduces it to one **block partial** minimum, and the partials are then combined across blocks. This two-level structure - reduce within a block, then combine across blocks - is the defining shape of every CUDA reduction.</span>

<span style="font-size: 14px;">Within a block, each thread loads one or more elements with its global index `blockIdx.x * blockDim.x + threadIdx.x` into `__shared__` memory, then the block walks the tree. At each step the active half of the threads take `fminf` of their slot and one a fixed stride away, the stride halves, and a `__syncthreads()` separates the steps so every write of one level is visible before the next reads it. After $\log_2(\text{blockDim.x})$ steps, slot $0$ holds the block minimum.</span>

<span style="font-size: 14px;">A block size of 256 is conventional: a multiple of the 32-lane **warp** so no lanes are wasted, large enough to give the scheduler many warps for latency hiding, and small enough that several blocks fit per **SM (Streaming Multiprocessor)** to keep **occupancy** high. Having each thread fold two or more elements during the initial load halves the tree's starting width and improves work-per-thread.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Each input element is read from global memory exactly once, and consecutive threads in a warp read consecutive addresses, so the initial loads are fully **coalesced** and hit near-peak bandwidth. All of the tree-combining traffic afterward stays in `__shared__` memory, roughly an order of magnitude lower latency than global memory, so DRAM is never re-read.</span>

<span style="font-size: 14px;">The shared-memory indexing matters. The scheme where thread $t$ compares against the element at $t + \text{stride}$ keeps the active lanes contiguous and is **bank-conflict-free**: shared memory is split into 32 **banks**, and as long as the active lanes touch distinct banks their accesses complete in one cycle. An interleaved scheme that spreads active threads by a growing stride makes multiple lanes hit the same bank and serializes them. The contiguous, stride-halving layout is the one to use.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">The kernel performs roughly one comparison per 4-byte element loaded, an **arithmetic intensity** of about:</span>

$$
\frac{1 \text{ op}}{4 \text{ bytes}} = 0.25 \text{ op/byte}
$$

<span style="font-size: 14px;">On the **roofline** model the ridge point sits in the range of tens of operations per byte, so at $0.25$ the reduction is two orders of magnitude under the line: it is firmly **memory-bound**. The shared-memory tree and the comparison unit are never the bottleneck; the runtime is essentially $4N$ bytes divided by achievable bandwidth. The only optimizations that matter are those that move those bytes efficiently - coalescing and enough warps in flight - not the comparisons themselves.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Cross-Block Combine</span>

<span style="font-size: 14px;">The tree reduces one block, but the block partials still have to become one number, and partials in different blocks cannot see each other through shared memory. The combine for `min` is harder than for `sum` because CUDA has no native floating-point atomic minimum: `atomicAdd` exists for `float`, but `atomicMin` is provided only for integer types. Two strategies bridge the gap.</span>

<span style="font-size: 14px;">The first is a **CAS-based atomic-min**: build the operation out of `atomicCAS` in a loop, reading the current global value, computing `fminf` with the block partial, and compare-and-swapping it back, retrying if another block changed it in between. This works but adds a retry loop and, like any atomic, serializes concurrent writers to the one address. The two-level structure keeps it cheap: reduce the whole block in shared memory first, then issue one CAS-based update per block, turning $N$ contending operations into $\lceil N / \text{blockDim.x} \rceil$ of them.</span>

<span style="font-size: 14px;">The second is a **two-kernel partials approach**: the first kernel writes one partial minimum per block into a global buffer with no atomics, and a second launch reduces that small buffer. The kernel-launch boundary acts as the global barrier CUDA does not otherwise provide across blocks. For `min` this is often the cleaner choice precisely because the float atomic is not native - it sidesteps the CAS retry loop entirely, trading one extra launch for simpler, contention-free code.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive block reduction syncs after every one of the $\log_2(\text{blockDim.x})$ tree levels, including the last five where only a single warp is still active. Once the stride drops to 32 or below, all surviving lanes are inside one warp and execute in lockstep, so those `__syncthreads()` calls are pure overhead.</span>

<span style="font-size: 14px;">The optimized form replaces those final levels with a **warp-shuffle reduction**: the `__shfl_down_sync` intrinsic lets a lane read another lane's register directly, feeding `fminf` with no shared memory and no barrier. The block reduces down to one partial per warp in shared memory, then a single warp finishes the combine in registers via shuffles. This removes the tail of `__syncthreads()` calls and the shared-memory round-trips for the last $\log_2 32 = 5$ steps, the dominant cost once the data is small. Using branchless `fminf` rather than an `if (a < b)` also avoids **warp divergence** on the comparison.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a block of 8 threads reducing $[3, 1, 7, 5, 4, 9, 6, 2]$ in shared memory. The tree finishes in $\log_2 8 = 3$ steps, each followed by a `__syncthreads()`:</span>

* <span style="font-size: 14px;">**Step 1** (stride 4): threads $0..3$ take `fminf` with the slot 4 away. Slots become $[3, 1, 6, 2, \dots]$ - the upper half is now dead.</span>
* <span style="font-size: 14px;">**Step 2** (stride 2): threads $0,1$ take `fminf` with the slot 2 away. Slots become $[3, 1, \dots]$.</span>
* <span style="font-size: 14px;">**Step 3** (stride 1): thread $0$ takes `fminf` of slots $0$ and $1$. Slot $0$ holds $1$, the block minimum.</span>

<span style="font-size: 14px;">Eight values collapsed to one in three parallel steps instead of seven serial comparisons, with the active lanes halving each step. This single partial is then combined across blocks by a CAS-based atomic-min or written to the partials buffer.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong identity value.** Threads with no real element must seed their slot with $+\text{FLT\_MAX}$; initializing to $0$ silently corrupts the result for all-positive inputs.</span>
* <span style="font-size: 14px;">**Assuming a native float `atomicMin`.** CUDA provides it only for integers, so the cross-block combine needs an `atomicCAS` loop or a two-kernel buffer; calling a nonexistent overload fails to compile or quietly uses the wrong type.</span>
* <span style="font-size: 14px;">**Missing or misplaced `__syncthreads()`.** Reading a shared slot before the previous tree level wrote it is a race that yields nondeterministic results; a barrier inside divergent control flow can hang the block.</span>
* <span style="font-size: 14px;">**Shared-memory bank conflicts.** An interleaved active-lane stride makes multiple lanes hit the same bank and serializes; the contiguous stride-halving pattern keeps accesses conflict-free.</span>

---