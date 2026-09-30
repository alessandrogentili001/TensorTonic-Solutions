# Softmax

Softmax turns a 1-D array of $N$ scores into a normalized probability distribution. From a systems angle it is not a map but a **row-parallel normalization**: every output depends on global properties of the whole array (its maximum and the sum of its exponentials), so it forces two full-array **reductions** before a single output can be written. That coupling is what makes it interesting on a GPU - no thread can finish its element in isolation.

---

## The Operation

For an index $i$ in $[0, N)$, with $m = \max_j \text{input}[j]$ for numerical stability, the kernel computes:

$$
\text{output}[i] = \frac{\exp(\text{input}[i] - m)}{\sum_{j=0}^{N-1} \exp(\text{input}[j] - m)}
$$

The input is one contiguous, row-major buffer of $N$ 32-bit floats in global memory; the output is the same shape. The subtraction of $m$ does not change the mathematical result - it shifts every exponent down so the largest is $\exp(0) = 1$, keeping the values inside float range. That stability trick is also a memory decision, because it forces a first pass purely to discover $m$ before any exponential can be taken.

---

## Parallelization Strategy

The standard mapping is **one block per row** - here the single array is one row, so one block owns the whole computation. Within the block, threads split the $N$ elements among themselves, and each thread holds a idx = threadIdx.x plus a stride of blockDim.x so it can sweep an array longer than the block. The block is sized as a multiple of the 32-lane **warp** (256 or 512 is typical) so no lanes are wasted and the scheduler has many warps to hide latency.

One thread could scan the full vector, but a cooperative reduction distributes the work across a block. The work therefore proceeds in three passes separated by __syncthreads(): a reduction to find the max, a reduction to find the sum of shifted exponentials, then an independent normalize-and-write. The first two passes are cooperative; only the third is embarrassingly parallel.

The dependency chain is strict: the sum pass needs $m$ to compute each $\exp(\text{input}[j] - m)$, and the write pass needs the sum to divide by. This is why the kernel is fundamentally multi-pass and why a barrier sits between each pass - the structure mirrors the math, where the denominator is a function of the whole array.

Each reduction is a **tree reduction** in __shared__ memory: every thread first folds its strided slice into one local value, the partials are written to a shared array, and then the block halves the active lanes each step - thread $t$ combines shared[t] with shared[t + stride] for stride starting at half the block size and halving down to one. This collapses the partials to a single value in $\log_2(\text{blockDim.x})$ steps instead of a linear scan.

---

## Memory Hierarchy and Access Pattern

The input is read from global memory more than once: once during the max pass and once during the exp-sum pass, and a third time only if the kernel does not cache it. Consecutive threads of a warp read consecutive addresses on each strided step, so the global loads are fully **coalesced** - the memory controller serves each warp request in the minimum number of transactions.

The two scalars that bind the array together, $m$ and the sum, live in __shared__ memory during their tree reductions and are broadcast to every thread once final. Shared memory is roughly an order of magnitude lower latency than global DRAM, so doing the reductions there avoids hammering global memory with the cross-thread communication. A common optimization keeps each thread's shifted exponentials in registers between the sum pass and the write pass, so the input is read twice from global rather than three times - trading register pressure for one fewer DRAM sweep.

---

## Memory-bound or compute-bound? Arithmetic intensity

Per element the kernel moves about 8 bytes of global memory (one 4-byte load, one 4-byte store) and does a handful of FLOPs - a subtract, an expf, and a divide. The expf is a transcendental that costs more than an add, so softmax sits slightly higher on the **arithmetic intensity** scale than vector addition, but still only a few FLOPs per byte:

$$
\frac{\sim 5 \text{ FLOPs}}{8 \text{ bytes}} \approx 0.6 \text{ FLOP/byte}
$$

On the **roofline** the ridge point sits in the tens of FLOPs per byte, so at $\approx 0.6$ softmax is firmly **memory-bound** at scale. The transcendental gives the adders something to chew on while loads are in flight, but the runtime is still governed by how fast the input can be streamed from DRAM. The optimization story is therefore about reducing global passes and keeping loads coalesced, not about cheaper arithmetic.

---

## Execution-Model Details

The __syncthreads() barriers between passes are load-bearing: the sum pass must not start until every thread has contributed to the max, and the write pass must not start until the sum is final. A barrier placed inside divergent control flow would deadlock, because lanes that skip the barrier leave the others waiting forever - so the syncs sit at uniform points all threads reach.

During the tree-reduction steps, more and more lanes fall idle as the active stride shrinks; the last few steps use one warp, but shared-memory communication still requires synchronization; replacing block barriers requires a warp-safe reduction. The reduction is also a **bank-conflict** concern: shared memory has 32 banks, and a stride pattern that lands multiple active lanes on the same bank serializes them, so the halving layout is chosen to keep accesses conflict-free.

**Occupancy** here is bounded by the shared-memory footprint of the reduction buffer rather than registers. Because only one block works the row, the block must be large enough to give the SM several resident warps for **latency hiding** while loads of the input are outstanding; otherwise the adders stall on DRAM with no other warp to switch to.

---

## Naive vs Optimized

A naive kernel makes three full global passes - max, sum, normalize - re-reading the input from DRAM each time. The optimized version cuts global traffic in two ways:

1. **Cache in registers or shared memory**: load each element once, keep the shifted exponential live, and reuse it for both the sum and the write, removing one of the three global reads.

2. **Warp-shuffle reductions**: replace the shared-memory tree with __shfl_down_sync so the last 32 lanes combine partials in registers without shared-memory exchange, using an appropriate participating-lane mask, shaving both latency and bank-conflict risk off each reduction.

Both shrink the constant in front of the same memory-bound runtime. The structural lesson stands: softmax cannot be one pass, because the normalization denominator is not known until every element has been seen.

---

## Worked Example

Take $N = 4$ with input $[1, 3, 2, 5]$ and a block of 4 threads. The three passes run as:

* **Max pass**: the tree reduction folds $[1,3,2,5]$ to $m = 5$ in $\log_2 4 = 2$ steps - step one gives $[\max(1,2), \max(3,5)] = [2,5]$, step two gives $5$.
* **Sum pass**: each thread computes $\exp(\text{input}[i] - 5)$, giving roughly $[0.018, 0.135, 0.050, 1.0]$, which the tree reduces to a sum of $\approx 1.203$.
* **Write pass**: each thread divides its own exponential by $1.203$, producing $\approx [0.015, 0.112, 0.041, 0.832]$, which sums to $1$.

The value exp(5) is representable in float32, but sufficiently large logits can overflow when exponentiated. Subtracting the maximum keeps the largest exponential at one without changing the probability ratios.

---

## Pitfalls

* **Skipping the max-subtract.** Exponentiating raw logits overflows to inf for large values, and the divide becomes inf/inf = NaN; the subtraction is mandatory for stability, not optional polish.
* **Missing __syncthreads() between passes.** Starting the sum before the max reduction completes, or the write before the sum is final, reads partial shared-memory values and yields nondeterministic wrong results.
* **Barrier inside divergent control flow.** Placing __syncthreads() where only some lanes reach it hangs the block, since the others wait on lanes that never arrive.
* **Bank conflicts in the reduction.** A shared-memory stride that maps several active lanes to one of the 32 banks serializes those accesses and slows every reduction step.

---