# <span style="font-size: 20px;">Mean and Variance</span>

<span style="font-size: 14px;">Computing the mean and population variance of an array collapses $N$ values into two scalars. Like any summary statistic over a whole array, it is a **reduction**: no single thread can see every element, so the work must collapse in stages across the memory hierarchy. The systems trick that makes it efficient is computing both statistics in a single pass using two parallel accumulators.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">The kernel produces the mean and the population (biased, divisor $N$) variance:</span>

$$
\mu = \frac{1}{N}\sum_{i=0}^{N-1} x_i, \qquad \sigma^2 = \frac{1}{N}\sum_{i=0}^{N-1} (x_i - \mu)^2
$$

<span style="font-size: 14px;">Both results are written to single-element device buffers. The naive reading suggests two passes - one to find $\mu$, a second to accumulate squared deviations - but the algebraic identity $\sigma^2 = \mathbb{E}[x^2] - \mu^2$ lets a single pass that accumulates both $\sum x_i$ and $\sum x_i^2$ recover the variance afterward.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Addition is associative, so each of the two sums can be reorganized into a **tree reduction** that combines partials in parallel and finishes in $\log_2 N$ steps rather than the $N$ steps of a serial loop. The array is mapped onto a grid where each block reduces a contiguous chunk, and every thread maintains two running register accumulators at once - one for $x_i$, one for $x_i^2$ - so a single sweep of the data feeds both statistics.</span>

<span style="font-size: 14px;">The block size is a multiple of the 32-lane **warp** (256 is conventional) for full lane utilization and enough resident warps to hide memory latency. Each block reduces its chunk in `__shared__` memory and emits one partial pair; a second stage combines the per-block partials into the global $\sum x_i$ and $\sum x_i^2$, after which a trivial final step divides by $N$ and applies the identity to produce $\mu$ and $\sigma^2$.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Each input element is read from global memory exactly once, and consecutive threads in a warp read consecutive addresses, so the loads are fully **coalesced**. The two partial sums live in `__shared__` memory during the in-block tree reduction, which is roughly an order of magnitude lower latency than global memory and avoids any re-read of DRAM. The squared term $x_i^2$ is formed in a register from the already-loaded value, so the single-pass design costs no extra memory traffic over a plain sum - only one extra multiply and one extra accumulator per thread.</span>

<span style="font-size: 14px;">This is the payoff of fusing the two reductions: a two-pass formulation would stream the entire array through global memory twice, doubling the dominant cost, whereas the single-pass sum-and-sum-of-squares reads it once.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the kernel moves 4 bytes from global memory and performs a small constant number of flops - one add for the sum, one multiply and one add for the sum of squares. That is an **arithmetic intensity** of roughly $0.75$ FLOPs per byte:</span>

$$
\frac{3 \text{ FLOPs}}{4 \text{ bytes}} \approx 0.75 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">That sits far under the **roofline** ridge point of tens of FLOPs per byte, so the kernel is **memory-bound**. The handful of extra flops for the squared accumulator are effectively free - they hide under the memory latency that already dominates. As with any reduction, the only levers that matter are coalesced access and enough warps in flight to keep the memory pipeline saturated.</span>

---

## <span style="font-size: 16px;">Cross-Block Combine</span>

<span style="font-size: 14px;">A tree reduction inside one block collapses that block's chunk, but the per-block partials still have to be merged. Two standard structures apply. The first is **atomics**: each block adds its partial $\sum x_i$ and $\sum x_i^2$ into two global accumulators with `atomicAdd`, then a one-thread finalize kernel divides by $N$ and computes $\sigma^2 = \tfrac{1}{N}\sum x_i^2 - \mu^2$. Doing one atomic per block (rather than one per thread) keeps contention negligible. The second is a **two-kernel** scheme: write each block's partials to a scratch array, then launch a single block to reduce that array. Both finish the same way; the atomic form is simpler when the partial count is modest.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A correct shared-memory tree reduction is already close to the bandwidth ceiling because the access is coalesced and the kernel is memory-bound. The remaining headroom is in the reduction itself: the final 32 elements of the in-block tree can be collapsed with **warp-shuffle** intrinsics, which exchange values directly between lanes through registers and need no `__shared__` memory or `__syncthreads()` for the last six steps. This removes synchronization overhead and shared-memory bank traffic from the tail of every block's reduction.</span>

<span style="font-size: 14px;">A second refinement is a grid-stride loop so each thread folds several elements into its registers before the tree begins, raising the work-per-thread and amortizing launch overhead for large $N$.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $x = [2, 4, 6, 8]$, $N = 4$. One block of 4 threads loads the values and maintains two accumulators.</span>

* <span style="font-size: 14px;">**Per-thread**: the four threads hold $(x, x^2)$ pairs $(2,4)$, $(4,16)$, $(6,36)$, $(8,64)$.</span>
* <span style="font-size: 14px;">**Tree reduction** (2 steps): stride 2 gives $(8, 40)$ and $(12, 100)$; stride 1 gives $(20, 140)$. So $\sum x = 20$, $\sum x^2 = 140$.</span>
* <span style="font-size: 14px;">**Finalize**: $\mu = 20/4 = 5$; $\sigma^2 = 140/4 - 5^2 = 35 - 25 = 10$.</span>

<span style="font-size: 14px;">The two scalars fall out of one sweep and a $\log_2 4 = 2$-step collapse, with no second pass over the data.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Catastrophic cancellation in the identity.** $\sigma^2 = \mathbb{E}[x^2] - \mu^2$ subtracts two large, nearly equal numbers when the data has a large mean and small variance, losing float32 precision. For such inputs a shifted or Welford-style accumulation is more stable than the raw identity.</span>
* <span style="font-size: 14px;">**Using sample instead of population variance.** This kernel divides by $N$, not $N-1$. Mixing up the divisor silently produces a result off by a factor of $N/(N-1)$, which the reference tolerance will reject.</span>
* <span style="font-size: 14px;">**Per-thread atomics instead of per-block.** Having every thread `atomicAdd` into the two global accumulators serializes the whole kernel on contention; reduce within the block first and emit one atomic pair per block.</span>
* <span style="font-size: 14px;">**Float accumulation order.** The parallel tree sums in a different order than a serial reference, so results differ in the last bits; this is expected and is why comparisons use a tolerance rather than bit-exactness.</span>

---