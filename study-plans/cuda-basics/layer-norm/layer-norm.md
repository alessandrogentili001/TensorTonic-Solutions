# <span style="font-size: 20px;">Layer Normalization</span>

<span style="font-size: 14px;">Layer normalization standardizes each row of a matrix to zero mean and unit variance, then applies a learned scale and shift. From a systems angle it is a **row-parallel normalization**: every output in a row depends on the row's mean and variance, which are global properties of that row. That coupling forces a reduction across the row before any element can be written, so the kernel cannot be a simple map.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For row $i$ and column $j$ of an $M \times N$ matrix, with row mean $\mu_i$, row variance $\sigma_i^2$, and stability constant $\epsilon$:</span>

$$
\text{output}[i,j] = \frac{x[i,j] - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \cdot \gamma[j] + \beta[j]
$$

<span style="font-size: 14px;">The input is a contiguous, row-major $M \times N$ buffer of 32-bit floats; `gamma` and `beta` are length-$N$ vectors reused across all $M$ rows, and `eps` is a scalar. Each row is normalized independently of the others, which is the structural fact that drives the decomposition.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">The standard mapping is **one block per row**: block $i$ owns row $i$, and the $M$ rows map to a grid of $M$ blocks that run independently with no cross-block communication. Within a block, threads split the $N$ columns among themselves with `idx = threadIdx.x` and a stride of `blockDim.x`, so a row wider than the block is swept in chunks. The block is sized as a multiple of the 32-lane **warp** (256 or 512) to waste no lanes and give the scheduler warps for latency hiding.</span>

<span style="font-size: 14px;">A single thread cannot normalize an element because $\mu_i$ and $\sigma_i^2$ are functions of the entire row, not of the one element the thread holds. The kernel therefore runs in passes separated by `__syncthreads()`: a reduction to accumulate statistics, then a fused normalize-and-affine write. The reduction is cooperative; the write is embarrassingly parallel once the two scalars are known.</span>

<span style="font-size: 14px;">A key trick collapses two reductions into one. Rather than reducing the sum, computing the mean, then reducing the squared deviations, the kernel accumulates both $\sum x$ and $\sum x^2$ in a single sweep and recovers variance as $\mathbb{E}[x^2] - \mu^2$. Each thread folds its strided slice into a local `(sum, sumsq)` pair, the partials go to `__shared__` memory, and a **tree reduction** halves the active lanes each step - combining `shared[t]` with `shared[t + stride]` for `stride` running $N/2, \ldots, 1$ - finishing in $\log_2(\text{blockDim.x})$ steps for both quantities at once.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The row is read from global memory at least twice: once during the statistics reduction and once during the normalize pass. Consecutive threads of a warp read consecutive columns on each strided step, so loads are fully **coalesced** and the memory controller serves each warp request in the minimum number of transactions. The `gamma` and `beta` vectors are read once per row from global memory and are prime candidates for caching, since all $M$ blocks reuse the same length-$N$ data.</span>

<span style="font-size: 14px;">The reduced statistics live in `__shared__` memory during the tree reduction and are broadcast to every thread once final, which is roughly an order of magnitude lower latency than re-reading DRAM. An optimization keeps each thread's loaded `x` values in registers between the statistics pass and the write pass, so the row is read once from global instead of twice - trading register pressure for one fewer DRAM sweep, the same memory-traffic decision that recurs across all row normalizations.</span>

<span style="font-size: 14px;">Because `gamma` and `beta` are read-only and shared by every block, they are ideal for the cache: the first few blocks pull them through L2 and later blocks hit it instead of DRAM, so their cost amortizes toward zero as $M$ grows. The matrix payload, by contrast, is touched exactly once per element and cannot be amortized - it sets the bandwidth floor.</span>

---

## Memory-bound or compute-bound? Arithmetic intensity

<span style="font-size: 14px;">Per element the kernel moves about 8 bytes of global memory (one 4-byte load, one 4-byte store) plus the amortized `gamma`/`beta` traffic, and performs a small fixed number of FLOPs - an add and a multiply-add in the statistics pass, then a subtract, a multiply by the reciprocal standard deviation, a scale, and a shift in the write. That is a handful of FLOPs per byte:</span>

$$
\frac{\sim 6 \text{ FLOPs}}{8 \text{ bytes}} \approx 0.75 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline** the ridge point sits in the tens of FLOPs per byte, so layer norm is firmly **memory-bound** at scale. The runtime is governed by how fast each row streams from DRAM, so the optimizations that matter are reducing global passes and keeping loads coalesced - not cheaper arithmetic. Fusing the affine scale-and-shift into the normalize pass is itself a bandwidth win, because it avoids a separate kernel that would re-read the whole matrix.</span>

---

## <span style="font-size: 16px;">Execution-Model Details</span>

<span style="font-size: 14px;">The `__syncthreads()` between the statistics reduction and the write is load-bearing: the write pass must not start until $\mu_i$ and $\sigma_i^2$ are final, or threads would normalize against partial sums. The barrier must sit at a uniform point all lanes reach, because a `__syncthreads()` inside divergent control flow hangs the block - lanes that skip it leave the others waiting forever.</span>

<span style="font-size: 14px;">As the tree-reduction stride shrinks, active lanes halve each step until the final steps run inside one warp where the 32 lanes execute in lockstep. The reduction is a **bank-conflict** concern: shared memory has 32 banks, and a stride that lands several active lanes on one bank serializes them, so the halving layout keeps accesses conflict-free. Packing `sum` and `sumsq` into a small struct or two parallel shared arrays lets both statistics reduce in one barrier sequence.</span>

<span style="font-size: 14px;">With **one block per row**, **occupancy** depends on the shared-memory budget of the reduction buffers and the per-thread register count. The block must hold enough resident warps that the scheduler always has a ready warp to issue while loads of the row are outstanding - this **latency hiding** is what keeps the memory pipeline saturated when no single warp can proceed.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A naive kernel makes three global passes - one to reduce the sum for the mean, a second to reduce squared deviations for the variance, a third to write - re-reading the row from DRAM each time. The optimized version cuts traffic in two ways:</span>

<span style="font-size: 14px;">1. **One-pass statistics**: accumulate $\sum x$ and $\sum x^2$ together and recover variance as $\mathbb{E}[x^2] - \mu^2$, collapsing two reductions into one sweep and removing a full global read.</span>

<span style="font-size: 14px;">2. **Warp-shuffle reductions and caching**: replace the shared-memory tree with `__shfl_down_sync` for the final warp, and keep loaded `x` in registers so the write reuses them - leaving a single global read of the row plus one write.</span>

<span style="font-size: 14px;">Both shrink the constant in front of the same memory-bound runtime. The structural lesson holds: layer norm needs the row's statistics before any element can be normalized, so it is inherently a reduce-then-write, never a single map.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take one row $x = [2, 4, 4, 6]$ with $N = 4$, $\gamma = [1,1,1,1]$, $\beta = [0,0,0,0]$, $\epsilon \approx 0$, in a block of 4 threads:</span>

* <span style="font-size: 14px;">**Statistics pass**: the tree reduces $\sum x = 16$ and $\sum x^2 = 4+16+16+36 = 72$ in $\log_2 4 = 2$ steps. Then $\mu = 16/4 = 4$ and $\sigma^2 = 72/4 - 4^2 = 18 - 16 = 2$.</span>
* <span style="font-size: 14px;">**Write pass**: each thread computes $(x[j] - 4)/\sqrt{2}$, giving $\approx [-1.41, 0, 0, 1.41]$, then applies the identity affine. The result has mean $0$ and unit variance.</span>

<span style="font-size: 14px;">Computing variance from $\mathbb{E}[x^2] - \mu^2$ let the block gather both moments in one reduction rather than waiting for $\mu$ before a second sweep - the single-pass saving that motivates the fused statistics.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting `eps` under the square root.** A row with near-zero variance produces a divide-by-zero and `inf`/`NaN` outputs; the $\epsilon$ inside the `sqrt` is mandatory for stability.</span>
* <span style="font-size: 14px;">**Missing `__syncthreads()` before the write.** Normalizing against partial sums before the statistics reduction completes reads stale shared memory and yields nondeterministic wrong results.</span>
* <span style="font-size: 14px;">**Catastrophic cancellation in $\mathbb{E}[x^2] - \mu^2$.** When the mean is large relative to the variance, subtracting two big numbers loses precision; the convenience of one-pass statistics carries a numerical cost.</span>
* <span style="font-size: 14px;">**Bank conflicts in the reduction.** A shared-memory stride mapping several active lanes onto one of the 32 banks serializes those accesses and slows every reduction step.</span>

---