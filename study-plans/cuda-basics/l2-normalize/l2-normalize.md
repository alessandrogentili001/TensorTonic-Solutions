# <span style="font-size: 20px;">L2 Normalize</span>

<span style="font-size: 14px;">L2 normalization divides every element of an array by its Euclidean norm, the square root of the sum of squares, so the result has unit length. It is a **reduce-then-map**: phase one reduces the array to a single scalar denominator, phase two maps that scalar back across every element. The defining systems feature is the dependency between the phases - no element can be scaled until the full sum of squares is known, which forces the array to be read twice and a scalar to be broadcast to every thread.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an input of length $N$, each output is the input scaled by the reciprocal of the L2 norm:</span>

$$
\text{output}[i] = \frac{\text{input}[i]}{\sqrt{\sum_{j=0}^{N-1} \text{input}[j]^2}}
$$

<span style="font-size: 14px;">The input and output are contiguous row-major buffers of $N$ 32-bit floats in global memory. The denominator is a single scalar that depends on every element, so it is a global property of the array, not a per-element quantity.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">The kernel decomposes into two patterns run in sequence. Phase one is a **reduction**: each thread loads `input[idx]` with `idx = blockIdx.x * blockDim.x + threadIdx.x`, squares it, and contributes that to a **tree reduction** that collapses $N$ squares to the sum of squares in $\log_2 N$ steps per block, with block partials combined across blocks. Phase two is an **embarrassingly parallel map**: each thread reloads `input[idx]` and writes `input[idx] * inv_norm`.</span>

<span style="font-size: 14px;">A block size of 256 is conventional - a multiple of the 32-lane **warp**, enough warps per **SM (Streaming Multiprocessor)** for **occupancy**. The phases cannot share a single launch cleanly because the map needs the finished global sum: the reduction must complete across all blocks before any scaling starts. The standard structure is therefore two kernel launches, with the launch boundary acting as the global barrier that guarantees the denominator is ready.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The array is read twice - once in the reduction, once in the map - and written once. Every access is at consecutive `idx` across a warp, so all loads and stores are fully **coalesced**. The cost of the dependency is concrete: the input streams out of DRAM twice, so global traffic is roughly $3N$ floats (two reads, one write) rather than the $2N$ a single-pass kernel would move. That extra read is the structural price of needing the complete sum of squares before the first scale.</span>

<span style="font-size: 14px;">Partial sums for the reduction live in `__shared__` memory, an order of magnitude lower latency than global, with `__syncthreads()` between tree levels. The tree pairs `s` with `s + stride` and halves the stride each step, keeping surviving lanes contiguous to avoid **bank conflicts** - shared memory splits into 32 banks, and a stride that maps many lanes onto one bank serializes accesses meant to run in parallel.</span>

<span style="font-size: 14px;">The finished denominator is a single scalar; in the map phase every thread needs it, so it is **broadcast** from a small global scratch location (or constant memory) and read identically by all lanes. A warp reading one address is the broadcast case - it costs a single transaction, not 32, so the scalar is effectively free to distribute.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Across both phases the kernel moves about 12 bytes per element ($3N$ floats) and performs a handful of FLOPs - a square and an add in phase one, one multiply in phase two, plus a single `rsqrtf` for the whole array. Its **arithmetic intensity** is on the order of:</span>

$$
\frac{\sim 3 \text{ FLOP}}{12 \text{ bytes}} \approx 0.25 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">That places it far under the **roofline** ridge point, so L2 normalize is firmly **memory-bound**. The reciprocal square root is computed once per array on the special-function unit and the per-element arithmetic is trivial; runtime is set by DRAM bandwidth and the unavoidable double read. Optimization is about not wasting bandwidth: coalescing (already optimal) and enough warps to hide latency.</span>

---

## <span style="font-size: 16px;">The Reciprocal Square Root</span>

<span style="font-size: 14px;">The denominator involves a square root, which a naive kernel would compute as `sqrtf` and then divide by. Both square root and division are expensive special-function operations. The standard trick is `rsqrtf`, which computes $1/\sqrt{x}$ directly in one special-function-unit instruction, turning the per-element division into a multiply by the precomputed reciprocal. Because the reciprocal is computed exactly once for the whole array, this is not a hot path - but it is the canonical way to fold the square root and the division into a single cheap broadcast scalar.</span>

---

## <span style="font-size: 16px;">Cross-Block Combine and Atomics</span>

<span style="font-size: 14px;">The reduction's per-block partial sums of squares must be combined across blocks before the map can run. A **two-kernel combine** writes one partial per block to scratch, sums them, applies `rsqrtf`, and a later launch scales. Alternatively each block does one `atomicAdd` of its partial into a single global accumulator, keeping contention to `gridDim.x` writers instead of $N$. Either way the combine must finish globally before any thread scales, which is why the map cannot be folded into the reduction's launch.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">Both phases are streaming and hide global-memory latency through **massive multithreading**: a stalled warp yields to another resident warp, and high occupancy keeps the memory pipeline saturated. The map phase has no synchronization and no **warp divergence** - every active lane does one multiply - so it runs at full streaming throughput. The reduction phase has the usual short tail where the active set falls below a warp, addressed by warp-shuffle. The launch boundary, not occupancy, is the structural cost: it serializes the two phases so the sum of squares is complete and visible to every block before any scaling begins.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A naive implementation might recompute the norm inside every thread, reading the whole array $N$ times for $O(N^2)$ traffic - catastrophic. The correct structure computes the sum of squares once in a reduction, applies one `rsqrtf`, and reuses the scalar, cutting traffic to two reads. On top of that, the reduction's final intra-warp levels use **warp-shuffle** (`__shfl_down_sync`) to drop shared-memory round trips, and both phases can use vectorized `float4` loads to widen transactions.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take `input = [3, 4]`, $N = 2$, in one block.</span>

* <span style="font-size: 14px;">**Phase 1, reduce:** each lane squares its element: $[9, 16]$. The tree adds them in one step: $9 + 16 = 25$. The norm is $\sqrt{25} = 5$; `rsqrtf(25)` gives the reciprocal $0.2$, written to scratch.</span>
* <span style="font-size: 14px;">**Phase 2, map:** the second launch reads the reciprocal once and multiplies: $[3 \times 0.2, 4 \times 0.2] = [0.6, 0.8]$.</span>

<span style="font-size: 14px;">The output has unit length: $0.6^2 + 0.8^2 = 1$. The array was read in full in phase one and again in phase two; the scalar $0.2$ was broadcast identically to both lanes.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Scaling before the global sum is ready.** Folding map into the reduction launch lets a block scale by a partial sum; the denominator must be complete across all blocks, which the launch boundary guarantees.</span>
* <span style="font-size: 14px;">**Using `sqrtf` then dividing per element.** That pays two special-function ops per element; compute `rsqrtf` once and multiply instead.</span>
* <span style="font-size: 14px;">**Atomic contention on the accumulator.** Many threads `atomicAdd`-ing one address serializes; reduce within the block first, one atomic per block.</span>
* <span style="font-size: 14px;">**Division by zero on an all-zero input.** The norm can be $0$ and `rsqrtf(0)` is infinite; guard the reciprocal or accept the defined edge-case behavior rather than emitting NaNs.</span>

---