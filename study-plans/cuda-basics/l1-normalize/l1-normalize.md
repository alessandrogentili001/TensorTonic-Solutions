# <span style="font-size: 20px;">L1 Normalize</span>

<span style="font-size: 14px;">L1 normalization divides every element of an array by the sum of the absolute values of all elements, so the result sums in magnitude to one. It is a **reduce-then-map**: phase one reduces the array to a single scalar denominator, phase two maps that scalar back across every element. The defining systems feature is the dependency between the phases - no element can be divided until the full sum is known, which forces the array to be read twice and a scalar to be broadcast to every thread. Unlike a fused multiply-reduce, the two patterns here cannot collapse into one pass, because the map's input is a global property the reduction has not finished producing.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an input of length $N$, each output is the input scaled by the reciprocal of the L1 norm:</span>

$$
\text{output}[i] = \frac{\text{input}[i]}{\sum_{j=0}^{N-1} |\text{input}[j]|}
$$

<span style="font-size: 14px;">The input and output are contiguous row-major buffers of $N$ 32-bit floats in global memory. The denominator is a single scalar that depends on every element, so it is a global property of the array, not a per-element quantity.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">The kernel decomposes into two patterns run in sequence. Phase one is a **reduction**: each thread loads `input[idx]` with `idx = blockIdx.x * blockDim.x + threadIdx.x`, takes its absolute value with `fabsf`, and contributes that to a **tree reduction** that collapses $N$ values to the L1 norm in $\log_2 N$ steps per block, with block partials combined across blocks. Phase two is an **embarrassingly parallel map**: each thread reloads `input[idx]` and writes `input[idx] * inv_norm`.</span>

<span style="font-size: 14px;">A block size of 256 is conventional - a multiple of the 32-lane **warp**, enough warps per **SM (Streaming Multiprocessor)** for **occupancy**. The phases cannot share a single launch cleanly because the map needs the finished global sum: the reduction must complete across all blocks before any division starts. The standard structure is therefore two kernel launches, with the launch boundary acting as the global barrier that guarantees the denominator is ready.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The array is read twice - once in the reduction, once in the map - and written once. Every access is at consecutive `idx` across a warp, so all loads and stores are fully **coalesced**. The cost of the dependency is concrete: the input streams out of DRAM twice, so global traffic is roughly $3N$ floats (two reads, one write) rather than the $2N$ a single-pass kernel would move. That extra read is not waste to be optimized away - it is the structural price of needing the complete sum before the first division, and it is why the kernel is classified as two passes over the data.</span>

<span style="font-size: 14px;">Partial sums for the reduction live in `__shared__` memory, an order of magnitude lower latency than global, with `__syncthreads()` between tree levels. The tree pairs `s` with `s + stride` and halves the stride each step, keeping surviving lanes contiguous to avoid **bank conflicts** - shared memory splits into 32 banks, and a stride that maps many lanes onto one bank serializes accesses that were meant to run in parallel.</span>

<span style="font-size: 14px;">The finished denominator is a single scalar; in the map phase every thread needs it, so it is **broadcast** from a small global scratch location (or constant memory) and read identically by all lanes. A warp reading one address is the broadcast case - it costs a single transaction, not 32, so the scalar is effectively free to distribute. The structural asymmetry is the point: phase one collapses $N$ values to one, phase two fans that one value back out to $N$ threads.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Across both phases the kernel moves about 12 bytes per element ($3N$ floats) and performs a handful of FLOPs - an absolute value and an add in phase one, one multiply in phase two. Its **arithmetic intensity** is on the order of:</span>

$$
\frac{\sim 3 \text{ FLOP}}{12 \text{ bytes}} \approx 0.25 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">That places it far under the **roofline** ridge point, so L1 normalize is firmly **memory-bound**. The reciprocal is computed once per array and the per-element arithmetic is trivial; runtime is set by DRAM bandwidth and the unavoidable double read. The optimization story is entirely about not wasting bandwidth: coalescing (already optimal) and enough warps to hide latency.</span>

---

## <span style="font-size: 16px;">Cross-Block Combine and Atomics</span>

<span style="font-size: 14px;">The reduction's per-block partial sums must be combined across blocks before the map can run. A **two-kernel combine** writes one partial per block to scratch and sums them, then a third launch (or the second launch's map) divides. Alternatively each block does one `atomicAdd` of its partial into a single global accumulator, keeping contention to `gridDim.x` writers instead of $N$. Either way the combine must finish globally before any thread divides, which is why the map cannot be folded into the same launch as the reduction.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">Both phases are streaming and hide global-memory latency through **massive multithreading**: a stalled warp yields to another resident warp, and high occupancy keeps the memory pipeline saturated. The map phase has no synchronization and no divergence - every active lane does one multiply - so it runs at full streaming throughput. The reduction phase has the usual short tail where the active set falls below a warp, addressed by warp-shuffle.</span>

<span style="font-size: 14px;">The inter-phase dependency, not occupancy, is the structural cost. The launch boundary serializes the two phases globally, so the kernel cannot overlap the tail of the reduction with the start of the map; the sum must be complete and visible to every block first.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A naive implementation might recompute the norm inside every thread, reading the whole array $N$ times for $O(N^2)$ traffic - catastrophic. The correct structure computes the norm once in a reduction and reuses it, cutting that to two reads. On top of that, the reduction's final intra-warp levels use **warp-shuffle** (`__shfl_down_sync`) to drop shared-memory round trips, and both phases can use vectorized `float4` loads to widen transactions. The denominator broadcast is already optimal as a single cached read.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take `input = [1, -3, 2, -2]`, $N = 4$, in one block.</span>

* <span style="font-size: 14px;">**Phase 1, reduce:** each lane takes `fabsf`: $[1, 3, 2, 2]$. Tree step stride 2 gives $[3, 5]$, stride 1 gives $[8]$. The L1 norm is $8$; the reciprocal $1/8 = 0.125$ is written to scratch.</span>
* <span style="font-size: 14px;">**Phase 2, map:** the second launch reads the denominator once and divides: $[0.125, -0.375, 0.25, -0.25]$.</span>

<span style="font-size: 14px;">The magnitudes sum to $1$. Note the array was read in full during phase one and again during phase two; the scalar $0.125$ was broadcast identically to all four lanes. A single-pass attempt would fail here: the lane holding `input[0]` would have to divide before it had seen `input[3]`, so its denominator would be wrong.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Dividing before the global sum is ready.** Folding map into the reduction launch lets a block divide by a partial sum; the denominator must be complete across all blocks, which is what the launch boundary guarantees.</span>
* <span style="font-size: 14px;">**Recomputing the norm per thread.** Reading the whole array inside every thread turns $O(N)$ traffic into $O(N^2)$; compute the scalar once and broadcast it.</span>
* <span style="font-size: 14px;">**Atomic contention on the accumulator.** Many threads `atomicAdd`-ing one address serializes; reduce within the block first, one atomic per block.</span>
* <span style="font-size: 14px;">**Division by zero on an all-zero input.** The norm can be $0$; guard the reciprocal or accept the defined edge-case behavior rather than emitting NaNs.</span>

---