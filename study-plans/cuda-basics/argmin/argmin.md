# <span style="font-size: 20px;">Argmin</span>

<span style="font-size: 14px;">Argmin returns the index of the smallest element of an array rather than the value itself, breaking ties toward the lowest index. It is a **reduction**, but a special one: the quantity being compared (the value) is not the quantity being returned (the index). This makes it the canonical **payload reduction** - the tree does not collapse scalars, it collapses `(value, index)` pairs, comparing on the value with a `<` comparator while carrying the index along for the ride. Everything that makes a plain min reduction work still applies, but every combine step now moves twice the data and obeys a tie rule, which is where the systems interest lives.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an input of $N$ floats, the kernel produces a single integer:</span>

$$
\text{result} = \arg\min_{0 \le i < N} \text{input}[i]
$$

<span style="font-size: 14px;">The input is a contiguous row-major buffer of $N$ 32-bit floats in global memory; the output is one 32-bit int. The tie rule is fixed: when two elements are equal, the smaller index wins. That rule is not decoration - it is what forces the combine operator to be carefully ordered, because a naive comparison can silently pick either index on a tie.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">No single thread can see all $N$ elements, so the work collapses in stages. The kernel maps the array onto a grid where each block reduces a contiguous chunk to one local winner, then the per-block winners are combined into the global answer. Within a block the reduction is a **tree reduction**: $N$ candidates collapse to one in $\log_2 N$ parallel steps, each step halving the number of active lanes.</span>

<span style="font-size: 14px;">The unit of work is one pair per thread. Each thread loads its element with `idx = blockIdx.x * blockDim.x + threadIdx.x`, forms the pair `(input[idx], idx)`, and stages it. A block size of 256 is conventional - a multiple of the 32-lane **warp** so no lanes are wasted, and enough warps per **SM (Streaming Multiprocessor)** to keep **occupancy** high. With 256 threads per block the grid needs $\lceil N / 256 \rceil$ blocks to cover the array.</span>

<span style="font-size: 14px;">When the grid rounds up, the tail block has threads whose `idx` runs past $N$. Those surplus lanes must not contribute a real candidate, so they seed a **sentinel pair**: a value of positive infinity and an out-of-range index. Because the comparator is now `<`, the neutral element flips to $+\infty$ so the sentinel always loses. The guarded tail threads still participate in the tree shape without polluting the result. This is the payload-reduction equivalent of the simple `if (idx < N)` bounds check on a map - the bounds check still exists, it just resolves to a neutral element instead of an early exit.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Each input element is read from global memory exactly once, and the 32 threads of a warp read 32 consecutive addresses, so the loads are fully **coalesced** - the best case the memory controller offers. After the initial load the data never returns to DRAM.</span>

<span style="font-size: 14px;">The pairs live in `__shared__` memory for the in-block tree, which is roughly an order of magnitude lower latency than global memory. A payload reduction needs two `__shared__` arrays, one for values and one for indices, because each combine step must move both halves of the pair together. That doubles the per-block shared-memory footprint relative to a scalar min reduction, which matters because shared memory is one of the resources that caps occupancy: a fatter per-block allocation means fewer blocks resident on an SM at once.</span>

<span style="font-size: 14px;">Each tree step is separated by `__syncthreads()` so that every lane sees the previous step's writes before it reads them. The stride pattern matters too: pairing `s` with `s + stride` and halving the stride each step keeps the surviving lanes contiguous, which avoids **bank conflicts** in shared memory and keeps the active warps packed rather than scattered across the SIMT lanes.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the kernel moves 4 bytes from global memory and performs roughly one comparison. Its **arithmetic intensity** is on the order of:</span>

$$
\frac{1 \text{ compare}}{4 \text{ bytes}} \approx 0.25 \text{ op/byte}
$$

<span style="font-size: 14px;">That sits far under the **roofline** ridge point, so argmin is firmly **memory-bound**. The comparison logic and the index bookkeeping are nearly free; the runtime is set by how fast the input streams out of DRAM. The only levers that matter are coalesced access and enough warps in flight to hide global-memory latency, not the cleverness of the comparator.</span>

---

## <span style="font-size: 16px;">Cross-Block Combine and Atomics</span>

<span style="font-size: 14px;">A single kernel launch cannot synchronize across blocks, so the per-block winners must be combined separately. There are two standard mechanisms.</span>

<span style="font-size: 14px;">1. **Two-kernel combine**: the first kernel writes one `(value, index)` pair per block to a small global scratch buffer; a second launch reduces those few pairs to the final answer. The kernel boundary acts as a global barrier, which is the clean way to combine partials when the payload is a pair.</span>

<span style="font-size: 14px;">2. **Atomic combine**: a single `atomicMin`-style operation will not preserve the index, so a pair reduction instead packs value and index into one 64-bit word and uses a CAS loop, or uses one atomic per block to serialize only the handful of cross-block writers. Many threads contending on one address serialize, so the block-local tree first, one atomic per block second, keeps contention to `gridDim.x` writers instead of $N$.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global-memory load costs hundreds of cycles. The GPU hides that latency not with caches but with **massive multithreading**: when a warp issues its load of `input[idx]` and stalls, the SM scheduler switches to another resident warp. High occupancy means there is always another warp ready, so the memory pipeline stays saturated while individual loads are in flight. For the initial load phase, argmin behaves exactly like a streaming map - one coalesced read per thread - and occupancy is set simply by launching enough blocks.</span>

<span style="font-size: 14px;">The tree phase is different. As the stride shrinks, fewer lanes stay active, and once the active set drops below 32 the warp runs mostly idle lanes. This tail is short - only $\log_2$ of the block size - but it is why the warp-shuffle optimization below targets exactly those final levels. There is no **warp divergence** in the body, since every active lane takes the same comparison path; the only branch is the tie resolution, which is a branchless select on the index in a good implementation.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive shared-memory tree is correct but leaves the last few steps running with most lanes idle and pays a `__syncthreads()` per level. The optimized form replaces the final intra-warp levels with **warp-shuffle**: once the active set fits in one warp, lanes exchange their pair directly through registers with `__shfl_down_sync`, dropping the synchronization and the shared-memory round trips for the tail of the tree. The payload makes this slightly heavier than a scalar reduction because two values shuffle per step, but the structure is identical and still collapses in $\log_2 N$ steps.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take 8 elements `input = [5, 2, 2, 7, 6, 2, 8, 9]` reduced in one block, comparing pairs at stride 4, then 2, then 1. Each step keeps the smaller value, and on a tie keeps the lower index.</span>

* <span style="font-size: 14px;">**Step 1 (stride 4):** pairs $(0,4),(1,5),(2,6),(3,7)$. Lane 0 compares value 5 vs 6, keeps `(5, idx 0)`. Lane 1 compares 2 vs 2 - a tie, so it keeps the lower index `(2, idx 1)`. Lane 2 keeps `(2, idx 2)`, lane 3 keeps `(7, idx 3)`.</span>
* <span style="font-size: 14px;">**Step 2 (stride 2):** lane 0 compares `(5, idx 0)` vs `(2, idx 2)`, keeps `(2, idx 2)`. Lane 1 keeps `(2, idx 1)`.</span>
* <span style="font-size: 14px;">**Step 3 (stride 1):** lane 0 compares `(2, idx 2)` vs `(2, idx 1)` - a tie, keeps the lower index `(2, idx 1)`.</span>

<span style="font-size: 14px;">Result is index 1. The lesson: the combine must compare values but resolve ties on the index, or the answer drifts to whichever lane happened to write last.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Reducing values and dropping the index.** A plain `fminf` tree finds the minimum value but loses which lane held it; the pair must travel together through every step.</span>
* <span style="font-size: 14px;">**Wrong sentinel sign.** Padding the tail with $-\infty$ instead of $+\infty$ makes a phantom out-of-range lane win the min; the neutral element must be the largest representable value.</span>
* <span style="font-size: 14px;">**Inconsistent tie-breaking.** If the combine uses `<` in some steps and `<=` in others, equal values resolve to different indices nondeterministically; use a strict `<` on value with a lower-index preference everywhere.</span>
* <span style="font-size: 14px;">**Atomic contention on the global winner.** Every thread doing a CAS on one address serializes; reduce within the block first and contend with only `gridDim.x` writers.</span>

---