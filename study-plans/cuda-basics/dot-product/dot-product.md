# <span style="font-size: 20px;">Dot Product</span>

<span style="font-size: 14px;">The dot product multiplies two arrays element by element and sums the products into a single scalar. It is a **fused map-then-reduce**: a pointwise multiply (the map) feeding directly into a sum (the reduction), with no intermediate product array ever written to memory. Fusing the two patterns is the whole point - keeping the products in registers between the multiply and the add is what makes one pass over two input streams sufficient. Treating it as a map followed by a separate reduce would be correct but would double the global memory traffic, which is exactly the wrong move on a bandwidth-bound kernel.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For two inputs $A$ and $B$ of length $N$, the kernel produces one float:</span>

$$
\text{result} = \sum_{i=0}^{N-1} A[i] \cdot B[i]
$$

<span style="font-size: 14px;">Both inputs are contiguous row-major buffers of $N$ 32-bit floats in device global memory; the output is a single float. There are two input streams instead of one, but they are read in lockstep at the same index, so the access pattern of each is identical to a plain reduction.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Each thread owns one index. It loads `A[idx]` and `B[idx]` with `idx = blockIdx.x * blockDim.x + threadIdx.x`, computes the product, and contributes that product as its leaf in a **tree reduction**. The map half is **embarrassingly parallel** - every product is independent - but the reduce half forces cooperation, because no single thread sees all $N$ products. Each block reduces its chunk to one partial sum in $\log_2 N$ steps, then the partials are combined.</span>

<span style="font-size: 14px;">A block size of 256 is conventional: a multiple of the 32-lane **warp** so no lanes are wasted, and enough warps per **SM (Streaming Multiprocessor)** to keep **occupancy** high. With 256 threads per block the grid needs $\lceil N / 256 \rceil$ blocks. The tail block has threads with `idx >= N`; those guarded lanes contribute a product of zero, the additive identity, so the bounds check resolves to a neutral element rather than reading out of bounds.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The kernel reads two global arrays and writes one scalar, so it moves 8 bytes per element loaded. Both streams are read with consecutive `idx` values across a warp, so each is fully **coalesced** - 32 lanes touch 32 consecutive addresses of `A` and 32 of `B`, served in the minimum number of transactions. Each element of each input is read exactly once and never returns to DRAM.</span>

<span style="font-size: 14px;">The product lives only in a register; there is no reuse to cache, so no product array is materialized. The partial sums then live in `__shared__` memory during the in-block tree, roughly an order of magnitude lower latency than global memory, with a `__syncthreads()` between levels so each lane reads the previous step's writes safely. Fusion means the 8 bytes of input traffic per element buy one multiply and one add with zero extra memory writes for the intermediate.</span>

<span style="font-size: 14px;">The stride pattern in the tree matters for shared memory: pairing `s` with `s + stride` and halving the stride each step keeps the surviving lanes contiguous, which avoids **bank conflicts** - shared memory splits into 32 banks, and a tree that strided by a power of two into the same bank would serialize the accesses it was meant to parallelize.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the kernel moves 8 bytes (two 4-byte loads) and performs one multiply plus one add, so two FLOPs. Its **arithmetic intensity** is:</span>

$$
\frac{2 \text{ FLOP}}{8 \text{ bytes}} = 0.25 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">That is far below the **roofline** ridge point of tens of FLOPs per byte, so the dot product is firmly **memory-bound**. The multiply-add is essentially free; runtime is governed by how fast both input streams arrive from DRAM. The only optimizations that matter raise effective bandwidth - coalescing (already optimal) and enough warps in flight to hide latency - not arithmetic. A single fused multiply-add instruction (`fma`) already does the per-element work in one op, so there is nothing left to shave on the compute side.</span>

---

## <span style="font-size: 16px;">Cross-Block Combine and Atomics</span>

<span style="font-size: 14px;">A kernel launch cannot synchronize across blocks, so each block's partial sum must be combined separately. Two standard mechanisms apply. A **two-kernel combine** writes one partial per block to a small scratch buffer and a second launch sums those few values, using the kernel boundary as a global barrier. Alternatively each block does one `atomicAdd` of its partial into a single global accumulator, which keeps contention to `gridDim.x` writers instead of $N$. The block-local tree first, one atomic per block second, is the standard pattern; many threads atomically adding to one address would serialize and destroy throughput.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global load costs hundreds of cycles, and this kernel issues two of them per thread. The GPU hides that latency with **massive multithreading**: when a warp issues its loads of `A[idx]` and `B[idx]` and stalls, the SM scheduler runs another resident warp. The two independent loads can also be in flight simultaneously, so the latency of the second is hidden behind the first within the same thread. High occupancy keeps the memory pipeline saturated; for the load phase the kernel behaves like a streaming map and occupancy is set simply by launching enough blocks.</span>

<span style="font-size: 14px;">The reduce phase shrinks the active set each step, so once the stride drops below 32 the warp runs mostly idle lanes - a short tail of $\log_2$ of the block size that the warp-shuffle optimization targets directly. There is no **warp divergence** in the body: every active lane takes the same multiply-then-add path, and the only branch is the bounds check at the boundary block.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A naive approach computes a separate product array in global memory and then reduces it in a second kernel. That writes and re-reads $N$ floats of intermediate traffic for nothing, roughly doubling global memory volume. The fused kernel keeps each product in a register and feeds it straight into the tree, cutting that intermediate traffic entirely - the central win of map-reduce fusion.</span>

<span style="font-size: 14px;">On top of fusion, the optimized reduction replaces the final intra-warp tree levels with **warp-shuffle**: once the active set fits in one warp, lanes add their partials directly through registers with `__shfl_down_sync`, eliminating the shared-memory round trips and `__syncthreads()` for the tail. Vectorized `float4` loads can also widen each memory transaction, pushing the two input streams closer to the bandwidth ceiling.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 8$ in one block with $A = [1, 2, 3, 4, 5, 6, 7, 8]$ and $B = [1, 1, 1, 1, 1, 1, 1, 1]$.</span>

* <span style="font-size: 14px;">**Map:** each lane computes its product into a register: $[1, 2, 3, 4, 5, 6, 7, 8]$. No product array is written to global memory.</span>
* <span style="font-size: 14px;">**Reduce step 1 (stride 4):** lane $i$ adds lane $i+4$: $[6, 8, 10, 12]$.</span>
* <span style="font-size: 14px;">**Reduce step 2 (stride 2):** $[16, 20]$.</span>
* <span style="font-size: 14px;">**Reduce step 3 (stride 1):** $[36]$.</span>

<span style="font-size: 14px;">The result $36$ is reached in $\log_2 8 = 3$ synchronized steps after a single fused multiply per element. Each step is separated by `__syncthreads()`; the multiply never left the register file.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Materializing the product array.** Writing products to global memory and reducing in a second pass doubles memory traffic on a memory-bound kernel; keep the product in a register and fuse.</span>
* <span style="font-size: 14px;">**Wrong tail padding.** Guarded lanes with `idx >= N` must contribute a product of zero; reading past the buffer or adding garbage corrupts the sum.</span>
* <span style="font-size: 14px;">**Atomic contention on the accumulator.** Every thread `atomicAdd`-ing one global address serializes; reduce within the block first, then one atomic per block.</span>
* <span style="font-size: 14px;">**Float accumulation order.** The parallel tree sums products in a different order than a serial reference, so bit-exact equality fails; this is why tolerances exist.</span>

---