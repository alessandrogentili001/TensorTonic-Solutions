# <span style="font-size: 20px;">Matrix Multiplication (Naive GEMM)</span>

<span style="font-size: 14px;">Matrix multiplication computes $C[i,j] = \sum_k A[i,k]\, B[k,j]$, the **GEMM** at the heart of every dense neural-network layer. It is the textbook case where the naive parallelization is correct but slow: one thread per output cell loops over the shared dimension, rereading whole rows of $A$ and columns of $B$ straight from global memory. The arithmetic is plentiful, but the kernel spends its time fetching the same operands over and over, so it lands far under the compute roofline. This entry is about why the obvious kernel is **memory-bound**, and what the intensity argument says is being wasted.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For output indices $i$ in $[0, M)$ and $j$ in $[0, N)$:</span>

$$
C[i,j] = \sum_{k=0}^{K-1} A[i,k]\, B[k,j]
$$

<span style="font-size: 14px;">$A$ is $M \times K$ row-major, $B$ is $K \times N$ row-major, $C$ is $M \times N$. Element $C[i,j]$ is the inner product of row $i$ of $A$ (offsets `i*K + k`) with column $j$ of $B$ (offsets `k*N + j`). The total work is $2MNK$ FLOPs, which sounds compute-heavy and is exactly why the bandwidth bottleneck is so instructive.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">The decomposition is **one thread per output element** of $C$, a 2-D grid of `16x16` blocks tiling the $M \times N$ output. Each thread derives its target cell from block and lane indices:</span>

$$
j = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}, \quad i = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}
$$

<span style="font-size: 14px;">Thread $(i,j)$ runs a length-$K$ loop, multiplying `A[i*K+k]` by `B[k*N+j]` and accumulating into a register, then writes the single `C[i*N+j]`. A `16x16` block of 256 threads is a multiple of the 32-lane **warp** and packs many blocks per **SM (Streaming Multiprocessor)** for **occupancy**. Edge blocks overhang non-multiple shapes, so a 2-D bounds check guards the body. There is no `__syncthreads()` in the naive kernel: each output is computed in isolation.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The defining flaw is **redundant global traffic**. Every thread reads an entire row of $A$ and an entire column of $B$ from global memory, and nothing is cached or shared between threads. Across the block, the 256 threads of a `16x16` tile that share a row of $A$ each fetch that same row independently, and threads sharing a column of $B$ each fetch that same column independently. Each input element is therefore reread $O(N)$ times (for $A$) or $O(M)$ times (for $B$) over the whole launch.</span>

<span style="font-size: 14px;">The access patterns also differ in quality. The $B$ access `k*N+j` is **coalesced**: at a fixed $k$, consecutive lanes (consecutive $j$) hit consecutive addresses. The $A$ access `i*K+k` is worse across a warp: lanes share $k$ but differ in $i$, so they stride by $K$ down a column of $A$, an **uncoalesced** pattern. Either way the deeper problem is volume, not just stride - the same bytes cross the DRAM bus dozens or hundreds of times because there is no on-chip reuse.</span>

<span style="font-size: 14px;">Registers hold only the running accumulator. `__shared__` memory, which exists precisely to let a block reuse loaded operands, is left completely unused. That omission is the entire performance story, and fixing it is what the tiled variant does.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per inner-product step a thread does one multiply and one add - 2 FLOPs - while loading two 4-byte operands, 8 bytes. The naive **arithmetic intensity** is therefore about:</span>

$$
\frac{2 \text{ FLOP}}{8 \text{ bytes}} = 0.25 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline**, compute-bound territory begins past the ridge point of tens of FLOPs per byte. At $0.25$ the naive GEMM sits two orders of magnitude below that line: it is firmly **memory-bound**. The multiply-add hardware - the very FLOPs the GPU exists to deliver - idles while the kernel waits on DRAM. A device might offer thousands of GFLOP/s of fused multiply-add, yet the naive kernel achieves a small fraction of it because every operand is fetched from global memory instead of reused on chip.</span>

<span style="font-size: 14px;">This is the central lesson: GEMM is intrinsically a high-FLOP operation ($2MNK$), so it *can* be compute-bound, but only if the operands are reused enough on chip to raise the intensity above the ridge. The naive kernel forfeits that reuse and leaves the FLOPs roofline badly underused. Raising arithmetic intensity, not adding adders, is the path forward.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">Each step's two global loads cost hundreds of cycles, hidden by **massive multithreading**: the SM scheduler runs other resident warps while one waits. High **occupancy** is therefore essential, since the kernel issues a torrent of dependent loads. But latency hiding only keeps the memory system busy - it cannot raise the intensity, so even at full occupancy the naive kernel remains pinned to the bandwidth ceiling.</span>

<span style="font-size: 14px;">There is no warp divergence - every active thread runs the same $K$-length loop - and no synchronization, so the only structural bottleneck is the redundant, partly uncoalesced global traffic.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel just described reads $A$ and $B$ entirely from global memory, rereading each element $O(M)$ or $O(N)$ times, for total input traffic on the order of $2MNK$ element-loads. That redundancy is what holds intensity at $0.25$.</span>

<span style="font-size: 14px;">The optimized counterpart stages `TILE x TILE` sub-blocks of $A$ and $B$ into `__shared__` memory so each loaded element is reused `TILE` times before being discarded. That single change raises arithmetic intensity by roughly a factor of `TILE` and is enough to push GEMM across the ridge into **compute-bound** operation. The mechanics - cooperative tile loads, two `__syncthreads()` per tile step, bank-conflict padding, and the resulting global-traffic reduction - are the subject of the tiled-matmul problem; here the point is only to see clearly what the naive kernel wastes.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = N = K = 2$ with a tiny launch of 4 threads, one per output cell. Consider the loads with no caching.</span>

* <span style="font-size: 14px;">**Thread (0,0)** reads $A$ row 0 (`A[0,0], A[0,1]`) and $B$ column 0 (`B[0,0], B[1,0]`).</span>
* <span style="font-size: 14px;">**Thread (0,1)** reads the **same** $A$ row 0 again, plus $B$ column 1.</span>
* <span style="font-size: 14px;">**Thread (1,0)** reads $A$ row 1, plus the **same** $B$ column 0 already fetched by thread (0,0).</span>

<span style="font-size: 14px;">Row 0 of $A$ is read twice and column 0 of $B$ is read twice, even at this trivial size; at full scale each row of $A$ is reread $N$ times and each column of $B$ is reread $M$ times. With $A = \begin{bmatrix}1&2\\3&4\end{bmatrix}$ and $B = \begin{bmatrix}5&6\\7&8\end{bmatrix}$ the kernel produces $C = \begin{bmatrix}19&22\\43&50\end{bmatrix}$ - correct, but at the cost of fetching the same operands repeatedly from DRAM.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Mistaking high FLOP count for compute-bound.** GEMM has $2MNK$ FLOPs, but at intensity $0.25$ the naive kernel is memory-bound; the fix is on-chip reuse, not faster arithmetic.</span>
* <span style="font-size: 14px;">**Uncoalesced $A$ access.** Lanes stride by $K$ down a column of $A$ at each step, fragmenting the load; the tiled form folds this into coalesced tile fills.</span>
* <span style="font-size: 14px;">**Leaving `__shared__` memory unused.** The redundant rereads of $A$ rows and $B$ columns are exactly the reuse shared memory exists to capture; skipping it caps intensity.</span>
* <span style="font-size: 14px;">**Integer overflow in `i*K+k` or `k*N+j`.** Large matrices push the row-major offset past 32-bit `int`; compute indices in `size_t`.</span>

---