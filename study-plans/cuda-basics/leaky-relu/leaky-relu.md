# Leaky ReLU

Leaky ReLU passes positives through unchanged and scales negatives by a small slope $\alpha$, $\text{output}[i] = \text{input}[i] > 0\ ?\ \text{input}[i] : \alpha \cdot \text{input}[i]$. It is an **embarrassingly parallel map**: every output depends on one input at the same index, with zero communication between threads. The systems interest is twofold - how the scalar $\alpha$ is supplied to every thread as a kernel argument, and how to express the two-case select without serializing a warp.

---

## The Operation

For an index $i$ in $[0, N)$ and a slope $\alpha$, the kernel evaluates:

$$
\text{output}[i] = \begin{cases} \text{input}[i] & \text{input}[i] > 0 \\ \alpha \cdot \text{input}[i] & \text{input}[i] \le 0 \end{cases}
$$

Input and output are contiguous, row-major buffers of $N$ 32-bit floats in device (global) memory; $\alpha$ is a single float scalar (commonly $0.01$). Output element $i$ reads only input element $i$ and writes only output element $i$. Nothing is shared, reused, or reordered.

---

## Parallelization Strategy

Because the $N$ outputs are mutually independent, the natural decomposition is **one thread per element**. A one-dimensional grid of one-dimensional blocks covers the array, and each thread reconstructs its global position:

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

A block size of 256 threads is conventional: it is a multiple of the 32-lane **warp** so no lanes are wasted, it gives the scheduler many warps per block for latency hiding, and many such blocks fit on one **SM (Streaming Multiprocessor)**, keeping **occupancy** high. The grid needs $\lceil N / 256 \rceil$ blocks.

The body is guarded by if (idx < N) because rounding the grid up leaves surplus tail threads; without the check they read and write past the buffers. There is no __syncthreads() and no shared state - the whole computation is one flat wave of independent work.

---

## The Scalar $\alpha$: Broadcast By Value

Alpha is passed by value as a kernel argument rather than as an array. Every thread receives the same scalar; the compiler and device manage its parameter access without a separate per-element input buffer.

---

## Memory Hierarchy and Access Pattern

The kernel touches two global arrays per element: it loads input[idx] and stores output[idx]. There is no reuse, so nothing belongs in __shared__ memory; the selected value lives in a register for its brief lifetime. The access pattern is ideal for **coalescing**: the 32 threads of a warp hold consecutive idx values, so they read 32 consecutive addresses of input and write 32 consecutive addresses of output, served in the minimum number of transactions. With $\alpha$ in a register, the entire memory footprint per element is just the 8 bytes of load and store.

---

## Memory-Bound or Compute-Bound?

Per element the kernel moves 8 bytes and performs a compare, a select, and a multiply - two or three FLOPs. The **arithmetic intensity** is about:

$$
\frac{\sim 3 \text{ FLOP}}{8 \text{ bytes}} \approx 0.375 \text{ FLOP/byte}
$$

On the **roofline** the ridge point sits in the tens of FLOPs per byte, so at $\approx 0.375$ Leaky ReLU is **deeply memory-bound**. The compare-select-multiply is nearly free; the only levers that change runtime are coalesced access (already optimal) and enough warps to hide DRAM latency. The runtime is just $8N$ bytes divided by achievable bandwidth.

---

## Predication vs Warp Divergence

A short conditional may compile to a predicated selection instead of a control-flow branch. Both if/else and ternary source forms can compile this way; the generated instructions determine whether a warp diverges.

The fmaxf alternative agrees with Leaky ReLU only when alpha is between zero and one. For larger valid slopes, it selects the wrong branch, so the conditional definition remains the general form.

---

## Naive vs Optimized

A short conditional is sufficient for correctness; source syntax alone does not determine divergence. Two possible refinements are:

1. **Grid-stride loop**: launch a fixed, device-sized grid and let each thread process multiple elements by striding in steps of blockDim.x * gridDim.x, with $\alpha$ still resident in a register the whole time. One configuration handles any $N$.

2. **Vectorized loads**: reinterpreting the arrays as float4 lets each thread load and store 16 bytes per instruction and apply the select across the four components, issuing fewer, wider transactions and nudging the kernel toward the memory ceiling.

All sit on top of an already-saturated memory pipeline; for a map you can only approach the bandwidth roofline.

---

## Worked Example

Take $N = 6$, $\alpha = 0.01$, block size 4. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, for 8 threads total - two more than there are elements.

* **Block 0** (blockIdx.x = 0): threads compute idx $= 0, 1, 2, 3$ and write output[0..3].
* **Block 1** (blockIdx.x = 1): threads compute idx $= 4, 5, 6, 7$. Indices $4$ and $5$ write output[4..5]; indices $6$ and $7$ fail idx < 6 and exit.

With input $= [-3, 2, -1, 5, 0, -4]$ and $\alpha$ in every lane's register, the active threads produce output $= [-0.03, 2, -0.01, 5, 0, -0.04]$. Under predication each lane computes both x and 0.01 * x and keeps the one its sign selects, so the warp runs a single straight-line stream even though the signs alternate - an input pattern whose control-flow cost depends on the generated instructions.

---

## Pitfalls

* **Passing $\alpha$ through global memory.** A scalar kernel argument avoids adding a separate per-element input buffer. Its access is still managed by the compiler and device.
* **Forcing a heavy branch.** A bulky conditional body can introduce divergent control flow. Use the required conditional behavior and inspect generated code before drawing performance conclusions.
* **Omitting the bounds check.** When $N$ is not a multiple of the block size the grid rounds up; without if (idx < N) the tail threads read and write out of bounds.
* **Breaking coalescing.** Strided or misaligned indexing fragments a warp's 32 requests into many transactions and collapses effective bandwidth.

---