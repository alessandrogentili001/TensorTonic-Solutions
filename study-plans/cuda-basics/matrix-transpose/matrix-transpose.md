# <span style="font-size: 20px;">Matrix Transpose</span>

<span style="font-size: 14px;">Matrix transpose writes $B[j,i] = A[i,j]$, turning an $M \times N$ matrix into an $N \times M$ one. It performs no arithmetic at all - it only moves data - which makes it the **canonical coalescing lesson** of CUDA. Because reads and writes hit different layouts, one of the two directions is forced off the contiguous path, and the entire art of a fast transpose is recovering coalescing on both sides. Memory-bound from start to finish, it isolates one variable: how the index pattern meets the memory controller.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For a row $i$ in $[0, M)$ and column $j$ in $[0, N)$, the kernel evaluates:</span>

$$
B[j,i] = A[i,j]
$$

<span style="font-size: 14px;">Input $A$ is $M \times N$ row-major, so $(i,j)$ sits at `i * N + j`. Output $B$ is $N \times M$ row-major, so $(j,i)$ sits at `j * M + i`. The two offsets walk memory at completely different strides, and that mismatch is the whole problem.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">The decomposition is **one thread per element**, with a 2-D grid of `16x16` blocks tiling the input. Each thread derives its coordinate from block and lane indices:</span>

$$
j = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}, \quad i = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}
$$

<span style="font-size: 14px;">A `16x16` block holds 256 threads, a multiple of the 32-lane **warp**, and many such blocks fit on an **SM (Streaming Multiprocessor)** for good **occupancy**. Edge blocks overhang non-multiple shapes, so a bounds check on both axes guards the body. The interesting part is not the launch but what the read and write addresses do once threads start moving data.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">**Coalescing** is the central concept: when the 32 threads of a warp touch 32 consecutive global addresses, the controller serves them in one transaction; a strided pattern fragments that into up to 32 separate transactions and wastes most of the delivered bandwidth.</span>

<span style="font-size: 14px;">In the naive kernel a warp has consecutive `threadIdx.x`, hence consecutive $j$, hence consecutive input offsets `i*N+j` - the **read is perfectly coalesced**. But each thread then writes to `j*M+i`, and across the warp $j$ varies while $i$ is fixed, so consecutive lanes write addresses $M$ apart. The **write is column-strided and uncoalesced**: 32 lanes generate up to 32 transactions, and effective store bandwidth collapses. (Swapping which axis is contiguous merely moves the penalty to the read side; one direction is always strided.)</span>

<span style="font-size: 14px;">The fix is to route the data through `__shared__` memory. A block cooperatively loads a `TILE x TILE` tile of $A$ into a shared array with coalesced reads, then `__syncthreads()`, then writes the tile out to $B$ with the indices swapped on the way out of shared memory rather than on the way to global. With the right indexing both the global read and the global write become coalesced, and the strided access is confined to fast on-chip shared memory.</span>

---

## <span style="font-size: 16px;">Bank Conflicts and the Padded Tile</span>

<span style="font-size: 14px;">Shared memory is split into 32 **banks**; if multiple lanes of a warp hit the same bank in one access (and it is not a broadcast), the accesses serialize. A square `[TILE][TILE]` tile of 32 floats per row maps every column onto the same bank, so the transposed read - a column of the tile - is a 32-way bank conflict that serializes completely, undoing much of the gain from coalescing global memory.</span>

<span style="font-size: 14px;">The standard fix is to pad the tile to `[TILE][TILE+1]`. The extra unused column shifts each logical row by one bank, so a column of the tile now spans 32 distinct banks and is **conflict-free**. The cost is one wasted float per row of shared memory, a trivial price for eliminating serialization. This `+1` padding is the signature trick of a fast transpose.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Transpose does zero FLOPs. Per element it moves 8 bytes - one 4-byte load and one 4-byte store - and performs no arithmetic, giving an **arithmetic intensity** of:</span>

$$
\frac{0 \text{ FLOP}}{8 \text{ bytes}} = 0 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">It sits on the floor of the **roofline**: as **memory-bound** as a kernel can be. No arithmetic optimization exists to apply. Performance is governed entirely by how close effective bandwidth gets to the device peak, and that is decided purely by coalescing and by avoiding shared-memory bank conflicts. The whole speedup story is moving bytes efficiently, never computing them faster.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">Global latency is hundreds of cycles, hidden by **massive multithreading**: while one warp waits on its tile load, the SM scheduler runs another resident warp. The tiled kernel adds two synchronization points - one `__syncthreads()` after the shared load so every lane sees a complete tile before the transposed read, and the natural barrier at the tile boundary - so enough warps must be resident to hide the stall the barrier introduces.</span>

<span style="font-size: 14px;">The shared tile costs roughly `TILE*(TILE+1)*4` bytes per block. A `16x16` padded tile is about 1 KB, small enough that occupancy stays high; a careless `32x32` tile uses 4 KB+ and can cap the number of resident blocks, so the tile size is itself an occupancy knob.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel reads coalesced and writes strided (or the reverse), so roughly half of every memory round trip runs at a fraction of peak bandwidth - the strided side can take up to 32 transactions where the coalesced side takes one.</span>

<span style="font-size: 14px;">The optimized kernel stages a tile through padded `__shared__` memory so both global directions are coalesced and the on-chip transpose is conflict-free:</span>

<span style="font-size: 14px;">1. **Coalesced load**: the warp reads a `TILE x TILE` block of $A$ into `tile[ty][tx]`, consecutive lanes hitting consecutive addresses.</span>

<span style="font-size: 14px;">2. **`__syncthreads()`**: every lane must finish writing the tile before any lane reads a transposed element, or it reads stale shared memory.</span>

<span style="font-size: 14px;">3. **Coalesced store**: the warp reads `tile[tx][ty]` (the transposed direction, conflict-free thanks to the `+1` pad) and writes a contiguous run of $B$, again consecutive lanes to consecutive addresses.</span>

<span style="font-size: 14px;">The payoff is that the uncoalesced global traffic is eliminated; both reads and writes now run near peak, roughly doubling effective bandwidth versus the naive store-strided version.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $A$ as $2 \times 3$ ($M=2$, $N=3$), so $B$ is $3 \times 2$. Look at the warp covering row $0$, lanes for $j = 0,1,2$.</span>

* <span style="font-size: 14px;">**Reads** target `i*N+j` = `0,1,2` - three consecutive addresses, fully coalesced.</span>
* <span style="font-size: 14px;">**Naive writes** target `j*M+i` = `0,2,4` - addresses two apart, strided and uncoalesced.</span>

<span style="font-size: 14px;">With $A = \begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$ the result is $B = \begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}$. In the tiled kernel those same writes become contiguous because the transpose happened inside shared memory: the warp now stores a contiguous run of $B$ rather than a strided scatter.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Accepting the strided global access.** A naive transpose has one coalesced direction and one column-strided direction; the strided side multiplies transactions up to 32x. Staging through a shared tile is the fix, not a luxury.</span>
* <span style="font-size: 14px;">**Unpadded shared tile.** A square `[TILE][TILE]` array maps a tile column onto one bank, causing a 32-way bank conflict on the transposed read; pad to `[TILE][TILE+1]`.</span>
* <span style="font-size: 14px;">**Missing `__syncthreads()` between load and transposed read.** Reading the tile before all lanes have written it is a shared-memory race that yields nondeterministic wrong values.</span>
* <span style="font-size: 14px;">**Forgetting the bounds check on non-square or non-multiple shapes.** Edge tiles overhang $A$ or $B$; guard both the load and the store, since the valid ranges differ after the swap.</span>

---