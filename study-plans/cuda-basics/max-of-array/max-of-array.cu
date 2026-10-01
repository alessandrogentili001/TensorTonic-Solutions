#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_SM 2

// -------------------------------------------------------------
// Universal Float Atomic Max (atomicCAS loop works on sm_50+)
// -------------------------------------------------------------
__device__ __forceinline__ void atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        // If current value is already larger, no need to update
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(address_as_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

// -------------------------------------------------------------
// Warp-level Max Reduction using registers
// -------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// -------------------------------------------------------------
// Block-level Max Reduction (Warps -> Shared Memory -> Warp 0)
// -------------------------------------------------------------
__device__ inline float block_reduce_max_tid0(float val, float* warp_scratch) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        warp_scratch[wid] = val;
    }
    __syncthreads();

    float block_max = -FLT_MAX;
    if (wid == 0) {
        float b = (lane < num_warps) ? warp_scratch[lane] : -FLT_MAX;
        block_max = warp_reduce_max(b);
    }
    return block_max; // Valid on threadIdx.x == 0
}

// =============================================================
// Vectorized Multi-Block Max Kernel
// =============================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
max_kernel(const float* __restrict__ input, 
           float* __restrict__ result, 
           int N) 
{
    __shared__ float warp_scratch[32];

    int num_vec = N / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* in_vec = reinterpret_cast<const float4*>(input);

    float thread_max = -FLT_MAX;

    // 1. Vectorized 128-bit load loop
    for (int i = tid; i < num_vec; i += stride) {
        float4 v = in_vec[i];
        thread_max = fmaxf(thread_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }

    // 2. Scalar leftover loop
    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < N; i += stride) {
        thread_max = fmaxf(thread_max, input[i]);
    }

    // 3. Intra-block reduction across all threads in this block
    float block_max = block_reduce_max_tid0(thread_max, warp_scratch);

    // 4. Thread 0 of each block atomically merges into global result
    if (threadIdx.x == 0) {
        atomicMaxFloat(result, block_max);
    }
}

// =============================================================
// Host Entry Point
// =============================================================
extern "C" void solve(const float* input, float* result, int N) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    const int threads = THREADS_PER_BLOCK;
    int blocks_per_sm = 4;
    int target_blocks = prop.multiProcessorCount * blocks_per_sm;

    int elements_per_block = threads * 4; // 1024 floats per block per iteration
    int needed_blocks = (N + elements_per_block - 1) / elements_per_block;
    if (needed_blocks == 0) needed_blocks = 1;

    int blocks = (needed_blocks < target_blocks) ? needed_blocks : target_blocks;

    // Initialize global result with -FLT_MAX
    float neg_inf = -FLT_MAX;
    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);

    max_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}