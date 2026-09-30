#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_SM 2

// -------------------------------------------------------------
// Warp-level sum reduction using registers (__shfl_down_sync)
// -------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -------------------------------------------------------------
// Block-level sum reduction: Warps -> Shared Memory -> Warp 0
// -------------------------------------------------------------
__device__ inline float block_reduce_sum(float val, float* warp_scratch) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    // 1. Reduce within each warp
    val = warp_reduce_sum(val);

    // 2. Warp leader (lane 0) publishes warp aggregate to shared memory
    if (lane == 0) {
        warp_scratch[wid] = val;
    }
    __syncthreads();

    // 3. First warp reduces the aggregates from all warps
    if (wid == 0) {
        float b_val = (lane < num_warps) ? warp_scratch[lane] : 0.0f;
        b_val = warp_reduce_sum(b_val);
        if (lane == 0) {
            warp_scratch[0] = b_val;
        }
    }
    __syncthreads();

    return warp_scratch[0];
}

// =============================================================
// Vectorized Multi-Block Sum Reduction Kernel
// =============================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
sum_kernel(const float* __restrict__ input, 
           float* __restrict__ result, 
           int N) 
{
    __shared__ float warp_scratch[32]; // Accommodates up to 1024 threads

    int num_vec = N / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* in_vec = reinterpret_cast<const float4*>(input);

    float thread_sum = 0.0f;

    // 1. Vectorized 128-bit load accumulator loop
    for (int i = tid; i < num_vec; i += stride) {
        float4 v = in_vec[i];
        thread_sum += (v.x + v.y) + (v.z + v.w);
    }

    // 2. Process scalar leftover elements if N is not divisible by 4
    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < N; i += stride) {
        thread_sum += input[i];
    }

    // 3. Intra-block reduction across all threads in this block
    float block_sum = block_reduce_sum(thread_sum, warp_scratch);

    // 4. Thread 0 of each block atomically adds its partial sum to global result
    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
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

    // Launch enough blocks to saturate all SMs without creating excess atomics
    int blocks_per_sm = 4;
    int target_blocks = prop.multiProcessorCount * blocks_per_sm;

    int elements_per_block = THREADS_PER_BLOCK * 4; // 1024 floats/pass
    int needed_blocks = (N + elements_per_block - 1) / elements_per_block;
    if (needed_blocks == 0) needed_blocks = 1;

    int blocks = (needed_blocks < target_blocks) ? needed_blocks : target_blocks;

    // Clear accumulator in global memory
    cudaMemset(result, 0, sizeof(float));

    sum_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, result, N);
    cudaDeviceSynchronize();
}