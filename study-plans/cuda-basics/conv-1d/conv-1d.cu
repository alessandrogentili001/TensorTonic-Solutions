#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define ITEMS_PER_THREAD 4
#define TILE_OUT (THREADS_PER_BLOCK * ITEMS_PER_THREAD) // 1024 outputs per block

// Maximum filter size supported for static shared memory sizing (adjust as needed)
#define MAX_KERNEL_SIZE 128 

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
conv1d_kernel(const float* __restrict__ input, 
              const float* __restrict__ filter, 
              float* __restrict__ output, 
              int N, 
              int kN, 
              int outN) 
{
    // Shared memory holds the input tile needed for this block: TILE_OUT + kN - 1
    extern __shared__ float s_input[];

    int tid = threadIdx.x;
    int block_out_start = blockIdx.x * TILE_OUT;
    int tile_in_len = TILE_OUT + kN - 1;

    // -------------------------------------------------------------
    // 1. Cooperative Load: Input tile into shared memory (coalesced)
    // -------------------------------------------------------------
    for (int i = tid; i < tile_in_len; i += blockDim.x) {
        int in_idx = block_out_start + i;
        s_input[i] = (in_idx < N) ? input[in_idx] : 0.0f;
    }
    __syncthreads();

    // -------------------------------------------------------------
    // 2. Convolution: Each thread computes 4 consecutive outputs
    // -------------------------------------------------------------
    int thread_out_idx = tid * ITEMS_PER_THREAD;
    int global_out_idx = block_out_start + thread_out_idx;

    if (global_out_idx >= outN) return;

    // Registers for 4 output accumulators
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    // Shared memory pointer for this thread's sliding window
    const float* s_ptr = &s_input[thread_out_idx];

    #pragma unroll 4
    for (int k = 0; k < kN; ++k) {
        float w = filter[k]; // Cached in L1/Constant cache; broadcast across warp

        sum0 += s_ptr[k]     * w;
        sum1 += s_ptr[k + 1] * w;
        sum2 += s_ptr[k + 2] * w;
        sum3 += s_ptr[k + 3] * w;
    }

    // -------------------------------------------------------------
    // 3. Vectorized Writeback
    // -------------------------------------------------------------
    if (global_out_idx + 3 < outN) {
        float4 out_val = make_float4(sum0, sum1, sum2, sum3);
        reinterpret_cast<float4*>(output)[global_out_idx / 4] = out_val;
    } else {
        // Boundary fallback if outN is not a multiple of 4
        if (global_out_idx + 0 < outN) output[global_out_idx + 0] = sum0;
        if (global_out_idx + 1 < outN) output[global_out_idx + 1] = sum1;
        if (global_out_idx + 2 < outN) output[global_out_idx + 2] = sum2;
    }
}

extern "C" void solve(const float* input, const float* filter, float* output, int N, int kN) {
    int outN = N - kN + 1;
    if (outN <= 0) return;

    const int threads = THREADS_PER_BLOCK;
    int blocks = (outN + TILE_OUT - 1) / TILE_OUT;

    // Dynamic shared memory footprint: (1024 + kN - 1) * sizeof(float)
    size_t shared_mem_bytes = (TILE_OUT + kN - 1) * sizeof(float);

    conv1d_kernel<<<blocks, threads, shared_mem_bytes>>>(input, filter, output, N, kN, outN);
    cudaDeviceSynchronize();
}