#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
// On H100 (50MB L2 cache), keep direct L1 mode until N exceeds ~500k elements
#define LARGE_THRESHOLD 524288 

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
conv1d_adaptive_kernel(const float* __restrict__ input, 
                       const float* __restrict__ filter, 
                       float* __restrict__ output, 
                       int N, 
                       int kN, 
                       int outN,
                       bool use_tiling) 
{
    // =========================================================
    // REGIME 1: Massive inputs (N >= 500k, DRAM bound)
    // =========================================================
    if (use_tiling) {
        extern __shared__ float s_input[];

        const int ITEMS = 4;
        const int TILE_OUT = blockDim.x * ITEMS; // 1024
        int tid = threadIdx.x;
        int block_out_start = blockIdx.x * TILE_OUT;
        int tile_in_len = TILE_OUT + kN - 1;

        for (int i = tid; i < tile_in_len; i += blockDim.x) {
            int in_idx = block_out_start + i;
            s_input[i] = (in_idx < N) ? input[in_idx] : 0.0f;
        }
        __syncthreads();

        int thread_out_idx = tid * ITEMS;
        int global_out_idx = block_out_start + thread_out_idx;

        if (global_out_idx >= outN) return;

        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        const float* s_ptr = &s_input[thread_out_idx];

        #pragma unroll 4
        for (int k = 0; k < kN; ++k) {
            float w = filter[k];
            sum0 += s_ptr[k]     * w;
            sum1 += s_ptr[k + 1] * w;
            sum2 += s_ptr[k + 2] * w;
            sum3 += s_ptr[k + 3] * w;
        }

        if (global_out_idx + 3 < outN) {
            *reinterpret_cast<float4*>(&output[global_out_idx]) = make_float4(sum0, sum1, sum2, sum3);
        } else {
            if (global_out_idx + 0 < outN) output[global_out_idx + 0] = sum0;
            if (global_out_idx + 1 < outN) output[global_out_idx + 1] = sum1;
            if (global_out_idx + 2 < outN) output[global_out_idx + 2] = sum2;
        }
        return;
    }

    // =========================================================
    // REGIME 2: Cache-resident inputs (N < 500k, L1/L2 served)
    // 4-way ILP with float4 stores, zero __syncthreads()
    // =========================================================
    {
        const int ITEMS = 4;
        int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS;

        if (global_idx >= outN) return;

        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        const float* in_ptr = &input[global_idx];

        #pragma unroll 8
        for (int k = 0; k < kN; ++k) {
            float f = filter[k];
            sum0 += in_ptr[k]     * f;
            sum1 += in_ptr[k + 1] * f;
            sum2 += in_ptr[k + 2] * f;
            sum3 += in_ptr[k + 3] * f;
        }

        if (global_idx + 3 < outN) {
            *reinterpret_cast<float4*>(&output[global_idx]) = make_float4(sum0, sum1, sum2, sum3);
        } else {
            if (global_idx + 0 < outN) output[global_idx + 0] = sum0;
            if (global_idx + 1 < outN) output[global_idx + 1] = sum1;
            if (global_idx + 2 < outN) output[global_idx + 2] = sum2;
        }
    }
}

extern "C" void solve(const float* input, const float* filter, float* output, int N, int kN) {
    int outN = N - kN + 1;
    if (outN <= 0) return;

    // Shift threshold to 524,288 so 262,144 runs in direct L1 mode
    bool use_tiling = (outN >= LARGE_THRESHOLD);

    if (use_tiling) {
        const int threads = THREADS_PER_BLOCK;
        const int tile_out = threads * 4;
        int blocks = (outN + tile_out - 1) / tile_out;
        size_t shmem_bytes = (tile_out + kN - 1) * sizeof(float);

        conv1d_adaptive_kernel<<<blocks, threads, shmem_bytes>>>(
            input, filter, output, N, kN, outN, true
        );
    } else {
        // Dynamic threads: finer block granularity for small sizes
        int threads = (outN <= 4096) ? 64 : ((outN <= 65536) ? 128 : 256);
        const int items_per_thread = 4;
        int elements_per_block = threads * items_per_thread;
        int blocks = (outN + elements_per_block - 1) / elements_per_block;

        conv1d_adaptive_kernel<<<blocks, threads, 0>>>(
            input, filter, output, N, kN, outN, false
        );
    }

    cudaDeviceSynchronize();
}