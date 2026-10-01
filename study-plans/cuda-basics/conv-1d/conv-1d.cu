#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define LARGE_THRESHOLD 32768

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
    // REGIME 1: Large input (Shared memory tiled + float4 store)
    // =========================================================
    if (use_tiling) {
        extern __shared__ float s_input[];

        const int ITEMS = 4;
        const int TILE_OUT = blockDim.x * ITEMS; // 1024
        int tid = threadIdx.x;
        int block_out_start = blockIdx.x * TILE_OUT;
        int tile_in_len = TILE_OUT + kN - 1;

        // 1. Cooperative load into shared memory
        for (int i = tid; i < tile_in_len; i += blockDim.x) {
            int in_idx = block_out_start + i;
            s_input[i] = (in_idx < N) ? input[in_idx] : 0.0f;
        }
        __syncthreads();

        // 2. Compute 4 consecutive outputs per thread
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

        // 3. Vectorized float4 writeback
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
    // REGIME 2: Small input (Direct L1 reads, fine block wave)
    // =========================================================
    {
        const int ITEMS = 2;
        int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS;

        if (global_idx >= outN) return;

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        const float* in_ptr = &input[global_idx];

        #pragma unroll 8
        for (int k = 0; k < kN; ++k) {
            float f = filter[k];
            sum0 += in_ptr[k]     * f;
            sum1 += in_ptr[k + 1] * f;
        }

        output[global_idx] = sum0;
        if (global_idx + 1 < outN) {
            output[global_idx + 1] = sum1;
        }
    }
}

// =============================================================
// Host Entry Point
// =============================================================
extern "C" void solve(const float* input, const float* filter, float* output, int N, int kN) {
    int outN = N - kN + 1;
    if (outN <= 0) return;

    bool use_tiling = (outN > LARGE_THRESHOLD);

    if (use_tiling) {
        // High-throughput configuration for large N
        const int threads = THREADS_PER_BLOCK;
        const int tile_out = threads * 4; // 1024 outputs per block
        int blocks = (outN + tile_out - 1) / tile_out;
        size_t shmem_bytes = (tile_out + kN - 1) * sizeof(float);

        conv1d_adaptive_kernel<<<blocks, threads, shmem_bytes>>>(
            input, filter, output, N, kN, outN, true
        );
    } else {
        // High-occupancy configuration for small N (more blocks, zero __syncthreads)
        int threads = (outN <= 2048) ? 64 : 128;
        int items_per_thread = 2;
        int elements_per_block = threads * items_per_thread;
        int blocks = (outN + elements_per_block - 1) / elements_per_block;

        conv1d_adaptive_kernel<<<blocks, threads, 0>>>(
            input, filter, output, N, kN, outN, false
        );
    }

    cudaDeviceSynchronize();
}