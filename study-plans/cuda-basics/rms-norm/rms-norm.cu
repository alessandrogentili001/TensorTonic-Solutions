#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------------------------------
// Warp-level sum reduction primitive
// -------------------------------------------------------------
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -------------------------------------------------------------
// Block-level sum reduction using warp shuffles + shared memory
// -------------------------------------------------------------
__device__ inline float block_reduce_sum(float val, float* warp_scratch) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    // 1. Intra-warp reduction
    val = warp_reduce_sum(val);

    // 2. Warp leaders store partial sum into shared memory
    if (lane == 0) {
        warp_scratch[wid] = val;
    }
    __syncthreads();

    // 3. Warp 0 reduces all warp totals
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
// RMSNorm Kernel: 1 Block = 1 Row
// Grid: dim3(M)
// Threads: 256 (or 512, 1024 depending on N)
// =============================================================
__global__ void rms_norm_kernel(const float* __restrict__ input, 
                                const float* __restrict__ gamma, 
                                float* __restrict__ output, 
                                int M, int N, float eps) 
{
    int row = blockIdx.x;
    if (row >= M) return;

    __shared__ float s_inv_rms;
    __shared__ float warp_scratch[32]; // Max 32 warps (1024 threads)

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // ---------------------------------------------------------
    // Phase 1: Compute sum of squares (x_i^2) with a strided loop
    // ---------------------------------------------------------
    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = row_in[i];
        thread_sum_sq += val * val;
    }

    // ---------------------------------------------------------
    // Phase 2: Block reduction to get row-wide sum of squares
    // ---------------------------------------------------------
    float block_sum_sq = block_reduce_sum(thread_sum_sq, warp_scratch);

    // Thread 0 computes 1.0 / sqrt(mean_square + eps) and broadcasts
    if (threadIdx.x == 0) {
        float mean_sq = block_sum_sq / (float)N;
        s_inv_rms = rsqrtf(mean_sq + eps); // Single instruction reciprocal sqrt
    }
    __syncthreads();

    // Load broadcasted scale factor into thread register
    float inv_rms = s_inv_rms;

    // ---------------------------------------------------------
    // Phase 3: Normalize, scale by gamma, and write back
    // ---------------------------------------------------------
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        row_out[i] = (row_in[i] * inv_rms) * gamma[i];
    }
}

// =============================================================
// Host Entry Point
// =============================================================
extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    if (N > 512) threads = 512;
    if (N > 1024) threads = 1024;

    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}