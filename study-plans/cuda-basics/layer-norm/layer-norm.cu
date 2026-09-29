#include <cuda_runtime.h>
#include <math.h>

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float block_reduce_sum(float val, float* warp_scratch) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) warp_scratch[wid] = val;
    __syncthreads();

    if (wid == 0) {
        float b_val = (lane < num_warps) ? warp_scratch[lane] : 0.0f;
        b_val = warp_reduce_sum(b_val);
        if (lane == 0) warp_scratch[0] = b_val;
    }
    __syncthreads();

    return warp_scratch[0];
}

__global__ void layer_norm_kernel(const float* __restrict__ input, 
                                  const float* __restrict__ gamma, 
                                  const float* __restrict__ beta, 
                                  float* __restrict__ output, 
                                  int M, int N, float eps) 
{
    int row = blockIdx.x;
    if (row >= M) return;

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float warp_scratch[32];

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // 1. Calculate Mean (mu)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        thread_sum += row_in[i];
    }
    float block_sum = block_reduce_sum(thread_sum, warp_scratch);
    if (threadIdx.x == 0) {
        s_mean = block_sum / (float)N;
    }
    __syncthreads();
    float mu = s_mean;

    // 2. Calculate Variance (sigma^2)
    float thread_sq_diff = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float diff = row_in[i] - mu;
        thread_sq_diff += diff * diff;
    }
    float block_sq_diff = block_reduce_sum(thread_sq_diff, warp_scratch);
    if (threadIdx.x == 0) {
        s_inv_std = rsqrtf((block_sq_diff / (float)N) + eps); // 1 / sqrt(sigma^2 + eps)
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // 3. Normalize, Scale (gamma), and Shift (beta)
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        row_out[i] = ((row_in[i] - mu) * inv_std) * gamma[i] + beta[i];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, 
                      float* output, int M, int N, float eps) {
    int threads = 256;
    if (N > 256) threads = 512;
    if (N > 512) threads = 1024;

    layer_norm_kernel<<<M, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}