#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

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

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_scratch[wid] = val;
    }
    __syncthreads();

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
// L1 Normalization Kernel (Single Block)
// =============================================================
__global__ void l1_normalize_kernel(const float* __restrict__ input, 
                                    float* __restrict__ output, 
                                    int N, 
                                    float eps = 1e-12f) 
{
    __shared__ float warp_scratch[32];
    __shared__ float s_inv_norm;

    int tid = threadIdx.x;
    int num_vec = N / 4;

    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    // ---------------------------------------------------------
    // Phase 1: Sum reduction of absolute values |x|
    // ---------------------------------------------------------
    float thread_l1 = 0.0f;

    // Vectorized 128-bit accumulation
    for (int i = tid; i < num_vec; i += blockDim.x) {
        float4 v = in_vec[i];
        thread_l1 += (fabsf(v.x) + fabsf(v.y)) + (fabsf(v.z) + fabsf(v.w));
    }

    // Scalar leftover accumulation
    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < N; i += blockDim.x) {
        thread_l1 += fabsf(input[i]);
    }

    // Intra-block reduction
    float block_l1 = block_reduce_sum(thread_l1, warp_scratch);

    // Thread 0 computes inverse multiplier and broadcasts via shared memory
    if (tid == 0) {
        s_inv_norm = __fdividef(1.0f, block_l1 + eps);
    }
    __syncthreads();

    float inv_norm = s_inv_norm;

    // ---------------------------------------------------------
    // Phase 2: Scale and write back
    // ---------------------------------------------------------
    for (int i = tid; i < num_vec; i += blockDim.x) {
        float4 v = in_vec[i];
        float4 res;
        res.x = v.x * inv_norm;
        res.y = v.y * inv_norm;
        res.z = v.z * inv_norm;
        res.w = v.w * inv_norm;
        out_vec[i] = res;
    }

    for (int i = tail_start + tid; i < N; i += blockDim.x) {
        output[i] = input[i] * inv_norm;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    if (N > 512)  threads = 512;
    if (N > 1024) threads = 1024;

    l1_normalize_kernel<<<1, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}