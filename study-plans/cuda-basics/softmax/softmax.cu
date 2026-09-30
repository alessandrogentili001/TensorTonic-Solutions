#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

// Helper struct: Holds running max (m) and running sum of exps (d)
struct MD {
    float m;
    float d;
};

// Combine two online Softmax states: (m1, d1) and (m2, d2)
__device__ __forceinline__ MD combine_md(MD a, MD b) {
    if (a.m > b.m) {
        return {a.m, a.d + b.d * __expf(b.m - a.m)};
    } else {
        return {b.m, b.d + a.d * __expf(a.m - b.m)};
    }
}

// -------------------------------------------------------------
// Warp-level fused (max, sum) reduction
// -------------------------------------------------------------
__device__ inline MD warp_reduce_md(MD val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        MD other;
        other.m = __shfl_down_sync(0xffffffff, val.m, offset);
        other.d = __shfl_down_sync(0xffffffff, val.d, offset);
        val = combine_md(val, other);
    }
    return val;
}

// -------------------------------------------------------------
// Block-level fused reduction using shared memory scratchpad
// -------------------------------------------------------------
__device__ inline MD block_reduce_md(MD val, MD* warp_scratch) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    // 1. Intra-warp reduction
    val = warp_reduce_md(val);

    // 2. Warp leaders store partial states
    if (lane == 0) {
        warp_scratch[wid] = val;
    }
    __syncthreads();

    // 3. Warp 0 aggregates all warp states
    if (wid == 0) {
        MD b_val = (lane < num_warps) ? warp_scratch[lane] : MD{-1e38f, 0.0f};
        b_val = warp_reduce_md(b_val);
        if (lane == 0) {
            warp_scratch[0] = b_val;
        }
    }
    __syncthreads();

    return warp_scratch[0];
}

// =============================================================
// Fast Online Softmax Kernel (Single-Block for 1D Vector of size N)
// =============================================================
__global__ void softmax_kernel(const float* __restrict__ input, 
                               float* __restrict__ output, 
                               int N) 
{
    __shared__ MD warp_scratch[32]; // Scratchpad for up to 1024 threads
    __shared__ float s_max;
    __shared__ float s_inv_sum;

    int tid = threadIdx.x;

    // ---------------------------------------------------------
    // Phase 1: 1-Step Fused Online Reduction for Max & Exp-Sum
    // ---------------------------------------------------------
    MD local_md = {-1e38f, 0.0f};

    for (int i = tid; i < N; i += blockDim.x) {
        float x = input[i];
        MD elem = {x, 1.0f};
        local_md = combine_md(local_md, elem);
    }

    // Block-wide reduction across all warps
    MD global_md = block_reduce_md(local_md, warp_scratch);

    // Broadcast finalized m and 1/d to the block
    if (tid == 0) {
        s_max = global_md.m;
        // Fast reciprocal unit
        s_inv_sum = __fdividef(1.0f, global_md.d);
    }
    __syncthreads();

    float m = s_max;
    float inv_sum = s_inv_sum;

    // ---------------------------------------------------------
    // Phase 2: Compute final normalized values and store
    // ---------------------------------------------------------
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input[i];
        output[i] = __expf(x - m) * inv_sum;
    }
}

// =============================================================
// Host Launcher
// =============================================================
extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    if (N > 512)  threads = 512;
    if (N > 1024) threads = 1024;

    // For a single flat vector, a single cooperative block handles reduction & normalization
    int blocks = 1;

    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}