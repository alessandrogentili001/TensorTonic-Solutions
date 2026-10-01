#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>

#define THREADS_PER_BLOCK 256

struct ArgMinVal {
    float val;
    int idx;
};

__device__ __forceinline__ ArgMinVal combine_argmin(ArgMinVal a, ArgMinVal b) {
    if (a.val > b.val) return b;
    if (b.val > a.val) return a;
    // Break ties with smaller index; sentinel INT_MAX will always lose
    return (a.idx < b.idx) ? a : b;
}

__device__ __forceinline__ ArgMinVal warp_reduce_argmin(ArgMinVal v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        ArgMinVal other;
        other.val = __shfl_down_sync(0xffffffff, v.val, offset);
        other.idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        v = combine_argmin(v, other);
    }
    return v;
}

__global__ void argmin_kernel(const float* __restrict__ input, 
                              int* __restrict__ out_idx, 
                              int N) 
{
    __shared__ ArgMinVal warp_scratch[32];

    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    int num_vec = N / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(input);

    // Initialize with INT_MAX so dummy values never win ties
    ArgMinVal thread_min = {FLT_MAX, INT_MAX};

    // 1. Vectorized 128-bit load loop
    for (int i = tid; i < num_vec; i += blockDim.x) {
        float4 v = in_vec[i];
        int base = i * 4;
        thread_min = combine_argmin(thread_min, {v.x, base + 0});
        thread_min = combine_argmin(thread_min, {v.y, base + 1});
        thread_min = combine_argmin(thread_min, {v.z, base + 2});
        thread_min = combine_argmin(thread_min, {v.w, base + 3});
    }

    // 2. Scalar leftover loop (FIX: was completely missing!)
    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < N; i += blockDim.x) {
        thread_min = combine_argmin(thread_min, {input[i], i});
    }

    // 3. Intra-warp reduction
    thread_min = warp_reduce_argmin(thread_min);

    if (lane == 0) {
        warp_scratch[wid] = thread_min;
    }
    __syncthreads();

    // 4. Warp 0 final reduction
    if (wid == 0) {
        ArgMinVal b = (lane < num_warps) ? warp_scratch[lane] : ArgMinVal{FLT_MAX, INT_MAX};
        b = warp_reduce_argmin(b);
        if (lane == 0) {
            *out_idx = b.idx;
        }
    }
}

extern "C" void solve(const float* input, int* out_idx, int N) {
    int threads = 256;
    if (N > 512)  threads = 512;
    if (N > 1024) threads = 1024;

    argmin_kernel<<<1, threads>>>(input, out_idx, N);
    cudaDeviceSynchronize();
}