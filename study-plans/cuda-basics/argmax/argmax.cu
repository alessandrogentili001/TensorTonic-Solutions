#include <cuda_runtime.h>
#include <float.h>

#define THREADS_PER_BLOCK 256

struct ArgMaxVal {
    float val;
    int idx;
};

__device__ __forceinline__ ArgMaxVal combine_argmax(ArgMaxVal a, ArgMaxVal b) {
    if (a.val > b.val) return a;
    if (b.val > a.val) return b;
    return (a.idx < b.idx) ? a : b;
}

__device__ __forceinline__ ArgMaxVal warp_reduce_argmax(ArgMaxVal v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        ArgMaxVal other;
        other.val = __shfl_down_sync(0xffffffff, v.val, offset);
        other.idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        v = combine_argmax(v, other);
    }
    return v;
}

__global__ void argmax_kernel(const float* __restrict__ input, 
                              int* __restrict__ out_idx, 
                              int N) 
{
    __shared__ ArgMaxVal warp_scratch[32];

    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    int num_vec = N / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(input);

    ArgMaxVal thread_max = {-FLT_MAX, -1};

    // 1. Vectorized 128-bit load loop
    for (int i = tid; i < num_vec; i += blockDim.x) {
        float4 v = in_vec[i];
        int base = i * 4;
        thread_max = combine_argmax(thread_max, {v.x, base + 0});
        thread_max = combine_argmax(thread_max, {v.y, base + 1});
        thread_max = combine_argmax(thread_max, {v.z, base + 2});
        thread_max = combine_argmax(thread_max, {v.w, base + 3});
    }

    // 2. Scalar leftovers
    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < N; i += blockDim.x) {
        thread_max = combine_argmax(thread_max, {input[i], i});
    }

    // 3. Intra-warp reduction
    thread_max = warp_reduce_argmax(thread_max);

    if (lane == 0) {
        warp_scratch[wid] = thread_max;
    }
    __syncthreads();

    // 4. Warp 0 final reduction
    if (wid == 0) {
        ArgMaxVal b = (lane < num_warps) ? warp_scratch[lane] : ArgMaxVal{-FLT_MAX, -1};
        b = warp_reduce_argmax(b);
        if (lane == 0) {
            // Write directly to device output pointer
            *out_idx = b.idx;
        }
    }
}

extern "C" void solve(const float* input, int* out_idx, int N) {
    int threads = 256;
    if (N > 512)  threads = 512;
    if (N > 1024) threads = 1024;

    argmax_kernel<<<1, threads>>>(input, out_idx, N);
    cudaDeviceSynchronize();
}