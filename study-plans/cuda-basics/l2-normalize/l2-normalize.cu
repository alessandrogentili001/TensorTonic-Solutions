#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduction returning the final scalar ONLY on thread 0
__device__ inline float block_reduce_sum_tid0(float val, float* warp_scratch) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_scratch[wid] = val;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (wid == 0) {
        float b = (lane < num_warps) ? warp_scratch[lane] : 0.0f;
        block_sum = warp_reduce_sum(b);
    }
    return block_sum; // Valid only on threadIdx.x == 0
}

// Pass 1: Vectorized Multi-block squared sum reduction
__global__ void reduce_sq_sum_multi_block(const float* __restrict__ input, 
                                          float* __restrict__ sumv, 
                                          int N) 
{
    __shared__ float warp_scratch[32];

    int num_vec = N / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* in4 = reinterpret_cast<const float4*>(input);

    float thread_sum = 0.0f;

    // Vectorized 128-bit loads
    for (int i = tid; i < num_vec; i += stride) {
        float4 v = in4[i];
        thread_sum += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    }

    // Scalar leftovers
    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < N; i += stride) {
        float x = input[i];
        thread_sum += x * x;
    }

    float block_sum = block_reduce_sum_tid0(thread_sum, warp_scratch);

    if (threadIdx.x == 0) {
        atomicAdd(sumv, block_sum);
    }
}

// Pass 2: Vectorized 128-bit scale kernel
__global__ void divide_by_sqrt(const float* __restrict__ input, 
                               float* __restrict__ output, 
                               const float* __restrict__ sumv, 
                               int N, float eps) 
{
    float inv_norm = rsqrtf(*sumv + eps);

    int num_vec = N / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);

    for (int i = tid; i < num_vec; i += stride) {
        float4 v = in4[i];
        float4 res;
        res.x = v.x * inv_norm;
        res.y = v.y * inv_norm;
        res.z = v.z * inv_norm;
        res.w = v.w * inv_norm;
        out4[i] = res;
    }

    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < N; i += stride) {
        output[i] = input[i] * inv_norm;
    }
}

__device__ float d_sum; // Static global device memory (zero cudaMalloc overhead)

extern "C" void solve(const float* input, float* output, int N) {
int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int threads = 256;
    int blocks = prop.multiProcessorCount * 4;
    if (blocks > 512) blocks = 512;

    // Zero out device scalar directly via pointer lookup or memset
    float* d_sum_ptr;
    cudaGetSymbolAddress((void**)&d_sum_ptr, d_sum);
    cudaMemsetAsync(d_sum_ptr, 0, sizeof(float));

    reduce_sq_sum_multi_block<<<blocks, threads>>>(input, d_sum_ptr, N);
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum_ptr, N, 1e-12f);

    cudaDeviceSynchronize();
}