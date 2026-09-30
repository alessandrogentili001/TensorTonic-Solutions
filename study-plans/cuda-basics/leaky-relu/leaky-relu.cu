#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float leaky_relu_op(float x, float alpha) {
    // Generates a single conditional selection (FSEL / predicate) without branching
    return (x > 0.0f) ? x : (x * alpha);
    
    // Alternatively, if alpha <= 1.0f:
    // return fmaxf(x, x * alpha);
}

__global__ void leaky_relu_kernel(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  float alpha, 
                                  int N) 
{
    int num_vec = N / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    // 1. Vectorized 128-bit loop (processes 4 floats per iteration)
    for (int i = idx; i < num_vec; i += stride) {
        float4 in = in_vec[i];
        float4 out;

        out.x = leaky_relu_op(in.x, alpha);
        out.y = leaky_relu_op(in.y, alpha);
        out.z = leaky_relu_op(in.z, alpha);
        out.w = leaky_relu_op(in.w, alpha);

        out_vec[i] = out;
    }

    // 2. Handle tail elements if N is not divisible by 4
    int tail_start = num_vec * 4;
    for (int i = tail_start + idx; i < N; i += stride) {
        output[i] = leaky_relu_op(input[i], alpha);
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    const int threads = 256;
    
    int num_vec = N / 4;
    int blocks = (num_vec + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    if (blocks > 2048) blocks = 2048; // Grid-stride loop cap

    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}