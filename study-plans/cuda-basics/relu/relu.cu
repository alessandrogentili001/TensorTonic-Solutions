#include <cuda_runtime.h>
#include <math.h>

__global__ void relu_kernel(const float* __restrict__ input, 
                            float* __restrict__ output, 
                            int N) 
{
    int num_vec = N / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    // 1. Vectorized 128-bit loop (processes 4 floats per step)
    for (int i = idx; i < num_vec; i += stride) {
        float4 in = in_vec[i]; // Single LDG.E.128 instruction
        float4 out;

        out.x = fmaxf(0.0f, in.x);
        out.y = fmaxf(0.0f, in.y);
        out.z = fmaxf(0.0f, in.z);
        out.w = fmaxf(0.0f, in.w);

        out_vec[i] = out;      // Single STG.E.128 instruction
    }

    // 2. Handle leftover tail elements if N is not a multiple of 4
    int tail_start = num_vec * 4;
    for (int i = tail_start + idx; i < N; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    const int threads = 256;
    
    // Each thread processes 4 elements per iteration
    int num_vec = N / 4;
    int blocks = (num_vec + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    if (blocks > 2048) blocks = 2048; // Grid-stride cap to maximize SM occupancy

    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}