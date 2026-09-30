#include <cuda_runtime.h>

__global__ void vector_add(const float* __restrict__ A, 
                           const float* __restrict__ B, 
                           float* __restrict__ C, 
                           int N) 
{
    int num_vec = N / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Reinterpret 32-bit float pointers as 128-bit float4 pointers
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    // 1. Vectorized loop for the bulk elements (chunks of 4)
    for (int i = idx; i < num_vec; i += stride) {
        float4 a = A4[i]; // Single 128-bit load instruction
        float4 b = B4[i]; // Single 128-bit load instruction
        float4 c;

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        C4[i] = c;        // Single 128-bit store instruction
    }

    // 2. Handle scalar leftover tail elements if N is not divisible by 4
    int tail_start = num_vec * 4;
    for (int i = tail_start + idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int threads = 256;
    
    // Each thread processes 4 elements per iteration
    int num_vec = N / 4;
    int blocks = (num_vec + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    if (blocks > 2048) blocks = 2048; // Cap grid size for the grid-stride loop

    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}