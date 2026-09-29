#include <cuda_runtime.h>

__global__ void outer_product_kernel(const float* a, const float* b, float* C, int M, int N) {
    // Write code here
    
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col<N && row<M) {
        C[row*N + col] = a[row] * b[col];
    }
}

extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    outer_product_kernel<<<blocks, threads>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}
