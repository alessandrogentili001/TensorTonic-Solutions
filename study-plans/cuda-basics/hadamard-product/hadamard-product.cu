#include <cuda_runtime.h>

__global__ void hadamard_kernel(const float* A, const float* B, float* C, int M, int N) {

    // Write code here
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row<M && col<N) {
        C[row*N + col] = A[row*N + col] * B[row*N + col];
    }
}



extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {

    dim3 threads(16, 16);

    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    hadamard_kernel<<<blocks, threads>>>(A, B, C, M, N);

    cudaDeviceSynchronize();

} 

