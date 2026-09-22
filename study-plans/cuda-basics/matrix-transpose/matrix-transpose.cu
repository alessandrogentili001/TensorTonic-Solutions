#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    // Write code here

    __shared__ float tile[16][16+1];

    int t_row = threadIdx.y;
    int t_col = threadIdx.x;
    
    int in_col = blockDim.x * blockIdx.x + t_col;
    int in_row = blockDim.y * blockIdx.y + t_row;

    if (in_col<N && in_row<M) {
        tile[t_row][t_col] = A[in_row * N + in_col];
    }
    __syncthreads();

    int out_col = blockDim.x * blockIdx.y + t_col;
    int out_row = blockDim.y * blockIdx.x + t_row;

    if (out_col<M && out_row<N) {
        B[out_row*M + out_col] = tile[t_col][t_row];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
