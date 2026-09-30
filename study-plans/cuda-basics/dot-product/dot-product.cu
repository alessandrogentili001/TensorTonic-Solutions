#include <cuda_runtime.h>

#include <cuda_runtime.h>

// -------------------------------------------------------------
// Warp-level sum reduction primitive
// -------------------------------------------------------------
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -------------------------------------------------------------
// Block-level sum reduction using warp shuffles + shared memory
// -------------------------------------------------------------
__device__ inline float block_reduce_sum(float val, float* warp_scratch) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    // 1. Intra-warp reduction
    val = warp_reduce_sum(val);

    // 2. Warp leaders publish to scratchpad
    if (lane == 0) {
        warp_scratch[wid] = val;
    }
    __syncthreads();

    // 3. Warp 0 reduces the scratchpad values
    if (wid == 0) {
        float b_val = (lane < num_warps) ? warp_scratch[lane] : 0.0f;
        b_val = warp_reduce_sum(b_val);
        if (lane == 0) {
            warp_scratch[0] = b_val;
        }
    }
    __syncthreads();

    return warp_scratch[0];
}

// =============================================================
// DOT Product Kernel: x = A * B
// Matrix A: M x N, Vector x: N x 1, Vector y: M x 1
// Grid: M blocks (1 block per row)
// =============================================================
__global__ void dot_kernel(const float* A, const float* B, float* result, int N)

{
    __shared__ float warp_scratch[32]; // Max 32 warps (1024 threads)

    // Step 1: Compute partial dot product in strided loop
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        thread_sum += A[i] * B[i];
    }

    // Step 2: Intra-block reduction across all threads in the block
    float dot_product = block_reduce_sum(thread_sum, warp_scratch);

    // Step 3: Thread 0 writes the scalar result for row m to DRAM
    if (threadIdx.x == 0) {
        *result = dot_product;
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    if (N > 512) threads = 512;
    if (N > 1024) threads = 1024;    
    int blocks = 1; // (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    dot_kernel<<<blocks, threads>>>(A, B, result, N);
    cudaDeviceSynchronize();
}

