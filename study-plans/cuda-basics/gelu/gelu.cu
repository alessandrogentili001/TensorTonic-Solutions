#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_SM 2

// -------------------------------------------------------------
// Exact Analytical GELU using CUDA's erff intrinsic
// Formula: 0.5f * x * (1.0f + erff(x * (1.0f / sqrt(2))))
// -------------------------------------------------------------
__device__ __forceinline__ float gelu_exact_op(float x) {
    const float INV_SQRT2 = 0.7071067811865475f; // 1.0f / sqrtf(2.0f)
    return 0.5f * x * (1.0f + erff(x * INV_SQRT2));
}

__device__ __forceinline__ float4 process_float4(float4 in) {
    float4 out;
    out.x = gelu_exact_op(in.x);
    out.y = gelu_exact_op(in.y);
    out.z = gelu_exact_op(in.z);
    out.w = gelu_exact_op(in.w);
    return out;
}

// =============================================================
// Exact GELU Kernel with 2x float4 ILP Unrolling
// =============================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
gelu_exact_kernel(const float* __restrict__ input, 
                  float* __restrict__ output, 
                  int N) 
{
    const int ELEMENTS_PER_THREAD = 8;
    int num_unrolled_chunks = N / ELEMENTS_PER_THREAD;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    // Main dual-vector unrolled loop
    for (int i = tid; i < num_unrolled_chunks; i += stride) {
        int idx4 = i * 2;

        float4 in0 = in_vec[idx4];
        float4 in1 = in_vec[idx4 + 1];

        float4 out0 = process_float4(in0);
        float4 out1 = process_float4(in1);

        out_vec[idx4]     = out0;
        out_vec[idx4 + 1] = out1;
    }

    // Tail 1: Process leftover float4 chunk
    int processed_floats = num_unrolled_chunks * ELEMENTS_PER_THREAD;
    int total_float4s = N / 4;

    for (int i = (processed_floats / 4) + tid; i < total_float4s; i += stride) {
        out_vec[i] = process_float4(in_vec[i]);
    }
    processed_floats = total_float4s * 4;

    // Tail 2: Scalar leftovers (< 4 elements)
    for (int i = processed_floats + tid; i < N; i += stride) {
        output[i] = gelu_exact_op(input[i]);
    }
}

// =============================================================
// Host Launcher with SM Wave Quantization
// =============================================================
extern "C" void solve(const float* input, float* output, int N) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int blocks_per_sm = 4;
    int target_blocks = prop.multiProcessorCount * blocks_per_sm;

    int elements_per_block = THREADS_PER_BLOCK * 8; // 2048 floats/block
    int needed_blocks = (N + elements_per_block - 1) / elements_per_block;
    if (needed_blocks == 0) needed_blocks = 1;

    int blocks = (needed_blocks < target_blocks) ? needed_blocks : target_blocks;

    gelu_exact_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}