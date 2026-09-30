#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_SM 2

// -------------------------------------------------------------
// Fast Swish / SiLU Operator using SFU __expf
// -------------------------------------------------------------
__device__ __forceinline__ float swish_op(float x) {
    // x / (1.0f + exp(-x)) -> x * sigmoid(x)
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float4 process_float4(float4 in) {
    float4 out;
    out.x = swish_op(in.x);
    out.y = swish_op(in.y);
    out.z = swish_op(in.z);
    out.w = swish_op(in.w);
    return out;
}

// =============================================================
// Swish Kernel: ILP Dual-Vector float4 Grid-Stride Loop
// =============================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
swish_kernel(const float* __restrict__ input, 
             float* __restrict__ output, 
             int N) 
{
    // 2x float4 = 8 floats per thread per unrolled iteration
    const int ELEMENTS_PER_THREAD = 8;
    int num_unrolled_chunks = N / ELEMENTS_PER_THREAD;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    // ---------------------------------------------------------
    // Main ILP Loop: Issue 2x 128-bit loads, compute, 2x stores
    // ---------------------------------------------------------
    for (int i = tid; i < num_unrolled_chunks; i += stride) {
        int idx4 = i * 2;

        // Concurrent pipelined 128-bit loads (LDG.E.128)
        float4 in0 = in_vec[idx4];
        float4 in1 = in_vec[idx4 + 1];

        // Evaluate Swish with SFU __expf
        float4 out0 = process_float4(in0);
        float4 out1 = process_float4(in1);

        // Concurrent 128-bit stores (STG.E.128)
        out_vec[idx4]     = out0;
        out_vec[idx4 + 1] = out1;
    }

    // ---------------------------------------------------------
    // Tail 1: Process leftover float4 chunk (if 4 <= remaining < 8)
    // ---------------------------------------------------------
    int processed_floats = num_unrolled_chunks * ELEMENTS_PER_THREAD;
    int total_float4s = N / 4;

    for (int i = (processed_floats / 4) + tid; i < total_float4s; i += stride) {
        out_vec[i] = process_float4(in_vec[i]);
    }
    processed_floats = total_float4s * 4;

    // ---------------------------------------------------------
    // Tail 2: Scalar leftovers (< 4 elements)
    // ---------------------------------------------------------
    for (int i = processed_floats + tid; i < N; i += stride) {
        output[i] = swish_op(input[i]);
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

    // Full SM wave quantization: 4 blocks per SM ensures latency hiding
    int blocks_per_sm = 4;
    int target_blocks = prop.multiProcessorCount * blocks_per_sm;

    int elements_per_block = THREADS_PER_BLOCK * 8; // 2048 elements/block
    int needed_blocks = (N + elements_per_block - 1) / elements_per_block;
    if (needed_blocks == 0) needed_blocks = 1;

    int blocks = (needed_blocks < target_blocks) ? needed_blocks : target_blocks;

    swish_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}