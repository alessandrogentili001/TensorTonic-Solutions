#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

// State tuple for parallel Welford reduction
struct WelfordData {
    float n;   // Count of elements
    float mu;  // mean_out
    float m2;  // Sum of squared differences from mean_out (M2)
};

// Merge two Welford states
__device__ __forceinline__ WelfordData combine_welford(WelfordData a, WelfordData b) {
    if (a.n == 0.0f) return b;
    if (b.n == 0.0f) return a;

    float new_n = a.n + b.n;
    float delta = b.mu - a.mu;
    float new_mu = a.mu + delta * (b.n / new_n);
    float new_m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / new_n);

    return {new_n, new_mu, new_m2};
}

// Update running state with a single scalar element x
__device__ __forceinline__ WelfordData update_welford(WelfordData cur, float x) {
    float new_n = cur.n + 1.0f;
    float delta = x - cur.mu;
    float new_mu = cur.mu + delta / new_n;
    float new_m2 = cur.m2 + delta * (x - new_mu);
    return {new_n, new_mu, new_m2};
}

// -------------------------------------------------------------
// Warp-level Welford reduction using register shuffles
// -------------------------------------------------------------
__device__ inline WelfordData warp_reduce_welford(WelfordData val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        WelfordData other;
        other.n  = __shfl_down_sync(0xffffffff, val.n, offset);
        other.mu = __shfl_down_sync(0xffffffff, val.mu, offset);
        other.m2 = __shfl_down_sync(0xffffffff, val.m2, offset);
        val = combine_welford(val, other);
    }
    return val;
}

// -------------------------------------------------------------
// Block-level Welford reduction
// -------------------------------------------------------------
__device__ inline WelfordData block_reduce_welford(WelfordData val, WelfordData* warp_scratch) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    val = warp_reduce_welford(val);

    if (lane == 0) {
        warp_scratch[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        WelfordData b_val = (lane < num_warps) ? warp_scratch[lane] : WelfordData{0.0f, 0.0f, 0.0f};
        b_val = warp_reduce_welford(b_val);
        if (lane == 0) {
            warp_scratch[0] = b_val;
        }
    }
    __syncthreads();

    return warp_scratch[0];
}

// =============================================================
// mean_out & var_outiance Kernel (Vectorized 128-bit Single-Block)
// =============================================================
__global__ void mean_out_var_outiance_kernel(const float* __restrict__ input, 
                                float* __restrict__ mean_out, 
                                float* __restrict__ var_out, 
                                int N) 
{
    __shared__ WelfordData warp_scratch[32];

    int tid = threadIdx.x;
    int num_vec = N / 4;
    const float4* in4 = reinterpret_cast<const float4*>(input);

    WelfordData local_welford = {0.0f, 0.0f, 0.0f};

    // 1. Vectorized 128-bit loads
    for (int i = tid; i < num_vec; i += blockDim.x) {
        float4 v = in4[i];
        local_welford = update_welford(local_welford, v.x);
        local_welford = update_welford(local_welford, v.y);
        local_welford = update_welford(local_welford, v.z);
        local_welford = update_welford(local_welford, v.w);
    }

    // 2. Scalar leftover tail
    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < N; i += blockDim.x) {
        local_welford = update_welford(local_welford, input[i]);
    }

    // 3. Block-wide reduction
    WelfordData final_welford = block_reduce_welford(local_welford, warp_scratch);

    // 4. Thread 0 writes final mean_out and population/sample var_outiance to DRAM
    if (tid == 0) {
        *mean_out = final_welford.mu;
        // Population var_outiance: M2 / N (use final_welford.n - 1.0f if unbiased sample var_outiance is requested)
        *var_out  = (final_welford.n > 0.0f) ? (final_welford.m2 / final_welford.n) : 0.0f;
    }
}

// =============================================================
// Host Entry Point (Conforms to Harness: solve(input, mean_out, var_out, N))
// =============================================================
extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    if (N > 512)  threads = 512;
    if (N > 1024) threads = 1024;

    mean_out_var_outiance_kernel<<<1, threads>>>(input, mean_out, var_out, N);
    cudaDeviceSynchronize();
}

