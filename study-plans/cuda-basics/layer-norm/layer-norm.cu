#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float shared_data[];
    float* s_sum = shared_data;
    float* s_sq_sum = &shared_data[blockDim.x];

    float sum = 0.0f;
    float sq_sum = 0.0f;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = input[row * N + i];
        sum += val;
        sq_sum += val * val;
    }

    s_sum[threadIdx.x] = sum;
    s_sq_sum[threadIdx.x] = sq_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sq_sum[threadIdx.x] += s_sq_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)N;
    float var = (s_sq_sum[0] / (float)N) - (mean * mean);
    float inv_std = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float normalized = (input[row * N + i] - mean) * inv_std;
        output[row * N + i] = normalized * gamma[i] + beta[i];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    size_t shared_size = 2 * threads * sizeof(float);
    
    layer_norm_kernel<<<blocks, threads, shared_size>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}