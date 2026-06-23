#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    atomicAdd(result,input[idx]);
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
