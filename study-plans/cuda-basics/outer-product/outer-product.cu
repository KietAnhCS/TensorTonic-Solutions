#include <cuda_runtime.h>

__global__ void outer_product_kernel(const float* a, const float* b, float* C, int M, int N) {
    int col = blockIdx.x *blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;

    if (row<M && col <N) {
        C[row*N+col] = a[row]*b[col];
    }
}

extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    outer_product_kernel<<<blocks, threads>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}
