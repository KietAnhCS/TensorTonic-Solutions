#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__device__ float atomicMinFloat(float* addr, float val) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i, assumed;
    do {
        assumed = old; 
        old = atomicCAS(addr_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__global__ void min_kernel(const float* input, float* result, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < N) ? input[i] : FLT_MAX;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMinFloat(result, sdata[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    init_result<<<1, 1>>>(result);
    
    min_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, result, N);
    
    cudaDeviceSynchronize();
}