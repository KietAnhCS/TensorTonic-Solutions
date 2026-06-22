#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    __shared__ float shared_max;
    __shared__ float shared_sum;

    int tid = threadIdx.x;

    // --- BƯỚC 1: TÌM MAX TOÀN CỤC ---
    float local_max = -INFINITY;
    for (int idx = tid; idx < N; idx += BLOCK_SIZE) {
        local_max = fmaxf(local_max, input[idx]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_max = sdata[0];
    }
    __syncthreads();
    float max_val = shared_max;

    // --- BƯỚC 2: TÌM TỔNG TẤT CẢ HÀM MŨ ---
    float local_sum = 0.0f;
    for (int idx = tid; idx < N; idx += BLOCK_SIZE) {
        local_sum += expf(input[idx] - max_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_sum = sdata[0];
    }
    __syncthreads();
    float sum_val = shared_sum;

    // --- BƯỚC 3: TÍNH TOÁN SOFTMAX ĐẦU RA ---
    for (int idx = tid; idx < N; idx += BLOCK_SIZE) {
        output[idx] = expf(input[idx] - max_val) / sum_val;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    // Ép hệ thống chỉ chạy đúng 1 Block duy nhất để đồng bộ an toàn bằng __syncthreads()
    int threads = BLOCK_SIZE;
    int blocks = 1; 
    
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}