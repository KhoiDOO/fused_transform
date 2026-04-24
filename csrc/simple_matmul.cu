#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <simple_matmul.h>
#include <cuda_runtime.h>

// Simple CUDA kernel for matrix multiplication
__global__ void simple_matmul_kernel(const float *A, const float *B, float *C, int m, int k, int n) {

    // m: # of rows in A and C
    // k: # of columns in A and rows in B
    // n: # of columns in B and C

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

void simple_matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k) {
    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        static_cast<unsigned int>((n + block_dim.x - 1) / block_dim.x),
        static_cast<unsigned int>((m + block_dim.y - 1) / block_dim.y));

    simple_matmul_kernel<<<grid_dim, block_dim>>>(
        A,
        B,
        C,
        static_cast<int>(m),
        static_cast<int>(k),
        static_cast<int>(n));
}