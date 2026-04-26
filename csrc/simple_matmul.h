#ifndef SIMPLE_MATMUL_H
#define SIMPLE_MATMUL_H

#include <stddef.h>
#include <cuda_runtime.h>

void simple_matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k, cudaStream_t stream);

#endif // SIMPLE_MATMUL_H