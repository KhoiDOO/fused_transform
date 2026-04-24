#ifndef SIMPLE_MATMUL_H
#define SIMPLE_MATMUL_H

#include <stddef.h>

void simple_matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k);

#endif // SIMPLE_MATMUL_H