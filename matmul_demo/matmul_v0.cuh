#ifndef MATMUL_V0_CUH_
#define MATMUL_V0_CUH_

#include <cstddef>
#include <cuda_runtime.h>

template <typename T>
__global__ void matmul_v0_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t rowIdx = idx / col;
    std::size_t colIdx = idx % col;
    // std::size_t colIdx = (idx & ((2 * col) - 1));

    if (rowIdx < row && colIdx < col) {
        T sum = 0;
        for (size_t k = 0; k < depth; ++k) {
            sum += A[rowIdx * depth + k] * B[k * col + colIdx];
        }
        C[rowIdx * col + colIdx] = sum;
    }
}

#endif // MATMUL_V0_CUH_