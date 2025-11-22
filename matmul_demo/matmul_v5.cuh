#ifndef MATMUL_V5_CUH_
#define MATMUL_V5_CUH_

#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename T>
__global__ void matmul_v5_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{

    extern __shared__ T sharedMem[];
    T* sharedA = sharedMem;
    T* sharedB = sharedMem + blockDim.x * blockDim.y;
    const auto blockRow = blockIdx.y * blockDim.y;
    const auto blockCol = blockIdx.x * blockDim.x;
    auto rowId = threadIdx.y;
    auto colId = threadIdx.x;
    T sum {};
    for (size_t j = 0; j < depth; j += blockDim.x) {
        sharedA[rowId * blockDim.x + colId] = A[(rowId + blockRow) * depth + j + colId];
        // sharedA[rowId * blockDim.x + colId] = static_cast<T>(0);
        sharedB[rowId * blockDim.x + colId] = B[(j + rowId) * col + colId + blockCol];
        // sharedB[rowId * blockDim.x + colId] = static_cast<T>(0);
        __syncthreads();
        for (size_t k = 0; k < blockDim.x; ++k) {
            sum += sharedA[rowId * blockDim.x + k] * sharedB[k * blockDim.x + colId];
        }
        __syncthreads();
    }
    // printf("blockRow: %d, blockCol: %d, rowId: %d, colId: %d, sum: %f\n", blockRow, blockCol, rowId, colId, sum);
    if (rowId + blockRow < row && colId + blockCol < col) {
        C[(rowId + blockRow) * col + colId + blockCol]
            = sum;
    }
}

#endif // MATMUL_V5_CUH_