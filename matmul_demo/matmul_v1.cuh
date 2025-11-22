#ifndef MATMUL_V1_CUH_
#define MATMUL_V1_CUH_

#include <cstddef>
#include <cuda_runtime.h>

template <typename T>
__global__ void matmul_v1_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;

    for (auto i = bid; i < row; i += gridDim.x) {
        // for ()
        for (auto j = tid; j < col; j += blockDim.x) {
            T sum {};
            for (size_t k = 0; k < depth; ++k) {
                sum += A[i * depth + k] * B[k * col + j];
            }
            C[i * col + j] = sum;
        }
    }
}

#endif // MATMUL_V1_CUH_