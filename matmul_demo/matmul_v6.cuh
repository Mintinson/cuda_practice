#ifndef MATMUL_V6_CUH_
#define MATMUL_V6_CUH_

#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename T>
__global__ void matmul_v6_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth, size_t sharedSize)
{
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    
    extern __shared__ T sharedData[];
    for (auto i = bid; i < row; i += gridDim.x) {
        for (auto p = tid; p < depth; p += sharedSize) {
            sharedData[tid] = A[i * depth + p];

            for (auto j = tid; j < col; j += blockDim.x) {
                T tmp { C[i * col + j] };
                for (size_t k = 0; k < sharedSize; ++k) {
                    tmp += A[i * depth + k] * B[k * col + j];
                }
                C[i * col + j] = tmp;
            }
        }
    }
}

#endif // MATMUL_V6_CUH_