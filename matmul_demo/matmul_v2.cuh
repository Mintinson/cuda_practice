#ifndef MATMUL_V2_CUH_
#define MATMUL_V2_CUH_

#include <cstddef>
#include <cuda_runtime.h>

template <typename T>
__global__ void matmul_v2_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    extern __shared__ T sharedData[];
    for (auto cId = tid; cId < depth; cId += blockDim.x)
    {
        sharedData[cId] = A[bid * depth + cId];
    }
    __syncthreads();

    // for (auto i = bid; i < row; i += gridDim.x)
    // {
    for (auto j = tid; j < col; j += blockDim.x)
    {
        T sum{};
        for (size_t k = 0; k < depth; ++k)
        {
            sum += sharedData[k] * B[k * col + j];
        }
        C[bid * col + j] = sum;
    }
    // }
}

#endif // MATMUL_V2_CUH_
