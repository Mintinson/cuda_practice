//
// Created by asus on 2025/4/20.
//

#ifndef MATMUL_V7_CUH
#define MATMUL_V7_CUH
#include <cuda_runtime.h>

template <std::size_t N, typename T>
__global__ void matmul_v7_kernel(const T* A, const T* B, T* C, const size_t row, const size_t col, const size_t depth
)
{
    extern __shared__ T sharedMem[];
    T* sharedA = sharedMem;
    T* sharedB = sharedMem + (blockDim.x * blockDim.y) * N;
    const auto blockRow = blockIdx.y * blockDim.y * N;
    const auto blockCol = blockIdx.x * blockDim.x * N;
    auto rowId = threadIdx.y;
    auto colId = threadIdx.x;
    T sum[N * N] = {};
    for (size_t j = 0; j < depth; j += 1 * blockDim.x)
    {
        for (size_t i = 0; i < N; i++)
        {
            sharedA[(rowId + (i * blockDim.y)) * blockDim.x + colId] =
                A[(rowId + (i * blockDim.y) + blockRow) * depth + j + colId];
            sharedB[rowId * blockDim.x * N + colId + i * blockDim.x] =
                B[(j + rowId) * col + colId + blockCol + i * blockDim.y];
        }

        __syncthreads();
        for (int ii = 0; ii < N; ii++)
        {
            for (int jj = 0; jj < N; jj++)
            {
                for (size_t k = 0; k < blockDim.x; ++k)
                {
                    sum[ii * N + jj] +=
                        sharedA[(rowId + ii * blockDim.y) * blockDim.x + k]
                        * sharedB[k * blockDim.x * N + colId + jj * blockDim.x];
                }
            }
        }
        __syncthreads();
    }
    for (size_t i = 0; i < N; ++i)
    {
        std::size_t rid = (rowId + (i * blockDim.y) + blockRow);
        if (rid < row)
        {
            for (int j = 0; j < N; ++j)
            {
                std::size_t cid = colId + blockCol + j * blockDim.y;
                if (cid < col)
                {
                    C[rid * col + cid]
                        = sum[i * N + j];
                }
            }
        }

    }
}


#endif //MATMUL_V7_CUH
