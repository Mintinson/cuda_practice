#ifndef MAT_TRANSPOSE_V0_CUH
#define MAT_TRANSPOSE_V0_CUH

#include <cstddef>
#include <cuda_runtime.h>

template <typename T>
__global__ void mat_trans_kernel_v0(const T* src, T* dst, size_t m, size_t n)
{
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        dst[col * m + row] = src[row * n + col];
    }
}

#endif // MAT_TRANSPOSE_V0_CUH