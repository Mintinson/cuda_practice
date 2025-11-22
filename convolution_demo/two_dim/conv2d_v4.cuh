#ifndef CONV2D_V4_CUH
#define CONV2D_V4_CUH

// #include <__clang_cuda_builtin_vars.h>
#include <cstddef>
#include <cuda_runtime.h>

#include <cstdlib>
#include <sys/types.h>

namespace two_dim {
namespace v4 {
    template <typename T>
    __constant__ T dKernel[25 * 25];
}
template <typename T>
__global__ void conv_kernel_v4_pad(
    const T* input, T* output,
    std::size_t m, std::size_t n,
    std::size_t kWidth)
{
    auto tidY = threadIdx.y;
    auto tidX = threadIdx.x;
    auto baseIdY = blockDim.y * blockIdx.y;
    auto baseIdX = blockDim.x * blockIdx.x;
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    extern __shared__ T sInput[];
    const dim3 regionWidth(blockDim.x + kWidth - 1, blockDim.y + kWidth - 1);
    const size_t halfKSize = kWidth / 2;

    sInput[(tidY + halfKSize) * regionWidth.y + (tidX + halfKSize)] = input[(idy + halfKSize) * (n + kWidth - 1) + (idx + halfKSize)];
    int offsetX = 0;
    int offsetY = 0;
    if (tidX < halfKSize) {
        offsetX = -halfKSize;
    } else if (tidX >= blockDim.x - halfKSize) {
        offsetX = halfKSize;
    }
    if (tidY < halfKSize) {
        offsetY = -halfKSize;
    } else if (tidY > blockDim.y - halfKSize) {
        offsetY = tidY + halfKSize - blockDim.y;
    }
    if (offsetX != 0 || offsetY != 0) {
        // if (offsetX < 0 && offsetY < 0) {
        size_t sY = tidY + halfKSize + offsetY;
        size_t sX = tidX + halfKSize + offsetX;
        size_t iY = idy + halfKSize + offsetY;
        size_t iX = idx + halfKSize + offsetX;
        sInput[(sY)*regionWidth.y + (sX)]
            = input[(iY) * (n + kWidth - 1) + (iX)];
        // }
    }
    // if (idy < kWidth - 1) {
    //     sInput[(tidY + blockDim.y) * regionWidth.y + tidX]
    //         = input[(baseIdY + tidY + blockDim.y) * (n + kWidth - 1) + idx];
    // } else if (tidY < (2 * kWidth - 2)) {
    //     auto tidL = tidX;
    //     auto tidK = tidY - (kWidth - 1) + blockDim.x;
    //     sInput[tidK * regionWidth.y + tidL] = input[(baseIdY + tidK) * (n + kWidth - 1) + baseIdX + tidL];
    // } else if (tidY < (3 * kWidth - 3) && tidX < kWidth - 1) {
    //     auto tidK = tidY - 2 * (kWidth - 1) + blockDim.y;
    //     auto tidL = tidX + blockDim.x;
    //     sInput[tidK * regionWidth.y + tidL] = input[(baseIdY + tidK) * (n + kWidth - 1) + baseIdX + tidL];
    // }
    __syncthreads();

    for (size_t row = idy; row < m; row += blockDim.y * gridDim.y) {
        for (size_t col = idx; col < n; col += blockDim.x * gridDim.x) {

            T tmp {};
            for (size_t i = 0; i < kWidth; ++i) {
                for (size_t j = 0; j < kWidth; ++j) {
                    tmp += sInput[(tidY + i) * regionWidth.y + (tidX + j)] * v4::dKernel<T>[i * kWidth + j];
                }
            }
            output[row * n + col] = tmp;
        }
    }
}
}

#endif // CONV2D_V4_CUH