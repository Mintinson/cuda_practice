#ifndef CONV2D_V2_CUH
#define CONV2D_V2_CUH

#include <cstddef>
#include <cuda_runtime.h>

#include <cstdlib>

namespace two_dim {
namespace v2 {
    template <typename T>
    __constant__ T dKernel[25 * 25];
}
template <typename T>
__global__ void conv_kernel_v2_pad(
    const T* input, T* output,
    std::size_t m, std::size_t n,
    std::size_t kWidth)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    for (size_t row = idy; row < m; row += blockDim.y * gridDim.y) {
        for (size_t col = idx; col < n; col += blockDim.x * gridDim.x) {

            T tmp {};
            for (size_t i = 0; i < kWidth; ++i) {
                for (size_t j = 0; j < kWidth; ++j) {
                    tmp += input[(row + i) * (n + kWidth - 1) + (j + col)] * v2::dKernel<T>[i * kWidth + j];
                }
            }
            output[row * n + col] = tmp;
        }
    }
}
}

#endif // CONV2D_V2_CUH