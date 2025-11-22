#ifndef CONV_V2_CUH
#define CONV_V2_CUH

#include <cstddef>
#include <cuda_runtime.h>

namespace one_dim {

template <typename T>
__constant__ T dKernel[256];
template <typename T>
__global__ void conv_kernel_v2(const T* input, T* output, const size_t sz, const size_t ksz)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < sz) {
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {

        const size_t halfKsz = ksz / 2;
        int startIdx = j - halfKsz;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            if (startIdx + i >= 0 && startIdx + i < sz) {
                tmp += input[startIdx + i] * dKernel<T>[i];
            }
        }
        output[j] = tmp;
    }
}

template <typename T>
__global__ void conv_kernel_v2_pad(const T* input, T* output, size_t sz, size_t ksz)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {

        // const size_t halfKsz = ksz / 2;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            tmp += input[j + i] * dKernel<T>[i];
        }
        output[j] = tmp;
    }
}
}
#endif // CONV_V2_CUH