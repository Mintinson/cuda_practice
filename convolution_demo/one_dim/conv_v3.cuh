#ifndef CONV_V3_CUH
#define CONV_V3_CUH

#include <cstddef>
#include <cuda_runtime.h>

namespace one_dim {

template <typename T>
__global__ void conv_kernel_v3(const T* input, T* output, const size_t sz, const T* kernel, const size_t ksz)
{
    extern __shared__ T dKernel[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < ksz) {
        dKernel[threadIdx.x] = kernel[threadIdx.x];
    }
    __syncthreads();
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {

        const size_t halfKsz = ksz / 2;
        int startIdx = j - halfKsz;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            if (startIdx + i >= 0 && startIdx + i < sz) {
                tmp += input[startIdx + i] * dKernel[i];
            }
        }
        output[j] = tmp;
    }
}

template <typename T>
__global__ void conv_kernel_v3_pad(const T* input, T* output, const size_t sz, const T* kernel, const size_t ksz)
{
    extern __shared__ T dKernel[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < ksz) {
        dKernel[threadIdx.x] = kernel[threadIdx.x];
    }
    __syncthreads();
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {

        // const size_t halfKsz = ksz / 2;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            tmp += input[j + i] * dKernel[i];
        }
        output[j] = tmp;
    }
}
}
#endif // CONV_V3_CUH