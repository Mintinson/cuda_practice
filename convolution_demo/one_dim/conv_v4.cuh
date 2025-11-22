#ifndef CONV_V4_CUH
#define CONV_V4_CUH

#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

namespace one_dim {

namespace v4 {
    template <typename T>
    __constant__ T dKernel[256];

}
template <typename T>
__global__ void conv_kernel_v4(const T* input, T* output, const size_t sz, const size_t ksz)
{
    extern __shared__ T shared[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t halfKsz = ksz / 2;
    // int startIdx = (blockIdx.x - 1) * blockDim.x;
    auto endIdx = (blockIdx.x + 1) * blockDim.x;
    auto baseId = blockIdx.x * blockDim.x;
    auto tid = threadIdx.x;
    if (tid < halfKsz) {
        // shared[halfKsz - tid - 1] = idx < 1 + tid ? T {} : input[idx - 1 - tid];
        shared[tid] = baseId < halfKsz - tid ? T {} : input[baseId + tid - halfKsz];
    } else if (tid >= blockDim.x - halfKsz) {
        auto increment = tid + halfKsz - blockDim.x;
        shared[halfKsz + blockDim.x + increment] = endIdx + increment >= sz ? T {} : input[endIdx + increment];
    }
    if (idx < sz) {

        shared[tid + halfKsz] = input[idx];
        // shared[0] = input[idx];
    }
    __syncthreads();

    if (idx < sz) {
        // const size_t halfKsz = ksz / 2;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            tmp += shared[tid + i] * v4::dKernel<T>[i];
        }
        output[idx] = tmp;
    }
}

template <typename T>
__global__ void conv_kernel_v4_pad(const T* input, T* output, size_t sz, size_t ksz)
{
    extern __shared__ T shared[];
    // auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto baseId = blockIdx.x * blockDim.x;
    auto tid = threadIdx.x;

    for (size_t j = baseId; j < sz; j += blockDim.x * gridDim.x) {
        shared[tid] = input[j + tid];
        if (tid < ksz - 1) {
            auto tidk = tid + blockDim.x;
            shared[tidk] = input[j + tidk];
        }
        __syncthreads();
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            tmp += shared[tid + i] * v4::dKernel<T>[i];
        }
        output[j + tid] = tmp;
    }
}
}
#endif // CONV_V4_CUH