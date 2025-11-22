#ifndef BRENT_KUNG_V0_CUH
#define BRENT_KUNG_V0_CUH

// #include <__clang_cuda_builtin_vars.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <functional>

template <typename T, typename Binary = std::plus<>>
__global__ void brent_kung_kernel_v0(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[];
    const auto tid = threadIdx.x;
    auto idx = blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += blockDim.x * gridDim.x) {
        shared[tid] = src[idx];
        // sweep up
        for (size_t stride = 1; stride <= blockDim.x; stride <<= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x) {
                shared[index] = binary(shared[index], shared[index - stride]);
            }
        }
        // sweep down
        for (size_t stride = blockDim.x / 4; stride > 0; stride >>= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if ((index + stride) < blockDim.x) {
                shared[index + stride] = binary(shared[index], shared[index + stride]);
            }
        }
        __syncthreads();
        dst[idx] = shared[tid];
    }
}

template <typename T, typename Binary = std::plus<>>
__global__ void brent_kung_kernel_v1(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[];
    const auto tid = threadIdx.x;
    auto idx = 2 * blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += 2 * blockDim.x * gridDim.x) {
        shared[tid] = src[idx];
        shared[tid + blockDim.x] = src[idx + blockDim.x];
        // sweep up
        for (size_t stride = 1; stride <= blockDim.x; stride <<= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x * 2) {
                shared[index] = binary(shared[index], shared[index - stride]);
            }
        }
        // sweep down
        for (size_t stride = (blockDim.x * 2) / 4; stride > 0; stride >>= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if ((index + stride) < blockDim.x * 2) {
                shared[index + stride] = binary(shared[index], shared[index + stride]);
            }
        }
        __syncthreads();
        dst[idx] = shared[tid];
        dst[idx + blockDim.x] = shared[tid + blockDim.x];
    }
}
#define BANK_INDEX(x) ((x) + (x) / 32)
template <typename T, typename Binary = std::plus<>>
__global__ void brent_kung_kernel_v2(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[];
    const auto tid = threadIdx.x;
    auto idx = 2 * blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += 2 * blockDim.x * gridDim.x) {
        shared[BANK_INDEX(tid)] = src[idx];
        shared[BANK_INDEX(tid + blockDim.x)] = src[idx + blockDim.x];
        // sweep up
        for (size_t stride = 1; stride <= blockDim.x; stride <<= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x * 2) {
                shared[BANK_INDEX(index)] = binary(shared[BANK_INDEX(index)],
                    shared[BANK_INDEX(index - stride)]);
            }
        }
        // sweep down
        for (size_t stride = (blockDim.x * 2) / 4; stride > 0; stride >>= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if ((index + stride) < blockDim.x * 2) {
                shared[BANK_INDEX(index + stride)] = binary(shared[BANK_INDEX(index)],
                    shared[BANK_INDEX(index + stride)]);
            }
        }
        __syncthreads();
        dst[idx] = shared[BANK_INDEX(tid)];
        dst[idx + blockDim.x] = shared[BANK_INDEX(tid + blockDim.x)];
    }
}

template <typename T>
__global__ void brent_kung_collect_kernel(const T* src, size_t n, T* dst, const size_t blockSize)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[(idx + 1) * 2 * blockDim.x - 1];
    }
}
template <typename T, typename Binary = std::plus<>>
__global__ void brent_kung_distribute_kernel(
    const T* src, size_t n, T* dst,
    T init = {}, Binary binary = {})
{
    __shared__ T shared[1];
    auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    auto tid = threadIdx.x;

    for (; idx < n; idx += 2 * blockDim.x * gridDim.x) {
        if (tid == 0) {
            shared[0] = blockIdx.x > 0 ? src[idx / (blockDim.x * 2) - 1] : T {};
        }
        __syncthreads();
        dst[idx] = binary(dst[idx], init);
        dst[idx + blockDim.x] = binary(dst[idx + blockDim.x], init);
        if (idx >= blockDim.x) {
            dst[idx] = binary(shared[0], dst[idx]);
            dst[idx + blockDim.x] = binary(shared[0], dst[idx + blockDim.x]);
        }
    }
}

#endif //  BRENT_KUNG_V0_CUH