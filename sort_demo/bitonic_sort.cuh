//
// Created by asus on 2025/4/14.
//

#ifndef BITONIC_SORT_CUH
#define BITONIC_SORT_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include <functional>

namespace cuda_sort {
namespace bit_ns {
    // because std::swap is not constexpr nor __device__ marked, therefore we implement swap ourselves
    template <typename T>
    __device__ void swap(T& a, T& b) noexcept
    {
        T temp = std::move(a);
        a = std::move(b);
        b = std::move(temp);
    }
}

template <typename T, typename Compare = std::less<>>
__global__ void bitonic_sort_shared(T* srcKey, T* dstKey, std::size_t n, Compare comp = {})
{
    extern __shared__ T sharedKey[];

    auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    auto tid = threadIdx.x;
    sharedKey[tid] = srcKey[idx];
    sharedKey[tid + blockDim.x] = srcKey[idx + blockDim.x];

    for (std::size_t sz = 2; sz < n; sz <<= 1) {
        bool direction = (tid & (sz / 2)) != 0; // 0: ascend, 1: descend

        for (auto stride = sz >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            auto pos = 2 * tid - (tid & (stride - 1));
            if (direction
                    ? comp(sharedKey[pos], sharedKey[pos + stride])
                    : comp(sharedKey[pos + stride], sharedKey[pos])) {
                bit_ns::swap(sharedKey[pos], sharedKey[pos + stride]);
            }
        }
    }
    for (auto stride = n / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        auto pos = 2 * tid - (tid & (stride - 1));
        if (comp(sharedKey[pos + stride], sharedKey[pos])) {
            bit_ns::swap(sharedKey[pos], sharedKey[pos + stride]);
        }
    }
    __syncthreads();
    dstKey[idx] = sharedKey[tid];
    dstKey[idx + blockDim.x] = sharedKey[tid + blockDim.x];
}

// make every (4 * blockDim.x) sequence is bitonic sequence
template <typename T, typename Compare = std::less<>>
__global__ void bitonic_sort_sharedBlock(T* srcKey, T* dstKey, std::size_t n, Compare comp = {})
{
    extern __shared__ T sharedKey[];

    auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    auto tid = threadIdx.x;
    sharedKey[tid] = srcKey[idx];
    sharedKey[tid + blockDim.x] = srcKey[idx + blockDim.x];

    for (std::size_t sz = 2; sz < 2 * blockDim.x; sz <<= 1) {
        bool direction = (tid & (sz / 2)) != 0; // 0: ascend, 1: descend

        for (auto stride = sz >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            auto pos = 2 * tid - (tid & (stride - 1));
            if (direction
                    ? comp(sharedKey[pos], sharedKey[pos + stride])
                    : comp(sharedKey[pos + stride], sharedKey[pos])) {
                bit_ns::swap(sharedKey[pos], sharedKey[pos + stride]);
            }
        }
    }
    bool direction = blockIdx.x & 1; // here is the different, the event block ascend, while the odd block descend
    for (auto stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();
        auto pos = 2 * tid - (tid & (stride - 1));
        if (direction
                ? comp(sharedKey[pos], sharedKey[pos + stride])
                : comp(sharedKey[pos + stride], sharedKey[pos])) {
            bit_ns::swap(sharedKey[pos], sharedKey[pos + stride]);
        }
    }
    __syncthreads();
    dstKey[idx] = sharedKey[tid];
    dstKey[idx + blockDim.x] = sharedKey[tid + blockDim.x];
}

template <typename T, typename Compare = std::less<>>
__global__ void bitonic_merge_global(T* src, T* dst, std::size_t n, std::size_t sz, std::size_t stride,
    Compare comp = {})
{
    const auto globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto comparatorI = globalIdx & (n / 2 - 1);

    bool direction = (comparatorI & (sz / 2)) != 0;
    auto pos = 2 * globalIdx - (globalIdx & (stride - 1));
    T srcA = src[pos];
    T srcB = src[pos + stride];
    if (direction
            ? comp(srcA, srcB)
            : comp(srcB, srcA)) {
        bit_ns::swap(srcB, srcA);
    }
    dst[pos] = srcA;
    dst[pos + stride] = srcB;
}

template <typename T, typename Compare = std::less<>>
__global__ void bitonic_merge_shared(T* src, T* dst, std::size_t n, std::size_t sz,
    Compare comp = {})
{
    extern __shared__ T sharedKey[];
    auto tid = threadIdx.x;
    auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sharedKey[tid] = src[idx];
    sharedKey[tid + blockDim.x] = src[idx + blockDim.x];

    auto globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    auto comparatorI = globalIdx & (n / 2 - 1);

    bool direction = (comparatorI & (sz / 2)) != 0;
    for (auto stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();
        auto pos = 2 * tid - (tid & (stride - 1));
        if (direction
                ? comp(sharedKey[pos], sharedKey[pos + stride])
                : comp(sharedKey[pos + stride], sharedKey[pos])) {
            bit_ns::swap(sharedKey[pos], sharedKey[pos + stride]);
        }
    }
    __syncthreads();
    dst[idx] = sharedKey[tid];
    dst[idx + blockDim.x] = sharedKey[tid + blockDim.x];
}
} // namespace cuda_sort
#endif // BITONIC_SORT_CUH
