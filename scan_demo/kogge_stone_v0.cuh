#ifndef KOGGE_STONE_V0_CUH
#define KOGGE_STONE_V0_CUH

#include "checker.cuh"
#include <cstddef>
#include <cstdio>
#include <functional>

template <typename T, typename Binary = std::plus<>>
__global__ void kogge_stone_kernel_v0(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[];
    const auto tid = threadIdx.x;
    auto idx = blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += blockDim.x * gridDim.x) {
        shared[tid] = src[idx];
        __syncthreads();
        for (size_t stride = 1; stride <= tid; stride *= 2) {
            // printf("stride=%d\n", stride);
            shared[tid] = binary(shared[tid - stride], shared[tid]);
            __syncthreads();
        }
        dst[idx] = shared[tid];
    }
}
template <typename T, typename Binary = std::plus<>>
__global__ void kogge_stone_kernel_v1(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[]; // 2 * blockDim.x
    const auto tid = threadIdx.x;
    auto idx = blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += blockDim.x * gridDim.x) {
        shared[tid] = src[idx];
        bool pout = false;
        bool pin = true;
        for (size_t stride = 1; stride <= blockDim.x; stride *= 2) {
            // printf("stride=%d\n", stride);
            pout = !pout;
            pin = !pout;
            __syncthreads();
            shared[pout * blockDim.x + tid] = shared[pin * blockDim.x + tid];
            if (tid >= stride) {

                shared[pout * blockDim.x + tid] = binary(shared[pout * blockDim.x + tid],
                    shared[pin * blockDim.x + tid - stride]);
            }
        }
        __syncthreads();
        dst[idx] = shared[pout * blockDim.x + tid];
    }
}
template <typename T>
__global__ void kogge_stone_collect_kernel(const T* src, size_t n, T* dst)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[(idx + 1) * blockDim.x - 1];
    }
}
template <typename T, typename Binary = std::plus<>>
__global__ void kogge_stone_distribute_kernel(const T* src, size_t n, T* dst, T init = {}, Binary binary = {})
{
    __shared__ T shared[2];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto tid = threadIdx.x;

    for (; idx < n; idx += blockDim.x * gridDim.x) {
        if (tid == 0) {
            shared[0] = blockIdx.x > 0 ? src[idx / blockDim.x - 1] : T {};
            // shared[1] = blockIdx.x > 0 ? src[idx / blockDim.x - 1 + blockDim.x] : T {};
        }
        __syncthreads();
        dst[idx] = binary(dst[idx], init);
        if (idx >= blockDim.x) {
            dst[idx] = binary(shared[0], dst[idx]);
        }
    }
}
template <typename T, typename Binary = std::plus<>>
__global__ void kogge_stone_single_kernel(const T* src, size_t n, T* dst, Binary binary = {})
{
    dst[0] = src[0];
    for (size_t i = 1; i < n; ++i) {
        dst[i] = binary(dst[i - 1], src[i]);
    }
}
template <typename T, typename Binary = std::plus<>>
__device__ T scan_warp(volatile T* sharedPartials, Binary binary = {})
{
    const size_t tid = threadIdx.x;
    const size_t lane = tid & 0x1f; // % 31
    if (lane >= 1) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 1));
    }
    if (lane >= 2) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 2));
    }
    if (lane >= 4) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 4));
    }
    if (lane >= 8) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 8));
    }
    if (lane >= 16) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 16));
    }
    return sharedPartials[0];
}
template <typename T, typename Binary = std::plus<>>
__device__ T scan_block(volatile T* sharedPartials, Binary binary = {})
{
    extern __shared__ T warpPartials[];
    const size_t tid = threadIdx.x;
    const size_t lane = tid & 0x1f; // %32
    const size_t warpId = tid >> 5; // /32
    T sum = scan_warp(sharedPartials, binary);
    __syncthreads();
    if (lane == 31) {
        warpPartials[16 + warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        scan_warp(16 + warpPartials + tid);
    }
    __syncthreads();
    if (warpId > 0) {
        sum = binary(sum, warpPartials[16 + warpId - 1]);
    }
    __syncthreads();
    *sharedPartials = sum;
    __syncthreads();
    return sum;
}
template <typename T, typename Binary = std::plus<>>
__global__ void scan_and_write_partials(const T* src, size_t n, T* dst, T* gPartials, size_t numBlocks, bool writeSpine, Binary binary = {})
{
    extern volatile __shared__ T sharedPartials[];
    const size_t tid = threadIdx.x;
    volatile T* myShared = sharedPartials + tid;

    for (size_t bid = blockIdx.x; bid < numBlocks; bid += gridDim.x) {
        size_t index = bid * blockDim.x + tid;

        *myShared = (index < n) ? src[index] : T {};
        __syncthreads();

        T sum = scan_block(myShared, binary);

        __syncthreads();
        if (index < n) {
            dst[index] = *myShared;
        }
        if (writeSpine && threadIdx.x == blockDim.x - 1) {
            gPartials[bid] = sum;
        }
    }
}
template <typename T, typename Binary = std::plus<>>
__global__ void scan_add_base_sums(T* baseSums, size_t n, T* dst, size_t numBlocks, T init = {}, Binary binary = {})
{
    const size_t tid = threadIdx.x;
    T fanValue { init };
    for (size_t bid = blockIdx.x; bid < numBlocks; bid += gridDim.x) {
        size_t index = bid * blockDim.x + tid;
        if (bid > 0) {
            fanValue = binary(fanValue, baseSums[bid - 1]);
        }
        dst[index] = binary(dst[index], fanValue);
    }
}
template <typename T, typename Binary = std::plus<>>
void warp_scan_fan(const T* src, size_t n, T* dst, size_t blockSize, T init = {}, Binary binary = {})
{
    if (n <= blockSize) {
        scan_and_write_partials<T><<<1, blockSize, blockSize * sizeof(T)>>>(
            src, n, dst,
            nullptr, 1, false, binary);
        return;
    }
    T* gPartials = nullptr;
    size_t numPartials = (n + blockSize - 1) / blockSize;
    // const size_t maxBlocks = 250;
    // size_t numBlocks = std::min(maxBlocks, numPartials);
    // size_t numBlocks = numPartials;

    checkCudaErrors(cudaMalloc(&gPartials, numPartials * sizeof(T)));
    scan_and_write_partials<<<numPartials, blockSize, blockSize * sizeof(T)>>>(src, n, dst, gPartials, numPartials, true, binary);
    warp_scan_fan(gPartials, numPartials, gPartials, blockSize, {}, binary);
    scan_add_base_sums<<<numPartials, blockSize>>>(gPartials, n, dst, numPartials, init, binary);
    checkCudaErrors(cudaFree(gPartials));
}

#endif // KOGGE_STONE_V0_CUH