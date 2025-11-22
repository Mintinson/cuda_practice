//
// Created by asus on 2025/4/12.
//

#ifndef COMPACT_CUDA_CUH
#define COMPACT_CUDA_CUH

#include "cuda_runtime.h"
#include "vector_types.h"
#include <cstddef>

namespace compact_v0
{
    template <typename T>
    __device__ constexpr auto& fetch_vec4(T* ptr)
    {
        using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
        if constexpr (std::is_same_v<DecayType, float>)
        {
            return reinterpret_cast<float4*>((ptr))[0];
        }
        else if constexpr (std::is_same_v<DecayType, double>)
        {
            return reinterpret_cast<double4*>((ptr))[0];
        }
        else if constexpr (std::is_same_v<DecayType, int>)
        {
            return reinterpret_cast<int4*>((ptr))[0];
        }
        else if constexpr (std::is_same_v<DecayType, unsigned int>)
        {
            return reinterpret_cast<uint4*>((ptr))[0];
        }
        else if constexpr (std::is_same_v<DecayType, char>)
        {
            return reinterpret_cast<char4*>((ptr))[0];
        }
    }

    template <typename T, typename U, typename Pred, std::enable_if_t<std::is_arithmetic_v<T>, void>* = nullptr>
    __global__ void pred_map_vec(const T* src, const size_t size, U* dst, Pred pred)
    {
        auto idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        auto reg_src = fetch_vec4(const_cast<T*>(src + idx));
        // auto reg_b = fetch_vec4(d_b + idx);

        // decltype(reg_src) reg_out;
        char4 reg_dst;
        reg_dst.x = pred(reg_src.x);
        reg_dst.y = pred(reg_src.y);
        reg_dst.z = pred(reg_src.z);
        reg_dst.w = pred(reg_src.w);
        fetch_vec4(dst + idx) = reg_dst;
    }

    template <typename T, typename U, typename Pred>
    __global__ void pred_map(const T* src, const size_t size, U* dst, Pred pred)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            dst[idx] = pred(src[idx]);
        }
    }

#define BANK_INDEX(x) ((x) + (x) / 32)

    template <typename T, typename U, typename Binary = std::plus<U>>
    __global__ void brent_kung_kernel_block(const T* src, size_t n, U* dst, Binary binary = {})
    {
        extern __shared__ U shared[];
        const auto tid = threadIdx.x;
        auto idx = 2 * blockIdx.x * blockDim.x + tid;

        for (; idx < n; idx += 2 * blockDim.x * gridDim.x)
        {
            shared[BANK_INDEX(tid)] = static_cast<U>(src[idx]);
            shared[BANK_INDEX(tid + blockDim.x)] = static_cast<U>(src[idx + blockDim.x]);
            // sweep up
            for (size_t stride = 1; stride <= blockDim.x; stride <<= 1)
            {
                __syncthreads();
                size_t index = (tid + 1) * 2 * stride - 1;
                if (index < blockDim.x * 2)
                {
                    shared[BANK_INDEX(index)] = binary(shared[BANK_INDEX(index)],
                                                       shared[BANK_INDEX(index - stride)]);
                }
            }
            // sweep down
            for (size_t stride = (blockDim.x * 2) / 4; stride > 0; stride >>= 1)
            {
                __syncthreads();
                size_t index = (tid + 1) * 2 * stride - 1;
                if ((index + stride) < blockDim.x * 2)
                {
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
        if (idx < n)
        {
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

        for (; idx < n; idx += 2 * blockDim.x * gridDim.x)
        {
            if (tid == 0)
            {
                shared[0] = blockIdx.x > 0 ? src[idx / (blockDim.x * 2) - 1] : T{};
            }
            __syncthreads();
            dst[idx] = binary(dst[idx], init);
            dst[idx + blockDim.x] = binary(dst[idx + blockDim.x], init);
            if (idx >= blockDim.x)
            {
                dst[idx] = binary(shared[0], dst[idx]);
                dst[idx + blockDim.x] = binary(shared[0], dst[idx + blockDim.x]);
            }
        }
    }

    template <typename T, typename Binary = std::plus<>>
    __global__ void scan_single_kernel(const T* src, size_t n, T* dst, Binary binary = {})
    {
        dst[0] = src[0];
        for (size_t i = 1; i < n; ++i)
        {
            dst[i] = binary(dst[i - 1], src[i]);
        }
    }

    template <typename T>
    __global__ void compact_distribute_kernel(const std::size_t* indices, const char* preds, std::size_t n,
                                              const T* src, T* dst)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        for (; idx < n; idx += blockDim.x * gridDim.x)
        {
            if (preds[idx])
            {
                dst[indices[idx] - 1] = src[idx];
            }
            else
            {
                dst[idx + indices[n - 1] - indices[idx]] = src[idx];
            }
        }
    }
}
#endif //COMPACT_CUDA_CUH
