//
// Created by asus on 2025/4/12.
//

#ifndef RADIX_SORT_CUH
#define RADIX_SORT_CUH

#include <cstddef>
#include <type_traits>

namespace cuda_sort
{
    template <typename T, bool Ascend = true>
    __device__ void radix_sort(T* arr, std::size_t n, T* tmp, std::size_t tid, std::size_t tdim)
    {
        static_assert(std::is_integral_v<T>, "T must be an integral type");
        if constexpr (std::is_unsigned_v<T>)
        {
            constexpr std::size_t numBits = 8 * sizeof(T);
            for (std::size_t bit = 0; bit < numBits; ++bit)
            {
                std::size_t index0 = 0;
                std::size_t index1 = 0;
                T mask = static_cast<T>(1) << bit;
                for (std::size_t i = 0; i < n; i += tdim)
                {
                    auto x = arr[i + tid];
                    if ((x & mask))
                    {
                        tmp[index1 + tid] = x;
                        index1 += tdim;
                    }
                    else
                    {
                        arr[index0 + tid] = x;
                        index0 += tdim;
                    }
                }
                for (std::size_t i = 0; i < index1; i += tdim)
                {
                    arr[i + index0 + tid] = tmp[i + tid];
                }
            }
        }
    }

    template <typename T, bool Ascend = true>
    __device__ void radix_merge_one(T* arr, std::size_t n, T* dst, std::size_t tid, std::size_t tdim)
    {
        extern __shared__ std::size_t listIndices[];
        listIndices[tid] = 0;
        __syncthreads();

        if (tid == 0)
        {
            auto n1 = n / tdim;
            for (std::size_t i = 0; i < n; ++i)
            {
                T minVal = std::numeric_limits<T>::max();
                std::size_t minIndex = 0;
                for (std::size_t list = 0; list < tdim; ++list)
                {
                    if (listIndices[list] < n1)
                    {
                        auto idx = list + (listIndices[list] * tdim);
                        auto x = arr[idx];
                        if (x < minVal)
                        {
                            minVal = x;
                            minIndex = list;
                        }
                    }
                }
                listIndices[minIndex]++;
                dst[i] = minVal;
            }
        }
    }
    template <typename T, bool Ascend = true>
    __global__ void radix_sort(T* arr, std::size_t tdim, std::size_t n)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        // __shared__  T
    }
} // namespace cuda_sort

#endif //RADIX_SORT_CUH
