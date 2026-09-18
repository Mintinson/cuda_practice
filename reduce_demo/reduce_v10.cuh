#ifndef REDUCE_V10_CUH_
#define REDUCE_V10_CUH_

#include "helper.cuh"
#include <cstddef>
#include <cuda_runtime.h>

template <typename T>
__device__ constexpr auto *fetch_vec4_ptr(T *ptr)
{
    using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<DecayType, float>)
    {
        return reinterpret_cast<float4 *>(ptr);
    }
    else if constexpr (std::is_same_v<DecayType, double>)
    {
        return reinterpret_cast<double4 *>(ptr);
    }
    else if constexpr (std::is_same_v<DecayType, int>)
    {
        return reinterpret_cast<int4 *>(ptr);
    }
    else if constexpr (std::is_same_v<DecayType, unsigned int>)
    {
        return reinterpret_cast<uint4 *>(ptr);
    }
}

// V10: float4 向量化加载 + Grid Stride Loop + Warp Shuffle
template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v10(T *input, T *output, size_t sz, T init, ReduceOp op)
{
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    // float4 加载：每线程每次处理 4 个 float
    auto *input4 = fetch_vec4_ptr(input);
    size_t n4 = sz / 4; // float4 的元素数量

    T val = init;

    // Grid Stride Loop：每个线程以 gridDim.x * blockDim.x 为步长迭代
    // 注意，这里每个线程都有这个循环，因此一次核函数，就将所有元素合并到了
    // gridDim.x * blockDim.x 个线程中的 val 中
    for (size_t idx = blockIdx.x * blockDim.x + tid;
         idx < n4;
         idx += gridDim.x * blockDim.x)
    {
        // 连取 4 个元素
        auto data = input4[idx];
        val = op(val, op(data.x, op(data.y, op(data.z, data.w))));
    }

    // 处理 n 不是 4 的倍数时的尾部元素
    size_t tail_start = n4 * 4;
#pragma unroll
    for (size_t idx = tail_start + blockIdx.x * blockDim.x + tid;
         idx < sz;
         idx += gridDim.x * blockDim.x)
    {
        // val += input[idx];
        val = op(val, input[idx]);
    }

    // Warp 内规约
    // 将 gridDim.x * blockDim.x 个线程中的 val 按照 warp 合并
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val = op(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    __shared__ T warp_results[32];
    // warp 中第一个元素负责将其放到 shared mem 中
    if (lane == 0)
        warp_results[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    // 每个block中的第一个 warp 的线程负责从 shared memo 中取值
    if (wid == 0)
    {
        // Combine the results from all warps
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            val = op(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }

    // 每个block中的第一个线程负责保存最终值
    if (tid == 0)
        output[blockIdx.x] = val;

    // 最终output 保存 grid_size 个值
    // 因此后续还要做一次长度为 grid_size 的 reduce
}
#endif // REDUCE_V10_CUH_