#ifndef REDUCE_V9_CUH_
#define REDUCE_V9_CUH_

#include "helper.cuh"
#include <cstddef>
#include <cuda_runtime.h>

template <typename T, typename ReduceOp>
__device__ INLINE T warp_reduce_sum(T sum, size_t blockSize, ReduceOp op)
{
    if (blockSize >= 32)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 16));
    if (blockSize >= 16)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 8));
    if (blockSize >= 8)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 4));
    if (blockSize >= 4)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 2));
    if (blockSize >= 2)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 1));
    return sum;
}

template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v9(const T* input, T* output, size_t sz, T init, ReduceOp op)
{
    T sum = init;
    constexpr size_t kWarpSize = 32;
    // each thread loads one element from global memory to shared mem
    auto idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    auto tid = threadIdx.x;
    auto blockSize = blockDim.x;

    // similar to previous load two elements from global memory
#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        // sum += d_in[i + iter * blockSize];
        size_t bIdx = idx + iter * blockSize;
        sum = op(sum, (bIdx < sz ? input[bIdx] : init));
    }

    static __shared__ T warpLevelSums[kWarpSize];
    const auto laneId = tid % kWarpSize;
    const auto warpId = tid / kWarpSize;

    sum = warp_reduce_sum(sum, blockSize, op);

    if (laneId == 0)
        warpLevelSums[warpId] = sum;
    __syncthreads();
    // only the first warp need to cross-warp reduce
    sum = (tid < blockSize / kWarpSize) ? warpLevelSums[laneId] : init;
    if (warpId == 0)
        sum = warp_reduce_sum(sum, blockSize / kWarpSize, op);
    if (tid == 0)
        output[blockIdx.x] = sum;
}

#endif // REDUCE_V9_CUH_