#ifndef SOFTMAX_V5_CUH_
#define SOFTMAX_V5_CUH_

#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include "softmax_helper.cuh"
#include "softmax_v4.cuh"

namespace cudda
{
    constexpr int TILE_SIZE = 4;
    /*
    Instead of having one block calculate only one row of the output matrix, one block
    will compute `TILE_SIZE` number of rows. This way we will have fewer blocks, and more
    computations per block. 2D threads will process elements. `tx` will process elements and
    `ty` will process rows in a block

    We will need partial block-wide reduction with width `tx` threads participating in the reduction
    for each row's maximum and norm value.
    */
    template <typename T>
    __global__ void softmax_v5_kernel(T *__restrict__ matd, T *__restrict__ resd, int M, int N)
    {
        const int ty = threadIdx.y;
        const int tx = threadIdx.x;
        const int row = blockIdx.x * TILE_SIZE + ty;
        if (row >= M)
            return;

        T local_max = std::numeric_limits<T>::lowest();
        T local_norm = 0.f;

        for (int i = tx; i < N; i += blockDim.x)
        {
            const T x = matd[row * N + i];
            if (x > local_max)
            {
                local_norm *= exp_op(local_max - x);
                local_max = x;
            }
            local_norm += exp_op(x - local_max);
        }

        const T thread_max = local_max;
        local_max = warpReduceMax(local_max);
        const T row_max = __shfl_sync(0xffffffffu, local_max, 0);
        local_norm *= exp_op(thread_max - row_max);
        local_norm = warpReduceSum(local_norm);
        const T row_norm = __shfl_sync(0xffffffffu, local_norm, 0);

        for (int i = tx; i < N; i += blockDim.x)
            resd[row * N + i] = exp_op(matd[row * N + i] - row_max) / row_norm;
    }
} // namespace cudda

#endif // SOFTMAX_V5_CUH_
