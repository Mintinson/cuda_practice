#ifndef SOFTMAX_V5_CUH_
#define SOFTMAX_V5_CUH_

#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
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
        int bx = blockDim.x;

        // ty equals TILE_SIZE
        int ty = threadIdx.y;
        int tx = threadIdx.x;

        // result matrix's row
        int row = (bx * TILE_SIZE + ty);
        if (row >= M)
            return;

        // one for each row
        T local_maxs[TILE_SIZE] = {std::numeric_limits<T>::min()};
        T local_norms[TILE_SIZE] = {0.f};
        T x[TILE_SIZE] = {0.f};

        for (int i = tx; i < N; i += bx)
        {
#pragma unroll
            for (int j = 0; j < TILE_SIZE; j++)
            {
                x[j] = matd[row * N + i];
                if (x[j] > local_maxs[j])
                {
                    local_norms[j] *= expf(local_maxs[j] - x[j]);
                    local_maxs[j] = x[j];
                }
                local_norms[j] += expf(x[j] - local_maxs[j]);
            }
        }
        __syncthreads();

        for (int tile = 0; tile < TILE_SIZE; tile++)
        {
            T lm = local_maxs[tile];
            local_maxs[tile] = warpReduceMax(lm);
            local_norms[tile] *= expf(lm - local_maxs[tile]);
            local_norms[tile] = warpReduceSum(local_norms[tile]);
        }

        // finally, compute softmax
        for (int i = tx; i < N; i += bx)
            for (int tile = 0; tile < TILE_SIZE; tile++)
                resd[row * N + i] = expf(matd[row * N + i] - local_maxs[tile]) / local_norms[tile];
    }
} // namespace cudda

#endif // SOFTMAX_V5_CUH_