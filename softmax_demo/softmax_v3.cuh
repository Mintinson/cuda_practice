#ifndef SOFTMAX_V3_CUH_
#define SOFTMAX_V3_CUH_

#include <cstddef>
#include <cuda_runtime.h>
#include "softmax_helper.cuh"

namespace cudda
{
    template <typename T>
    __global__ void softmax_v3_kernel(T *__restrict__ matd, T *__restrict__ resd, int M, int N)
    {
        // max and norm reduction will happen in shared memory (static)
        extern __shared__ T smem[];

        int row = blockIdx.x;
        int tid = threadIdx.x;
        // number of threads in a warp
        unsigned int warp_size = 32;
        if (row >= M)
            return;

        T *input_row = matd + row * N;
        T *output_row = resd + row * N;
        T local_max = -std::numeric_limits<T>::infinity();
        T local_norm = 0.0f;

        for (int i = tid; i < N; i += blockDim.x)
        {
            T x = input_row[i];
            if (x > local_max)
            {
                local_norm *= exp_op(local_max - x);
                local_max = x;
            }
            local_norm += exp_op(x - local_max);
        }

        T val = local_max;
        for (int offset = warp_size / 2; offset > 0; offset /= 2)
        {
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        }

        // when blockDim is greater than 32, we need to do a block level reduction
        // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
        // the global max will be stored in the first warp
        if (blockDim.x > warp_size)
        {
            if (tid % warp_size == 0)
            {
                // which warp are we at?
                // store the value in its first thread index
                smem[tid / warp_size] = val;
            }
            __syncthreads();

            // first warp will do global reduction only
            // this is possible because we stored the values in the shared memory
            // so the threads in the first warp will read from it and then reduce
            if (tid < warp_size)
            {
                val = (tid < (blockDim.x + warp_size - 1) / warp_size) ? smem[tid] : std::numeric_limits<T>::lowest();
                for (int offset = warp_size / 2; offset > 0; offset /= 2)
                {
                    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
                }
                if (tid == 0)
                    smem[0] = val;
            }
        }
        else
        {
            // this is for when the number of threads in a block are not
            // greater than the warp size, in that case we already reduced
            // so we can store the value
            if (tid == 0)
                smem[0] = val;
        }
        __syncthreads();

        // we got the global row max now
        T row_max = smem[0];
        // smem is reused for the sum reduction; protect the row_max read from
        // being raced by lane 0 of an earlier-scheduled warp.
        __syncthreads();

        // same reduction algorithm as above, but instead of max reduction
        // we do a sum reduction i.e. we accumulate the values
        // val = smem[tid];
        val = local_norm * exp_op(local_max - row_max);
        for (int offset = warp_size / 2; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (blockDim.x > warp_size)
        {
            if (tid % warp_size == 0)
            {
                smem[tid / warp_size] = val;
            }
            __syncthreads();

            // first warp will do global reduction
            if (tid < warp_size)
            {
                val = (tid < (blockDim.x + warp_size - 1) / warp_size) ? smem[tid] : 0.0f;
                for (int offset = warp_size / 2; offset > 0; offset /= 2)
                {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                if (tid == 0)
                    smem[0] = val;
            }
        }
        else
        {
            if (tid == 0)
                smem[0] = val;
        }
        __syncthreads();

        T row_norm = smem[0];
        // __syncthreads();

        // finally, compute softmax
        for (int i = tid; i < N; i += blockDim.x)
        {
            output_row[i] = exp_op(input_row[i] - row_max) / row_norm;
        }
    }
} // namespace cudda

#endif // SOFTMAX_V3_CUH_
