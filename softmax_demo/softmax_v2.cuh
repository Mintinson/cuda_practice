#ifndef SOFTMAX_V2_CUH_
#define SOFTMAX_V2_CUH_

#include <cstddef>
#include <cuda_runtime.h>
#include "softmax_helper.cuh"

namespace cudda
{
    /*
    How this works:
    One thread processes one entire row, but instead of 3 passes we do only 2 passes.
    This is possible due to the property of exponentials.
    We are parallelizing over the rows.
    */
    template <typename T>
    __global__ void softmax_v2_kernel(T *__restrict__ matd, T *__restrict__ resd, int M, int N)
    {
        extern __shared__ T smem[];

        int row = blockIdx.x;
        int tid = threadIdx.x;

        // edge condition (we don't process further)
        if (row >= M)
            return;

        T *input_row = matd + row * N;
        T *output_row = resd + row * N;
        T local_max = std::numeric_limits<T>::lowest();
        T local_norm = 0.0f;

        // compute local max and norm for each thread
        // and then finally have a sync barrier before moving on
        for (int i = tid; i < N; i += blockDim.x)
        {
            T x = input_row[i];
            if (x > local_max)
            {
                // local_norm *= expf(local_max - x);
                local_norm *= exp_op(local_max - x);
                local_max = x;
            }
            local_norm += exp_op(x - local_max);
        }
        // __syncthreads();

        // each thread will have its own local max
        // we store it in the tid of the shared memory
        smem[tid] = local_max;
        __syncthreads();

        // block-level reduction in O(log(N)) time over all threads
        // is faster than linear reduction over all threads
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (tid < stride)
            {
                smem[tid] = max(smem[tid], smem[tid + stride]);
            }
            // sync barrier before next iteration to ensure correctness
            __syncthreads();
        }

        // the first element after max reduction from all threads
        // will contain the global max for the row
        T row_max = smem[0];
        // Every thread must consume smem[0] before smem is reused below.
        __syncthreads();

        // each thread will have its own local norm
        // we will store the corrected local norm in the shared memory
        // again, exploits property of exponentials
        smem[tid] = local_norm * exp_op(local_max - row_max);
        __syncthreads();

        // sum reduction similar to above for global norm factor
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }
        T row_norm = smem[0];
        __syncthreads();

        // finally, compute softmax
        for (int i = tid; i < N; i += blockDim.x)
        {
            output_row[i] = exp_op(input_row[i] - row_max) / row_norm;
        }
    }
} // namespace cudda

#endif // SOFTMAX_V2_CUH_
