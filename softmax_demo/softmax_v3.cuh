#ifndef SOFTMAX_V3_CUH_
#define SOFTMAX_V3_CUH_

#include <cstddef>
#include <cuda_runtime.h>

namespace cudda
{
    /*
        This one is largely similar to the above kernel. The difference is instead of accessing
        shared memory and having sync barrier overhead, we will use warp-level primitives (then
        block-level) for performing max and sum reductions. The benefit is: it is faster than shared
        memory access and also does not need syncing since each warp (group of 32 threads) execute
        an instuction parallely on GPU so no chance of race conditions.
    */
    template <typename T>
    __global__ void softmax_v3_kernel(T *__restrict__ matd, T *__restrict__ resd, int M, int N)
    {
        // max and norm reduction will happen in shared memory (static)
        __shared__ T smem[1024];

        int row = blockIdx.x;
        int tid = threadIdx.x;
        // number of threads in a warp
        unsigned int warp_size = 32;
        if (row >= M)
            return;

        T *input_row = matd + row * N;
        T *output_row = resd + row * N;
        T local_max = std::numeric_limits<T>::min();
        T local_norm = 0.0f;

        for (int i = tid; i < N; i += blockDim.x)
        {
            T x = input_row[i];
            if (x > local_max)
            {
                local_norm *= expf(local_max - x);
                local_max = x;
            }
            local_norm += expf(x - local_max);
        }
        __syncthreads();

        // each thread will have its own local max
        // we store it in shared memory for reduction
        // smem[tid] = local_max;
        // __syncthreads();

        // warp level reduction using XOR shuffle ('exchanges' the values in the threads)
        // note: if there are 256 threads in one block (8 warps of 32 threads each)
        // the following for loop reduces the value in all the 8 warps
        // the 8 warps contain the 8 maximum values of the 32 threads that reside in those warps
        // T val = smem[tid];
        T val = local_max;
        for (int offset = warp_size / 2; offset > 0; offset /= 2)
        {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
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
                val = (tid < (blockDim.x + warp_size - 1) / warp_size) ? smem[tid] : std::numeric_limits<T>::min();
                for (int offset = warp_size / 2; offset > 0; offset /= 2)
                {
                    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
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
        __syncthreads();

        // each thread will have its own local_norm
        // we will store the corrected local_norm in the shared memory
        // smem[tid] = local_norm * expf(local_max - row_max);
        // __syncthreads();

        // same reduction algorithm as above, but instead of max reduction
        // we do a sum reduction i.e. we accumulate the values
        // val = smem[tid];
        val = local_norm * expf(local_max - row_max);
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
        __syncthreads();

        // finally, compute softmax
        for (int i = tid; i < N; i += blockDim.x)
        {
            output_row[i] = expf(input_row[i] - row_max) / row_norm;
        }
    }
} // namespace cudda

#endif // SOFTMAX_V3_CUH_