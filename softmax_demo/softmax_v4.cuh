#ifndef SOFTMAX_V4_CUH_
#define SOFTMAX_V4_CUH_

#include <cstddef>
#include <cuda_runtime.h>
#include <limits>

namespace cudda
{

    constexpr int WARP_SIZE = 32;

    // warp reduce for any Op
    template <typename T, typename Op>
    __device__ __forceinline__ T warpReduce(T val, Op op, unsigned int mask = 0xffffffffu)
    {
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
            val = op(val, __shfl_down_sync(mask, val, offset));
        return val;
    }

    // warp reduce for sum and max using lambda functions
    template <typename T>
    __device__ __forceinline__ T warpReduceSum(T val)
    {
        return warpReduce(val, [] __device__(T a, T b)
                          { return a + b; });
    }

    template <typename T>
    __device__ __forceinline__ T warpReduceMax(T val)
    {
        return warpReduce(val, [] __device__(T a, T b)
                          { return a > b ? a : b; });
    }

    template <typename T, typename Op>
    __device__ __forceinline__ void blockReduce(T val, T *smem, T identity, Op op)
    {
        int tx = threadIdx.x;
        int wid = tx / WARP_SIZE;
        int lane = tx % WARP_SIZE;

        val = warpReduce(val, op);

        // when blockDim is greater than 32, we need to do a block level reduction
        // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
        // the global max will be stored in the first warp
        if (blockDim.x > WARP_SIZE)
        {
            if (lane == 0)
            {
                // which warp are we at?
                // store the value in its first thread index
                smem[wid] = val;
            }
            __syncthreads();

            // first warp will do global reduction only
            // this is possible because we stored the values in the shared memory
            // so the threads in the first warp will read from it and then reduce
            if (tx < WARP_SIZE)
            {
                val = (tx < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? smem[tx] : identity;
                val = warpReduce(val, op);
                if (tx == 0)
                    smem[0] = val;
            }
        }
        else
        {
            // this is for when the number of threads in a block are not
            // greater than the warp size, in that case we already reduced
            // so we can store the value
            if (tx == 0)
                smem[0] = val;
        }
    }

    template <typename T>
    __device__ __forceinline__ void blockReduceSum(T val, T *smem, T identity)
    {
        return blockReduce(
            val, smem, identity, [] __device__(T a, T b)
            { return a + b; });
    }

    template <typename T>
    __device__ __forceinline__ void blockReduceMax(T val, T *smem, T identity)
    {
        return blockReduce(
            val, smem, identity, [] __device__(T a, T b)
            { return a > b ? a : b; });
    }

    template <typename T>
    __device__ constexpr auto *fetch_vec4_ptr(T *ptr)
    {
        using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
        if constexpr (std::is_same_v<DecayType, T>)
        {
            return reinterpret_cast<float4 *>((ptr));
        }
        else if constexpr (std::is_same_v<DecayType, double>)
        {
            return reinterpret_cast<double4 *>((ptr));
        }
        else if constexpr (std::is_same_v<DecayType, int>)
        {
            return reinterpret_cast<int4 *>((ptr));
        }
        else if constexpr (std::is_same_v<DecayType, unsigned int>)
        {
            return reinterpret_cast<uint4 *>((ptr));
        }
    }
    /*
        Instead of accessing shared memory and having sync barrier overhead, we will use warp-level primitives (then
        block-level) for performing max and sum reductions. The benefit is: it is faster than shared
        memory access and also does not need syncing since each warp (group of 32 threads) execute
        an instruction parallely on GPU so no chance of race conditions.

        We will also use vectorized loads and stores.
    */
    template <typename T>
    __global__ void softmax_v4_kernel(T *__restrict__ matd, T *__restrict__ resd, int M, int N)
    {
        // max and norm reduction will happen in shared memory (static)
        extern __shared__ T smem[];

        int row = blockIdx.x;
        int tid = threadIdx.x;
        if (row >= M)
            return;

        T *input_row = matd + row * N;
        T *output_row = resd + row * N;
        T local_max = std::numeric_limits<T>::min();
        T local_norm = 0.0f;

        // cast as float4
        int n_float4s = N / 4;
        int tail = N % 4;
        auto *input_row_vec = fetch_vec4_ptr(input_row);
        auto *output_row_vec = fetch_vec4_ptr(output_row);
        T maxval = std::numeric_limits<T>::min();

#pragma unroll
        for (int i = tid; i < n_float4s; i += blockDim.x)
        {
            float4 elem = input_row_vec[i];

            maxval = fmaxf(maxval, elem.x);
            maxval = fmaxf(maxval, elem.y);
            maxval = fmaxf(maxval, elem.z);
            maxval = fmaxf(maxval, elem.w);
            if (maxval > local_max)
            {
                local_norm *= __expf(local_max - maxval);
                local_max = maxval;
            }
            local_norm += __expf(elem.x - maxval);
            local_norm += __expf(elem.y - maxval);
            local_norm += __expf(elem.z - maxval);
            local_norm += __expf(elem.w - maxval);
        }

        // handle extra row elements
        if (tail && tid < tail)
        {
            T val = input_row[n_float4s * 4 + tid];
            if (val > local_max)
            {
                local_norm *= __expf(local_max - val);
                local_max = val;
            }
            local_norm += __expf(val - local_max);
        }
        __syncthreads();

        // warp level reduction using XOR shuffle ('exchanges' the values in the threads)
        // note: if there are 256 threads in one block (8 warps of 32 threads each)
        // the following for loop reduces the value in all the 8 warps
        // the 8 warps contain the 8 maximum values of the 32 threads that reside in those warps
        // T val = smem[tid];
        blockReduceMax<T>(local_max, smem, std::numeric_limits<T>::min());
        __syncthreads();

        // we got the global row max now
        T row_max = smem[0];
        __syncthreads();

        // each thread will have its own local_norm
        // we will store the corrected local_norm and reduce it
        // same reduction algorithm as above, but instead of max reduction
        // we do a sum reduction i.e. we accumulate the values
        T val = local_norm * expf(local_max - row_max);
        blockReduceSum<T>(val, smem, 0.0f);
        __syncthreads();

        T row_norm = smem[0];
        __syncthreads();

// finally, compute softmax
#pragma unroll
        for (int i = tid; i < n_float4s; i += blockDim.x)
        {
            auto elem = input_row_vec[i];
            elem.x = __expf(elem.x - row_max) / row_norm;
            elem.y = __expf(elem.y - row_max) / row_norm;
            elem.z = __expf(elem.z - row_max) / row_norm;
            elem.w = __expf(elem.w - row_max) / row_norm;

            output_row_vec[i] = elem;
        }
        // write tail elements
        if (tail && tid < tail)
        {
            T val = input_row[n_float4s * 4 + tid];
            output_row[n_float4s * 4 + tid] = __expf(val - row_max) / row_norm;
        }
    }
} // namespace cudda

#endif // SOFTMAX_V4_CUH_