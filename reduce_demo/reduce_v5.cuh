#ifndef REDUCE_V5_CUH_
#define REDUCE_V5_CUH_

#include <cstddef>
#include <cuda_runtime.h>

template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v5(const T* input, T* output, size_t sz, ReduceOp op)
{
    extern __shared__ T shared_mem[];
    // auto baseId = blockIdx.x * blockDim.x;
    auto idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    shared_mem[threadIdx.x] = op((idx < sz ? input[idx] : T {}),
        (idx + blockDim.x < sz ? input[idx + blockDim.x] : T {}));
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] = op(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

#endif // REDUCE_V5_CUH_