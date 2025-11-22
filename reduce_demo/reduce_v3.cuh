#ifndef REDUCE_V3_CUH_
#define REDUCE_V3_CUH_

#include <cstddef>
#include <cuda_runtime.h>

template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v3(const T* input, T* output, size_t sz, ReduceOp op)
{
    extern __shared__ T shared_mem[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared_mem[threadIdx.x] = idx < sz ? input[idx] : T {};
    __syncthreads();

    for (size_t i = 1; i < blockDim.x; i <<= 1) {
        // if (threadIdx.x % (2 * i) == 0) {
        auto index = 2 * i * threadIdx.x;
        if (index < blockDim.x) {

            shared_mem[index] = op(shared_mem[index], shared_mem[index + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

#endif // REDUCE_V3_CUH_