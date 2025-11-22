#ifndef REDUCE_V6_CUH_
#define REDUCE_V6_CUH_

#include <cstddef>
#include <cuda_runtime.h>

template <typename T, typename ReduceOp>
__device__ void warpReduce(volatile T* data, size_t id, ReduceOp op)
{
    data[id] = op(data[id], data[id + 32]);
    data[id] = op(data[id], data[id + 16]);
    data[id] = op(data[id], data[id + 8]);
    data[id] = op(data[id], data[id + 4]);
    data[id] = op(data[id], data[id + 2]);
    data[id] = op(data[id], data[id + 1]);
}
template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v6(const T* input, T* output, size_t sz, ReduceOp op)
{
    extern __shared__ T shared_mem[];
    // auto baseId = blockIdx.x * blockDim.x;
    auto idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    shared_mem[threadIdx.x] = op((idx < sz ? input[idx] : T {}),
        (idx + blockDim.x < sz ? input[idx + blockDim.x] : T {}));
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] = op(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        warpReduce(shared_mem, threadIdx.x, op);
        // no syncthreads()
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

#endif // REDUCE_V6_CUH_