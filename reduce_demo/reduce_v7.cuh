#ifndef REDUCE_V7_CUH_
#define REDUCE_V7_CUH_

#include <cstddef>
#include <cuda_runtime.h>

template <typename T, typename ReduceOp>
__device__ void warpReduce_v7(volatile T* data, size_t id, ReduceOp op)
{
    data[id] = op(data[id], data[id + 32]);
    data[id] = op(data[id], data[id + 16]);
    data[id] = op(data[id], data[id + 8]);
    data[id] = op(data[id], data[id + 4]);
    data[id] = op(data[id], data[id + 2]);
    data[id] = op(data[id], data[id + 1]);
}
template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v7(const T* input, T* output, size_t sz, ReduceOp op)
{
    extern __shared__ T shared_mem[];
    auto tid = threadIdx.x;
    const auto blockSize = blockDim.x;
    auto idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    shared_mem[tid] = op((idx < sz ? input[idx] : T {}),
        (idx + blockDim.x < sz ? input[idx + blockDim.x] : T {}));
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) {
            shared_mem[tid] = op(shared_mem[tid], shared_mem[tid + 256]);
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            shared_mem[tid] = op(shared_mem[tid], shared_mem[tid + 128]);
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            shared_mem[tid] = op(shared_mem[tid], shared_mem[tid + 64]);
        }
        __syncthreads();
    }
    if (tid < 32) {
        warpReduce_v7(shared_mem, tid, op);
        // no syncthreads()
    }
    if (tid == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

#endif // REDUCE_V7_CUH_