#ifndef REDUCE_V11_CUH_
#define REDUCE_V11_CUH_

#include <cstddef>
#include <cuda_runtime.h>

// __reduce_add_sync is available for signed/unsigned 32-bit integers on sm_80+.
// All 32 lanes participate, including lanes whose loads are out of range.
template <int ThreadsPerBlock>
__global__ void reduce_kernel_v11_ampere(const int *input, int *output, size_t size)
{
    static_assert(ThreadsPerBlock % 32 == 0 && ThreadsPerBlock <= 1024);
    constexpr int warp_size = 32;
    __shared__ int warp_sums[ThreadsPerBlock / warp_size];

    const int tid = threadIdx.x;
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;
    const size_t index = static_cast<size_t>(blockIdx.x) * (2 * ThreadsPerBlock) + tid;

    int sum = index < size ? input[index] : 0;
    const size_t second = index + ThreadsPerBlock;
    sum += second < size ? input[second] : 0;
    sum = __reduce_add_sync(0xffffffffu, sum);
    if (lane == 0)
        warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0)
    {
        sum = lane < ThreadsPerBlock / warp_size ? warp_sums[lane] : 0;
        sum = __reduce_add_sync(0xffffffffu, sum);
        if (lane == 0)
            output[blockIdx.x] = sum;
    }
}

#endif // REDUCE_V11_CUH_
