#ifndef NAIVE_CUH_
#define NAIVE_CUH_

#include <cuda_runtime.h>

template <typename T, typename Operator>
__global__ void element_wise_naive_kernel(T* d_a, T* d_b, T* d_out, size_t n, Operator op)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    // c[idx] = a[idx] + b[idx];
    if (idx < n) {
        d_out[idx] = op(d_a[idx], d_b[idx]);
    }
}

#endif // NAIVE_CUH_