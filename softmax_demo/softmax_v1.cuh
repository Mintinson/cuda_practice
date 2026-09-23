#ifndef SOFTMAX_V1_CUH_
#define SOFTMAX_V1_CUH_

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
__global__ void softmax_v1_kernel(T* __restrict__ matd, T* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        T m = -std::numeric_limits<T>::infinity();
        T L = {};

        // compute max and norm factor in one pass only
        // by exploiting the property of exponentials
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            T curr = matd[i];
            if (curr > m) {
                L = L * exp_op(m - curr);
                m = curr;
            }
            L += expf(curr - m);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            resd[i] = exp_op(matd[i] - m) / L;
        }
    }
}
} // namespace cudda

#endif // SOFTMAX_V1_CUH_