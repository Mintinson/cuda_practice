#ifndef SOFTMAX_V0_CUH_
#define SOFTMAX_V0_CUH_

#include <cstddef>
#include <cuda_runtime.h>

#include "softmax_helper.cuh"

// The naive cuda softmax implementation
// step1: Calculation of the maximum
// step2: Calculation of the exponential and sum
// step3: Calculation of the final result

namespace cudda {
template <typename T>
__global__ void softmax_v0_kernel(const T *__restrict__ A, T *__restrict__ C,
                                  int m, int n) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < m) {
    // max
    T max_val = -std::numeric_limits<T>::infinity();
    // norm factor
    T sum = 0.0f;

    // 3 passes (not optimal)
    for (int col = 0; col < n; col++) {
      int i = row * n + col;
      max_val = max(max_val, A[i]);
    }
    for (int col = 0; col < n; col++) {
      int i = row * n + col;
      sum += exp_op(A[i] - max_val);
    }
    for (int col = 0; col < n; col++) {
      int i = row * n + col;
      C[i] = exp_op(A[i] - max_val) / sum;
    }
  }
}

} // namespace cudda

#endif // SOFTMAX_V0_CUH_