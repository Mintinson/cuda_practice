#ifndef SOFTMAX_HELPER_CUH_
#define SOFTMAX_HELPER_CUH_

#include <cuda_runtime.h>

namespace cudda {
template <typename T> __device__ inline T exp_op(T x) {
  if constexpr (std::is_same_v<T, double>)
    return exp(x);
  else
    return expf(x);
}

} // namespace cudda

#endif // SOFTMAX_HELPER_CUH_