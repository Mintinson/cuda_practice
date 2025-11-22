#ifndef MAT_TRANS_V2_CUH
#define MAT_TRANS_V2_CUH

#include <cstddef>
#include <cuda_runtime.h>
namespace transv2 {

constexpr size_t TILE_WIDTH = 16;
}
template <typename T>
__global__ void mat_trans_kernel_v2(const T* src, T* dst, size_t m, size_t n)
{
    __shared__ T tile[transv2::TILE_WIDTH][transv2::TILE_WIDTH + 1];

    const size_t row = blockIdx.y * transv2::TILE_WIDTH + threadIdx.y;
    const size_t col = blockIdx.x * transv2::TILE_WIDTH + threadIdx.x;

    if (row < m && col < n) {
        tile[threadIdx.y][threadIdx.x] = src[row * n + col];
    }
    __syncthreads();

    const size_t newRow = blockIdx.x * transv2::TILE_WIDTH + threadIdx.y;
    const size_t newCol = blockIdx.y * transv2::TILE_WIDTH + threadIdx.x;

    if (newRow < n && newCol < m) {
        dst[newRow * m + newCol] = tile[threadIdx.x][threadIdx.y];
    }
}
#endif // MAT_TRANS_V2_CUH
