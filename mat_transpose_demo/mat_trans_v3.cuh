#ifndef MAT_TRANS_V3_CUH
#define MAT_TRANS_V3_CUH

#include <cstddef>
#include <cuda_runtime.h>
namespace transv3 {

constexpr size_t TILE_WIDTH = 16;
}
template <typename T>
__global__ void mat_trans_kernel_v3(const T* src, T* dst, size_t m, size_t n)
{
    __shared__ T tile[transv3::TILE_WIDTH][transv3::TILE_WIDTH + 1];

    const size_t mm = (m + transv3::TILE_WIDTH - 1) / transv3::TILE_WIDTH;
    const size_t nn = (n + transv3::TILE_WIDTH - 1) / transv3::TILE_WIDTH;

    for (auto blockY = blockIdx.y; blockY < mm; blockY += gridDim.y) {
        for (auto blockX = blockIdx.x; blockX < nn; blockX += gridDim.x) {
            const size_t row = blockY * transv3::TILE_WIDTH + threadIdx.y;
            const size_t col = blockX * transv3::TILE_WIDTH + threadIdx.x;
            if (row < m && col < n) {
                tile[threadIdx.y][threadIdx.x] = src[row * n + col];
            }
            __syncthreads();
            const size_t newRow = blockX * transv3::TILE_WIDTH + threadIdx.y;
            const size_t newCol = blockY * transv3::TILE_WIDTH + threadIdx.x;

            if (newRow < n && newCol < m) {
                dst[newRow * m + newCol] = tile[threadIdx.x][threadIdx.y];
            }
        }
    }
}
#endif // MAT_TRANS_V3_CUH