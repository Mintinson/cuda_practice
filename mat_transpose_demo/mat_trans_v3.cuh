#ifndef MAT_TRANS_V3_CUH
#define MAT_TRANS_V3_CUH

#include <cstddef>
#include <cuda_runtime.h>
namespace transv3 {

constexpr size_t TILE_WIDTH = 16;
// 固定申请的最大 block 数量（每个维度），实际大小由 benchmark 决定
constexpr unsigned MAX_GRID_DIM = 64;
}

template <typename T>
__global__ void mat_trans_kernel_v3(const T* src, T* dst, size_t m, size_t n)
{
    __shared__ T tile[transv3::TILE_WIDTH][transv3::TILE_WIDTH + 1];

    const size_t mm = (m + transv3::TILE_WIDTH - 1) / transv3::TILE_WIDTH;
    const size_t nn = (n + transv3::TILE_WIDTH - 1) / transv3::TILE_WIDTH;

    for (size_t blockY = blockIdx.y; blockY < mm; blockY += gridDim.y) {
        for (size_t blockX = blockIdx.x; blockX < nn; blockX += gridDim.x) {
            const size_t row = blockY * transv3::TILE_WIDTH + threadIdx.y;
            const size_t col = blockX * transv3::TILE_WIDTH + threadIdx.x;
            // 越界槽位写 0 是防御性写法，不是正确性要求：只有满足
            // newRow<n && newCol<m 的线程才会写 dst，而该条件等价于下面
            // tile[threadIdx.x][threadIdx.y] 所对应的源元素两个下标都在界内，
            // 即真正会被读取的槽位一定已被写入，未写入的槽位只会被丢弃。
            tile[threadIdx.y][threadIdx.x] = (row < m && col < n) ? src[row * n + col] : T{};
            __syncthreads();

            const size_t newRow = blockX * transv3::TILE_WIDTH + threadIdx.y;
            const size_t newCol = blockY * transv3::TILE_WIDTH + threadIdx.x;

            if (newRow < n && newCol < m) {
                dst[newRow * m + newCol] = tile[threadIdx.x][threadIdx.y];
            }

            // 防御性同步：同一个 block 会复用这块共享内存处理多个 tile。
            // 语义上其实不需要——线程必须读完本轮 tile 才能到达下一轮开头的
            // barrier，所以下一轮的写入不会覆盖未读完的数据；这里显式写出
            // 是为了让"共享内存复用"不依赖对 barrier 语义的推理。
            __syncthreads();
        }
    }
}
#endif // MAT_TRANS_V3_CUH
