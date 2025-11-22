# Matrix Transpose 的实现与优化

## CPU 实现

首先是 CPU 实现，CPU的实现非常简单直观，当然，这里可以细分为读连续和写连续两种实现，见下列代码：

```cpp
// src: m * n, dst: n * m, and reading from src is continuous, if ContiguousR is true
template <typename T, bool ContiguousR = true>
void cpu_mat_trans(const T* src, T* dst, int m, int n)
{
    if constexpr (ContiguousR) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                dst[j * m + i] = src[i * n + j];
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                dst[i * m + j] = src[j * n + i];
            }
        }
    }
}
```

完整代码见 [cpu_mat_trans](cpu_mat_trans.hpp)

实验表明，两者的性能是相近的，但是整体上，读连续的实现性能略高（但是差距不大）。

## GPU 实现

### 优化0 二维线程块实现

由于矩阵转置实际上是一个一一映射的操作，所以简单的GPU实现很直接，直接将每个元素映射到对应的位置即可。

这里我们采用二维的线程块实现，当然也可以选用一维线程块：

```cpp
template <typename T>
__global__ void mat_trans_kernel_v0(const T* src, T* dst, size_t m, size_t n)
{
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        dst[col * m + row] = src[row * n + col];
    }
}
```

完整代码见 [mat_trans_v0](mat_trans_v0.cuh)

这一简单的实现就比CPU 的实现性能高很多。

### 优化1 使用共享内存优化

上述实现最大的问题就是对全局内存进行了多次读写，而且相邻线程的写操作不是连续的（相邻线程的写操作表现在矩阵上跨行操作，因此是不连续的），导致无法使用内存合并。

为了缓解这一问题，我们可以使用共享内存优化，将这些不连续的访存操作转移到访存更快的存储单元中（比如共享内存），已达到减少开销的目的。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504070946272.png)

如图所示，从原矩阵中连续地读取数据，保存到共享内存中，然后连续地写回到目标位置。只不过写会时读取的共享内存位置是转置后的位置。

```cpp
constexpr size_t TILE_WIDTH = 16;

template <typename T>
__global__ void mat_trans_kernel_v1(const T* src, T* dst, size_t m, size_t n)
{
    __shared__ T tile[transv1::TILE_WIDTH][transv1::TILE_WIDTH];

    const size_t row = blockIdx.y * transv1::TILE_WIDTH + threadIdx.y;
    const size_t col = blockIdx.x * transv1::TILE_WIDTH + threadIdx.x;

    if (row < m && col < n) {
        tile[threadIdx.y][threadIdx.x] = src[row * n + col];
    }
    __syncthreads();

    const size_t newRow = blockIdx.x * transv1::TILE_WIDTH + threadIdx.y;
    const size_t newCol = blockIdx.y * transv1::TILE_WIDTH + threadIdx.x;

    if (newRow < n && newCol < m) {
        dst[newRow * m + newCol] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 优化3 避免 bank conflict

上述代码对共享内存是按列访问，容易导致 bank conflict，见 [bank conflict](../reduce_demo/note.md#优化4-解决-bank-冲突)

这里解决bank冲突的方法很简单，就是在定义共享内存时，多加一列数据，错开不同的bank。

```cpp
    __shared__ T tile[transv2::TILE_WIDTH][transv2::TILE_WIDTH + 1];
```

下面的图片解释了这一优化方式

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250419111636.png)

### 优化4 针对大规模数据

当矩阵增大的时候，上面的方法会导致线程块变多，共享内存使用也变多，而共享内存时有限的，因此申请固定的blocks数量，通过在线程内循环完成转置。

```cpp
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
```

详见 [mat_trans_v3](mat_trans_v3.cuh)

至于最大的 Block 数量，最好通过实验来确定。