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

在 benchmark 中，CPU 版本只作为**正确性参考**和**数量级对照**：它不参与 GPU 的带宽对比（见下文"CPU 基线"一节）。

## GPU 实现

### v0 二维线程块实现

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

这一简单的实现就比CPU 的实现性能高很多。**注意**：每个元素只被读一次、写一次，所以问题不在于"多次读写全局内存"，而在于**写操作无法合并**：同一个 warp 内相邻线程的 `col` 连续，但写入地址 `col * m + row` 相差 `m` 个元素，一个 warp 的 32 次写会落在 32 个不同的 32 字节 sector 上，写带宽被严重浪费。读是合并的，写不是，这就是 v0 的瓶颈。

### v1 使用共享内存优化

为了缓解这一问题，我们可以使用共享内存优化，将这些不连续的访存操作转移到访存更快的存储单元中（比如共享内存），已达到减少开销的目的。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504070946272.png)

如图所示，从原矩阵中连续地读取数据，保存到共享内存中，然后连续地写回到目标位置。只不过写回时读取的共享内存位置是转置后的位置。

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

### v2 避免 bank conflict

上述代码对共享内存是按列访问，容易导致 bank conflict，见 [bank conflict](../reduce_demo/note.md#优化4-解决-bank-冲突)

这里解决bank冲突的方法很简单，就是在定义共享内存时，多加一列数据，错开不同的bank。

```cpp
    __shared__ T tile[transv2::TILE_WIDTH][transv2::TILE_WIDTH + 1];
```

下面的图片解释了这一优化方式

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250419111636.png)

### v3 控制 block 数量：grid-stride 复用 tile

先纠正一个常见的误解：**共享内存是"每个驻留 block"的片上资源，不是整个 grid 共享的资源池**。`__shared__ T tile[...]` 的占用是按 resident block 计算的（每个 SM 能同时驻留多少 block，取决于每个 block 的共享内存与寄存器用量），grid 里 block 总数再多，也不会把共享内存"一次性全部申请掉"。所以"矩阵变大 → block 变多 → 共享内存不够用"这个推理是不成立的。

v3 真正做的事是：把 grid 的维度**人为限制**在一个固定上限（`MAX_GRID_DIM`），然后让每个 block 用 grid-stride 循环处理多个 tile，并复用同一块共享内存。这样做的动机是**调度与复用**：

* 减少需要调度/发射的 block 数量，降低 block 启动与调度开销；
* 每个 block 处理更多 tile，共享内存与寄存器在多次迭代中被摊销；
* grid-stride 循环本身也给编译器更多机会做 ILP。

代价是并行度下降（活跃 block 变少），所以收益与矩阵规模、以及 `MAX_GRID_DIM` 和实际 tile 数的匹配关系强相关。实测它的结果**不是单调的**：有的形状变快、有的变慢，而且这台机器上的运行间抖动大于该效应（见文末实测与结论）。因此 v3 的定位应当是一个**可调的调度/复用实验**，而不是"解决共享内存总量不足"的方案。

```cpp
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
            tile[threadIdx.y][threadIdx.x] = (row < m && col < n) ? src[row * n + col] : T{};
            __syncthreads();

            const size_t newRow = blockX * transv3::TILE_WIDTH + threadIdx.y;
            const size_t newCol = blockY * transv3::TILE_WIDTH + threadIdx.x;

            if (newRow < n && newCol < m) {
                dst[newRow * m + newCol] = tile[threadIdx.x][threadIdx.y];
            }

            __syncthreads();
        }
    }
}
```

详见 [mat_trans_v3](mat_trans_v3.cuh)。这里的两个细节容易被误判为"必须"，实际上都不是正确性的必要条件，但当前实现仍然都做了，理由如下：

1. **越界时不写 0 是安全的**。`tile[threadIdx.y][threadIdx.x]` 在越界时不写入，但转置读取 `tile[threadIdx.x][threadIdx.y]` 是跨线程的：读取者拿到的是源 `src[(blockY*TILE+threadIdx.x)*n + blockX*TILE+threadIdx.y]`。而只有满足 `newRow = blockX*TILE+threadIdx.y < n` 且 `newCol = blockY*TILE+threadIdx.x < m` 的线程才会写 `dst`，这两个条件正好等价于"该源元素的两个下标都在界内"。**所以任何真正会写 `dst` 的线程，读到的 tile 槽位都必然是别的线程已经写入过的**；没被写入的槽位只会被那些越界（不写 `dst`）的线程读到，那些值被直接丢弃。当前实现里写 `T{}` 属于**防御性写法**（让 shared memory 内容确定、便于调试），换成 predicated store 不会改变结果。
2. **每轮 tile 之间的第二次 `__syncthreads()` 是防御性的**。单次 barrier 已经足够：设本轮为 k，下一次迭代为 k+1。barrier 的语义是"所有线程都到达后才能通过"，而线程只有在**做完了本轮对 `tile` 的读取**之后才能到达 k+1 开头的那个 barrier；因此等任何一个线程真正开始写 k+1 的 `tile` 时，其他线程本轮对 `tile` 的读取必然已经结束。所以本轮末尾那次 barrier 在语义上是冗余的。保留它是为了把"共享内存复用"这件事显式写出来，代价是每轮多一次 barrier 开销。

## cuBLAS 实现

cuBLAS 默认使用**列主序**，而这里的输入和输出都是 C/C++ 常用的行主序。可以利用这一点完成转置：把行主序的 `m * n` 输入内存看成列主序的 `n * m` 矩阵，再调用 `cublasSgeam` 做转置。

`cublasSgeam` 的功能是矩阵加法：

```text
C = alpha * op(A) + beta * op(B)
```

本例只需要转置，因此设置 `alpha = 1`、`beta = 0`：

```cpp
float alpha = 1.0f;
float beta = 0.0f;

const cublasStatus_t status = cublasSgeam(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    m, n,
    &alpha, input, n,
    &beta, dummy, m,
    output, m);
```

这里的 `input` 在 cuBLAS 看来是一个列主序的 `n * m` 矩阵，转置后得到列主序的 `m * n` 矩阵；这块内存按行主序解释时，正好就是我们想要的 `n * m` 转置结果。`B` 的内容不会参与计算，但仍然准备一个合法的 `dummy` 指针更稳妥。

### cuBLAS 使用时的注意点

1. **不要直接套用行主序的 lda/ldb/ldc**。这些 leading dimension 是按 cuBLAS 的列主序定义的，本例中对应 `input: n`、`dummy: m`、`output: m`。
2. **检查 cuBLAS 的返回值**。`cublasSgeam` 返回的是 `cublasStatus_t`，不能只依赖后面的 `cudaGetLastError()`。
3. **handle 只创建一次并复用**。benchmark 结束后调用 `cublasDestroy(handle)`，不要每个矩阵尺寸都重复创建。
4. **计时时要保证在同一个 stream 上**。cuBLAS 默认使用 default stream；如果 kernel 使用了其他 stream，需要通过 `cublasSetStream` 设置到同一个 stream。
5. `cublasSgeam` 本质上是“转置 + 矩阵加法”，不是专门的 transpose kernel。这里利用 `beta = 0` 把它当作转置接口使用，适合用作性能参考。

## 实验结果分析

benchmark 只统计 kernel 时间，按照一次读输入、一次写输出计算 effective bandwidth。以当前设备的理论带宽 `256.032 GB/s` 为参考，几个较大矩阵的结果如下：

| 矩阵尺寸 | v0 | v1 | v2 | v3 | cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 x 2048 | 98.1 | 192.8 | 195.0 | 204.8 | 226.0 |
| 4096 x 4096 | 98.9 | 219.9 | 220.0 | 212.8 | 231.6 |
| 512 x 4096 | 97.0 | 182.0 | 183.3 | 193.0 | 215.6 |

单位为 `GB/s`，具体数据见 [mat_transpose_benchmark.csv](mat_transpose_benchmark.csv)。

可以得到几个直观结论：

1. v0 的瓶颈确实是转置写入不连续，较大矩阵只有约 `100 GB/s`。
2. v1 通过共享内存把全局内存访问变得连续，性能接近提升一倍。v2 的 padding 在本次设备上的收益很小，说明 bank conflict 并不是所有场景下的主要瓶颈。
3. v3 的结果不是单调变好：2048 x 2048 上优于 v2，但 4096 x 4096 上反而略慢。限制 grid 数量会减少调度开销，但也可能降低并行度，因此需要结合矩阵规模调参。
4. cuBLAS 在大矩阵上整体最好，4096 x 4096 达到约 `231.6 GB/s`，约为理论带宽的 `90%`。对于这种主要受内存带宽限制的操作，已经比较接近设备上限。
5. 小矩阵的 kernel 时间很短，启动开销和运行抖动占比很大，不能过度解读 v1、v2、v3 之间几个百分点的差距。

CPU 结果只用于正确性和数量级对照，不应和 GPU 的 effective bandwidth 直接比较；GPU benchmark 没有把主机和设备之间的数据传输计入时间。
