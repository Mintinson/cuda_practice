# Matrix Multiplication 的实现与优化

## CPU 实现

cpu 实现完整代码见 [cpu_matmul.hpp](cpu_matmul.hpp)

### 最直接的矩阵乘法

```cpp
template <typename T>
void matmul_general(const T* a, const T* b, T* c, int m, int n, int l)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            T tmp {};
            for (int k = 0; k < l; k++) {
                tmp += a[i * l + k] * b[k * n + j];
            }
            c[i * n + j] = tmp;
        }
    }
}
```

这种矩阵乘法存在缺陷，最内部的循环对矩阵 `b` 的元素是按列读取的，在C++中，元素是按行存储的，这很有可能导致编译器对内存访问的优化很有限，即不能向量化对B的读取。

### 循环交换矩阵乘法

针对一般矩阵乘法中存在的访问不连续，无法向量化的问题，一种改进方法是交换内部两层循环，使得数据访问连续。

如图所示，这种矩阵乘法的核心思想是每次读取 a 矩阵的一个元素，与B矩阵对应的一行做乘积，然后将乘积的结果累加到 c 矩阵的对应位置上。

![此次应该有图]()

```cpp
// if isInit is true, function assume that c is all zero, so we won't fill c with zero first
template <typename T, bool isInit = true>
void matmul_swap_loop(const T* a, const T* b, T* c, std::size_t m, std::size_t n, std::size_t l)
{
    if constexpr (!isInit) {

        std::fill(c, c + m * n, T {});
    }
    for (std::size_t i = 0; i < m; i++) {
        for (std::size_t k = 0; k < l; k++) {
            T tmp = a[i * l + k];
            for (std::size_t j = 0; j < n; j++) {
                c[i * n + j] += tmp * b[k * n + j];
            }
        }
    }
}
```

### 转置矩阵乘法

首先将矩阵B转置，然后进行矩阵乘法。这样只有在转置过程中对 B 有不连续的访问，因此可以提高内存访问效率。

```cpp
// if isTranspose is true, function assume that b is n x l, so we won't transpose b
template <typename T, bool isTranspose = true>
void matmul_transpose(const T* a, const T* b, T* c, std::size_t m, std::size_t n, std::size_t l)
{
    T* transB = const_cast<T*>(b);
    if constexpr (!isTranspose) {
        transB = static_cast<T*>(malloc(sizeof(T) * n * l));
        for (std::size_t i = 0; i < n; i++) {
            for (std::size_t j = 0; j < l; j++) {
                transB[i * l + j] = b[j * n + i];
            }
        }
    }
    for (std::size_t i = 0; i < m; i++) {
        for (std::size_t j = 0; j < n; j++) {
            T tmp {};
            for (std::size_t k = 0; k < l; k++) {
                tmp += a[i * l + k] * transB[j * l + k];
            }
            c[i * n + j] = tmp;
        }
    }
    if constexpr (isTranspose) {
        free(transB);
    }
}
```

在某些编译器上，转置矩阵乘法可以提高性能。但是在 GCC9 中，循环交换矩阵乘法的性能相对高些。

## GPU 实现

### 优化0： grid 线程循环矩阵乘法

最简单直接的实现是将 [最简单直接的矩阵乘法](#最直接的矩阵乘法) 中外部两个循环分别用 grid 和 block 分别进行划分，这样每个
线程都可以独立计算一个 c 矩阵的元素。

```cpp
template <typename T>
__global__ void matmul_v0_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t rowIdx = idx / col;
    std::size_t colIdx = idx % col;
    // std::size_t colIdx = (idx & ((2 * col) - 1));

    if (rowIdx < row && colIdx < col) {
        T sum = 0;
        for (size_t k = 0; k < depth; ++k) {
            sum += A[rowIdx * depth + k] * B[k * col + colIdx];
        }
        C[rowIdx * col + colIdx] = sum;
    }
}

matmul_v0_kernel<<<(row * col + BlockSize - 1) / BlockSize, BlockSize>>>(
        d_input_a.data, d_input_b.data, d_output.data, row, col, depth);
```

这个 kernel 的算术强度极低。计算一个 $C[i][j]$ ：

- 运算量： $2K$ FLOPs
- 数据搬运：读取 $A$ 的一行（ $K$ 个 float）+ $B$ 的一列（ $K$ 个 float）= $8K$ Bytes

$$
\text{算术强度} = \frac{2K}{8K} = 0.25 \text{ FLOPs/Byte}
$$

远低于平衡点，严重受带宽限制。实测通常只有理论峰值的 6%~11%。

### 优化1： block 线程循环矩阵乘法

对上述方法做进一步改进，每个block计算C矩阵的一行，block内的 thread 以固定跳步步长 `blockDim.x` 的方法循环计算 C 矩阵的一行；

每一行启动一个 block，共计 m 个 block。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504060853726.png)

```cpp
template <typename T>
__global__ void matmul_v1_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;

    for (auto i = bid; i < row; i += gridDim.x) {
        // for ()
        for (auto j = tid; j < col; j += blockDim.x) {
            T sum {};
            for (size_t k = 0; k < depth; ++k) {
                sum += A[i * depth + k] * B[k * col + j];
            }
            C[i * col + j] = sum;
        }
    }
}

```

### 优化2：行共享存储矩阵乘法

可以用共享存储来优化矩阵乘法，将矩阵 A 的一行存储在共享内存中，然后让每个线程计算一行的 C 矩阵元素。

```cpp
template <typename T>
__global__ void matmul_v2_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    extern __shared__ T sharedData[];
    for (auto cId = tid; cId < depth; cId += blockDim.x)
    {
        sharedData[cId] = A[bid * depth + cId];
    }
    __syncthreads();

    for (auto j = tid; j < col; j += blockDim.x)
    {
        T sum{};
        for (size_t k = 0; k < depth; ++k)
        {
            sum += sharedData[k] * B[k * col + j];
        }
        C[bid * col + j] = sum;
    }

}
```

### 优化3：使用二维的malloc来实现对齐访问

如果参与运算的矩阵不是 BlockSize 的整数倍，此时矩阵存储不对齐，将影响访存的性能。

为了优化对齐访问，可以在矩阵空间分配时每行后添加空白空间，从而保证下一行数据的访存对齐。

CUDA 提供了 `cudaMallocPitch()` 函数，可以分配一个自动对齐的二维数组，并返回数组的 pitch，即数组的行首地址的间隔。（即行与行之间间隔的字节数）

而拷贝二维数组，也可以用 `cudaMemcpy2D()`, 注意二维只是逻辑上的二维，实际数组依然是一个一维数组。

关于上述两个函数的用法见 [cuda_runtime_API, memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY).

这里注意的是，`pitch`是字节数，而不是元素个数，因此需要除以`sizeof(T)`。

```cpp
template <typename T>
void matmul_gpu_v3(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    size_t pitchA;
    size_t pitchB;
    size_t pitchC;
    T* d_a;
    T* d_b;
    T* d_c;
    checkCudaErrors(cudaMallocPitch(&d_a, &pitchA, depth * sizeof(T), row));
    checkCudaErrors(cudaMallocPitch(&d_b, &pitchB, col * sizeof(T), depth));
    checkCudaErrors(cudaMallocPitch(&d_c, &pitchC, col * sizeof(T), row));
    checkCudaErrors(cudaMemcpy2D(d_a, pitchA, A, depth * sizeof(T), depth * sizeof(T), row,
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_b, pitchB, B, col * sizeof(T), col * sizeof(T), depth,
        cudaMemcpyHostToDevice));
    matmul_v2_kernel<<<row, BlockSize, depth * sizeof(T)>>>(d_a, d_b, d_c,
        row, pitchC / sizeof(T), pitchA / sizeof(T));  // 使用的实际上是 v2 的方法，核函数不变
    checkCudaErrors(cudaMemcpy2D(C, col * sizeof(T), d_c, pitchC, col * sizeof(T), row,
        cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

完整代码见 [matmul_v3](matmul_main.cu)


### 优化6：重新改进 优化2的共享内存分配

> [!WARNING]
> 这一节的 `matmul_v6_kernel` 实现有 bug：`k` 循环固定从 `0` 累加到 `sharedSize`，且每个 `p` 分段都重复累加同一段，`depth > sharedSize` 时结果不正确。（旧版 benchmark 把 v6 的结果误与 v5 的输出缓冲比较，掩盖了这个问题。）因此它没有纳入[基准测试](#基准测试)，修复它留作练习。

做实验的时候会发现，[优化2](#优化2行共享存储矩阵乘法)
，虽然使用共享存储，但是并没有比 [优化1](#优化1-block-线程循环矩阵乘法) 有太大的优势，甚至有时候更差。

这是因为当矩阵A的列数过大的时候，过大的共享存储会影响活跃warp的数量，从而影响性能。

因此我们可以固定共享数组的大小，然后加入一层循环，表示按照A的列移动，并将中间计算结果保存在矩阵C中。

```c++
template <typename T>
__global__ void matmul_v6_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth, size_t sharedSize)
{
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    extern __shared__ T sharedData[];
    for (auto i = bid; i < row; i += gridDim.x) {
        for (auto p = tid; p < depth; p += sharedSize) {
            sharedData[tid] = A[i * depth + p];

            for (auto j = tid; j < col; j += blockDim.x) {
                T tmp { C[i * col + j] };
                for (size_t k = 0; k < sharedSize; ++k) {
                    tmp += A[i * depth + k] * B[k * col + j];
                }
                C[i * col + j] = tmp;
            }
        }
    }
}
```

## 优化 4：使用棋盘阵列的优化方法

优化 3 方法存在的最大问题是需要占用太大的 共享内存空间（如果矩阵A的列数过大的话），同时 B 矩阵依然位于全局内存中，因此会带来额外的访问时间。

因此另一种改进手法是使用两个共享存储，一个存储A的一块，一个存储B的一块，A的一块会在A矩阵沿着行跳步移动，B的一块会在B句子沿着列跳步移动，计算的结果累加之后，
就可以将结果存储到C中。因此，一个线程块处理的是A的若干行加B的若干列。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504061054324.png)

### 初步计算

将矩阵 $C$ 划分成大小为 $Bm \times Bn$ 的小块，总共有 $\frac{M}{Bm} \times \frac{N}{Bn}$ 个 Thread Block1。每个 Thread Block 沿 $K$ 维度按步长 $Bk$ 迭代，共需迭代 $\frac{K}{Bk}$ 次。

单次迭代的搬运量 : A : $Bm \times Bk$ + B: $Bk \times Bn$ = $Bk \times (Bm + Bn)$, 因此单个 block 的访问量是 $\frac{K}{Bk} \times Bk \times (Bm + Bn) = K \times (Bm + Bn)$, 全局的访问量为 $K \times (Bm + Bn) \times\frac{M}{Bm} \times \frac{N}{Bn} = MNK \cdot \left(  \frac{Bm + Bn}{Bm \cdot Bn}\right)$

因此全局总访存为:

$$
\text{Total}_{\text{tiling}} = MNK \cdot \left( \frac{Bm + Bn}{Bm \cdot Bn} \right) = MNK \cdot \left( \frac{1}{Bm} + \frac{1}{Bn} \right)
$$

在实际优化中，通常选择 $BM = BN = B_{\text{tile}}$（例如 $BM = BN = 128$。将上式代入可得：

$$
\text{Total}_{\text{tiling}} = MNK \cdot \left( \frac{2}{B_{\text{tile}}} \right) = \frac{2 MNK}{B_{\text{tile}}}
$$

量级分析：与朴素实现的 $2MNK$ 相比，Global Memory 的访问量降低到了原先的 $\frac{1}{B_{\text{tile}}}$。

因此，沿着 K 切分能够有效提高复用率，降低全局内存的访问。

再计算算术强度：

$$
I = \frac{2MNK}{4 \times MNK \cdot \left( \frac{1}{Bm} + \frac{1}{Bn} \right)} = \frac{1}{2 \times \left(\frac{1}{Bm} + \frac{1}{Bn}\right)}
$$

将分子分母同除以 $Bm \times Bn$ 后可以看出：在 $Bm + Bn$ （即 Shared Memory 占用）固定的约束下，由均值不等式知 $\frac{1}{Bm} + \frac{1}{Bn}$ 在 $BM = BN$ 时取最小值，即 **当 $Bm$ 与 $Bn$ 越接近时，计算访存比越大** 。这也就是为什么大多数优化将两者取相等。

取 $BM = BN = 128$ ：算术强度 = $128 \times 128 / (2 \times 256) = 32$ FLOPs/Byte，对于 A100，已经是大幅超越平衡点了。

### 实现

#### 简单实现

这里我们要使用二维线程块和二维grid。

```cpp
    dim3 block(blockSize, blockSize);
    dim3 grid((col + block.x - 1) / block.x, (row + block.y - 1) / block.y);
    matmul_v4_kernel<<<grid, block, 2 * (blockSize * blockSize) * sizeof(T)>>>(
        d_input_a.data, d_input_b.data, d_output.data, row, col, depth);
```

核函数为, 详情见 [matmul_v4](matmul_v4.cuh)：

```cpp
template <typename T>
__global__ void matmul_v4_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{

    extern __shared__ T sharedMem[];
    T* sharedA = sharedMem;
    T* sharedB = sharedMem + blockDim.x * blockDim.y;
    const auto blockRow = blockIdx.y * blockDim.y;
    const auto blockCol = blockIdx.x * blockDim.x;
    auto rowId = threadIdx.y;
    auto colId = threadIdx.x;
    T sum {};
    for (size_t j = 0; j < depth; j += blockDim.x) {
        if (rowId + blockRow < row && j + colId < depth) {
            sharedA[rowId * blockDim.x + colId] = A[(rowId + blockRow) * depth + j + colId];
        } else {
            sharedA[rowId * blockDim.x + colId] = static_cast<T>(0);
        }
        if (colId + blockCol < col && j + rowId < depth) {
            sharedB[rowId * blockDim.x + colId] = B[(j + rowId) * col + colId + blockCol];
        } else {
            sharedB[rowId * blockDim.x + colId] = static_cast<T>(0);
        }
        __syncthreads();
        for (size_t k = 0; k < blockDim.x; ++k) {
            sum += sharedA[rowId * blockDim.x + k] * sharedB[k * blockDim.x + colId];
        }
        __syncthreads();
    }
    if (rowId + blockRow < row && colId + blockCol < col) {
        C[(rowId + blockRow) * col + colId + blockCol]
            = sum;
    }
}
```

该实现比较简单，但是暴露了很多问题，如果尝试运行，会发现似乎速度更慢。

#### 优化1：降低条件分支

简单实现的判断过多，且每次判断位于循环内，导致减小了并行性。因此我们可以采用类似[优化3](#优化3使用二维的malloc来实现对齐访问)
的方法，采用二维的malloc来实现对齐访问和填充0，减少条件判断。

```cpp
const size_t blockSize = 32;
    col = pitchC / sizeof(T);
    depth = pitchA / sizeof(T);
    dim3 block(blockSize, blockSize);
    dim3 grid((col + block.x - 1) / block.x, (row + block.y - 1) / block.y);
    matmul_v5_kernel<<<grid, block, 2 * (blockSize * blockSize) * sizeof(T)>>>(
        d_a, d_b, d_c, row, col, depth);
```

```cpp
template <typename T>
__global__ void matmul_v5_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{

    extern __shared__ T sharedMem[];
    T* sharedA = sharedMem;
    T* sharedB = sharedMem + blockDim.x * blockDim.y;
    const auto blockRow = blockIdx.y * blockDim.y;
    const auto blockCol = blockIdx.x * blockDim.x;
    auto rowId = threadIdx.y;
    auto colId = threadIdx.x;
    T sum {};
    for (size_t j = 0; j < depth; j += blockDim.x) {
        sharedA[rowId * blockDim.x + colId] = A[(rowId + blockRow) * depth + j + colId];
        sharedA[rowId * blockDim.x + colId] = static_cast<T>(0);
        sharedB[rowId * blockDim.x + colId] = B[(j + rowId) * col + colId + blockCol];
        sharedB[rowId * blockDim.x + colId] = static_cast<T>(0);
        __syncthreads();
        for (size_t k = 0; k < blockDim.x; ++k) {
            sum += sharedA[rowId * blockDim.x + k] * sharedB[k * blockDim.x + colId];
        }
        __syncthreads();
    }
    // printf("blockRow: %d, blockCol: %d, rowId: %d, colId: %d, sum: %f\n", blockRow, blockCol, rowId, colId, sum);
    if (rowId + blockRow < row && colId + blockCol < col) {
        C[(rowId + blockRow) * col + colId + blockCol]
            = sum;
    }
}
```

#### 优化2：增加单个线程的计算量

### 优化 7 增加单个线程的计算量

该方法实在 [优化5](#优化5针对优化4减小条件判断)的基础上的改进，我们可以通过让一个 Block 共享两个 A 块矩阵和 B 块矩阵，让一个
线程计算两个结果来增加单个线程的计算量.

```c++
template <std::size_t N, typename T>
__global__ void matmul_v7_kernel(const T* A, const T* B, T* C, const size_t row, const size_t col, const size_t depth
)
{
    extern __shared__ T sharedMem[];
    T* sharedA = sharedMem;
    T* sharedB = sharedMem + (blockDim.x * blockDim.y) * N;
    const auto blockRow = blockIdx.y * blockDim.y * N;
    const auto blockCol = blockIdx.x * blockDim.x * N;
    auto rowId = threadIdx.y;
    auto colId = threadIdx.x;
    T sum[N * N] = {};
    for (size_t j = 0; j < depth; j += 1 * blockDim.x)
    {
        for (size_t i = 0; i < N; i++)
        {
            sharedA[(rowId + (i * blockDim.y)) * blockDim.x + colId] =
                A[(rowId + (i * blockDim.y) + blockRow) * depth + j + colId];
            sharedB[rowId * blockDim.x * N + colId + i * blockDim.x] =
                B[(j + rowId) * col + colId + blockCol + i * blockDim.y];
        }

        __syncthreads();
        for (int ii = 0; ii < N; ii++)
        {
            for (int jj = 0; jj < N; jj++)
            {
                for (size_t k = 0; k < blockDim.x; ++k)
                {
                    sum[ii * N + jj] +=
                        sharedA[(rowId + ii * blockDim.y) * blockDim.x + k]
                        * sharedB[k * blockDim.x * N + colId + jj * blockDim.x];
                }
            }
        }
        __syncthreads();
    }
    for (size_t i = 0; i < N; ++i)
    {
        std::size_t rid = (rowId + (i * blockDim.y) + blockRow);
        if (rid < row)
        {
            for (int j = 0; j < N; ++j)
            {
                std::size_t cid = colId + blockCol + j * blockDim.y;
                if (cid < col)
                {
                    C[rid * col + cid]
                        = sum[i * N + j];
                }
            }
        }

    }
}

```

详见 [matmul_v7](matmul_v7.cuh)

#### 优化3：增大 Bm 和 Bk

前面都是一个线程块一个 tile，受限于 GPU 单个线程块的最大线程数，这导致我们的 $Bm = Bn = 32$, 使得计算强度只有 $1 / (\frac{4}{32})=8$, 非常低，这也就是为什么对于小矩阵，上述方法还没有共享内存存储一整行来的快。

可以让tile大一些，同时，缩小 $Bk$,而一个线程块通过跳步来计算，这样即使 BlockSize 小一点，也能充分获得计算强度。

##### 协作加载的线程排布

同时，加载 tileA 和 tileB 时，线程的排列方式应当保证 **Global Memory 的合并访问** （Coalesced Access）——即同一个 warp 内相邻线程访问连续内存地址。

以 $Bm=Bn = 128, Bk=8, \text{Block Size}=256$ 为例：

* 由于 tileA 和 tileB 的元素个数都是 128*8=1024 个，而线程数是256个，所以搬运 tileA 和 tileB 都需要每个线程循环搬运 $1024 / 256 = 4$ 次
- **加载 tileA** （ $128 \times 8$ ）：一行只有 8 个元素，如果按照常规方法让一个 warp 32 个线程去读一行，内存的不连续会导致合并不成功。应让 8 个相邻线程负责一行。256 个线程重排为 $32 \times 8$ （32 行 × 8 列），每轮加载 32 行，循环 4 次覆盖 128 行。因此 `A_COLS_PER_PASS= 8`, `A_ROWS_PER_PASS= 256/8 =32`
- **加载 tileB** （ $8 \times 128$ ）：一行有 128 个元素，足够让warp中的线程连续访问。因此应让 32 个相邻线程负责一行的连续段。256 个线程重排为 $8 \times 32$ （8 行 × 32 列），每轮加载所有 8 行，每行循环 4 次覆盖 128 列。即 列方向长度 `B_COLS_PER_PASS= 32`，行方向长度 `B_ROWS_PER_PASS = 256 / 32 = 8`

💡 **提示** ：这种排布方式使同一个 warp（32 个连续 tid）在加载 tileB 时恰好覆盖一行的 32 个连续 float，实现完美合并访问。加载 tileA 时，warp 内的 32 个线程覆盖连续 4 行 × 8 列，每行的 8 个元素也是连续的。

线程重排的索引计算：

```c++
const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

// 加载 tileA 时的线程坐标（8列×32行）
const int a_col = thread_id % A_COLS_PER_PASS;             // A 分块内的列
const int a_row = thread_id / A_COLS_PER_PASS;             // A 分块内的行（步长起点）

// 加载 tileB 时的线程坐标（32列×8行）
const int b_col = thread_id % B_COLS_PER_PASS;
const int b_row = thread_id / B_COLS_PER_PASS;
```

对于 tileA 加载，单轮迭代只能覆盖 32 行，而 tileA 一共有 128 行。所以需要在行方向上做跨步为 $A\_BLOCK\_Y = 32$ 的循环：

```c++
// 协作加载 tileA：i 依次为 a_thread_y, a_thread_y + 32, a_thread_y + 64, a_thread_y + 96
#pragma unroll
        for (int i = a_row; i < BLOCK_M; i += A_ROWS_PER_PASS)
        {
            const int global_row = global_row0 + i;
            const int global_col = k0 + a_col;
            smem_A[i][a_col] =
                (global_row < M && global_col < K)
                    ? A[global_row * K + global_col]
                    : T(0);
        }
```



对 tileB 也同理

```c++
// 协作加载 tileB：j 依次为 b_thread_x, b_thread_x + 32, b_thread_x + 64, b_thread_x + 96
#pragma unroll
        for (int j = b_col; j < BLOCK_N; j += B_COLS_PER_PASS)
        {
            const int global_row = k0 + b_row;
            const int global_col = global_col0 + j;
            smem_B[b_row][j] =
                (global_row < K && global_col < N)
                    ? B[global_row * N + global_col]
                    : T(0);
        }
```

**输出 tileC** （ $128 \times 128$ ）：将tileC 划分为若干个 `C_COLS_PER_PASS ` $\times$ `C_ROWS_PER_PASS` = `BLOCK_SIZE` ($256$)的 `C_BLOCK` ，这里取 `C_COLS_PER_PASS = C_ROWS_PER_PASS = 16` ，则X方向有 ` TILE_M = BLOCK_M / C_ROWS_PER_PASS` 个 `C_BLOCK` ， Y方向有 `TILE_N = BLOCK_N / C_COLS_PER_PASS` 个 `C_BLOCK` ，每个线程需要在 `TILE_M ` $\times$ `TILE_N ` 个 `C_BLOCK` 的相同位置输出一个值，总共 `TILE_M ` $\times$ `TILE_N ` 个值

```c++
// ---- C 分块计算：形状 [BLOCK_M][BLOCK_N] 摊平为网格 ----
constexpr int C_COLS_PER_PASS = 16;
constexpr int C_ROWS_PER_PASS = THREADS / C_COLS_PER_PASS;
const int c_col = thread_id % C_COLS_PER_PASS;
const int c_row = thread_id / C_COLS_PER_PASS;

// 每线程在 C 中负责的寄存器 tile 大小
constexpr int TILE_M = BLOCK_M / C_ROWS_PER_PASS;  // 8
constexpr int TILE_N = BLOCK_N / C_COLS_PER_PASS;  // 8

// 每线程私有的输出累加器
T accum[TILE_M][TILE_N] = {T(0)};
```

⚠️ **注意** ：这里每个线程负责的 8×8 个输出元素并非连续的，而是以 `C_COLS_PER_PASS =16` 行、 `C_ROWS_PER_PASS=16` 列为间距的 **跨步分布** 。这是因为 256 个线程排成 $16 \times 16$ 的网格，每个线程在 M 和 N 方向各”跳”8 次覆盖 $128 \times 128$ 的 tileC。

这一部分理解起来有点难，可以看我用 AI 生成的 [可交互式页面](./gemm_block_tiling_interactive.html) 并结合 [代码](matmul_coal.cuh)

### 5. Thread 级 Tiling 优化：寄存器级数据复用

到这里，我们的优化形成了一条清晰的 **三级数据搬运链** ：

```mermaid
graph LR
    A[Global Memory] -->|Block 协作加载 BM×BK / BK×BN| B[Shared Memory]
    B -->|Thread 独立加载 TM+TN 个 float| C[Register]
    C -->|TM×TN 次 FMA| D[计算结果]
```

每一级搬运都遵循相同原则：沿 K 维度迭代，每次搬运一小块，在当前存储层级上最大化复用后再搬运下一块。现在我们聚焦从 Shared Memory 到 Register 这一级。

#### 5.1 问题：Shared Memory 仍是瓶颈

上一步的 kernel（Section 4.5）虽然已经让每个线程负责 $TM \times TN$ (8 x 8) 个输出，但代码中对 `As[row][p]` 和 `Bs[p][col]` 的访问直接写在三层循环内部——编译器可能会优化，但从代码结构上看，每次 FMA （融合乘加）都伴随着对 Shared Memory 的隐式读取依赖。当 256 个线程同时高频访问 Shared Memory 时，会出现 **MIO Throttle Stall** 。

>  **MIO**​ 是 **Memory Input/Output**（内存输入输出）的缩写，特指 GPU 中负责处理**共享内存（Shared Memory）**、**常量内存（Constant Memory）**以及部分特殊 I/O 操作的专用流水线（MIO Pipeline）。
> **MIO Throttle Stall**​ 指的是：当 Warp 需要执行访问共享内存或常量内存的指令时，由于 **MIO 流水线或指令队列已经满载/繁忙**，Warp 调度器无法将指令发射出去，导致该 Warp 被迫进入停顿（Stall）等待状态

```c++
for (int kk = 0; kk < BLOCK_K; ++kk)
{
    for (int i = 0; i < TILE_M; ++i)
    {
        const int local_row = c_row + i * C_ROWS_PER_PASS;
        const T a_val = smem_A[local_row][kk];           // 提升到寄存器
        for (int j = 0; j < TILE_N; ++j)
        {
            const int local_col = c_col + j * C_COLS_PER_PASS;
            accum[i][j] += a_val * smem_B[kk][local_col]; // ← 每次都读 smem_B
        }
    }
}
```

#### 5.2 解决思路

解决思路：将”从 Shared Memory 读取”和”FMA 计算”显式分离——先批量读取到寄存器数组（ `reg_A` 、 `reg_B` ），再在寄存器上做外积。这样每次 p-Loop 迭代只需 $TM + TN$ 次 Shared Memory 读取，就完成 $TM \times TN$ 次 FMA。

对比：**Block Tiling**

|访问对象|访问次数|说明|
|---|---|---|
|`smem_A`|`BLOCK_K × TILE_M`|每个 `(kk, i)` 读一次|
|`smem_B`|`BLOCK_K × TILE_M × TILE_N`|每个 `(kk, i, j)` 都读一次|
而 **Thread Tiling**

|访问对象|访问次数|说明|
|---|---|---|
|`smem_A`|`BLOCK_K × THREAD_TILE_M`|每个 `(kk, i)` 读一次|
|`smem_B`|`BLOCK_K × THREAD_TILE_N`|每个 `(kk, j)` 读一次|

关键在于循环顺序。传统内积写法（K-先）导致 $A$ 和 $B$ 的元素无法复用：

```c++
// 内积：每个线程独立累加一个 C 元素 -> 复用率低
for k: sum += A[row][k] * B[k][col]
```

改为 **外积** 写法（K 在最外层）：

```c++
// 外积：每个线程负责 TM×TN 个 C 元素
for k:
    load A_frag[TM] from smem  // 读一次 A 的 TM 个元素
    load B_frag[TN] from smem  // 读一次 B 的 TN 个元素
    for i in TM:
        for j in TN:
            accum[i][j] += reg_A[i] * reg_B[j]  // TM×TN 次 FMA
```

**计算-访存比分析** ：每轮 K 迭代中，从 Shared Memory 读取 $TM + TN$ 个 float（即 $4(TM + TN)$ 字节），完成 $TM \times TN$ 次 FMA（即 $2 \times TM \times TN$ FLOPs）。则：


$$
\text{算术强度} = \frac{2 \times TM \times TN}{4 \times (TM + TN)} = \frac{TM \times TN}{2 \times (TM + TN)} = \frac{1}{2 \times \left(\frac{1}{TM} + \frac{1}{TN}\right)}
$$

形式与 前面完全一致——这并非巧合，而是因为外积本质上是把 Block 分块的思想下沉到了 Register 层级。同样由均值不等式可知：在 $TM + TN$ （寄存器占用近似项）固定时， ** $TM = TN$ 时计算访存比最大** 。

**为什么取** $TM = TN = 8$ ：

- 单线程寄存器占用： $a_{frag}[TM] + b_{frag}[TN] + c_{frag}[TM][TN] = TM + TN + TM \times TN$ 。 $TM=TN=8$ 时为 $8 + 8 + 64 = 80$ 个 float。
- 加上索引、循环变量等开销，单线程寄存器数约 90~120，刚好接近 SM 单线程寄存器上限（255）的 1/2，能保证 occupancy（每个 SM 至少驻留 2 个 Block）。
- 若取 $TM=TN=16$ ，则 $c_{frag}$ 就要占 256 个寄存器，会触发寄存器溢出（spill 到 Local Memory），反而劣化性能。

核心思路与之前是一样的，具体代码见[代码](matmul_reg.cuh)

### 6. Warp 级分块优化

在 Thread Tiling 之后，SGEMM 的优化看似已经到位：每个数据块只从全局内存加载一次、从共享内存读取一次，其余都发生在寄存器里。

但是——**共享内存的访问还没到极限**。

Thread Tiling 只解决了"每个线程读多少"的问题，没有回答"整个 warp 一起读的时候，能否利用广播进一步降低共享内存的带宽压力"。这一步优化，就是从 Thread Tiling 到 Warp Tiling 的关键跨越。

#### 6.1 动机

CUDA 共享内存有一个非常关键的特性：**同一个 warp 内的 32 个线程，如果访问同一个地址，会被硬件合并为一次广播（broadcast）**，而不是 32 次访问。

这意味着：如果 warp 内多个线程恰好需要**同一个元素**，硬件只会付出 1 次访问的代价。这正是 Warp 级优化的核心杠杆。

回顾 Thread Tiling 里每个线程的读取模式：

```c++
for (int i = 0; i < THREAD_TILE_M; ++i) reg_A[i] = smem_A[tile_row0 + i][kk];
for (int j = 0; j < THREAD_TILE_N; ++j) reg_B[j] = smem_B[kk][tile_col0 + j];
```

- `reg_A[i]` 只与 `tile_row0 + i` 有关，**与线程在 N 方向的排列无关**
- `reg_B[j]` 只与 `tile_col0 + j` 有关，**与线程在 M 方向的排列无关**

也就是说：**如果 warp 内多个线程在同一个 M 位置（或 N 位置），它们的 `reg_A`（或 `reg_B`）读取就是完全相同的地址**——硬件可以直接广播。

现在问题变成：**warp 内的 32 个线程应该怎么排布，才能让广播带来的收益最大化？**

#### 6.2 数学推导

设 warp 内的 32 个线程排成 `x × y` 的网格（`x` 是 M 方向线程数，`y` 是 N 方向线程数），满足：
$$
x \cdot y = 32
$$


每个线程负责 `THREAD_TILE_M × THREAD_TILE_N` 的输出子块。

考虑 warp 中所有线程在第 `kk` 次 K 步的读取：

**A 的读取**：线程在 M 方向的坐标决定它读哪些 `smem_A` 元素。
- 同一个 M 位置上的 `y` 个线程（N 方向有 `y` 个线程）会读取**完全相同的** `THREAD_TILE_M` 个 A 元素
- M 方向共有 `x` 个位置
- 所以 warp 实际从 `smem_A` 读取的**不同元素**数为 `x × THREAD_TILE_M`

**B 的读取**：对称地，
- N 方向共有 `y` 个位置
- warp 实际从 `smem_B` 读取的**不同元素**数为 `y × THREAD_TILE_N`

每次 K 步中：

- **计算量**：`32 × THREAD_TILE_M × THREAD_TILE_N × 2` FLOPs（FMA = 2 FLOP）
- **共享内存读取量**：`(x × THREAD_TILE_M + y × THREAD_TILE_N) × 4` Bytes（假设 `T = float`）
比值：
$$
\text{ratio} = \frac{32 \cdot TM \cdot TN \cdot 2}{(x \cdot TM + y \cdot TN) \cdot 4}
$$
令 $TM=TN$， 则有：
$$
\text{ratio} = \frac{32 \cdot TM^2 \cdot 2}{(x + y) \cdot TM \cdot 4} = \frac{16 \cdot TM}{x + y}
$$

有均值不等式可得（加上 $x \cdot y = 32$ 这一约束），当 $(x, y) = (4,8) \, \text{or} \, (8,4)$ 的时候，访存效率最高，是 10.67. 

>  **直觉理解**：warp 越"方"，每个线程在 M 和 N 方向都只有少数线程共享同坐标，广播覆盖率就越高。

#### 6.3 Warp 内线程坐标计算

以 `BLOCK_M = BLOCK_N = 128`、`BLOCK_K = 8`、`THREAD_TILE_M = THREAD_TILE_N = 8`、warp 形状 `4 × 8` 为例。

首先推导一下：

```
每个 warp 内的线程布局      : 4 × 8
每个线程负责的输出子块      : 8 × 8
→ 每个 warp 覆盖 C 分块的    : (4 × 8) × (8 × 8) = 32 × 64
→ 整个 block (128 × 128) 需要的 warp 数: (128/32) × (128/64) = 4 × 2 = 8
```

正好 256 线程 = 8 warp ✓

**线程 ID 分解**

```c++
const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

// 1. 分解到 warp_id / lane_id
const int warp_id = thread_id >> 5;     // thread_id / 32
const int lane_id = thread_id & 31;     // thread_id % 32

// 2. warp 在 block 中的位置（4 × 2 网格）
const int warp_grid_row = warp_id / WARPS_N;   // M 方向：0~3
const int warp_grid_col = warp_id % WARPS_N;   // N 方向：0~1

// 3. lane 在 warp 内的位置（4 × 8 网格）
const int lane_grid_row = lane_id / WARP_N;    // M 方向：0~3
const int lane_grid_col = lane_id % WARP_N;    // N 方向：0~7

// 4. 线程负责的 THREAD_TILE_M × THREAD_TILE_N 子块
const int tile_row0 = (warp_grid_row * WARP_M + lane_grid_row) * THREAD_TILE_M;
const int tile_col0 = (warp_grid_col * WARP_N + lane_grid_col) * THREAD_TILE_N;
```

#### 为什么要这样安排 warp 网格？

- **M 方向**：4 warp × 4 lane-row × 8 行/线程 = 128 行 ✓
- **N 方向**：2 warp × 8 lane-col × 8 列/线程 = 128 列 ✓
    
如果用 `8 × 1` 的 warp 网格：

- M：8 × 4 × 8 = 256 行（超了）
- N：1 × 8 × 8 = 64 列（不够）
所以 `4 × 2` 是唯一能让 warp 子块恰好铺满 128 × 128 的网格形状。

代码见 []

### 7. 向量化访存：float4 优化

从 Global Memory 加载数据到 Shared Memory 时，如果每次只搬运一个 float（32 bit），需要执行大量 LDG/STS 指令。GPU 的内存系统支持一次搬运 128 bit（即一个 `float4` ），这能将指令数量减少为原来的 1/4，显著降低指令发射压力。

**问题**：向量化的前提是**访问地址在内存中连续**，且起始地址 **16 字节对齐**。

回顾 Thread Tiling / Warp Tiling 的共享内存读取模式：

```c++
for (int i = 0; i < THREAD_TILE_M; ++i)
    reg_A[i] = smem_A[tile_row0 + i][kk];   // ← 跨步访问，无法向量化

for (int j = 0; j < THREAD_TILE_N; ++j)
    reg_B[j] = smem_B[kk][tile_col0 + j];   // ← 连续访问，可以向量化
```

`smem_A` 是行主序 `[BLOCK_M][BLOCK_K]`，同一个 `kk` 列在不同行之间的元素，地址相差 `BLOCK_K × sizeof(T)`。地址不连续，无法拼接成 `float4`。

**解决方法**：在共享内存中转置 Aб要让 `reg_A` 的读取地址连续，把 A 在共享内存中**转置存储**：

```
原始布局 smem_A[BLOCK_M][BLOCK_K]   →   转置布局 smem_A_T[BLOCK_K][BLOCK_M]
   ┌──────────────┐                        ┌──────────────┐
   │  r0  r1  ... │                        │  c0  c1  ... │
   │  r0  r1  ... │                        │  r0  r1  ... │
   └──────────────┘                        └──────────────┘
   行主序，跨步读列不连续                    转置后，读一行即读连续地址
```

---

### 8. Z-order 映射避免 bank-conflict

解析见 [bank_conflic](bank_conflict.md), 代码见 [matmul_bank](matmul_bank.cuh)

### 9. 双缓冲与流水线：让数据搬运和计算真正重叠

前八节我们把 SGEMM 的**访存量**压到了最低：Block Tiling 减少全局访问、Thread Tiling 减少共享内存访问、Warp Tiling + float 4 减少指令、Z-order 消除 bank conflict。

但是，**访存量少不等于访存不耗时**。

只要 Global Memory 访问延迟还是 300~500 个时钟周期，只要 Shared Memory 读取还要等 20~30 个周期，计算单元在这些等待期间就会白白空转。**双缓冲与流水线，就是把这些"等待期"填满的技术。**

#### 9.1 延迟问题的本质

##### 9.1.1 无缓冲时的时序

回顾之前的实现，每次 K 循环迭代都是**严格的串行**：

```
时间 →
┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
│  Global →    │  │  Shared → Reg    │  │  外积累加    │  │  __syncthreads() │
│  Shared      │→ │  (a_frag,        │→ │  (FFMA)      │→ │                  │
│  (LDG + STS) │  │   b_frag)        │  │              │  │                  │
└──────────────┘  └──────────────────┘  └──────────────┘  └──────────────────┘
      ↑                    ↑                    ↑
  300~500 cycle        20~30 cycle          FMA 满载
      ↓                    ↓                    ↓
   计算单元              计算单元              计算单元
   完全空闲              部分空闲              全速工作
```

📌 **关键观察**：**每个阶段都在等前一个阶段完成**。计算单元只在第三段真正工作，其余时间要么在等 Global Memory 数据到达，要么在等 Shared Memory 读取完成。

##### 9.1.2 双缓冲的核心思想：乒乓球

双缓冲的思路非常朴素——**准备两套缓冲区，一套用于当前计算，另一套用于预加载下一轮数据**：

```text

时间 →
┌──────────────────────────────────────────────────────────────┐
│  阶段 A：用 Buffer 0 计算                                     │
│         同时往 Buffer 1 加载下一轮数据                        │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  阶段 B：用 Buffer 1 计算                                     │
│         同时往 Buffer 0 加载下一轮数据                        │
└──────────────────────────────────────────────────────────────┘
                          ↓
                        循环往复...
```
**两套缓冲区交替使用，就像打乒乓球一样**：

```text

        ┌────────────────────────────────────────────┐
        │                                            │
        ▼                                            │
    ┌─────────┐   切换    ┌─────────┐   切换    ┌─────────┐
   │ Buffer 0 │ ────────► │ Buffer 1 │ ────────► │ Buffer 0 │ ...
   │ (计算)  │           │ (计算)  │           │ (计算)  │
   └─────────┘           └─────────┘           └─────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐           ┌─────────┐           ┌─────────┐
   │ Buffer 1 │           │ Buffer 0 │           │ Buffer 1 │
   │ (加载)  │           │ (加载)  │           │ (加载)  │
   └─────────┘           └─────────┘           └─────────┘
```

**收益**：

- 计算和加载在不同执行单元上**并行进行**
- Global Memory 的 300+ 周期延迟被计算过程**隐藏**
- Shared Memory 的 20~30 周期延迟被 FFMA **隐藏**
    

---

#### 9.2 两个层级的双缓冲

在 SGEMM 中，双缓冲作用于**两个不同的层级**：

##### 9.2.1 层级结构

```text

┌─────────────────────────────────────────────────────────────┐
│  Level 1: Global Memory → Shared Memory                     │
│   延迟: 300~500 cycle                                        │
│   对策: K-Loop 级双缓冲（Shared Memory 分两份）              │
├─────────────────────────────────────────────────────────────┤
│  Level 2: Shared Memory → Register                          │
│   延迟: 20~30 cycle                                          │
│   对策: p-Loop 级双缓冲（Register 分两份）                   │
├─────────────────────────────────────────────────────────────┤
│  Level 3: Register → FMA（计算）                            │
│   延迟: 4~6 cycle                                            │
│   对策: ILP（依赖编译器展开 + 多累加器）                     │
└─────────────────────────────────────────────────────────────┘
```

**两个层级可以叠加使用**——外层隐藏 Global 延迟，内层隐藏 Shared 延迟。

##### 9.2.2 p-Loop 级双缓冲（Shared → Register）

在 K 循环的**内层**（我们称之为 p-Loop），每一步需要：

1. 从 Shared Memory 读取 `reg_A[0..TM-1]`、`reg_B[0..TN-1]`
2. 在寄存器上做 `TM × TN` 次外积

如果严格串行，每次 FFMA 都要等 LDS 完成：
```cpp

// 串行版本
for (int kk = 0; kk < BLOCK_K; ++kk) {
    for (int i = 0; i < TM; ++i) reg_A[i] = smem_A_T[kk][tile_row 0 + i];  // LDS
    for (int j = 0; j < TN; ++j) reg_B[j] = smem_B[kk][tile_col 0 + j];    // LDS
    // 上面 LDS 完成才能开始计算
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            accum[i][j] += reg_A[i] * reg_B[j];  // FFMA，需要等 LDS 结果
}
```

**双缓冲版本**：准备两组寄存器 `reg_A[2][TM]`、`reg_B[2][TN]`，在计算第 `k` 步时预取第 `k+1` 步：

```cpp

float reg_A[2][TM], reg_B[2][TN];
// 预取 k = 0
load_shared_to_reg(smem_A_T, smem_B, reg_A[0], reg_B[0], 0);
#pragma unroll
for (int kk = 0; kk < BLOCK_K; ++kk) {
    // 计算用 kk 的数据
    // 预取用 kk+1 的数据，写入另一组寄存器
    if (kk + 1 < BLOCK_K) {
        load_shared_to_reg(smem_A_T, smem_B, reg_A[(kk+1) & 1], reg_B[(kk+1) & 1], kk + 1);
    }
    // 计算 kk 步
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            accum[i][j] += reg_A[kk & 1][i] * reg_B[kk & 1][j];
}
```
**时序对比**：

```text

串行：
  LDS(k)  →  等待  →  FFMA(k)  →  LDS(k+1)  →  等待  →  FFMA(k+1) → ...
  ↓             ↓         ↓            ↓          ↓          ↓
  [20-30 c]    [空转]    [FMA]       [20-30 c]   [空转]     [FMA]
双缓冲：
  LDS(k)  →  FFMA(k-1)  →  LDS(k+1)  →  FFMA(k)  →  ...
  ↓           ↓              ↓            ↓
  [20-30 c]   [FMA]         [20-30 c]     [FMA]
             ↑ 计算时 LDS 也在进行，LDS 延迟被 FFMA 隐藏
```

##### 9.2.3 K-Loop 级双缓冲（Global → Shared）

在内层之外，K 循环每次迭代还需要从 Global Memory 加载下一块 A、B tile：

```cpp

for (int k 0 = 0; k 0 < K; k 0 += BLOCK_K) {
    // 1. Global → Shared
    load_tile_A<...>(A, smem_A, ...);
    load_tile_B<...>(B, smem_B, ...);
    __syncthreads();
    // 2. 计算
    for (int kk = 0; kk < BLOCK_K; ++kk) {
        // ... 外积
    }
    __syncthreads();
}
```

**问题**：加载和计算严格串行，`__syncthreads()` 两侧都是等待。

**双缓冲版本**：把 Shared Memory 分成两份：

```cpp

__shared__ float smem_A[2][BLOCK_K][BLOCK_M];  // 两份
__shared__ float smem_B[2][BLOCK_K][BLOCK_N];
```
K 循环中，用 Buffer 0 做计算的同时，往 Buffer 1 加载下一轮：

```text
时间 →
┌───────────────────────────────────────────────────────────┐
│ 迭代 0:                                                    │
│   [Global → smem_A[0], smem_B[0]]  [计算用 buf 0]        │
│                                    [同时加载 → buf 1]     │
├───────────────────────────────────────────────────────────┤
│ 迭代 1:                                                    │
│   [Global → smem_A[1], smem_B[1]]  [计算用 buf 1]        │
│                                    [同时加载 → buf 0]     │
└───────────────────────────────────────────────────────────┘
```
**关键**：加载下一个 tile 的指令（`LDG`）可以**提前发射**，让它在后台执行，线程继续做当前 tile 的 FFMA 计算。当计算结束时，下一 tile 的数据通常已经就绪。

---

#### 9.3 流水线编排：指令级别的时序

流水线的效果取决于**指令排布顺序**。下面是一次主循环迭代的推荐顺序：

```text

┌────────────────────────────────────────────────────────────────┐
│  1. 发射 LDG 指令（加载下一轮 Global → Register 暂存）           │
│     ↓ 异步，LDG 发出后线程继续执行后续指令                       │
│                                                                │
│  2. 执行 FFMA 计算（使用当前 Buffer 的 Shared Memory 数据）     │
│     ↓ 计算期间，LDG 的 Global 数据在后台传输                     │
│                                                                │
│  3. 执行 STS 指令（将暂存的寄存器写入另一 Buffer 的 Shared）    │
│     ↓ LDG 数据已到达，写入 Shared Memory                        │
│                                                                │
│  4. __syncthreads() 切换 Buffer                                │
│     ↓ 所有线程都完成写入后，安全切换到新 Buffer                  │
└────────────────────────────────────────────────────────────────┘
```
##### 9.3.1 为什么顺序如此重要

**LDG 是异步的**：发出后线程立即继续执行，不需要等数据返回。这就是流水线的物理基础。

**FFMA 是计算密集的**：TM × TN 个 FFMA 会占用数十个周期。这段时间足够 LDG 把 Global 数据搬到 L 2 甚至 L 1。

**STS 必须等 LDG 返回**：因为要把 LDG 的结果写到 Shared Memory，所以 STS 只能在 LDG 数据到达后执行。

**时序示意**：

```text

Cycle:    0    50   100   150   200   250   300   350   400   450   500
          │    │    │     │     │     │     │     │     │     │     │
LDG:      ████████████████████████████████████  (300+ cycle 传输)
          ↑ 发射
FFMA:          ████████████████████████████████  (计算密集)
               ↑ 立即开始，不等 LDG
STS:                                                ████  (LDG 数据到达后)
                                                    ↑
Sync:                                                    █
                                                         ↑
实际有效时间：从 0 到 500，但计算单元在 50~450 全速工作
—— LDG 的 300+ cycle 延迟被 FFMA 完全隐藏
```

##### 9.3.2 Prologue（预热）

流水线启动时需要"填满"它。第一轮迭代没有"上一轮"可用，所以需要：

1. **先把第一个 tile 从 Global 加载到 Shared**（同步阻塞）
2. **把第一组 Shared 数据预取到寄存器**（同步阻塞）
3. 然后才进入主循环
    
这就是**流水线的填充阶段（Prologue）**。

---

#### 9.4 完整代码框架

下面给出一个整合了**两级双缓冲**的 SGEMM 框架。命名沿用之前规范：`BLOCK_M/BLOCK_N/BLOCK_K`、`THREAD_TILE_M/N`、`smem_A_T/smem_B`、`reg_A/reg_B`、`accum`。

完整代码见 [matmul_buffer](matmul_buffer.cuh)

> 上面两个辅助函数把"LDG 发射"和"STS 写回"分开，是为了**让 LDG 和后续 FFMA 之间没有依赖关系**，硬件才能真正异步执行。如果 LDG 之后立即 STS，LDG 就变成了同步的，流水线就断了。

---

#### 9.5 时序可视化

##### 9.5.1 整体流水线时序

```
text

                        迭代 t=0          迭代 t=1          迭代 t=2
                     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
LDG (next tile)      │ LDG(t=1)     │ │ LDG(t=2)     │ │ LDG(t=3)     │
                     │ ████████████ │ │ ████████████ │ │ ████████████ │
                     └──────────────┘ └──────────────┘ └──────────────┘
                                     ↑ 300+ cycle 传输，全程异步
FFMA (current tile)  │    FFMA(t=0) │ │    FFMA(t=1) │ │    FFMA(t=2) │
                     │ ████████████ │ │ ████████████ │ │ ████████████ │
                     └──────────────┘ └──────────────┘ └──────────────┘
                     ↑ 立即开始，不等 LDG
STS (write smem)     │            ▓ │ │            ▓ │ │            ▓ │
                     └──────────────┘ └──────────────┘ └──────────────┘
                     ↑ LDG 数据到达后立即写入
__syncthreads()      │             S│ │             S│ │             S│
                     └──────────────┘ └──────────────┘ └──────────────┘
有效计算时间:          ├──────全速────┤ ├──────全速────┤ ├──────全速────┤
```

📌 **关键结论**：整个 K 循环中，计算单元**从未空闲**。Global Memory 的延迟、Shared Memory 的延迟，都被后续的 FFMA 隐藏。

### 9.5.2 p-Loop 内的双缓冲时序

```text
k=0:  LDS(0) [已预取]  →  FFMA(0)  +  LDS(1) 预取
k=1:  LDS(1) [已就绪]  →  FFMA(1)  +  LDS(2) 预取
k=2:  LDS(2) [已就绪]  →  FFMA(2)  +  LDS(3) 预取
...
时间轴：
  ┌─── LDS(1) ───┐        ┌─── LDS(2) ───┐
  │ ████████████ │        │ ████████████ │
  └──────────────┘        └──────────────┘
      ┌───────────────────────────────┐
      │        FFMA(0) 全速           │
      └───────────────────────────────┘
```

  LDS 和 FFMA 分属不同执行单元，硬件并行调度 → LDS 延迟被隐藏

---

#### 9.6 同步优化

##### 9.6.1 无缓冲：两次同步

```cpp

for (int k 0 = 0; k 0 < K; k 0 += BLOCK_K) {
    load_tile(A, B, smem, ...);       // Global → Shared
    __syncthreads();                  // 同步 1：等加载完成
    for (int kk = 0; kk < BLOCK_K; ++kk) { /* 计算 */ }
    __syncthreads();                  // 同步 2：等计算完成才能覆盖 smem
}
```

**为什么需要两次？**

- 同步 1：计算线程要等加载线程把 smem 写完
- 同步 2：下一轮加载线程要等所有计算线程读完了旧 smem，才能覆盖
    

##### 9.6.2 双缓冲：只需要一次同步

```cpp
for (int k 0 = 0; k 0 < K; k 0 += BLOCK_K) {
    // 计算用 buf，加载用 next_buf（互不干扰）
    compute(smem[buf], ...);          // 读 buf
    load_to(smem[next_buf], ...);     // 写 next_buf
    __syncthreads();                  // 唯一同步：切换 buffer
    buf = next_buf;
}
```

**为什么只需要一次？**

- 计算读 `buf`、加载写 `next_buf`，**两者操作的是不同的内存区域，无竞争**
    
- 唯一的同步点是在"切换"瞬间——确保所有线程都完成本轮工作，才能进入下一轮
    

📌 **同步次数从 2 降到 1，同步开销减少一半**。对于 K 循环迭代很多次的情况（例如 `K=4096, BLOCK_K=8`，迭代 512 次），这能累积出可观的收益。

##### 9.6.3 `__syncthreads()` 的代价

|维度|代价|
|---|---|
|延迟|每个同步等待所有 warp 到达，耗时数十周期|
|流水线冲刷|同步点会清空流水线，之后需要重新填充|
|阻塞|到达早的 warp 必须空等，浪费调度机会|

**双缓冲把两次同步压到一次，是 SGEMM 优化中"省下来就是赚到"的典型**。


### 10. Ampere 异步拷贝流水线：从"软件乒乓"到"硬件异步引擎"

第 9 节我们用"双缓冲 + 寄存器暂存"实现了流水线，计算和搬运确实重叠了。但仔细看那条流水线，有一个隐藏的成本：**Global 数据必须先进入寄存器，再写入 Shared Memory**。

这个"寄存器中转"占用了宝贵的寄存器资源，而且 LDG → 等待 → STS 的链条让流水线深度受限，只能做到 2~3 级。

**Ampere（SM 8.0）引入的 `cp.async`，彻底改变了这个局面。**

---

#### 10.1 cp.async：Global → Shared 的直接通路

##### 10.1.1 传统路径 vs cp.async 路径

先对比两条数据通路：

```
传统路径（第 9 节的做法）：
┌────────────┐    LDG     ┌──────────┐    STS     ┌──────────────┐
│ Global Mem │ ─────────► │ Register │ ─────────► │ Shared Memory│
└────────────┘            └──────────┘            └──────────────┘
                               ↑
                        占用寄存器暂存
                        LDG 与 STS 必须串行

cp.async 路径（Ampere 新特性）：
┌────────────┐  cp.async   ┌──────────────┐
│ Global Mem │ ──────────► │ Shared Memory│
└────────────┘             └──────────────┘
                    ↑
          硬件异步引擎直接搬运
          不经过寄存器
```

**关键收益**：

| 维度 | 传统 LDG+STS | cp.async |
|------|-------------|----------|
| 寄存器占用 | 需 8~16 个暂存寄存器/线程 | **0** |
| 指令数 | 2 条（LDG + STS） | 1 条 |
| 流水线深度 | 受寄存器限制，通常 2~3 级 | 可达 **4~6 级** |
| 发起线程阻塞 | STS 需等 LDG 返回 | 发起后立即返回 |

##### 10.1.2 cp.async 的三种变体

PTX 提供了三种 `cp.async` 变体：

```ptx
// 变体 1：.ca —— 缓存在 L1 和 L2
cp.async.ca.shared.global [smem_ptr], [gmem_ptr], 4;   // 4B, 8B, 16B 均可

// 变体 2：.cg —— 只缓存在 L2，不分配 L1（16B 是唯一合法宽度）
cp.async.cg.shared.global [smem_ptr], [gmem_ptr], 16;

// 变体 3：带 src-size 的变体（用于尾部处理，不足部分填 0）
cp.async.ca.shared.global [smem_ptr], [gmem_ptr], 16, src_size;
```

**GEMM 中通常使用 `.cg` 变体**：
- 只走 L 2，不污染 L 1（L 1 留给后续的 Shared Memory 和寄存器访问）
- 16 字节（`float4`）是 `.cg` 唯一合法的宽度，天然向量化
- 数据只被读一次（无 L 1 复用价值），`.cg` 避免了无效的 L 1 分配

##### 10.1.3 异步拷贝的完整协议：三条 PTX 指令

`cp.async` 不是一条"发射即完成"的指令，它需要三条 PTX 指令配合才能构成完整的异步拷贝协议：

```
┌─────────────────────────────────────────────────────────────────┐
│  第一步：cp.async.cg.shared.global                              │
│    发起异步拷贝，立即返回，数据在后台传输                        │
│                                                                 │
│  第二步：cp.async.commit_group                                  │
│    把之前发出的所有未提交 cp.async 打包成一个"组"（group）      │
│    每个线程独立维护自己的组                          │
│                                                                 │
│  第三步：cp.async.wait_group N                                  │
│    阻塞直到最多还有 N 个组未完成                   │
│    控制 N 即可实现多级流水线                                    │
└─────────────────────────────────────────────────────────────────┘
```

**用生活化的比喻**：

- `cp.async` = 把包裹交给快递员（发出即走，不等待）
- `commit_group` = 告诉快递员"这批包裹算一个订单"（打包成组）
- `wait_group N` = "我等前几个订单，但允许还有 N 个在路上"（控制流水线深度）

##### 10.1.4 一个最小示例：两阶段 cp.async 流水线

```cpp
// 两阶段流水线：一边用 buf0 计算，一边用 cp.async 往 buf1 加载
__shared__ float smem[2][BLOCK_K * BLOCK_M];

// 预加载第一个 tile 到 buf0
for (int i = tid; i < BLOCK_K * BLOCK_M / 4; i += blockDim.x)
{
    cp_async_cg(&smem[0][i * 4], &A[...], 16);
}
cp_async_commit_group();
cp_async_wait_group(0);   // 等待 buf0 就绪
__syncthreads();

for (int k0 = BLOCK_K; k0 < K; k0 += BLOCK_K)
{
    int buf = (k0 / BLOCK_K) & 1;
    int next_buf = 1 - buf;

    // 发起下一 tile 的 cp.async（异步）
    for (int i = tid; i < BLOCK_K * BLOCK_M / 4; i += blockDim.x)
    {
        cp_async_cg(&smem[next_buf][i * 4], &A[...], 16);
    }
    cp_async_commit_group();

    // 用当前 buf 计算
    compute(smem[buf]);

    // 等待下一 tile 就绪
    cp_async_wait_group(0);
    __syncthreads();
}
```

对应的宏定义：

```cpp
#define cp_async_cg(dst, src, bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" \
                 :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), \
                    "l"(src), "n"(bytes))

#define cp_async_commit_group() \
    asm volatile("cp.async.commit_group;\n" ::)

#define cp_async_wait_group(n) \
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n))
```

##### 10.1.5 跨线程可见性：wait_group 之后必须加 `__syncthreads()`

有一个容易踩的坑：** `cp.async.wait_group` 是 per-thread 的**，它只保证**当前线程**提交的 cp.async 组完成了。

但 Shared Memory 是被整个 block 共享的——**线程 A 写入的 smem 数据，线程 B 要读，必须经过一次 block 级别的同步**。

因此正确的模式永远是：

```cpp
cp_async_wait_group(0);
__syncthreads();   // ← 这行不能省
```

`wait_group` 保证"数据到了"，`__syncthreads` 保证"所有人都能看到"。

---

#### 10.2 Pipeline 同步模型：控制流水线深度

### 10.2.1 wait_group N 的语义

`cp.async.wait_group N` 的行为是：**阻塞直到最多还有 N 个组未完成**。

```
组编号:   G0   G1   G2   G3   G4
         │    │    │    │    │
时间 →   ▼    ▼    ▼    ▼    ▼

wait_group(0)：等所有组完成
wait_group(1)：等 G0~G3 完成，允许 G4 还在飞
wait_group(2)：等 G0~G2 完成，允许 G3、G4 还在飞
```

##### 10.2.2 多级流水线的时序

以 **3 级流水线**（`kStages = 3`）为例：

```
Stage:     S0        S1        S2
          ┌────┐    ┌────┐    ┌────┐
tile 0:   │计算│    │    │    │    │
          └────┘    └────┘    └────┘
tile 1:   │加载│    │计算│    │    │
          └────┘    └────┘    └────┘
tile 2:   │加载│    │加载│    │计算│
          └────┘    └────┘    └────┘
tile 3:   │加载│    │加载│    │加载│
          └────┘    └────┘    └────┘

主循环中：
  发射 tile k+2 的 cp.async  →  commit_group
  计算 tile k                 →  使用 S[k % 3]
  wait_group(1)              →  等 tile k+1 就绪（允许 k+2 还在飞）
  __syncthreads()
```

** `wait_group` 的参数 N 决定了允许有多少个 tile 在"飞行中"**：

| kStages | 主循环中 wait_group 的参数 | 允许在飞的 tile 数 |
|---------|--------------------------|------------------|
| 2 | `wait_group(0)` | 1 |
| 3 | `wait_group(1)` | 2 |
| 4 | `wait_group(2)` | 3 |

##### 10.2.3 为什么 3 stage 通常是甜点

**更多 stage 的好处**：能隐藏更长的延迟。
**更多 stage 的代价**：消耗更多 Shared Memory，可能降低 occupancy。

CUTLASS 在 Ampere 上的默认配置是 **3 stage**，实践中也是最佳平衡点：

```
Shared Memory 用量（BLOCK_M=128, BLOCK_N=128, BLOCK_K=8, fp32）：

  2 stage: 2 × (8×128 + 8×128) × 4B = 16 KB
  3 stage: 3 × (8×128 + 8×128) × 4B = 24 KB
  4 stage: 4 × (8×128 + 8×128) × 4B = 32 KB

Ampere SM 有 164 KB Shared Memory → 3 stage 时仍可容纳 4~6 个 block
                                 → 4 stage 时 block 数减少，occupancy 下降
```

---

#### 10.3 CUDA C++ Pipeline API

PTX 汇编虽然强大，但可移植性差。CUDA 11+ 提供了更高层的 C++ 抽象：`cuda::pipeline` + `cuda::memcpy_async`。

##### 10.3.1 基础用法

```cpp
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void kernel_with_pipeline(const float *in, float *out, int N)
{
    auto block = cg::this_thread_block();

    // 声明 pipeline 状态（2 级流水线）
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    __shared__ float smem[2][BLOCK_SIZE];

    // 发射异步拷贝
    pipe.producer_acquire();
    cuda::memcpy_async(block, smem[0], in, sizeof(float) * BLOCK_SIZE, pipe);
    pipe.producer_commit();

    // 等待数据就绪
    pipe.consumer_wait();
    // ... 使用 smem[0] 计算 ...
    pipe.consumer_release();
}
```

** `cuda::memcpy_async` 的优势**：它会自动选择最优的实现——在 Ampere+ 上用 `cp.async`，在更老的架构上退化为 `LDG + STS`。

##### 10.3.2 Producer-Consumer 模式

对于更复杂的流水线，可以使用 producer-consumer 模式，把拷贝和计算分给不同的线程：

```cpp
__device__ void produce(cuda::pipeline<cuda::thread_scope_block> &pipe,
                        int num_stages, int stage, int batch,
                        float *buffer, int buffer_len,
                        float *in, int N)
{
    pipe.producer_acquire();
    cuda::memcpy_async(buffer + stage * buffer_len + threadIdx.x,
                       in + batch * buffer_len + threadIdx.x,
                       cuda::aligned_size_t<4>(sizeof(float)), pipe);
    pipe.producer_commit();
}

__device__ void consume(cuda::pipeline<cuda::thread_scope_block> &pipe,
                        int stage, int buffer_len,
                        float *buffer, float *out, int N)
{
    pipe.consumer_wait();
    // ... 从 buffer[stage] 读取并计算 ...
    pipe.consumer_release();
}
```

##### 10.3.3 对比 PTX 内联汇编

| 维度 | PTX 内联汇编 | C++ Pipeline API |
|------|------------|-----------------|
| 可读性 | 低（汇编） | 高（C++） |
| 架构兼容性 | 仅 SM 8.0+ | 自动降级到 LDG+STS |
| 控制粒度 | 精确到指令 | 较粗，但足够 |
| 性能 | 最优 | 接近最优 |
| 推荐场景 | 极限调优 | 通用开发 |

---

#### 10.4 CUTLASS 多级流水线

CUTLASS 是 NVIDIA 官方的 GEMM 模板库，在 Ampere 上默认使用 **3~4 级 Shared Memory buffer**。

##### 10.4.1 CUTLASS 流水线的数据流

```
          ┌─────────────┐
          │ Global Mem  │
          └──────┬──────┘
                 │ cp.async
                 ▼
    ┌────────────────────────────┐
    │   Shared Memory (3 stages) │
    │  ┌──────┐ ┌──────┐ ┌──────┐│
    │  │Stage0│ │Stage1│ │Stage2││
    │  └──────┘ └──────┘ └──────┘│
    └────────────────────────────┘
                 │ ldmatrix / LDS
                 ▼
          ┌─────────────┐
          │  Register   │
          └──────┬──────┘
                 │ mma.sync
                 ▼
          ┌─────────────┐
          │ Tensor Core │
          └─────────────┘
```

##### 10.4.2 CUTLASS 中的 Pipeline 抽象

CUTLASS 提供 `PipelineTmaAsync`、`PipelineAsync` 等类，封装了 `cp.async` 的 commit/wait 逻辑。用户只需要：

```cpp
// CUTLASS 风格（简化）
using MainloopPipeline = cutlass::PipelineAsync<NumStages>;
MainloopPipeline pipeline(smem_storage, make_shape(0), cluster_shape);

// Producer 端
pipeline.producer_acquire(state);
cutlass::arch::cp_async_zfill<16>(
    &smem[stage][...], &gmem[...], pred);
pipeline.producer_commit(state);

// Consumer 端
pipeline.consumer_wait(state);
// ... 从 smem[stage] 读取并计算 ...
pipeline.consumer_release(state);
```

##### 10.4.3 多级流水线的 Shared Memory 布局

```
NumStages = 3 时的 Shared Memory 分配：

    smem_A[3][BLOCK_K][BLOCK_M]     ← 3 份 A tile
    smem_B[3][BLOCK_K][BLOCK_N]     ← 3 份 B tile

    偏移计算：
        A 的 stage i: smem_A + i * (BLOCK_K * BLOCK_M)
        B 的 stage i: smem_B + i * (BLOCK_K * BLOCK_N)
```

---

#### 10.5 Hopper（SM 90）：TMA 与 WGMMA

Hopper 把异步化推向了新的高度，引入了两个关键硬件单元：

##### 10.5.1 TMA（Tensor Memory Accelerator）

**TMA 是什么**：一个专用硬件单元，处理**多维张量**的地址计算和搬运。

```
传统 cp.async：
    每个线程手动计算地址 → 发 cp.async → 硬件搬运
    地址计算由软件完成，消耗指令和寄存器

TMA：
    线程发一条 TMA 指令（带 descriptor）→ 硬件自动计算多维地址 → 搬运
    地址计算由硬件完成，线程零开销
```

**TMA 的 descriptor** 描述了张量的形状、步长、边界等信息。发起线程只需给出"要搬哪个 tile"，硬件自动完成所有地址计算和边界处理。

##### 10.5.2 WGMMA（Warpgroup MMA）

**WGMMA 是什么**：warp-group 级别（128 线程 = 4 warps）的异步矩阵乘指令。

```
传统 mma.sync：
    warp 级别（32 线程）
    操作数必须先进寄存器（需要 ldmatrix 中转）
    同步执行，发射后需要等待结果

WGMMA (wgmma.mma_async)：
    warpgroup 级别（128 线程 = 4 warps）
    A/B 操作数可直接从 Shared Memory 读取（通过 descriptor）
    异步执行，发射后可以立即做别的事
    用 wgmma.wait_group 收割结果
```

##### 10.5.3 Hopper 的 Producer-Consumer 流水线

FlashAttention-3 是 Hopper 异步化的典型应用。它把一个 threadblock 的 warpgroup 分成两类角色：

```
┌─────────────────────────────────────────────────────────────┐
│                    Threadblock                               │
│                                                             │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │  Producer (WG0)     │    │  Consumer (WG1, WG2)     │   │
│  │  只用 TMA 搬数据     │    │  只用 WGMMA 计算         │   │
│  │  不参与计算          │    │  不直接访问 HBM          │   │
│  └─────────┬───────────┘    └────────────┬─────────────┘   │
│            │                             │                  │
│            ▼                             ▼                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Shared Memory (多级 Pipeline Buffer)          │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐               │  │
│  │  │Stage0│ │Stage1│ │Stage2│ │Stage3│               │  │
│  │  └──────┘ └──────┘ └──────┘ └──────┘               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Producer 和 Consumer 通过 named barrier 同步**，而不是 `__syncthreads()` ——这样可以让"部分线程"同步，实现更精细的 handshake。

##### 10.5.4 Hopper 的异步意义

```
传统同步模型：
    搬数据 → 等数据到 → 算 GEMM → 等算完 → 再搬下一块
    Tensor Core 和访存单元互相等待

Hopper 异步模型：
    TMA 搬数据（异步，发起后立即返回）
    WGMMA 算 GEMM（异步，发起后立即返回）
    softmax 计算（与上述重叠）
    → 搬数据、算 GEMM-I、算 softmax、算 GEMM-II 全部在时间上重叠
```

这就是 FlashAttention-3 能在 H 100 上达到约 75% peak（约 740 TFLOPS）的原因。

---

#### 10.6 完整代码：基于 cp.async 的多级流水线 SGEMM

见 [matmul_cp_async](matmul_cp_async.cuh)

##### 10.6.1 代码要点：与第 9 节双缓冲的对比

| 维度 | 第 9 节双缓冲 | 本节 cp.async 流水线 |
|------|-------------|-------------------|
| 数据通路 | Global → Register → Shared | **Global → Shared（直达）** |
| 寄存器暂存 | 需要 `ldg_A_stash` / `ldg_B_stash` | **不需要** |
| 流水线级数 | 固定 2 级 | ** `NUM_STAGES` 可配置** |
| 主循环同步 | 1 次 `__syncthreads()` | 1 次 `cp_async_wait_group` + 1 次 `__syncthreads()` |
| 跨线程可见性 | `__syncthreads()` 自动保证 | `wait_group` 后必须跟 `__syncthreads()` |
| 尾部 tile | 需单独处理 | cp.async 可配合 `src_size` 处理边界 |

##### 10.6.2 Prologue 的流水线填充

```
NUM_STAGES = 3 时：

Prologue:
    cp.async(tile 0) → commit
    cp.async(tile 1) → commit        ← 预加载前 2 个 tile
    cp.async(tile 2) → commit        ← 第 3 个 tile 也发了，但不等它
    wait_group(1)                    ← 等 tile 0 就绪（允许 tile 1, 2 在飞）
    __syncthreads()

主循环 k_tile = 0:
    发射 tile 3 的 cp.async → commit
    计算 tile 0
    wait_group(1)                    ← 等 tile 1 就绪
    __syncthreads()

主循环 k_tile = 1:
    发射 tile 4 的 cp.async → commit
    计算 tile 1
    wait_group(1)                    ← 等 tile 2 就绪
    __syncthreads()

...
```

##### 10.6.3 多级流水线的 Shared Memory 开销

| NUM_STAGES | Shared Memory（BLOCK_M=BLOCK_N=128, BLOCK_K=8, fp 32） |
|-----------|------------------------------------------------------|
| 2 | 2 × (8×128 + 8×128) × 4 B = 16 KB |
| 3 | 3 × (8×128 + 8×128) × 4 B = **24 KB** |
| 4 | 4 × (8×128 + 8×128) × 4 B = 32 KB |

Ampere SM 有 164 KB Shared Memory。3 stage 时仍可容纳 6 个 block（6 × 24 = 144 KB），occupancy 足够高。

##### 10.6.4 实战：A 的转置难题与 XOR swizzle

把 cp.async 用到真实的 SGEMM 里会遇到一个绕不开的问题：**cp.async 只能做 16B 连续直拷，无法在搬运时转置**。

B tile 没有这个问题：它在全局内存里是行主序 `B[k][n]`，读 reg_B 时也是按 `n` 方向连续读 float4，cp.async 沿 `n` 方向整块拷进去即可。A tile 则不然——全局里 `A[m][k]` 行主序，而 reg_A 需要按 `m` 方向连续读（`smem_A[k][m]` 的转置布局），"搬运时的内存布局"和"读取时的内存布局"差了一个转置。

可行的出路是让 A 在 smem 里保持行主序 `smem_A[m][k]`，reg_A 改为"固定 k、跨行读"：

```
reg_A[i] = smem_A[tile_row0 + i][kk]     // 8 次标量读取
```

但这会引入 bank conflict：行距 = `BLOCK_K` 个 float，`bank(row) = (row * BLOCK_K + kk) % 32`。Z-order 线程映射下，一个 warp 的 32 个 lane 恰好覆盖 32 个连续的行，行距 8 时（`BLOCK_K=8`）所有行组都落进同一个 bank，最坏 8 路 conflict。

解法是给每行的 16B 块做 **XOR 旋转**（CUTLASS 的 swizzle 思想的最小版本）：

```cpp
// 存储端：chunk 位置按行号旋转
chunk' = chunk ^ ((row >> 3) & 3);
// 读取端：用同一公式还原出列号
col = ((kk >> 2) ^ ((row >> 3) & 3)) * 4 + (kk & 3);
reg_A[i] = smem_A[row][col];
```

一个 warp 内 4 个行组的 `row >> 3` 互不相同，被旋转到 4 个不同的 16B 块上，bank 互不冲突；存储端每个 warp 恰好覆盖 8 行、旋转量在 warp 内为常数，store 同样零 conflict。这要求每行至少有 4 个 16B 块，即 **`BLOCK_K >= 16`**——这正好和"流水线粒度"的要求合流：`BLOCK_K=16` 时每级流水线 16 KB、每次迭代 1024 次 FMA/线程，`NUM_STAGES=3` 时共享内存恰好 48 KB（静态上限）。

在 RTX 4060 Laptop（`256 GB/s`）上的实测（`M=N=K=4096`，同轮对比）：

| 版本 | GFLOPS | 说明 |
| --- | ---: | --- |
| 修复前（只有 B 走 cp.async，A 同步 LDG+STS） | ~4890 | 见下文"根因" |
| 修复后（A/B 全异步 + swizzle + `BLOCK_K=16`） | ~7370 | 追平到 double-buffer 的 2% 以内 |
| `NUM_STAGES=2` | ~7270 | 预取深度不足 |
| double-buffer（`BLOCK_K=8`） | ~7650 | 同轮参照 |
| cuBLAS（纯 FP32 路径） | ~7790 | 上限参照 |

修复前慢的根因值得记住：旧版主循环里下一块 A 用的是**同步** `LDG → STS`（只有 B 走了 cp.async），而且 STS 放在迭代开头——stage 下标是运行期值（`k_tile % NUM_STAGES`），编译器无法证明它与本次计算的 stage 不别名，不敢把 STS 重排到 p-Loop 之后，于是每次迭代 warp 先停在 A 的全局内存延迟上。**流水线只修了一半，另一半反而比全软件流水线更差。**修成全异步后，稳态下线程只剩三件事：发拷贝、算外积、等 group。

剩余的微小差距来自 reg_A 的逐标量读取（cp.async 无法转置的固有代价）；要再往上就得动用 `ldmatrix` / Tensor Core，超出本节"同步优化"的范围。

---

#### 10.7 从 cp.async 到 TMA：演进路线

```
SM 7.0 (Volta)  →  LDG + STS，软件乒乓
                    需要寄存器暂存，流水线深度受限

SM 8.0 (Ampere) →  cp.async
                    Global → Shared 直达，不占寄存器
                    支持 3~4 级流水线

SM 9.0 (Hopper) →  TMA + WGMMA
                    TMA：硬件自动计算多维地址
                    WGMMA：warpgroup 级异步矩阵乘，操作数直接从 SMEM 读
                    Producer-Consumer warp specialization
```

**每一代的核心趋势都是一致的：让硬件承担更多搬运和计算的工作，把软件从地址计算、寄存器中转、同步等待中解放出来。**

---

## 使用第三方库

### cublas 矩阵乘法

cublas 提供 `[cublas<t>gemm](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)` 来计算矩阵乘法。

注意：由于cublas内部的算子使用Fortran写的，而Fortran的矩阵是按列存储，因此我们 $C = A \times B$
实际上为 $C^T = B^T \times A^T$

```cpp
template <typename T>
void matmul_gpu_cublas(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{

    T* d_a;
    T* d_b;
    T* d_c;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc(&d_a, row * depth * sizeof(T));
    cudaMalloc(&d_b, depth * col * sizeof(T));
    cudaMalloc(&d_c, row * col * sizeof(T));

    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);

    cublasSetVector(row * depth, sizeof(T), A, 1, d_a, 1);
    cublasSetVector(depth * col, sizeof(T), B, 1, d_b, 1);

    if constexpr (std::is_same_v<T, float>) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, col, row, depth,
            &alpha, d_b, col, d_a, depth,
            &beta, d_c, col);
    } else if constexpr (std::is_same_v<T, double>) {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, col, row, depth,
            &alpha, d_b, col, d_a, depth,
            &beta, d_c, col);
    }
    cublasGetVector(row * col, sizeof(T), d_c, 1, C, 1);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
}
```

当矩阵很大的时候，cublas才能发挥优势。

## 基准测试

上面各版本的完整对比由 [matmul_main.cu](matmul_main.cu) 驱动，结果写入
[matmul_benchmark.csv](matmul_benchmark.csv)，再用 [plot.py](plot.py) 画成
[matmul_benchmark.png](matmul_benchmark.png)（耗时、有效带宽、实测算力、Roofline 四个面板）。

### 计时与指标口径

公平对比的关键是**所有方法用同一套口径**：

1. **只计 kernel 时间**。设备缓冲一次性分配并上传，所有方法复用同一份；D2H 取回和正确性校验都在计时区间之外。（旧版把每次调用的分配 + H2D + D2H 都计入了时间，既不公平也不准确。）
2. **5 次 warmup + 20 次测量取平均**，每次测量前先把 L2 缓存刷掉（刷新本身不计时），避免上一个方法的残留缓存加速下一个方法。
3. **有效带宽用"理论下限流量"**：读 A 一次 + 读 B 一次 + 写 C 一次，即 `(M*K + K*N + M*N) * sizeof(T)`。数据复用差的方法实际流量更高，会表现为有效带宽偏低——口径对所有方法一致。
4. **计算强度**（arithmetic intensity）`AI = 2*M*N*K / 流量字节`，只与形状有关：立方矩阵（`M=N=K=n`、float）的 `AI = n/6`。实测算力 `GFLOPS = 2*M*N*K / 时间` 才是因方法而异的指标。
5. **cuBLAS 关闭 TF32**：Ampere 以上默认允许 cuBLAS 用 TF32 Tensor Core 做 FP32 GEMM，那不是纯 FP32。`cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)` 强制走 FP32 CUDA Core，与手写 kernel 同精度。
6. **正确性**：每个方法的结果都与一份双精度 CPU 参考比较（容差随 K 放宽，`K=1024` 时为 `1e-1`），不通过直接抛异常。

### 运行方式

```bash
cmake --preset cuda-msvc-release
cmake --build --preset cuda-msvc-release --target matmul_bench
# 在 matmul_demo 目录下运行，CSV 输出到当前目录
./out/build/cuda-msvc-release/matmul_demo/matmul_bench.exe
# 用 new_python 环境画图
python plot.py
```

### 实验结果

设备：RTX 4060 Laptop GPU，理论带宽 `256.032 GB/s`。`M=N=K=4096`（`AI = 682.7 FLOP/Byte`）的结果如下：

| 方法 | 时间 (ms) | GFLOPS | 有效带宽 (GB/s) | 带宽利用率 |
| --- | ---: | ---: | ---: | ---: |
| CPU swap-loop（基线） | 41014 | 3.4 | — | — |
| GPU v0 naive | 207.3 | 663 | 0.97 | 0.4% |
| GPU v1 block-loop | 171.9 | 800 | 1.17 | 0.5% |
| GPU v2 row-smem | 158.9 | 865 | 1.27 | 0.5% |
| GPU v3 pitch | 159.1 | 864 | 1.27 | 0.5% |
| GPU v4 checkerboard | 187.6 | 733 | 1.07 | 0.4% |
| GPU v5 pitch-nocheck | 185.4 | 741 | 1.09 | 0.4% |
| GPU v7 pitch-subtile | 151.4 | 908 | 1.33 | 0.5% |
| GPU block-tiling | 36.3 | 3787 | 5.55 | 2.2% |
| GPU thread-tiling | 37.6 | 3653 | 5.35 | 2.1% |
| GPU warp-tiling | 23.9 | 5762 | 8.44 | 3.3% |
| GPU warp-float4 | 21.5 | 6394 | 9.37 | 3.7% |
| GPU swizzle-zorder | 20.2 | 6816 | 9.98 | 3.9% |
| GPU double-buffer | 18.3 | 7530 | 11.03 | 4.3% |
| GPU cp.async-3stage | 18.6 | 7373 | 10.80 | 4.2% |
| GPU cublas fp32 | 17.6 | 7795 | 11.42 | 4.5% |

其余尺寸的完整数据见 [matmul_benchmark.csv](matmul_benchmark.csv)。几个直观结论：

1. **分块带来量级跃迁**。v0~v7 每一版只改善访存模式、没有成块的数据复用，始终停在 `0.5~0.9 TFLOPS`；block-tiling 让 A/B 子块进共享内存后一步跨到 `3.8 TFLOPS`；再到 warp 分块 + float4 向量化 + 双缓冲，达到 `7.5 TFLOPS`，约为 cuBLAS（纯 FP32 路径）的 `97%`。
2. **带宽面板呈"山峰"状**：有效带宽在 `512³` 附近最高、之后随尺寸回落。这不是访存变差，而是 `AI = n/6` 随尺寸增长，kernel 从带宽受限转入计算受限——分母（时间）被算力压住后，"下限流量/时间"自然下降。
3. **Roofline 的解读**：本机 FP32 实测算力约 `7.8 TFLOPS`，除以 `256 GB/s` 得到 ridge point 约 `30 FLOP/Byte`，位于 `256³`（`AI = 42.7`）左侧，所以整个测试范围都处于**计算受限区**：带宽上限线远在数据点上方，真正压住数据的是"实测最优"那条水平线。立方矩阵越大越靠右，小矩阵在左下方——既没吃满带宽也没吃满算力。
4. **cp.async 的问题出在实现而不是机制**。最初的 cp.async 版本只有 4.9 TFLOPS，根因是：主循环里下一块 A 用的是**同步** `LDG → STS`（只有 B 走了 cp.async），且 STS 放在迭代开头，编译器无法跨 stage 重排，每次迭代 warp 先停一次全局内存延迟——半成品的流水线反而不如全软件双缓冲。修成 A/B 全异步（A 行主序进 smem + XOR swizzle + `BLOCK_K=16`，见 [10.6.4](#1064-实战a-的转置难题与-xor-swizzle)）后达到 `7.4 TFLOPS`（+51%），追平到 double-buffer 的 2% 以内；剩余微小差距来自 reg_A 的逐标量读取，再往上要 `ldmatrix` / Tensor Core。同轮调参：`NUM_STAGES=3` 优于 `2`；`BLOCK_K=16` 对 double-buffer 也有收益但被测量噪声掩盖。
5. **v5 相对 v4 几乎没有收益**：去掉边界检查在现代编译器优化下优势有限，说明"少几个分支"通常不是瓶颈所在，瓶颈在数据复用。
6. **小尺寸的排名不可过度解读**：`256³` 时 kernel 只有 `0.04 ms` 量级，启动开销与测量抖动占比很大。
7. **跨轮次比较无效**：笔记本 GPU 的 boost 频率随功耗/温度漂移，同一份数据两次全量运行的整体成绩可以差 `5%~15%`。本节所有结论都基于**同一轮内**的方法间对比；调参时也务必把候选配置放进同一轮。

CPU 结果只用于正确性和数量级对照，不应和 GPU 指标直接比较；GPU benchmark 没有把主机和设备之间的传输计入时间。

> [!NOTE]
> v6 / v8 两个 kernel 因正确性问题未纳入对比，原因见[优化6](#优化6重新改进-优化2的共享内存分配)一节的警告。

