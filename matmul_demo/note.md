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

针对一般矩阵乘法中存在的访问不连续，无法向量化的问题，一种改进方法是交换内部两层循环，使得数据范文连续。

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
block 都可以独立计算一个 c 矩阵的元素。

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
    for (auto i = bid; i < row; i += gridDim.x) {
        for (auto cId = tid; cId < row; cId += blockDim.x) {
            sharedData[cId] = A[bid * depth + cId];
        }
        __syncthreads();
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

### 优化 4：使用棋盘阵列的优化方法

优化 3 方法存在的最大问题是需要占用太大的 共享内存空间（如果矩阵A的列数过大的话），同时 B 矩阵依然位于全局内存中，因此会带来额外的访问时间。

因此另一种改进手法是使用两个共享存储，一个存储A的一块，一个存储B的一块，A的一块会在A矩阵沿着行跳步移动，B的一块会在B句子沿着列跳步移动，计算的结果累加之后，
就可以将结果存储到C中。因此，一个线程块处理的是A的若干行加B的若干列。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504061054324.png)

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

### 优化5：针对优化4，减小条件判断

判断过多，且每次判断位于循环内，导致减小了并行性。因此我们可以采用类似[优化3](#优化3使用二维的malloc来实现对齐访问)
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

### 优化6：重新改进 优化2的共享内存分配

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


### 优化 8 使用 vec4, 来加速对全局内存的读写

在上面的优化 7 中，一个线程读取了四个离散的数据，写入了四个离散的数据，有没有可能利用 `float4`, 一次性读取四个连续的地址，这样无疑会更快。

下

## 使用第三方库

### cublas 矩阵乘法

cublas 提供 [cublas<t>gemm](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm) 来计算矩阵乘法。

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

