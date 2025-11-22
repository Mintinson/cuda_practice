# Element Wise 操作的实现与优化

本目录存放的是对 GPU 中的 element_wise 类进行操作的优化。作为CUDA练习的一个热身。

## 逐元素优化

首先最简单也是最直接的方法是使用逐点进行运算。其代码实现也比较简单：

```cpp
template <typename T, typename Operator>
__global__ void element_wise_naive_kernel(T* d_a, T* d_b, T* d_out, size_t n, Operator op)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    // c[idx] = a[idx] + b[idx];
    if (idx < n) {
        d_out[idx] = op(d_a[idx], d_b[idx]);
    }
}
```

详细代码保存在 [naive.cuh](naive.cuh) 中.

## 向量优化

### 使用长度为 2 的向量优化

```cpp
template <typename T>
__device__ constexpr auto& fetch_vec2(T* ptr)
{
    using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<DecayType, float>) {
        return reinterpret_cast<float2*>((ptr))[0];
    } else if constexpr (std::is_same_v<DecayType, double>) {
        return reinterpret_cast<double2*>((ptr))[0];

    } else if constexpr (std::is_same_v<DecayType, int>) {
        return reinterpret_cast<int2*>((ptr))[0];
    } else if constexpr (std::is_same_v<DecayType, unsigned int>) {
        return reinterpret_cast<uint2*>((ptr))[0];
    }
}

template <typename T, typename Operator>
__global__ void vec2_element_wise_kernel(T* d_a, T* d_b, T* d_out, size_t n, Operator op)
{
    auto idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    // c[idx] = a[idx] + b[idx];
    auto reg_a = fetch_vec2(d_a + idx);
    auto reg_b = fetch_vec2(d_b + idx);
    std::remove_reference_t<decltype(reg_a)> reg_out;
    reg_out.x = op(reg_a.x, reg_b.x);
    reg_out.y = op(reg_a.y, reg_b.y);

    fetch_vec2(d_out + idx) = reg_out;
}
```

详细代码保存在 [vec2_optimize.cuh](vec2_optimize.cuh) 和 [element_wise.cu](element_wise.cu)中.

### 使用长度为 4 的向量优化

详细代码保存在 [vec2_optimize.cuh](vec4_optimize.cuh) 和 [element_wise.cu](element_wise.cu)中.

### 向量优化的解释

以下解释来自于 `deepseek`

#### **合并内存访问（Coalesced Memory Access）** 

 * 逐点访问（如 `float*`）可能导致 非合并内存访问（*Non-Coalesced Access*），即每个线程单独发起一个 4 字节（float）的内存请求，导致显存带宽利用率低下。
 
 * 使用 `float2` 访问时：每个线程一次性读取/写入 8 字节（`float2` 的两个 `float`）。相邻线程的访问会自动合并为更少的显存事务（如 128 字节的缓存行）。显存带宽利用率提高 2 倍（理想情况下）。

---

#### **指令级并行（ILP）和向量化计算**

* 标量计算需要多条独立指令（如 ADD.F32）。

* `float2` 的运算可能被编译器优化为 SIMD 指令（如 `FMA.64` 或 `VADD.2F32`），减少指令发射次数。例如，`float2` 的加法可能被编译为一条指令同时处理 x 和 y 分量。


#### 寄存器使用优化

* 标量计算需要更多寄存器存储中间结果。

* `float2` 将两个 `float` 打包存储，减少寄存器占用。

例如，float2 仅占用 1 个寄存器（64 位），而两个 float 可能占用 2 个寄存器（32 位 x 2）。

#### 减少全局内存事务

* 标量访问需要多次全局内存读写。

* `float2` 一次性读写两个 `float`，减少全局内存事务总数。对 `c[idx]` 的写入从 2 次（标量）变为 1 次（向量）。

## 使用第三方库

### 使用 [cuBLAS](https://developer.nvidia.com/cublas) 

对于简单的类型(如 float 和 double)，以及简单的计算法则，比如加法和减法，CUBLAS 提供了预先实现的函数，可以减少开发成本。而且效率还不错。

```cpp
    cublasHandle_t handle;
    cublasCreate(&handle);
    checkCudaErrors(cudaMalloc((void**)&d_input_a, size * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_input_b, size * sizeof(T)));
    cublasSetVector(size, sizeof(T), input_a, 1, d_input_a, 1);
    cublasSetVector(size, sizeof(T), input_b, 1, d_input_b, 1);
    
    T alpha = static_cast<T>(1.0);
    // y = a*x + y
    cublasSaxpy_v2(handle, size, &alpha, d_input_a, 1, d_input_b, 1);
    cublasGetVector(size, sizeof(T), d_input_b, 1, output, 1);

    checkCudaErrors(cudaFree(d_input_a));
    checkCudaErrors(cudaFree(d_input_b));
    cublasDestroy(handle);
```

具体 API 见 [cuBLAS: cublas<t>axpy()](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-axpy)

### 使用 [Thrust](https://nvidia.github.io/cccl/thrust/index.html)

Thrust 是一个 C++ 库，用于在 GPU 上执行并行算法。它提供了许多用于 GPU 上的常见算法的函数，如排序、查找、归约、变换等。

如果了解 C++ 标准库算法，可以很快地上手使用 thrust。

下面是 elementwise 的示例代码：

```cpp
        thrust::device_vector<ValueType> d_input_a(randVec);
        thrust::device_vector<ValueType> d_input_b(randVec2);
        thrust::device_vector<ValueType> d_output(randVec.size());
        timer.start();
        thrust::transform(
            d_input_a.begin(), d_input_a.end(), d_input_b.begin(), d_output.begin(), oper);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "GPU (thrust): " << timer.elapsed() << "" << timer.unit() << std::endl;
        thrust::host_vector<ValueType> gpuThrust = d_output;
```



## 参考资料：

* [深入浅出GPU优化系列：elementwise优化及CUDA工具链介绍](https://zhuanlan.zhihu.com/p/488601925)


