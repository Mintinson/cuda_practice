# 一维卷积的实现和优化

一维卷积公式如下：

$$
y[i] = \sum_{j=0}^{K-1} x[i+j - K/2]k[j]
$$

其中 $k$ 为卷积核，$K$ 为卷积核的大小，这里假设卷积核大小都为奇数。

一维卷积的实现非常简单，复杂的是边界处理，这里只介绍两种简单的边界处理，分别是条件和填充。

条件指在每次遍历的时候判断是否到输入的边界，如果是则跳过。

```cpp
template <typename T>
void cpu_convolution(const T* input, T* output, std::size_t size, const T* kernel, std::size_t kSize)
{

    const size_t halfKSize = kSize / 2;
    for (size_t i = 0; i < size; ++i) {
        T sum {};
        if (i < halfKSize) {
            for (size_t j = halfKSize - i; j < kSize; ++j) {
                sum += input[(i + j) - halfKSize] * kernel[j];
            }
        } else if (i >= size - halfKSize) {
            for (size_t j = 0; j < size - i + halfKSize; ++j) {
                sum += input[i - halfKSize + j] * kernel[j];
            }
        } else {
            for (int j = 0; j < kSize; ++j) {
                sum += kernel[j] * input[i - halfKSize + j];
            }
        }
        output[i] = sum;
    }
}
```

而填充首先将输入的边界填充为0，然后进行无条件判断的卷积：

```cpp
template <typename T>
void cpu_convolution_pad(const T* input, T* output, std::size_t size, const T* kernel, std::size_t kSize)
{

    const size_t halfKSize = kSize / 2;
    T* newInput = new T[size + kSize - 1] {};
    for (size_t i = 0; i < size; ++i) {
        newInput[i + halfKSize] = input[i];
    }
    for (size_t i = 0; i < size; ++i) {
        T sum {};

        for (int j = 0; j < kSize; ++j) {
            sum += kernel[j] * newInput[i + j];
        }
        output[i] = sum;
    }
    delete[] newInput;
}
```

在CPU端，条件的速度可能更快。但是在 GPU端，填充的速度更快。因为都是要分配内存的，核大小往往比较小，所以分配多一点的GPU内存影响并不显著，反而核函数内的多次条件判断会导致频繁的分支跳转，影响性能。

## GPU 实现

### 优化 1 简单 GPU 实现

由于卷积的元素相互之间互不影响，因此可以为每个元素分配一个线程，进行卷积运算。

```cpp
template <typename T>
__global__ void conv_kernel_v1(const T* input, T* output, size_t sz, const T* kernel, size_t ksz)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < sz) {
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {

        const size_t halfKsz = ksz / 2;
        int startIdx = j - halfKsz;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            if (startIdx + i >= 0 && startIdx + i < sz) {
                tmp += input[startIdx + i] * kernel[i];
            }
        }
        output[j] = tmp;
    }
}

template <typename T>
__global__ void conv_kernel_v1_pad(const T* input, T* output, size_t sz, const T* kernel, size_t ksz)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {
        T tmp {};
        for (int i = 0; i < ksz; ++i) {

            tmp += input[j + i] * kernel[i];
        }
        output[j] = tmp;
    }
}
```

这是最简单的 GPU 实现，也很好理解。

### 优化 2 使用常量内存来访问核

由于卷积的核是固定的，因此可以将核放在常量内存中，常量内存的带框比全局内存要大很多。因此访问很快。

常量内存的设置在host端，通过`cudaMemcpyToSymbol` 实现，详情见 [cuda runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9bcf02b53644eee2bef9983d807084c7)

```cpp
template <typename T, bool pad = false>
void gpu_conv_v2(const T* input, T* output, std::size_t size, const T* kernel, std::size_t kSize)
{
    if constexpr (pad) {
        helper::DeviceDataHandler<T> d_input(size + kSize - 1, [&](T* data) {
            checkCudaErrors(cudaMemcpy(data + kSize / 2, input, size * sizeof(T), cudaMemcpyHostToDevice));
        });
        checkCudaErrors(cudaMemcpyToSymbol(one_dim::dKernel<T>, kernel, kSize * sizeof(T)));

        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v2_pad<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
            d_input.data, d_output.data, size, kSize);
        d_output.cpyToHost(output);
    } else {
        helper::DeviceDataHandler<T> d_input(input, size);
        // helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        checkCudaErrors(cudaMemcpyToSymbol(one_dim::dKernel<T>, kernel, kSize * sizeof(T)));
        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v2<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
            d_input.data, d_output.data, size, kSize);
        d_output.cpyToHost(output);
    }
}
```

```cpp
template <typename T>
__constant__ T dKernel[256];
template <typename T>
__global__ void conv_kernel_v2(const T* input, T* output, const size_t sz, const size_t ksz)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < sz) {
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {

        const size_t halfKsz = ksz / 2;
        int startIdx = j - halfKsz;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            if (startIdx + i >= 0 && startIdx + i < sz) {
                tmp += input[startIdx + i] * dKernel<T>[i];
            }
        }
        output[j] = tmp;
    }
}

template <typename T>
__global__ void conv_kernel_v2_pad(const T* input, T* output, size_t sz, size_t ksz)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {

        // const size_t halfKsz = ksz / 2;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            tmp += input[j + i] * dKernel<T>[i];
        }
        output[j] = tmp;
    }
}
```

详见代码 [conv_v2](conv_v2.cu)

### 优化 3 使用共享内存来访问核

同理，我们也可以使用共享内存来访问核，同样要比 (优化1) 要快得多。

```cpp
template <typename T>
__global__ void conv_kernel_v3(const T* input, T* output, const size_t sz, const T* kernel, const size_t ksz)
{
    extern __shared__ T dKernel[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < ksz) {
        dKernel[threadIdx.x] = kernel[threadIdx.x];
    }
    __syncthreads();
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {

        const size_t halfKsz = ksz / 2;
        int startIdx = j - halfKsz;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            if (startIdx + i >= 0 && startIdx + i < sz) {
                tmp += input[startIdx + i] * dKernel[i];
            }
        }
        output[j] = tmp;
    }
}

template <typename T>
__global__ void conv_kernel_v3_pad(const T* input, T* output, const size_t sz, const T* kernel, const size_t ksz)
{
    extern __shared__ T dKernel[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < ksz) {
        dKernel[threadIdx.x] = kernel[threadIdx.x];
    }
    __syncthreads();
    for (size_t j = idx; j < sz; j += blockDim.x * gridDim.x) {

        // const size_t halfKsz = ksz / 2;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            tmp += input[j + i] * dKernel[i];
        }
        output[j] = tmp;
    }
}
```

### 优化 4 使用常量内存来访问核，共享内存来访问输入

在上述优化中，始终存在对全局内存的频繁访问，即对输入 `input` 的访问。我们也可以将其优化到共享内存中。

将一个线程块所对应的输入数据以及其附近的数据拷贝到共享内存中(大小为 `BlockSize + KernelSize - 1`)，则后续的计算都可以只在共享内存中进行，速度要快很多。

由于加载到共享内存中的数据比线程块大小多`KernelSize - 1`个，因此需要注意如何加载数据：

```cpp
template <typename T>
__global__ void conv_kernel_v4(const T* input, T* output, const size_t sz, const size_t ksz)
{
    extern __shared__ T shared[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t halfKsz = ksz / 2;
    // int startIdx = (blockIdx.x - 1) * blockDim.x;
    auto endIdx = (blockIdx.x + 1) * blockDim.x;
    auto baseId = blockIdx.x * blockDim.x;
    auto tid = threadIdx.x;
    if (tid < halfKsz) {
        // shared[halfKsz - tid - 1] = idx < 1 + tid ? T {} : input[idx - 1 - tid];
        shared[tid] = baseId < halfKsz - tid ? T {} : input[baseId + tid - halfKsz];
    } else if (tid >= blockDim.x - halfKsz) {
        auto increment = tid + halfKsz - blockDim.x;
        shared[halfKsz + blockDim.x + increment] = endIdx + increment >= sz ? T {} : input[endIdx + increment];
    }
    if (idx < sz) {

        shared[tid + halfKsz] = input[idx];
        // shared[0] = input[idx];
    }
    __syncthreads();

    if (idx < sz) {
        // const size_t halfKsz = ksz / 2;
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            tmp += shared[tid + i] * v4::dKernel<T>[i];
        }
        output[idx] = tmp;
    }
}

template <typename T>
__global__ void conv_kernel_v4_pad(const T* input, T* output, size_t sz, size_t ksz)
{
    extern __shared__ T shared[];
    // auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto baseId = blockIdx.x * blockDim.x;
    auto tid = threadIdx.x;

    for (size_t j = baseId; j < sz; j += blockDim.x * gridDim.x) {
        shared[tid] = input[j + tid];
        if (tid < ksz - 1) {
            auto tidk = tid + blockDim.x;
            shared[tidk] = input[j + tidk];
        }
        __syncthreads();
        T tmp {};
        for (int i = 0; i < ksz; ++i) {
            tmp += shared[tid + i] * v4::dKernel<T>[i];
        }
        output[j + tid] = tmp;
    }
}
```