# Scan 的实现和优化

## Kogge-Stone scan

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504090909416.png)

### 单个 Block 内的简单实现

```c++
template <typename T, typename Binary = std::plus<>>
__global__ void kogge_stone_kernel_v0(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[];
    const auto tid = threadIdx.x;
    auto idx = blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += blockDim.x * gridDim.x) {  // 防止数据太多，一次核函数处理不完
        shared[tid] = src[idx];  // 读入数据
        __syncthreads();
        for (size_t stride = 1; stride <= tid; stride *= 2) { // 按照上图所示，步长由1开始，不断增大，由于我们这里的stride是-stride，所以 stride的终止条件是大于 tid
            // printf("stride=%d\n", stride);
            shared[tid] = binary(shared[tid - stride], shared[tid]);
            __syncthreads();
        }
        dst[idx] = shared[tid];
    }
}
```

上面的方法实际上是存在bug的，当block数量变多后，会发现后续的scan结果会有错误，因此需要采用下面的方法：

### 单个 Block 内开关实现

```c++
template <typename T, typename Binary = std::plus<>>
__global__ void kogge_stone_kernel_v1(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[]; // 2 * blockDim.x
    const auto tid = threadIdx.x;
    auto idx = blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += blockDim.x * gridDim.x) {
        shared[tid] = src[idx];
        bool pout = false;
        bool pin = true;
        for (size_t stride = 1; stride <= blockDim.x; stride *= 2) {
            // printf("stride=%d\n", stride);
            pout = !pout;
            pin = !pout;
            __syncthreads();
            // copy data to other zone
            shared[pout * blockDim.x + tid] = shared[pin * blockDim.x + tid];
            if (tid >= stride) {
                // pay attention to here
                // different input sources has different pin and pout
                shared[pout * blockDim.x + tid] = binary(shared[pout * blockDim.x + tid],
                    shared[pin * blockDim.x + tid - stride]);
            }
        }
        __syncthreads();
        dst[idx] = shared[pout * blockDim.x + tid];
    }
}
```

### 完整的 Kogge-Stone scan

上述的方法只能实现单个 Block 内的 scan，完整的Block需要重复上次操作。即每个 Block scan完后，将各个block最后一个数据迁移到一个新的数组中，如果该数组长度依然大于 `BlockSize` ，则继续重复上述操作，直到数组长度小于 `BlockSize` 为止。

随后，对这个小数组执行最后的scan，并将结果加到原数组中。

示意图如下：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504101139470.png)

因此这里有两个核心步骤，这里分别称为 `collect` (将各个block的最后一个数据迁移到一个新的数组中)，以及 `distribute`(将新的数组的结果分别加到对应的原数组中)

```cpp
template <typename T>
__global__ void kogge_stone_collect_kernel(const T* src, size_t n, T* dst)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[(idx + 1) * blockDim.x - 1];
    }
}
```

```cpp
template <typename T, typename Binary = std::plus<>>
__global__ void kogge_stone_distribute_kernel(const T* src, size_t n, T* dst, T init = {}, Binary binary = {})
{
    __shared__ T shared[2];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto tid = threadIdx.x;

    for (; idx < n; idx += blockDim.x * gridDim.x) {
        if (tid == 0) {
            shared[0] = blockIdx.x > 0 ? src[idx / blockDim.x - 1] : T {};
        }
        __syncthreads();
         // 这里特别处理了用户给定初始值的情况，
         // 思路为：先不管初始值，直到最后一个 distribute 阶段，所有数据都加上这个 init 值
        dst[idx] = binary(dst[idx], init); 
        if (idx >= blockDim.x) {
            dst[idx] = binary(shared[0], dst[idx]);
        }
    }
}
```

以上所有代码见 [kogge_stone_v0.cu](kogge_stone_v0.cu)

而host端函数相对比较复杂:

```cpp
template <bool inclusive, typename T, typename Binary = std::plus<>>
void kogge_stone_v1(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    auto [sumSize, cnt] = get_sum_size<BlockSize>(size);  // 计算总共需要分配的数组大小（包括上面的中间数组大小），以及需要 Block 内 scan 的迭代次数
    helper::DeviceDataHandler<T> d_input(sumSize, [&](T* data) {
        checkCudaErrors(cudaMemcpy(data, input, sizeof(T) * size, cudaMemcpyHostToDevice));
    });
    T* d_input_ptr = d_input.data;
    // 由上往下阶段
    for (size_t i = 0; i < cnt; ++i) {
        kogge_stone_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize, 2 * BlockSize * sizeof(T)>>>(
            d_input_ptr, size, d_input_ptr, binary);
        kogge_stone_collect_kernel<<<(size / BlockSize + BlockSize - 1) / BlockSize, BlockSize>>>(
            d_input_ptr, size / BlockSize, d_input_ptr + size);
        d_input_ptr += size;
        size /= BlockSize;
    }
    // 最后一个小于 BlockSize 的数组直接原地串行处理即可，还会更快
    kogge_stone_single_kernel<<<1, 1>>>(d_input_ptr, size, d_input_ptr, binary);

    // 由下往上阶段
    for (size_t i = 0; i < cnt; ++i) {
        size *= BlockSize;
        d_input_ptr -= size;
        // 最后一次 distribute 阶段，需要加上 init 
        if (i == cnt - 1)
            kogge_stone_distribute_kernel<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr, init, binary);
        else
            kogge_stone_distribute_kernel<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr, {}, binary);
    }
    // 如果是 inclusive，则直接拷贝到 output size个数据
    if constexpr (inclusive) {
        d_input.cpyToHost(output, 0, size);
    } else {
        // 否则则从 output+1 开始拷贝到 size-1 个数据，并且将 output[0] 设置为 init
        d_input.cpyToHost(output + 1, 0, size - 1);
        output[0] = init;
    }
}
```

详细代码见 [scan_main.cu](scan_main.cu)


## Brent-Kung scan

Brent-Kung scan 算法较为复杂，其分为自底向上和自顶向下两个阶段。

自底向上阶段，跨度值从1开始以2的幂次增加，直到达到数据量的一半；自顶向下过程，跨度值从最大值的 1/4 开始以2的幂次减少，直到1.

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504101154309.png)

首先是自底向上阶段，这里需要注意的是索引的计算，公式如下：

```cpp
        for (size_t stride = 1; stride <= blockDim.x; stride <<= 1) {
            size_t index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x) 
                shared[index] = binary(shared[index], shared[index - stride]);
            ...
        }
```

设 `blockDim.x = 16`, 则每次迭代输出的有效索引分别是：

```cpp
stride=1: 1, 3, 5, 7, 9, 11, 13, 15
stride=2: 3, 7, 11, 15
stride=4: 7, 11, 15
stride=8: 15
```

然后是自顶向下阶段，其计算公式如下：

```cpp
for (size_t stride = blockDim.x / 4; stride > 0; stride >>= 1) {
    __syncthreads();
    size_t index = (tid + 1) * 2 * stride - 1;
    if ((index + stride) < blockDim.x) {
        shared[index + stride] = binary(shared[index], shared[index + stride]);
    }
}
```

完整代码如下：

```cpp
template <typename T, typename Binary = std::plus<>>
__global__ void brent_kung_kernel_v0(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[];
    const auto tid = threadIdx.x;
    auto idx = blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += blockDim.x * gridDim.x) {
        shared[tid] = src[idx];
        // sweep up
        for (size_t stride = 1; stride <= blockDim.x; stride <<= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x) {
                shared[index] = binary(shared[index], shared[index - stride]);
            }
        }
        // sweep down
        for (size_t stride = blockDim.x / 4; stride > 0; stride >>= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if ((index + stride) < blockDim.x) {
                shared[index + stride] = binary(shared[index], shared[index + stride]);
            }
        }
        __syncthreads();
        dst[idx] = shared[tid];
    }
}
```

### 扩大块大小

由上述简单实现可以看到，至少有一半的线程实际上是不工作的，因此，我们可以优化一下，使得线程可以处理线程数量两倍的数据，即：

```cpp
template <typename T, typename Binary = std::plus<>>
__global__ void brent_kung_kernel_v1(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[];
    const auto tid = threadIdx.x;
    auto idx = 2 * blockIdx.x * blockDim.x + tid; // 注意这里的索引要乘以2

    for (; idx < n; idx += 2 * blockDim.x * gridDim.x) {
        shared[tid] = src[idx];
        shared[tid + blockDim.x] = src[idx + blockDim.x]; // load 两组数据
        // sweep up
        for (size_t stride = 1; stride <= blockDim.x; stride <<= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x * 2) // 这里也是 *2 
            {
                shared[index] = binary(shared[index], shared[index - stride]);
            }
        }
        // sweep down
        for (size_t stride = (blockDim.x * 2) / 4; stride > 0; stride >>= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if ((index + stride) < blockDim.x * 2) {
                shared[index + stride] = binary(shared[index], shared[index + stride]);
            }
        }
        __syncthreads();
        // 写入两组数据
        dst[idx] = shared[tid];
        dst[idx + blockDim.x] = shared[tid + blockDim.x];
    }
}
```

### 避免 bank conflict 

上述算法的运算和访存均位于索引 index 位置及其跨度的差值位置。当 block 内 thread 数量大于32时，会产生 bank conflict。避免 bank conflict 的方法之一是在每个 warp(32) 后填充一个无效数据。

```cpp
#define BANK_INDEX(x) ((x) + (x) / 32)
template <typename T, typename Binary = std::plus<>>
__global__ void brent_kung_kernel_v2(const T* src, size_t n, T* dst, Binary binary = {})
{
    extern __shared__ T shared[];
    const auto tid = threadIdx.x;
    auto idx = 2 * blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += 2 * blockDim.x * gridDim.x) {
        shared[BANK_INDEX(tid)] = src[idx];
        shared[BANK_INDEX(tid + blockDim.x)] = src[idx + blockDim.x];
        // sweep up
        for (size_t stride = 1; stride <= blockDim.x; stride <<= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x * 2) {
                shared[BANK_INDEX(index)] = binary(shared[BANK_INDEX(index)],
                    shared[BANK_INDEX(index - stride)]);
            }
        }
        // sweep down
        for (size_t stride = (blockDim.x * 2) / 4; stride > 0; stride >>= 1) {
            __syncthreads();
            size_t index = (tid + 1) * 2 * stride - 1;
            if ((index + stride) < blockDim.x * 2) {
                shared[BANK_INDEX(index + stride)] = binary(shared[BANK_INDEX(index)],
                    shared[BANK_INDEX(index + stride)]);
            }
        }
        __syncthreads();
        dst[idx] = shared[BANK_INDEX(tid)];
        dst[idx + blockDim.x] = shared[BANK_INDEX(tid + blockDim.x)];
    }
}
```

分配时也要注意分多几块共享内存：

```cpp
    brent_kung_kernel_v2<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize,
        (2 * BlockSize + (2 * BlockSize) / 32) * sizeof(T)>>>(
        d_input.data, size, d_output.data, binary);
```

### 完整的 Brent-Kung 算法

同 [kogge-stone-scan](#完整的-kogge-stone-scan) 一样的思路，只不过由于这次我们一次计算 BlockSize * 2 的数据，所以 `collect` 和 `distribute` 算法有所区别：

```cpp
template <typename T>
__global__ void brent_kung_collect_kernel(const T* src, size_t n, T* dst, const size_t blockSize)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[(idx + 1) * 2 * blockDim.x - 1];  // 注意这里 * 2
    }
}
```

```cpp
template <typename T, typename Binary = std::plus<>>
__global__ void brent_kung_distribute_kernel(
    const T* src, size_t n, T* dst,
    T init = {}, Binary binary = {})
{
    __shared__ T shared[1];
    auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // 注意这里 * 2
    auto tid = threadIdx.x;

    for (; idx < n; idx += 2 * blockDim.x * gridDim.x) {
        if (tid == 0) {
            shared[0] = blockIdx.x > 0 ? src[idx / (blockDim.x * 2) - 1] : T {}; // 注意这里 * 2
        }
        __syncthreads();
        // 连续计算两个数据
        dst[idx] = binary(dst[idx], init);
        dst[idx + blockDim.x] = binary(dst[idx + blockDim.x], init);
        if (idx >= blockDim.x) {
            dst[idx] = binary(shared[0], dst[idx]);
            dst[idx + blockDim.x] = binary(shared[0], dst[idx + blockDim.x]);
        }
    }
}
```

完整代码见 [scan_main.cu](scan_main.cu)

## Warp 内 Kogge-Stone Scan

前面的方法是在 Block内先实现scan，然后分散到其他数据，再收集回来。

我们也可以在 Warp 内实现 scan，然后分散到其他数据，再收集回来。

单个 warp 内的scan：

```cpp
template <typename T, typename Binary = std::plus<>>
__device__ T scan_warp(volatile T* sharedPartials, Binary binary = {})
{
    const size_t tid = threadIdx.x;
    const size_t lane = tid & 0x1f; // % 32
    if (lane >= 1) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 1));
    }
    if (lane >= 2) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 2));
    }
    if (lane >= 4) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 4));
    }
    if (lane >= 8) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 8));
    }
    if (lane >= 16) {
        sharedPartials[0] = binary(sharedPartials[0], *(sharedPartials - 16));
    }
    return sharedPartials[0];
}
```

然后再上一层，实现Block scan：

```cpp
template <typename T, typename Binary>
__device__ T scan_block(volatile T* sharedPartials, Binary binary = {}) {
    extern __shared__ T warpPartials[]; // 存储 Warp 的 Spine
    const size_t lane = threadIdx.x & 0x1f;
    const size_t warpId = threadIdx.x >> 5; // Warp ID

    // Step 1: Warp 内扫描
    T sum = scan_warp(sharedPartials, binary);
    __syncthreads();

    // Step 2: 收集 Warp 的 Spine（最后一个元素）
    if (lane == 31) warpPartials[16 + warpId] = sum; // 偏移16是因为 scan_warp 最多往前扫16个元素
    __syncthreads();

    // Step 3: 扫描 Warp 的 Spine
    if (warpId == 0) scan_warp(16 + warpPartials + tid);
    __syncthreads();

    // Step 4: 回填 Spine 到当前 Warp
    if (warpId > 0) sum = binary(sum, warpPartials[16 + warpId - 1]);
    __syncthreads();

    *sharedPartials = sum; // 更新共享内存
    __syncthreads();
    return sum;
}
```

`scan_and_write_partials`：Block 级扫描 + Spine 存储

```cpp
template <typename T, typename Binary>
__global__ void scan_and_write_partials(const T* src, size_t n, T* dst, T* gPartials, size_t numBlocks, bool writeSpine, Binary binary = {}) {
    extern volatile __shared__ T sharedPartials[];
    volatile T* myShared = sharedPartials + threadIdx.x;

    for (size_t bid = blockIdx.x; bid < numBlocks; bid += gridDim.x) {
        size_t index = bid * blockDim.x + threadIdx.x;
        *myShared = (index < n) ? src[index] : T{};
        __syncthreads();

        T sum = scan_block(myShared, binary); // Block 扫描
        __syncthreads();

        if (index < n) dst[index] = *myShared; // 写回结果
        if (writeSpine && threadIdx.x == blockDim.x - 1) {
            gPartials[bid] = sum; // 存储 Spine
        }
    }
}
```
`scan_add_base_sums`：Spine 回填

```cpp
template <typename T, typename Binary>
__global__ void scan_add_base_sums(T* baseSums, size_t n, T* dst, size_t numBlocks, T init = {}, Binary binary = {}) {
    T fanValue = init;
    for (size_t bid = blockIdx.x; bid < numBlocks; bid += gridDim.x) {
        size_t index = bid * blockDim.x + threadIdx.x;
        if (bid > 0) fanValue = binary(fanValue, baseSums[bid - 1]);
        dst[index] = binary(dst[index], fanValue);
    }
}
```

最后，利用递归进行全局调用：

```c++
template <typename T, typename Binary>
void warp_scan_fan(const T* src, size_t n, T* dst, size_t blockSize, T init = {}, Binary binary = {}) {
    if (n <= blockSize) { // 小数据直接处理
        scan_and_write_partials<<<1, blockSize, blockSize * sizeof(T)>>>(src, n, dst, nullptr, 1, false, binary);
        return;
    }

    T* gPartials;
    size_t numPartials = n / blockSize;
    cudaMalloc(&gPartials, numPartials * sizeof(T));

    // Step 1: Block 扫描 + 存储 Spine
    scan_and_write_partials<<<numPartials, blockSize, blockSize * sizeof(T)>>>(src, n, dst, gPartials, numPartials, true, binary);

    // Step 2: 递归扫描 Spine
    warp_scan_fan(gPartials, numPartials, gPartials, blockSize, {}, binary);

    // Step 3: 回填 Spine (只有最外层的spine回填要加init)
    scan_add_base_sums<<<numPartials, blockSize>>>(gPartials, n, dst, numPartials, init, binary);

    cudaFree(gPartials);
}
```
## 使用第三方库

### 使用 `thrust` 库实现

```c++
    thrust::device_vector<T> d_input(input, input + size);
    if constexpr (inclusive)
    {
        thrust::device_vector<T> d_output(size, init);
        thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), binary);
        thrust::copy(d_output.begin(), d_output.end(), output);
    }
    else
    {
        thrust::device_vector<T> d_output(size);
        thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init, binary);
        thrust::copy(d_output.begin(), d_output.end(), output);
    }
```

见 [scan_main.cu](scan_main.cu)

### 使用 `cub` 实现

```c++
template <bool inclusive, typename T, typename Binary = std::plus<>>
void cub_scan(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    helper::DeviceDataHandler<T> d_input(input, size);
    helper::DeviceDataHandler<T> d_ouput(size);
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    if constexpr (inclusive)
    {
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_input.data, d_ouput.data,
                                       binary, size);
        // allocate the temporary memory
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_input.data, d_ouput.data,
                                       binary, size);
    }
    else
    {
        cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_input.data, d_ouput.data,
                                       binary, init, size);
        // allocate the temporary memory
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_input.data, d_ouput.data,
                                       binary, init, size);
    }

    d_ouput.cpyToHost(output);
    checkCudaErrors(cudaFree(d_temp_storage));
}
```

见[scan_main.cu](scan_main.cu)
