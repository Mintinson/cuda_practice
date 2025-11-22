# Compact 操作以及优化

## Compact 操作的定义

在CUDA并行计算中，Compact操作（也称为流压缩，Stream Compaction）是一种高效处理数据筛选和压缩的技术，其核心目标是
**移除无效数据，仅保留有效数据，并将它们紧密排列。**

其类似于 `copy_if` 操作，即在从输入数据中筛选出满足条件的元素（如非零、符合特定阈值等），并将这些元素连续存储，消除空隙。

## cuda 实现

### 最基础的实现思想

最基础的实现思想如下：

1. 对数组中的每个元素，判断其是否满足条件，将结果写入一个数组中，这里称为 `predRes`
2. 对 `predRes` 求 [inclusive scan](../scan_demo/note.md), 得到一个由整数组成的数组。称为 `indices`
3. 对 `predRes` 中为 true 的位置，其对应的 `indices` 的整数为 `j`，则 `j-1` 就是结果该写入的索引位置。
4. *(optional)* 如果还要对 `predRes` 为 false 的部分做compact，即要求输入中不满足 predicate 的元素按顺序排列在所有满足
   predicate的元素后面。则对 `predRes` 中为 false 的位置，假设该位置原本的索引为i，对应的 `indices` 值为 j, 而`indices`
   最后一个元素（即满足predicate的数量)为`k`时，则该元素写入的位置为 ： `i-j+k`.

如下图所示 (假设 predicate 为 `x % 2 == 0`：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250413102536.png)

### 最基础的代码实现

下面是最基础的 cuda 代码实现 （详见 [compact_main.cu](compact_main.cu)

由于该方法包含其他算子（比如scan），因此比较复杂 （效率也不是很高）

```c++
template <typename T, typename Pred>
void cuda_compact_inplace_v0(T* data, std::size_t n, Pred pred)
{
    auto [sumSize, cnt] = get_sum_size<2 * BlockSize>(n);
    helper::DeviceDataHandler<T> input(data, n);
    helper::DeviceDataHandler<T> output(n);
    helper::DeviceDataHandler<char> predRes(n);
    helper::DeviceDataHandler<std::size_t> scanRes(sumSize + 1);

    // predicate
    compact_v0::pred_map<<<(n + BlockSize - 1) / BlockSize, BlockSize>>>(
        input.data, n, predRes.data, pred);


    // scan
    std::size_t* d_input_ptr = scanRes.data;
    std::size_t size = n;
    for (size_t i = 0; i < cnt; ++i)
    {
        if (i == 0)
        {
            compact_v0::brent_kung_kernel_block<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize,
                (2 * BlockSize + (2 * BlockSize) / 32) * sizeof(std::size_t)>>>(
                    predRes.data, size, d_input_ptr);
            // cudaDeviceSynchronize();
            // checkCudaErrors(cudaGetLastError());
        }
        else
        {
            compact_v0::brent_kung_kernel_block<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize,
                (2 * BlockSize + (2 * BlockSize) / 32) * sizeof(std::size_t)>>>(
                    d_input_ptr, size, d_input_ptr);
            // cudaDeviceSynchronize();
            // checkCudaErrors(cudaGetLastError());
        }
        compact_v0::brent_kung_collect_kernel<<<(size / (2 * BlockSize) + BlockSize - 1) / (BlockSize),
            BlockSize>>>(
                d_input_ptr, size / (2 * BlockSize), d_input_ptr + size, 2 * BlockSize);
        // cudaDeviceSynchronize();
        // checkCudaErrors(cudaGetLastError());
        d_input_ptr += size;
        size /= 2 * BlockSize;
    }
    compact_v0::scan_single_kernel<<<1, 1>>>(d_input_ptr, size, d_input_ptr);

    for (size_t i = 0; i < cnt; ++i)
    {
        size *= 2 * BlockSize;
        d_input_ptr -= size;
        if (i == cnt - 1)
            compact_v0::brent_kung_distribute_kernel<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr);
        else
            compact_v0::brent_kung_distribute_kernel<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr);
    }
    // end scn
    
    // distribute
    compact_v0::compact_distribute_kernel<<<(n + BlockSize - 1) / BlockSize, BlockSize>>>(
        scanRes.data, predRes.data, n, input.data, output.data);
    output.cpyToHost(data);
}
```

`pred_map`, `compact_distribute_kernel` 都定义在 [compact_cuda_v0](compact_cuda_v0.cuh) 中，而 scan
算法详情请参考 [scan_demo](../scan_demo/note.md)

## 使用第三方库

### `Thrust` 库的使用

可以使用 `thrust::copy_if` 或者 `thrust::stable_partition` 实现类似的功能：

```c++
template <typename T, typename Pred>
void thrust_compact_inplace_v0(T* data, std::size_t n, Pred pred)
{
    thrust::device_vector<T> d_input(data, data + n);
    thrust::stable_partition(d_input.begin(), d_input.end(), pred);
    thrust::copy(d_input.begin(), d_input.end(), data);
}

template <typename T, typename Pred>
void thrust_compact_to_v0(const T* data, std::size_t n, T* src, Pred pred)
{
    thrust::device_vector<T> d_input(data, data + n);
    thrust::device_vector<T> d_out(n);
    thrust::copy_if(d_input.begin(), d_input.end(), d_out.begin(), pred);
    thrust::copy(d_out.begin(), d_out.end(), src);
}

```

