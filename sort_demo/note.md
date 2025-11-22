# 各种排序算法

## 快速排序

思路：在数组中选择一个固定的相对位置（可以是中间位置，也可以是最后一个位置）的元素，作为 `pivot`,
将数组其他元素与其对比，比它小的元素放在它前面，比它大的元素放在它后面，从而形成两个子数组。其中前面的子数组都比 `pivot`
小，后面的子数组都比 `pivot`大。

对子数组重新执行上面的操作，直到子数组只有一个值的时候，此时排序完成。

### CPU 实现

下面是快速排序的一个CPU实现，该方法没有用到递归，而是使用了循环。

**（该方法由于使用 `int` 作为范围索引，因此计算范围有限，可以使用
`unsigned` 类型，但是注意边界溢出问题。）**

```c++
    template <typename Iterator, typename Compare = std::less<>>
    void quick_sort(Iterator arr, std::size_t n, Compare comp = {})
    {
        if (n <= 1)
            return;
        using PairType = int;
        std::vector<std::pair<PairType, PairType>> ranges;
        ranges.reserve(64);
        ranges.emplace_back(0, static_cast<PairType>(n - 1));

        while (!ranges.empty())
        {
            auto [beg, end] = ranges.back();
            ranges.pop_back();

            if (beg >= end)
                continue;


            auto pivotal = arr[end];
            auto left = beg;
            auto right = end;


            while (left < right)
            {
                while (comp(arr[left], pivotal) && left < right)
                    left++;
                while (!comp(arr[right], pivotal) && left < right)
                    right--;
                if (left < right)
                    std::swap(arr[left], arr[right]);
            }


            if (!comp(arr[left], arr[end]))
                std::swap(arr[left], arr[end]);
            else
                left++;


            if (left - 1 > beg)
                ranges.emplace_back(beg, left - 1);
            if (end > left + 1)
                ranges.emplace_back(left + 1, end);
        }
    }
```

### CUDA 实现

由于并行计算较为复杂，这里采用递归地方式。而在 CUDA 中，想要实现核函数的递归调用，必须启动 separate 编译模式。即编译时输入命令行

`-rdc=true`

或者cmake脚本：

```cmake
set_target_properties(sort_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

#### GPU 简单实现

一个非常简单的cuda实现：

```c++
    template <typename T, typename Compare = std::less<>>
    __device__ void selection_sort(T* data, int left, int right, Compare comp = {})
    {
        for (int i = left; i <= right; ++i)
        {
            T min_val = data[i];
            int min_idx = i;

            // Find the smallest value in the range [left, right].
            for (int j = i + 1; j <= right; ++j)
            {
                T val_j = data[j];

                if (comp(val_j, min_val))
                {
                    min_idx = j;
                    min_val = val_j;
                }
            }

            // Swap the values.
            if (i != min_idx)
            {
                data[min_idx] = data[i];
                data[i] = min_val;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Very basic quicksort algorithm, recursively launching the next level.
    ////////////////////////////////////////////////////////////////////////////////
    template <typename T, typename Compare = std::less<>>
    __global__ void cdp_simple_quicksort(T* data, int left, int right,
                                         int depth, Compare comp = {})
    {
        // If we're too deep or there are few elements left, we use an insertion
        // sort...
        constexpr int MAX_DEPTH = 32;
        constexpr int INSERTION_SORT = 32;
        // 为了防止递归层数太多，当递归太多的时候，采用选择排序
        if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT)
        {
            selection_sort(data, left, right);
            return;
        }

        T* lptr = data + left;
        T* rptr = data + right;
        T pivot = data[(left + right) / 2];

        // Do the partitioning.
        while (lptr <= rptr)
        {
            // Find the next left- and right-hand values to swap
            T lval = *lptr;
            T rval = *rptr;

            // Move the left pointer as long as the pointed element is smaller than the
            // pivot.
            while (comp(lval, pivot))
            {
                lptr++;
                lval = *lptr;
            }

            // Move the right pointer as long as the pointed element is larger than the
            // pivot.
            while (comp(pivot, rval))
            {
                rptr--;
                rval = *rptr;
            }

            // If the swap points are valid, do the swap!
            if (lptr <= rptr)
            {
                *lptr++ = rval;
                *rptr-- = lval;
            }
        }

        // Now the recursive part
        int nright = rptr - data;
        int nleft = lptr - data;

        // Launch a new block to sort the left part.
        if (left < (rptr - data))
        {
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
            cudaStreamDestroy(s);
        }

        // Launch a new block to sort the right part.
        if ((lptr - data) < right)
        {
            cudaStream_t s1;
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
            cudaStreamDestroy(s1);
        }
    }
```

其实质并没有运动到太多多线程的内容，一个子数组的判断和移动依然是用一个线程来完成。只不过对于后续前半段的排序和后半段的排序，
分别采用了两个线程来实现。因此不做过多介绍。

#### GPU 高级实现

以下代码来自 _CUDA Toolkits Samples_。用到了许多进阶内容，这里做一些解释。完整代码见 [quick_sort.cuh](quick_sort.cuh)

核心代码如下：

```c++
       /**
         *  Simplest possible implementation, does a per-warp quicksort with no
         *  inter-warp communication. This has a high atomic issue rate, but the
         *  rest should be fairly quick because of low work per thread.
         *
         *  A warp finds its section of the data, then writes all data < pivot to
         *  one buffer and all data > pivot to the other. Atomics are used to get
         *  a unique section of the buffer.
         *
         *  Obvious optimisation: do multiple chunks per warp, to increase in-flight
         *  loads and cover the instruction overhead.
        */
        __global__ void quick_sort_warp(unsigned* inData, unsigned* outData, unsigned int offset,
                                        unsigned int len,
                                        QSortAtomicData* atomicData, QSortRingBuf* atomicDataStack,
                                        unsigned int sourceIsInData, unsigned int depth)
        {

            // Find my data offset, based on warp ID
            auto tid = threadIdx.x;
            auto idx = tid + (blockIdx.x << details::QSortBlockSizeShift);
            auto laneId = tid & (warpSize - 1);

            // Exit if I'm outside the range of sort to be done
            if (idx >= len)
                return;

            // First part of the algorithm. Each warp counts the number of elements that
            // are greater/less than the pivot.
            // When a warp knows its count, it updates an atomic counter.
            auto pivot = inData[offset + len / 2];
            auto data = inData[offset + idx];

            // Count how many are <= and how many are > pivot.
            // If all are <= pivot then we adjust the comparison
            // because otherwise the sort will move nothing and
            // we'll iterate forever.
            unsigned greater = data > pivot;
            auto gtMask = __ballot_sync(0xffffffff, greater); // 这段代码计算一个warp中有多少个线程中的greater是 1
            // 比如 如果只有线程 1,7,23 是1， 则返回 00000000100000000000000010000010  (0x00800082)
            // auto gtMask = ballot(greater);
            
            // 放宽条件，否则会永远循环
            if (gtMask == 0)
            {
                greater = (data >= pivot);
                gtMask = __ballot_sync(0xffffffff, greater); // Must re-ballot for adjusted comparator
            }
            // lest than mask
            auto ltMask = __ballot_sync(0xffffffff, !greater);
            // __popc: Count the number of bits that are set to 1 in a 32-bit integer.
            // 计算当前 warp 有多少个元素小于 pivot，多少个元素大于 pivot
            unsigned gtCount = __popc(gtMask);
            unsigned ltCount = __popc(ltMask);

            // Atomically adjust the lt_ and gtOffsets by this amount. Only one thread
            // need do this. Share the result using shfl
            // offset 用于计算 数组的索引
            unsigned ltOffset{};
            unsigned gtOffset{};
            // 一个 warp 只有一个线程需要计算，其他线程共享该结果即可
            if (laneId == 0)
            {
                if (ltCount > 0)
                    ltOffset = atomicAdd(const_cast<unsigned int*>(&(atomicData->ltOffset)), ltCount);
                if (gtCount > 0)
                    gtOffset =
                        len - (atomicAdd(const_cast<unsigned int*>(&atomicData->gtOffset), gtCount) +
                            gtCount);
            }
            // Everyone pulls the offsets from lane 0
            ltOffset = __shfl_sync(0xfffffff, static_cast<int>(ltOffset), 0);
            gtOffset = __shfl_sync(0xfffffff, static_cast<int>(gtOffset), 0);
            __syncthreads();

            // Now compute my own personal offset within this. I need to know how many
            // threads with a lane ID less than mine are going to write to the same buffer
            // as me. We can use popc to implement a single-operation warp scan in this
            // case.
            unsigned lane_mask_lt;  // 表示当前线程的 Lane掩码（Lane Mask），用于标识同一 Warp 中 Lane ID 小于当前线程的所有线程
            // 若当前线程的 Lane ID 是 3（0-based），则 lane_mask_lt 的二进制为 00000000 00000000 00000000 00000111
            asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
            // 标识当前线程所属的分区（gtMask 或 ltMask）
            unsigned int myMask = greater ? gtMask : ltMask;  
            // 计算当前线程在所属分区内的 局部偏移量。
            unsigned int myOffset = __popc(myMask & lane_mask_lt);

            // Move data.
            myOffset += greater ? gtOffset : ltOffset;
            // 将排序的结果移动到目标位置（即大于pivot移动到右半边，小于pivot的移动到左半边，注意这里 outData 和 inData 并不是同一块内存
            // 采用双数组交替进行，通过 sourceIsInData 来判断当前输入 inData 是否是真的输入（即用户一开始输入的数组）
            outData[offset + myOffset] = data;

            // Count up if we're the last warp in. If so, then Kepler will launch the next
            // set of sorts directly from here.
            if (laneId == 0)
            {
                // Count "elements written". If I wrote the last one, then trigger the next
                // qsorts
                auto myCount = ltCount + gtCount;
                // 采用原子操作，在所有线程间同步，最后一个完成排序的线程负责为两边的数据排序启动线程
                if (atomicAdd(const_cast<unsigned int*>(&(atomicData->sortedCount)), myCount) + myCount == len)
                {
                    // We're the last warp to do any sorting. Therefore, it's up to us to
                    // launch the next stage.
                    auto ltLen = atomicData->ltOffset;
                    auto gtLen = atomicData->gtOffset;

                    cudaStream_t lStream;
                    cudaStream_t rStream;
                    cudaStreamCreateWithFlags(&lStream, cudaStreamNonBlocking);
                    cudaStreamCreateWithFlags(&rStream, cudaStreamNonBlocking);

                    // Begin by freeing our atomicData storage. It's better for the ringbuffer
                    // algorithm
                    // if we free when we're done, rather than re-using (makes for less
                    // fragmentation).
                    ring_buf_free<QSortAtomicData>(atomicDataStack, atomicData);

                    // Exceptional case: if "ltLen" is zero, then all values in the batch
                    // are equal. We are then done (may need to copy into correct buffer,
                    // though)
                    if (ltLen == 0)
                    {
                        if (sourceIsInData)
                            cudaMemcpyAsync(inData + offset, outData + offset, gtLen * sizeof(unsigned),
                                            cudaMemcpyDeviceToDevice, lStream);
                        return;
                    }
                    // Start with lower half first
                    if (ltLen > BitonicSortLen)
                    {
                        // If we've exceeded maximum depth, fall through to back up
                        // big_bitonicsort
                        if (depth >= QSortMaxDepth)
                        {
                            // The final bitonic stage sorts in-place in "outData". We therefore
                            // re-use "inData" as the out-of-range tracking buffer. For (2^n)+1
                            // elements we need (2^(n+1)) bytes of our buffer. The backup qsort
                            // buffer is at least this large when sizeof(T) >= 2.
                            // 如果 sourceIsInData, 则说明此时 inData 才是实际输入数组，因此输出到 inData上，
                            // 否则 outData 才是实际输入数组, 因此直接原地排序
                            big_bitonic_sort_kernel<<<1, BitonicSortLen, 0, lStream>>>(
                                outData, sourceIsInData ? inData : outData, 
                                inData, offset, ltLen);
                        }
                        else
                        {
                            // Launch another quicksort. We need to allocate more storage for the
                            // atomic data.
                            if ((atomicData = ring_buf_allocate<QSortAtomicData>(atomicDataStack)) ==
                                nullptr)
                            {
                                printf("Stack-allocation error. Failing left child launch.\n");
                            }
                            else
                            {
                                atomicData->ltOffset = atomicData->gtOffset = atomicData->sortedCount = 0;
                                unsigned int numBlocks =
                                    static_cast<unsigned int>(ltLen + (QSortBlockSize - 1)) >>
                                    QSortBlockSizeShift;
                                // 注意这里  !sourceIsInData，且输入 是outData，输出是 InData，因此是交替工作的
                                quick_sort_warp<<<numBlocks, QSortBlockSize, 0, lStream>>>(
                                    outData, inData, offset, ltLen, atomicData, atomicDataStack,
                                    !sourceIsInData, depth + 1);
                            }
                        }
                    }
                    else if (ltLen > 1)
                    {
                        // Final stage uses a bitonic sort instead. It's important to
                        // make sure the final stage ends up in the correct (original) buffer.
                        // We launch the smallest power-of-2 number of threads that we can.
                        unsigned bitonicLen = 1 << (quick_sflo(ltLen - 1U) + 1);
                        bitonic_sort_kernel<<<1, bitonicLen, 0, lStream>>>(
                            outData, sourceIsInData ? inData : outData, offset, ltLen);
                    }
                    // Finally, if we sorted just one single element, we must still make
                    // sure that it winds up in the correct place.
                    else if (sourceIsInData && (ltLen == 1))
                    {
                        inData[offset] = outData[offset];
                    }
                    // 右侧同理，不做过多解释
                    // Now the upper half.
                    if (gtLen > BitonicSortLen)
                    {
                        // If we've exceeded maximum depth, fall through to backup
                        // big_bitonicsort
                        if (depth >= QSortMaxDepth)
                        {
                            big_bitonic_sort_kernel<<<1, BitonicSortLen, 0, rStream>>>(
                                outData, sourceIsInData ? inData : outData, inData, offset + ltLen,
                                gtLen);
                        }
                        else
                        {
                            if ((atomicData = ring_buf_allocate<QSortAtomicData>(atomicDataStack)) ==
                                nullptr)
                            {
                                printf("stack allocation error! Failing right-side launch\n");
                            }
                            else
                            {
                                atomicData->ltOffset = atomicData->gtOffset = atomicData->sortedCount = 0;
                                unsigned int numBlocks =
                                    static_cast<unsigned int>(gtLen + (QSortBlockSize - 1)) >>
                                    QSortBlockSizeShift;
                                quick_sort_warp<<<numBlocks, QSortBlockSize, 0, rStream>>>(
                                    outData, inData, offset + ltLen, gtLen, atomicData, atomicDataStack,
                                    !sourceIsInData, depth + 1);
                            }
                        }
                    }
                    else if (gtLen > 1)
                    {
                        unsigned bitonicLen = 1 << (quick_sflo(gtLen - 1U) + 1);
                        bitonic_sort_kernel<<<1, bitonicLen, 0, rStream>>>(
                            outData, sourceIsInData ? inData : outData, offset + ltLen, gtLen);
                    }
                    else if (sourceIsInData && (gtLen == 1))
                    {
                        inData[offset + ltLen] = outData[offset + ltLen];
                    }
                }
            }
        }
```

## 双调排序

### 双调排序网络概述

#### 双调序列

双调序列，是指对于一个偶数个元素的序列，前一半是升序（降序），后一半是降序（升序），
*或者能够通过循环移位实现上述状态的*序列。比如：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250414092957.png)

注意，任意两个元素组成的序列必然是双调序列，这是双调排序网络工作的原理。

#### 双调归并网络

当一个序列是双调序列的时候，则可以对其进行双调归并算法，使其最终的结果为有序数列。

双调归并算法如下图所示，以升序为例，对前半段和后半段逐个元素进行比较，将小的替换到前面来，这样得到的两个序列（原序列长度的一半），依然是双调序列。
然后再对另一半做同样的双调归并，直到比较的元素是相邻元素，最终得到的序列就是有序序列。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250414095731.png)

#### 双调排序网络

双调排序网络的输入序列可以是任意序列，我们把上面的双调归并网络称为BM (bitonic merge),且示例中是大小为 16 的
BM，则对于任意的序列（偶数序列），我们可以执行下列操作：**由于两个元素的序列一定是双调序列**
,首先每两个两个元素执行双调归并（BM2）且相邻两个双调归并的排序相反（升降序），然后再每四个每四个执行双调归并（BM4），同样相邻的归并排序相反，
最终当双调归并只有一个的时候（即覆盖一整个序列的时候），即完成对任意序列的排序。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250414101947.png)

### CPU 实现

下列是双调网络的 CPU 实现，其：

```c++
template <typename Iterator, typename Compare = std::less<>>
    void bitonic_sort(Iterator arr, std::size_t n, Compare comp = {})
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        // bitonic merge size
        for (size_t k = 2; k <= n; k <<= 1)
        {
            // bitonic merge
            for (size_t j = k >> 1; j > 0; j >>= 1)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    auto ixj = i ^ j;
                    if (ixj > i)
                    {
                        if ((i & k) == 0 && comp(*(arr + ixj), *(arr + i)))
                        {
                            std::swap(*(arr + i), *(arr + ixj));
                        }
                        if ((i & k) != 0 && comp(*(arr + i), *(arr + ixj)))
                        {
                            std::swap(*(arr + i), *(arr + ixj));
                        }
                    }
                }
            }
        }
    }
```

详见 [cpp_sort.hpp](cpp_sort.hpp)

### cuda 实现

bitonic sort 天然适配并行计算。但是当数组很大的时候，由于cuda只能实现一个线程块的同步操作，因此需要分多个核函数来完成。

由上面的示意图可以看出，两个元素的比较和交换只需要一个线程即可，一个 BlockSize 的线程块可以处理 2*BlockSize 的数据。

```c++
    // make every (4 * blockDim.x) sequence is bitonic sequence
    template <typename T, typename Compare = std::less<>>
    __global__ void bitonic_sort_sharedBlock(T* srcKey, T* dstKey, std::size_t n, Compare comp = {})
    {
        extern __shared__ T sharedKey[];

        auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
        auto tid = threadIdx.x;
        sharedKey[tid] = srcKey[idx];
        sharedKey[tid + blockDim.x] = srcKey[idx + blockDim.x];

        for (std::size_t sz = 2; sz < 2*blockDim.x; sz <<= 1)
        {
            bool direction = (tid & (sz / 2)) != 0; // 0: ascend, 1: descend

            for (auto stride = sz >> 1; stride > 0; stride >>= 1)
            {
                __syncthreads();
                auto pos = 2 * tid - (tid & (stride - 1));
                if (direction
                        ? comp(sharedKey[pos], sharedKey[pos + stride])
                        : comp(sharedKey[pos + stride], sharedKey[pos]))
                {
                    bit_ns::swap(sharedKey[pos], sharedKey[pos + stride]);
                }
            }
        }
        bool direction = blockIdx.x & 1; // here is the different, the event block ascend, while the odd block descend
        for (auto stride = blockDim.x; stride > 0; stride >>= 1)
        {
            __syncthreads();
            auto pos = 2 * tid - (tid & (stride - 1));
            if (direction
                    ? comp(sharedKey[pos], sharedKey[pos + stride])
                    : comp(sharedKey[pos + stride], sharedKey[pos]))
            {
                bit_ns::swap(sharedKey[pos], sharedKey[pos + stride]);
            }
        }
        __syncthreads();
        dstKey[idx] = sharedKey[tid];
        dstKey[idx + blockDim.x] = sharedKey[tid + blockDim.x];
    }
```

注意这里中间的 `bool direction = blockIdx.x & 1`, 我们让第0,2,4，... 线程块按照原来的顺序排序，而第 1,3,5,...
线程块按倒序排序，这样可以保证每 4 * blockDim.x 的数据都是双调序列，即前 `2*BlockSize` 是升，后 `2*BlockSize` 是降。

现在我们每 `4*blockDim.x` 是双调的，就可以对这 `4*blockDim.x` 做双调合并，同样的，相邻两个双调合并的排序顺序必须相反，得到
`8*BlockSize` 的双调，以此类推，直到覆盖整个数组为止：

```c++
    for (auto size = 2 * 2 * BlockSize; size <= n; size <<= 1)
        {
            for (auto stride = size / 2; stride > 0; stride >>= 1)
            {
                if (stride >= 2 * BlockSize)
                {
                    cuda_sort::bitonic_merge_global<<<
                        grimSize, BlockSize
                        >>>(d_output.data, d_output.data, n, size, stride, comp);
                }
                // 当 stride 较小的时候，利用共享内存加速计算
                else
                {
                    cuda_sort::bitonic_merge_shared<<< grimSize, BlockSize,sharedSize>>>(
                        d_output.data, d_output.data, n, size, comp
                    );
                    break;
                }
            }
        }
```

其中 ：

```c++
    template <typename T, typename Compare = std::less<>>
    __global__ void bitonic_merge_global(T* src, T* dst, std::size_t n, std::size_t sz, std::size_t stride,
                                         Compare comp = {})
    {
        const auto globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
        const auto comparatorI = globalIdx & (n / 2 - 1);

        bool direction = (comparatorI & (sz / 2)) != 0;
        auto pos = 2 * globalIdx - (globalIdx & (stride - 1));
        T srcA = src[pos];
        T srcB = src[pos + stride];
        if (direction
                ? comp(srcA, srcB)
                : comp(srcB, srcA))
        {
            bit_ns::swap(srcB, srcA);
        }
        dst[pos] = srcA;
        dst[pos + stride] = srcB;
    }

    template <typename T, typename Compare = std::less<>>
    __global__ void bitonic_merge_shared(T* src, T* dst, std::size_t n, std::size_t sz,
                                         Compare comp = {})
    {
        extern __shared__ T sharedKey[];
        auto tid = threadIdx.x;
        auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
        sharedKey[tid] = src[idx];
        sharedKey[tid + blockDim.x] = src[idx + blockDim.x];

        auto globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
        auto comparatorI = globalIdx & (n / 2 - 1);

        bool direction = (comparatorI & (sz / 2)) != 0;
        for (auto stride = blockDim.x; stride > 0; stride >>= 1)
        {
            __syncthreads();
            auto pos = 2 * tid - (tid & (stride - 1));
            if (direction
                    ? comp(sharedKey[pos], sharedKey[pos + stride])
                    : comp(sharedKey[pos + stride], sharedKey[pos]))
            {
                bit_ns::swap(sharedKey[pos], sharedKey[pos + stride]);
            }
        }
        __syncthreads();
        dst[idx] = sharedKey[tid];
        dst[idx + blockDim.x] = sharedKey[tid + blockDim.x];
    }
```

详细代码见 [bitonic_sort.cuh](bitonic_sort.cuh) 和 [sort_main](sort_main.cu)

## 合并排序

合并排序是将两个有序的序列合并成一个更大的有序序列，对任意序列排序时，首先将原始序列不断拆分，
直到仅剩 1 个元素（此时为有序序列），然后再进行合并。从而得到更大的有序序列。

### CPU 实现

下面是 合并排序的 CPU 实现：

```c++
        // 这里是合并排序的核心
        template <typename Iterator, typename Compare = std::less<>>
        void merge(Iterator arr, std::size_t low, std::size_t mid, std::size_t high, Compare comp = {})
        {
            using T = typename std::iterator_traits<Iterator>::value_type;
            T* tmp = static_cast<T*>(malloc(sizeof(T) * (high - low + 1)));
            size_t i = low;
            size_t j = mid + 1;
            size_t k = 0;
            while (i <= mid && j <= high)
            {
                tmp[k++] = comp(arr[j], arr[i]) ? arr[j++] : arr[i++];
            }
            while (i <= mid)
            {
                tmp[k++] = arr[i++];
            }
            while (j <= high)
            {
                tmp[k++] = arr[j++];
            }
            for (k = 0, i = low; i <= high; ++i)
            {
                arr[i] = tmp[k++];
            }
            free(tmp);
        }

        template <typename Iterator, typename Compare = std::less<>>
        void merge_sort(Iterator arr, std::size_t low, std::size_t high, Compare comp = {})
        {   
            // 利用递归不断划分小序列
            if (low < high)
            {
                auto mid = low + (high - low) / 2;
                merge_sort(arr, low, mid, comp);
                merge_sort(arr, mid + 1, high, comp);
                merge(arr, low, mid, high, comp);
            }
        }
     
    // 程序从这里进入
    template <typename Iterator, typename Compare = std::less<>>
    void merge_sort(Iterator arr, std::size_t n, Compare comp = {})
    {
        details::merge_sort(arr, 0, n - 1, comp);
    }
```

### CUDA 实现

cuda 实现相对比较复杂，具体代码可以见 [merge_sort.cuh](merge_sort.cuh)

## 基数排序

在对非负整数进行基数排序时，需要首先将排序元素统一为相同的数位长度，数位不足的排序元素可以添加前导零的方式实现数位长度统一对齐；然后从排序元素的
最低位（个位）开始进行排序，直到完成对最高位的排序，此时即实现了对排序元素的整体排序。而对每个数位进行排序的过程则是通过（稳定的）计数排序完成的，故基数排序同样是稳定的。由于我们是从排序元素的最低位向最高位依次排序的，故这种方式被称为最低位（LSD，Least
Significant Digit）法；反之，如果是从排序元素的最高位向最低位依次排序的，则被称之为最高位（MSD，Most Significant Digit）法

实际上，基数排序算法对排序元素的类型不要求一定是非负整数才可以进行，其对于字符串、浮点数等类型均可适用。其关键在于要求排序元素的数位长度统一。
例如对整数排序时，如果排序元素中含有负数，则可以对排序元素均加上一个数使其全部为非负整数；如果元素类型是字符串的话，在计数排序过程中，
可以直接使用该位字符对应ASCII码值进行计数，对于长度不足的字符串，可直接在其后面补0实现长度对齐。即在计数排序过程中，如果发现某位字符是为对齐所填充的0的话，则可认为其对应的ASCII码值为0进行计数，因为字符'A'
所对应的ASCII码值是65，字符'0'所对应的ASCII码值是48，均比0大。这样即可保证基数排序的结果是符合字典序的

### CPU 实现

```c++
    template <typename Iterator, bool Ascend = true>
    void radix_sort(Iterator arr, std::size_t n)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        // static_assert(std::is_integral_v<T>, "Radix sort only works with integral types.");
        if constexpr (std::is_integral_v<T>)
        {
            if constexpr (std::is_unsigned_v<T>)
            {
                constexpr int num_bits = sizeof(T) * 8;
                // T* arr =
                for (int bit = 0; bit < num_bits; ++bit)
                {
                    auto second_part = std::stable_partition(arr, arr + n, [bit](T value)
                    {
                        return Ascend ^ ((value & (T(1) << bit)) != 0);
                    }); // Elements with the bit not set come first
                }
            }
            else
            {
                constexpr int num_bits = sizeof(T) * 8;
                for (int bit = 0; bit < num_bits - 1; ++bit)
                {
                    auto second_part = std::stable_partition(arr, arr + n, [bit](T value)
                    {
                        return Ascend ^ ((value & (T(1) << bit)) != 0);
                    }); // Elements with the bit not set come first
                }
                auto second_part = std::stable_partition(arr, arr + n, [bit = num_bits - 1](T value)
                {
                    return (!Ascend) ^ ((value & (T(1) << bit)) != 0);
                }); // Negative numbers come first
            }
        }
        else if constexpr (std::is_same_v<std::remove_reference_t<std::remove_cv_t<T>>, float>)
        {
            constexpr int num_bits = sizeof(T) * 8;
            for (int bit = 0; bit < num_bits - 1; ++bit)
            {
                auto second_part = std::stable_partition(arr, arr + n, [bit](T value)
                {
                    return Ascend ^ (((*reinterpret_cast<unsigned*>(&value)) & (unsigned(1) << bit)) != 0);
                }); // Elements with the bit not set come first
            }
            auto second_part = std::stable_partition(arr, arr + n, [bit = num_bits - 1](T value)
            {
                return (!Ascend) ^ (((*reinterpret_cast<unsigned*>(&value)) & (unsigned(1) << bit)) != 0);
            }); // Negative numbers come first
        }
    }
```

## 使用第三方库的排序

### 使用 `Thrust` 库的快速排序

```c++
template <typename T, typename Comp = std::less<>>
void thrust_sort(T* arr, std::size_t n, Comp comp = {})
{
    thrust::device_vector<T> d_input(arr, arr+n);
    thrust::sort(d_input.begin(), d_input.end(), comp);
    thrust::copy(d_input.begin(), d_input.end(), arr);
}
```

### 使用 `Cub` 库的基数排序

```c++
template <typename T, bool Ascend = true>
void cub_radix_sort(T* arr, std::size_t n)
{
    helper::DeviceDataHandler<T> d_keys_in(arr, n);
    helper::DeviceDataHandler<T> d_keys_out(n);

    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    if constexpr (Ascend)
    {
        cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in.data, d_keys_out.data, n);
    }
    else
    {
        cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys_in.data, d_keys_out.data, n);
    }
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if constexpr (Ascend)
    {
        cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in.data, d_keys_out.data, n);
    }
    else
    {
        cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys_in.data, d_keys_out.data, n);
    }
    d_keys_out.cpyToHost(arr);
    cudaFree(d_temp_storage);
}

```