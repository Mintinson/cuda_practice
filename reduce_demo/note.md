# Parallel Reduction 的实现与优化

这个目录主要包含的是利用 `cuda_runtime api` 实现的 `parallel reduction`, 并针对各种优化进行了分析。

## 优化1. shared memory 的使用

在 GPU 中，并行化 reduce 的一般步骤为分块 reduce，最终形成一种树状的结构。

以线程大小为 8 的线程块为例，如图：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504011718303.png)

每个块依次执行这样的操作后，将每个块的最终结果填入.

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504011724692.png)

表现在代码上如下所示：

```cpp
template <typename T, typename ReduceOp>
__global__ void reduce_rest_kernel(T* input, T* output, T init, size_t size, ReduceOp op)
{
    for (int i = 0; i < size; ++i) {
        init = op(init, input[i]);
    }
    output[0] = init;
}

template <typename T, typename ReduceOp>
T reduce_v1(const T* input, size_t size, T init, ReduceOp op)
{
    helper::DeviceDataHandler d_input(input, size);
    while (size > BlockSize) {

        reduce_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, op);  // 这里是核心计算区域，后续会讲
        size /= BlockSize;
    }
    // 最后不够一个块大小的数据采用串行的方式计算速度更快
    reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
    auto res = d_input.singleDataToHost(0);

    return res;
}
```

而对于核心的计算，其示意图如图所示，关键是要注意索引的正确：

![](https://pic1.zhimg.com/v2-3538ba3e8f3f0cc33c0efc45df7d3a64_1440w.jpg)

如图所示：
* 在第一轮迭代中，需要操作的是块中第 0，2，4，6，8，10，12，14 索引的数据，操作为与各自索引+1 的数据相加。
* 在第二轮迭代中，需要操作索引是 0，4，8，12，其操作为与各自索引+2 的数据相加
* 在第三轮迭代中，需要操作的索引是 0，8，其操作为与各自的索引+4 的数据相加
* 以此类推，直到整个块需要操作的索引只有一个，也只能是第 0 号位的数据，则此时整个块的 reduce 结果就是第 0 号位的结果。

因此我们可以看到，每个迭代，步长都乘以 2，因此我们迭代的代码可以写成：

```cpp
for (int i = 1; i < blockDim.x; i <<= 1){...}
```

然后需要操作的索引判断也很简单：

```cpp
if (threadIdx.x % (i*2) == 0) {...}
```

加上共享内存的运用，完整的核函数为 [reduce_v1](reduce_v1.cuh)：

```cpp
template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v1(const T* input, T* output, ReduceOp op)
{
    extern __shared__ T shared_mem[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared_mem[threadIdx.x] = input[idx];
    __syncthreads();

    for (size_t i = 1; i < blockDim.x; i <<= 1) {
        if (threadIdx.x % (2 * i) == 0) {
            shared_mem[threadIdx.x] = op(shared_mem[threadIdx.x], shared_mem[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_mem[0];  // only the first thread of the block is needed
    }
}
```

## 优化2. 取模的优化

在 [shared memory 的使用](#优化1-shared-memory-的使用化) 中，我们用了取模算法，但是取模在 GPU 运算中是非常耗时的，因此我们可以使用位运算来优化取模。

```cpp
        if ((threadIdx.x & ((2 * i) - 1)) == 0) 
```

完整程序可以看 [reduce_v2](reduce_v2.cuh)。

其余程序维持不变。

## 优化3. 减少 warp divergence

在 cuda 程序中，最常见的优化方式就是减小 warp divergence，即减少 warp（相邻 32 个线程） 之间 divergent 的情况。因为一个 warp 如果执行相同的指令，那么在硬件中，这些线程会一起执行，而如果线程之间 divergent，那么其他线程会等待 divergent 的线程执行完，再继续执行，从而导致性能的下降。

在上述方法中，对于迭代1，一个warp只有 0，2，4，6，8... 等线程执行计算，而其他相邻线程都是等待的。这导致明显的 warp divergence。因此值得优化。

最直接的思想是在每次迭代的时候，由相邻的线程执行计算，然后放置到正确的位置上，由下图所示：

![warp_divergence.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504020927907.png)

其中中间的方框代表线程，因此，其实质上与上述思路差不多：第一次迭代，线程0读取位置为0和1的数据，将其写入0位置。线程1读取位置为2和3的数据，将其写入2位置，线程2读取4和5的数据，将其写入4位置；到了第二次迭代，线程0读取位置为0和2的数据，将其写入0位置，线程1读取位置为4和6的数据，将其写入4位置，线程2读取位置为8和10的数据，将其写入8位置，以此类推，所以核心代码如下：

```cpp
    for (size_t i = 1; i < blockDim.x; i <<= 1) {
        // if (threadIdx.x % (2 * i) == 0) {
        auto index = 2 * i * threadIdx.x;
        if (index < blockDim.x) {

            shared_mem[index] = op(shared_mem[index], shared_mem[index + i]);
        }
        __syncthreads();
    }
```

即重新定义了一下映射关系。

分析一下代码，虽然代码依旧存在着 `if` 语句，但是却与 `reduce_v1, reduce_v2` 代码有所不同。我们继续假定 block 中存在 256 个t hread，即拥有256/32=8个 warp。当进行第1次迭代时，0-3 号 warp 的 `index<blockDim.x`， 4-7 号 warp 的 `index>=blockDim.x`。对于每个 warp 而言，都只是进入到一个分支内，所以并不会存在warp divergence的情况。当进行第2次迭代时，0、1号两个warp进入计算分支。当进行第3次迭代时，只有0号warp进入计算分支。当进行第4次迭代时，只有0号warp的前16个线程进入分支。此时开始产生 warp divergence。通过这种方式，我们消除了前3次迭代的warp divergence。

完整代码见 [reduce_v3](reduce_v3.cuh)

## 优化4. 解决 bank 冲突

在CUDA编程中，共享内存（Shared Memory）的Bank Conflict是性能瓶颈的常见原因。其核心机制与共享内存的硬件设计密切相关，

共享内存被划分为32个Bank，每个Bank宽度为4字节（或8字节，依GPU架构而定）。当线程访问地址时，地址对Bank数量的取模结果决定其映射的Bank：

* **地址计算公式**：Bank_ID = Address % 32
* **冲突条件**：同一Warp内的多个线程访问同一Bank的不同地址时，触发串行化访问。

在上述方法中，比如[warp-divergence](#优化3-减少-warp-divergence) 中的示意图，比如在第一次迭代中，线程0读取位置为0和1的数据，而同一个warp中的线程16，其读取位置为32和33的数据，因此，线程0和线程16的访问同一Bank的不同地址，触发串行化访问，从而导致bank冲突。

不仅如此，在第二次迭代时，线程0读取位置为0和2的数据，线程8读取位置为32和34的数据，线程16读取位置为64和66的数据，线程24读取位置为96和98的数据，发生了4路 bank 冲突。即迭代次数越多，bank冲突越为显著。

因此需要对算法进行优化，即让每个线程访问的 bank 不同，从而减少 bank 冲突。

一种思路是让线程0一直都读取 `x*32 + 0` 位置的数据，而线程 1 一直都读取 `x*32 + 1` 位置的数据，以此类推。

下图采用的思路是 步长由大到小，第一次迭代的时候，线程0 读取位置 `0` 和 `0 + BlockSize/2` 的数据，假设 `BlockSize = 256`, 则线程 0 读取 0 和 128， 刚好位于同一块 blank上；同理，线程 1 读取 1 和 129 位置，也位于 bank 1 上。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/202504021012106.png)

由于第一次迭代把有用的数据都放在了 `BlockSize / 2` 长度的 共享数据上，则第二次迭代只用利用这 `BlockSize / 2` 的数据，将步长缩小为 `BlockSize / 4`, 则线程0 读取 0 和 64 位置，线程 1 读取 1 和 65 位置，以此类推。

```cpp
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] = op(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
        }
        __syncthreads();
    }
```

核心代码在 [reduce_v4](reduce_v4.cuh) 中。

## 优化 5 解决 idle 线程

前面的方法都很不错，但是注意到有太多的线程是启动但是其实只是加载了数据到shared memory中，剩余的什么都不做。而GPU编程需要压榨 GPU的每个核心，使其尽可能地工作，即提高算法的计算密度。

因此，为了让线程多做点活，我们可以在加载数据的时候做些手脚。在加载数据的时候，让线程一次性载入原本两个 block 中的数据。这要求我们要么减小 blockSize，要么减小 block的数量。我们这里采用减小 block 数量来实现。

首先在调用`kernel`的时候, 代码变成：

```cpp
while (size > BlockSize * 2) {
        reduce_kernel_v5<<<(size + 2 * BlockSize - 1) / (BlockSize * 2),
            BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, size, op); 
        size = (size + 2 * BlockSize - 1) / (2 * BlockSize);
    }
```

注意这里我们传入的实际 `BlockSize` 是不变的，但是在每一次迭代推进的时候，按照 `BlockSize * 2` 来推进，这是因为核函数内部会将 `BlockDim.x` 乘以2来一次性计算两个 block 的数据。

接下来是核函数的实现：

```cpp
template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v5(const T* input, T* output, size_t sz, ReduceOp op)
{
    extern __shared__ T shared_mem[];
    // auto baseId = blockIdx.x * blockDim.x;
    auto idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    shared_mem[threadIdx.x] = op((idx < sz ? input[idx] : T {}),
        (idx + blockDim.x < sz ? input[idx + blockDim.x] : T {}));
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] = op(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}
```

其中的三元运算符是为了防止当 size 不是 `BlockSize * 2` 的倍数时，会多读取一些无效的数据。

完整代码见 [reduce_v5](reduce_v5.cuh)。

## 优化 6 人工展开最后一层循环以减少同步

显式同步对性能的影响较大；观察上面的算法，我们知道，当 `s <= 32` 时，实际上只有一个 Warp 在工作，而其他warp都在等待；但是我们依然需要 `__syncthreads()` 来同步，这降低了性能。

因此这里使用了一个非常牛的技巧，就是手动展开最后一层循环，从而减少同步。

这里一定要注意，当 `s = 32` 时，原算法还有6次循环，利用warp 在没有分支的情况下能够确保并行性的特点，我们的手动展开可以写成这样：

```cpp
// only work for threads that s = 32, and threadIdx.x < 32
template <typename T, typename ReduceOp>
__device__ void warpReduce(volatile T* data, size_t id, ReduceOp op)
{
    data[id] = op(data[id], data[id + 32]);
    data[id] = op(data[id], data[id + 16]);
    data[id] = op(data[id], data[id + 8]);
    data[id] = op(data[id], data[id + 4]);
    data[id] = op(data[id], data[id + 2]);
    data[id] = op(data[id], data[id + 1]);
}
```

其中 `volatile` 是为了避免编译器优化，是必须的。

核函数修改的地方为：

```cpp
   for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] = op(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        warpReduce(shared_mem, threadIdx.x, op);
        // no syncthreads()
    }
```

完整代码见 [reduce_v6](reduce_v6.cuh)。

## 优化 7 完全展开 for 循环减少同步

其实到了这一步，reduce的效率已经足够高了。再进一步优化其实已经非常困难了。为了探索极致的性能表现，Mharris接下来给出的办法是对for循环进行完全展开。我觉得这里主要是减少for循环的开销。Mharris的实验表明这种方式有着1.41x的加速比。但是用的机器是G80，十几年前的卡。性能数据也比较老了，至于能不能真的有这么好的加速比，我们拭目以待。

```cpp
template <typename T, typename ReduceOp>
__device__ void warpReduce_v7(volatile T* data, size_t id, ReduceOp op)
{
    data[id] = op(data[id], data[id + 32]);
    data[id] = op(data[id], data[id + 16]);
    data[id] = op(data[id], data[id + 8]);
    data[id] = op(data[id], data[id + 4]);
    data[id] = op(data[id], data[id + 2]);
    data[id] = op(data[id], data[id + 1]);
}
template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v7(const T* input, T* output, size_t sz, ReduceOp op)
{
    extern __shared__ T shared_mem[];
    auto tid = threadIdx.x;
    const auto blockSize = blockDim.x;
    auto idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    shared_mem[tid] = op((idx < sz ? input[idx] : T {}),
        (idx + blockDim.x < sz ? input[idx + blockDim.x] : T {}));
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) {
            shared_mem[tid] = op(shared_mem[tid], shared_mem[tid + 256]);
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            shared_mem[tid] = op(shared_mem[tid], shared_mem[tid + 128]);
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            shared_mem[tid] = op(shared_mem[tid], shared_mem[tid + 64]);
        }
        __syncthreads();
    }
    if (tid < 32) {
        warpReduce_v7(shared_mem, tid, op);
        // no syncthreads()
    }
    if (tid == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}
```

完整代码见 [reduce_v7](reduce_v7.cuh)。

## 优化 8 合理设置 Block Size

具体与硬件相关，这里不做过多解释。

### 优化 9 考虑使用 cuda shuffle 指令

> `warp shuffle` 函数是cuda提供的在warp内直接进行数据交换的函数，灵活地使用能够避免共享内存的开销，支持灵活的线程间的数据传递。

这里用到的是：

```cpp
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize); 
```

该函数从与当前线程偏移量为 `delta` 的同一个 warp 的线程中提取变量 `var` 的值，然后返回该值。其中，`mask` 参数是一个32位的整数，用于控制线程的访问权限，如果某个线程的 `mask` 值为0，则该线程不参与这个函数的运算。 


`Shuffle` 指令是一组针对 `warp` 的指令。`Shuffle` 指令最重要的特性就是 `warp` 内的寄存器可以相互访问。在没有 `shuffle` 指令的时候，各个线程在进行通信时只能通过 shared memory 来访问彼此的寄存器。而采用了 `shuffle` `指令之后，warp` 内的线程可以直接对其他线程的寄存器进行访存。通过这种方式可以减少访存的延时。除此之外，带来的最大好处就是可编程性提高了，在某些场景下，就不用 shared memory 了。毕竟，开发者要自己去控制 shared memory 还是挺麻烦的一个事。 

下面是核心代码：

```cpp
/**
 * @brief 这个函数在 warp 级别（通常32个线程）内对输入值 sum 进行归约
*/
template <typename T, typename ReduceOp>
__device__ INLINE T warp_reduce_sum(T sum, size_t blockSize, ReduceOp op)
{
    // 根据块大小 blockSize 动态决定执行哪些归约步骤
    // 实际上类似于 shared memory 中的 reduction，只不过这里是针对 warp 内的 memory，
    // 因此大小只能是 32 以内，且不用同步
    if (blockSize >= 32)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 16));
    if (blockSize >= 16)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 8));
    if (blockSize >= 8)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 4));
    if (blockSize >= 4)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 2));
    if (blockSize >= 2)
        sum = op(sum, __shfl_down_sync(0xffffffff, sum, 1));
    return sum;
}

template <typename T, typename ReduceOp>
__global__ void reduce_kernel_v9(const T* input, T* output, size_t sz, T init, ReduceOp op)
{
    T sum = init;
    constexpr size_t kWarpSize = 32;
    // each thread loads one element from global memory to shared mem
    auto idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    auto tid = threadIdx.x;
    auto blockSize = blockDim.x;

    // similar to previous load two elements from global memory
    // 二次加载
#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        // sum += d_in[i + iter * blockSize];
        size_t bIdx = idx + iter * blockSize;
        sum = op(sum, (bIdx < sz ? input[bIdx] : init));
    }

    static __shared__ T warpLevelSums[kWarpSize];
    const auto laneId = tid % kWarpSize;  // 线程在一个 warp 中的id
    const auto warpId = tid / kWarpSize;  // 线程所在 warp 的 id

    // warp level reduction
    sum = warp_reduce_sum(sum, blockSize, op);

    if (laneId == 0) // 每个 warp 内的第一个线程负责将 sum 写入 shared memory
        // 注意写入的是 warpId，因此最终一个 block 内计算的数据被写入到了一个最大为32的共享内存中，
        // 因此可以用一个 warp 的 shuffle 指令来做最后计算
        warpLevelSums[warpId] = sum;  
    __syncthreads();
    // 第一个 warp 的线程再将 shared memory 中的数据写入到warp的寄存器中，方便shuffle
    sum = (tid < blockSize / kWarpSize) ? warpLevelSums[laneId] : init;
    if (warpId == 0)
        // blockSize / kWarpSize，如果 blockSize = 1024， 则刚好32块共享内存都用上；否则只有 blockSize / kWarpSize 的数据需要计算
        sum = warp_reduce_sum(sum, blockSize / kWarpSize, op);
    if (tid == 0)
        output[blockIdx.x] = sum;
}
```

完整代码见 [reduce_v9](reduce_v9.cuh)。

## 使用第三方库

### 使用 [Thrust](https://nvidia.github.io/cccl/thrust/index.html)

代码如下：

```cpp
            thrust::device_vector<ValueType> d_input(randomVec);
            // thrust::device_vector<ValueType> );
            timer.start();

            auto resThrust = thrust::reduce(thrust::device, d_input.cbegin(), d_input.cend(), static_cast<ValueType>(0), op);
            timer.stop();
```

性能普遍在上述 [优化5](#优化-5-解决-idle-线程) 和 [优化6](#优化-6-人工展开最后一层循环以减少同步) 之间波动 (`gcc9 ubuntu22.04, cuda12.0`)

在 Windows (`_MSC_VER = 1942 cuda 11.8`) 上性能最高，且远大于其他 CUDA 实现。

## 扩展

同样的优化手段也适用于内积操作。这里不做过多介绍。

# 测试结果

