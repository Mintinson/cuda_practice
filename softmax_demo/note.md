# Parallel Softmax 的实现与优化

## Softmax 计算公式

对于向量 $x = [x_0, x_1, \dots, x_{N-1}]^T$

$$
\sigma(x_i) = \frac{e^{x_i}}{\sum_{j=0}^{N-1} e^{x_i}}
$$

实际计算中，为了维持数值稳定性，往往通过减最大值的方法:

$$
\sigma(x_i) = \frac{e^{x_i - c}}{\sum_{j=0}^{N-1} e^{x_i-c}}
$$
其中 $c = \max_j x_j$.

可以很容易证明两者的值是相等的。

Safe Softmax 的计算自然分为**三遍扫描**（Three-Pass）：

| 遍次 | 操作 | 数学表达式 |
|:--: | :--:| :--:|
|第 1 遍 | 求行最大值 | $c = \max_j x_j$ |
|第 2 遍 | 指数求和 | $d = \sum_j e^{x_j-c}$ |
|第 3 遍 | 归一化 | $y_i =e^{x_i - c} / d$ |

每一遍都需要读取整行数据，对于长度为 $N$ 的向量来说，总共需要 3N 此全局内存读取。因此 Softmax 属于典型的 **Memory-Bound** 操作，优化目标是**减少内存访问次数和提升带宽利用率**。

## 优化 v0: 简单的 GPU 实现 （一个线程处理一行）

最简单的并行化策略：每个线程独立处理输入矩阵的一行，串行完成该行的 max、exp-sum、normalize 三步。行与行之间完全独立，天然并行 [代码](./softmax_v0.cuh)。

```cpp
template <typename T>
__global__ void softmax_v0_kernel(const T *__restrict__ A, T *__restrict__ C,
                                  int m, int n) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < m) {
    // max
    T max_val = std::numeric_limits<T>::min();
    // norm factor
    T sum = 0.0f;

    // 3 passes (not optimal)
    for (int col = 0; col < n; col++) {
      int i = row * n + col;
      max_val = max(max_val, A[i]);
    }
    for (int col = 0; col < n; col++) {
      int i = row * n + col;
      sum += expf(A[i] - max_val);
    }
    for (int col = 0; col < n; col++) {
      int i = row * n + col;
      C[i] = expf(A[i] - max_val) / sum;
    }
  }
}
```

该方法简单，但是性能极差，原因：

1. 并行度不足：每行只有 1 个线程，N=4096 个元素被串行处理。GPU 的数千个 CUDA Core 绝大部分在空闲
2. 非合并访存：同一 Warp 内的 32 个线程处理 32 个不同行，它们在内循环中访问的地址相差 $N \times 4$ 个字节造成严重的非合并访存。
3. 冗余计算：`expf(x[i] - max_val)` 在 Pass 2 和 Pass 3 中重复计算了两次

## 优化 v1: 从数学上进行优化（一个线程处理一行）

如果数学或者计算机基本原理比较好的话，可以发现我们其实可以不用三此pass，我们可以边求和，边计算最大值，而通过指数的（乘积等于幂数相加），一旦我们找到新的最大值，可以立刻通过乘法调整。

假设 $x[j] = \max_{i=0..k-1}x[i]$, 而 $x[k] > x[j]$，且 $j < k$, 再计算到 $k-1$ 的时候，我们已经有
$$
s[k-1] = \sum_{i=0}^{k-1} e^{x[i] - x[j]}
$$

当到达 $k$ 后，我们可以计算：
$$
s[k] =  \sum_{i=0}^{k} e^{x[i] - x[k]} = \sum_{i=0}^{k-1} e^{x[i] - x[k]} + 1 = 
 \left(\sum_{i=0}^{k-1} e^{x[i] - x[j]} \right ) \cdot e^{x[j] - x[k]} + 1 = s[k-1]  \cdot e^{x[j] - x[k]} +1
$$

因此我们实际上只要记录最大值和累加值 1 次 pass 就能计算出两者的最终值

```cpp
/*
How this works:
One thread processes one entire row, but instead of 3 passes we do only 2 passes.
This is possible due to the property of exponentials.
We are parallelizing over the rows.
*/
template <typename T>
__global__ void softmax_v1_kernel(T* __restrict__ matd, T* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        T m = std::numeric_limits<T>::min();
        T L = {};

        // compute max and norm factor in one pass only
        // by exploiting the property of exponentials
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            T curr = matd[i];
            if (curr > m) {
                L = L * expf(m - curr);
                m = curr;
            }
            L += expf(curr - m);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            resd[i] = expf(matd[i] - m) / L;
        }
    }
}
```

## 优化 v2: 一个 Block 处理一行

当前的核心问题是并行度不够——一行 $N$ 个元素只用 1 个线程处理。解决方案是让一个 Block 内的多个线程协作处理同一行。每个线程负责一行中的一段数据，通过 Shared Memory 做并行规约求出 max 和 sum。

这等于把 Softmax 分解为两个 Reduce 操作（max-reduce + sum-reduce）加一个 element-wise 操作（normalize），每个 Reduce 都使用我们在 [reduce](../readme.md) 优化文章中学到的技术。

```cpp
    /*
    How this works:
    One thread processes one entire row, but instead of 3 passes we do only 2 passes.
    This is possible due to the property of exponentials.
    We are parallelizing over the rows.
    */
    template <typename T>
    __global__ void softmax_v2_kernel(T *__restrict__ matd, T *__restrict__ resd, int M, int N)
    {
        __shared__ T smem[1024];

        int row = blockIdx.x;
        int tid = threadIdx.x;

        // edge condition (we don't process further)
        if (row >= M)
            return;

        T *input_row = matd + row * N;
        T *output_row = resd + row * N;
        T local_max = std::numeric_limits<T>::min();
        T local_norm = 0.0f;

        // compute local max and norm for each thread
        // and then finally have a sync barrier before moving on
        for (int i = tid; i < N; i += blockDim.x) // note the step = blockDim.x
        {
            T x = input_row[i];
            if (x > local_max)
            {
                local_norm *= expf(local_max - x);
                local_max = x;
            }
            local_norm += expf(x - local_max);
        }
        __syncthreads();

        // each thread will have its own local max
        // we store it in the tid of the shared memory
        smem[tid] = local_max;
        __syncthreads();

        // block-level reduction in O(log(N)) time over all threads
        // is faster than linear reduction over all threads
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (tid < stride)
            {
                smem[tid] = max(smem[tid], smem[tid + stride]);
            }
            // sync barrier before next iteration to ensure correctness
            __syncthreads();
        }

        // the first element after max reduction from all threads
        // will contain the global max for the row
        T row_max = smem[0];
        // __syncthreads();

        // each thread will have its own local norm
        // we will store the corrected local norm in the shared memory
        // again, exploits property of exponentials
        smem[tid] = local_norm * expf(local_max - row_max);
        __syncthreads();

        // sum reduction similar to above for global norm factor
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }
        T row_norm = smem[0];
        __syncthreads();

        // finally, compute softmax
        for (int i = tid; i < N; i += blockDim.x)
        {
            output_row[i] = expf(input_row[i] - row_max) / row_norm;
        }
    }
```

仍存在的问题：

* Shared Memory 规约需要多轮 `__syncthreads()`

### 优化 v3：使用 warp shuffle 优化规约

类似 reduce，我们可以利用 warp shuffle 来减少 shared memory 的使用，从而减少 `__syncthreads()` 的使用，毕竟显式同步往往是比较耗时的。

```cpp

    template <typename T>
    __global__ void softmax_v3_kernel(T *__restrict__ matd, T *__restrict__ resd, int M, int N)
    {
        // max and norm reduction will happen in shared memory (static)
        extern __shared__ T smem[];

        int row = blockIdx.x;
        int tid = threadIdx.x;
        // number of threads in a warp
        unsigned int warp_size = 32;
        if (row >= M)
            return;

        T *input_row = matd + row * N;
        T *output_row = resd + row * N;
        T local_max = -std::numeric_limits<T>::infinity();
        T local_norm = 0.0f;

        // 每个线程各自计算自己的最大值 和 累积和
        for (int i = tid; i < N; i += blockDim.x)
        {
            T x = input_row[i];
            if (x > local_max)
            {
                local_norm *= exp_op(local_max - x);
                local_max = x;
            }
            local_norm += exp_op(x - local_max);
        }

        // warp 内求最大值
        T val = local_max;
        for (int offset = warp_size / 2; offset > 0; offset /= 2)
        {
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        }

        // when blockDim is greater than 32, we need to do a block level reduction
        // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
        // the global max will be stored in the first warp
        // warp 间求最大值
        if (blockDim.x > warp_size)
        {
            // 每个 warp 中的第一个线程将自己warp中的最大值放在共享mem中
            if (tid % warp_size == 0)
            {
                // which warp are we at?
                // store the value in its first thread index
                smem[tid / warp_size] = val;
            }
            __syncthreads();

            // first warp will do global reduction only
            // this is possible because we stored the values in the shared memory
            // so the threads in the first warp will read from it and then reduce
            // 再由第一个 warp 中的线程通过warp内reduce取出最大值 
            if (tid < warp_size)
            {
                val = (tid < (blockDim.x + warp_size - 1) / warp_size) ? smem[tid] : std::numeric_limits<T>::lowest();
                for (int offset = warp_size / 2; offset > 0; offset /= 2)
                {
                    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
                }
                // 第一个线程保存最大值
                if (tid == 0)
                    smem[0] = val;
            }
        }
        else
        {
            // this is for when the number of threads in a block are not
            // greater than the warp size, in that case we already reduced
            // so we can store the value
            if (tid == 0)
                smem[0] = val;
        }
        __syncthreads();

        // we got the global row max now
        // 取出当前行最大值
        T row_max = smem[0];

        // same reduction algorithm as above, but instead of max reduction
        // we do a sum reduction i.e. we accumulate the values
        // val = smem[tid];
        // 有了当前行最大值，就可以计算真正的累积和了
        // 这同样是个 reduce 操作，因此可以用 warp shuffle
        val = local_norm * exp_op(local_max - row_max);
        for (int offset = warp_size / 2; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (blockDim.x > warp_size)
        {
            if (tid % warp_size == 0)
            {
                smem[tid / warp_size] = val;
            }
            __syncthreads();

            // first warp will do global reduction
            if (tid < warp_size)
            {
                val = (tid < (blockDim.x + warp_size - 1) / warp_size) ? smem[tid] : 0.0f;
                for (int offset = warp_size / 2; offset > 0; offset /= 2)
                {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                if (tid == 0)
                    smem[0] = val;
            }
        }
        else
        {
            if (tid == 0)
                smem[0] = val;
        }
        __syncthreads();
        // 取出累计值
        T row_norm = smem[0];
        // __syncthreads();

        // finally, compute softmax
        for (int i = tid; i < N; i += blockDim.x)
        {
            output_row[i] = exp_op(input_row[i] - row_max) / row_norm;
        }
    }
```

### 优化 v4: 使用向量化加载

V3 的瓶颈已经转移到全局内存加载阶段。每次循环中，每个线程只加载 1 个 float（4 字节），指令调度开销相对于数据吞吐比例偏高。使用 float4 向量化加载，每条指令搬运 16 字节（4 个 float），可以：

* 减少加载指令总数（减少 4 倍的循环迭代）
* 提升内存事务的利用效率
* 增加指令级并行（ILP）


由于代码量较大，因此这里不放出来，具体代码见 [softmax_v4](./softmax_v4.cuh)

### 优化 v5：一个 Block 处理多行

v4 每个 Block 处理一行；v5 将 4 行作为一个 tile，由一个 Block 处理。Block 使用二维线程布局：每个 warp 对应一行，线程沿列方向分工，并通过 warp shuffle 分别归约行最大值与归一化因子。这样减少了 Block 数量，也无需跨 warp 的共享内存规约。

在线计算每个线程负责列上的局部最大值和指数和；warp 求出行最大值后，先校正局部和，再归约得到行和，最后写出 softmax 结果。行索引为 `blockIdx.x * TILE_SIZE + threadIdx.y`，当前 tile 大小为 4，见 [softmax_v5](./softmax_v5.cuh)。

多行分块能减少调度开销，但不一定总比 v4 快：性能仍取决于列数、寄存器使用和 GPU 占用率。
