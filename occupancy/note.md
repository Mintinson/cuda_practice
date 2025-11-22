
# Occupancy 计算和分析

## Occupancy 概念

一个 SM 中理论最大支持的 `active warps` 数为 `M`，实际上只有 N 个 warps 活跃，则 `N/M` 就是 Occupancy。

GPU 的 SM 可能因为资源限制，比如执行单元，寄存器，共享对称使用太多，导致并不是所有 warp 都同时活跃。

- 用户限制的 block 大小会影响实际 SM 可以容纳的 warp 数量。
- CUDA 如果一个 warp 处于等待，则会立刻切换到另一个 warp，这个过程可以看做是零开销的（与 CPU 不一样），另外，处于等待的 warp 也是 active warp。
- 共享内存和寄存器类似，每个 SM 有固定大小，所以也影响 Warp （Block 块）的数量。
- 在结合木桶效应，取最小值可以得到每个 SM 能容纳的 warp 数量。

## 理论 Occupancy 计算例子

### Block Size 对 Occupancy 的影响

在 RTX 4060 中，规则：

| property                             | limit |
| ------------------------------------ | ----- |
| Threads Per Warp                     | 32    |
| Max warps per multiprocessor         | 48    |
| Max Thread Blocks Per Multiprocessor | 24    |

假设我们每个 thread 都只使用少量的 smem 和 register，因此没有超出关于 shaerd memory 和 register 的限制，而设置 block 大小为 `(32 x 5)`

则一个 Block 的线程数为 160，需要 warp 5 个，而一个 Block 的所有线程必须都在一个 SM 上，而一个 SM 最多涵盖 48 个 warps，因此一个 SM 最多只能分配 $\lfloor 48 / 5 \rfloor = 9$ 个 blocks 即 45 个 warps，因此占用率是 `45 / 48 = 0.937`

而如果设置 block 为 32，则一个 block 只需要一个 warp，但是由于一个 SM 最多只能含有 24 个 Blocks，因此只有 24 个 warp 用上了，占用率是 0.5.

### Register 对 Occupancy 的影响

| property                        | limit |
| ------------------------------- | ----- |
| Register per multiprocessor     | 65536 |
| Register Allocation Unit Size   | 256   |
| Register Allocation Granularity | Warp  |
| Warp AllocationGranularity      | 4     |

其中

- Register Allocation Unit Size 指的是寄存器一次分配只能按照 256 的倍数分配。
- Register Allocation Granularity 和 Warp AllocationGranularity 意思是寄存器分配是 4 个 warp，4 个 warp 进行分配。

例如，对于 BlockSzie = 128，如果不计较寄存器，则此时占有率为 100%。

如果一个线程要 90 个寄存器，则此时一个 warp 总共需要 32 x 90 = 2880 个，但是要根据 256 分，因此需要分配 $\lceil(2880 / 256) \rceil\times 256 = 3072$ 个寄存器。。一个 warp 要 3072 个寄存器，因此一个 SM 最多是 65536 个寄存器，因此一个 SM 上最多只能有 $\lfloor 65536 / 3072 \rfloor = 21$ 个 warps，但是要按照 4 个 warps 来分配，因此最终只能有 20 个 warps，因此占用率是 $20/48 = 0.416$

### Shared Memory 对 Occupancy 的影响

| property                                       | limit |
| ---------------------------------------------- | ----- |
| Shared Memory per multiprocessor (bytes)       | 16384 |
| Max Shared Memory per Block                    | 16284 |
| Shared Memory Allocation Unit Size             | 128   |
| Shared Memory Per Block (bytes) (CUDA runtime) | 1024  |

其中 Shared Memory Allocation Unit Size 指的是 Shared memory 只能按照 128 字节进行分配，而 Shared Memory Per Block (bytes) (CUDA runtime) 指的是对于所有 block，即使没有显示使用共享内存，cuda runtime 也会分配 1024 个字节来启动该核。

例如，对于一个 128 个线程的 Block，其中每个 Block 需要显式使用 5000 Bytes 共享内存，则每个 Block 总共需要 1024 + 5000 = 6024 bytes 的共享内存。而只能按照 128 分配的话，也就是 `6024/128 = 47.0625 -> 48 * 128 = 6144 bytes` 的共享内存，而一个 SM 最多只能有 16384 bytes，因此一个 SM 只能有 $\lfloor 16384/6144  \rfloor = 2$ 个 Block，也就是 `2 x 128 / 32 = 8` 个 warp，因此占用率为 $4 / 48 = 0.167$

注意，Shared Memory per multiprocessor (bytes) 是可以配置的，对于计算能力 8.9 的设备，其配置范围为：$[8192, 102400]$, 每次按 2 倍递增。因此，如果我们配置为 102400，则上述其他配置不变的情况下，占用率为 100%。

---

注意：较高的占用率并不总是能带来更高的性能，不过，较低的占用率总会降低隐藏延迟的能力。进而导致整体性能下降。上述计算的都是理论占用率，我们可以通过 Nsight Compute 中的 Occupancy Calculator 自动计算。实际占用率往往与理论占用率存在差异，如果差异较大，说明工作负载高度不均匀。

## 实际 Occupancy 影响因素

![](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250428112610.png)

首先如上图所示，Ada 结构下（4060）， 一个SM有4个 Warp Scheduler，而 一个 warp 可以运行 48 个 warps，因此一个 warp scheduler 最多利用控制 12 个 warps。

一个 warp scheduler 上面有 12 个 warp slots，用于存放等待调度并执行的warp（用于记录每个 warp 的槽位），一个warp 有两种状态，激活的和未被激活的，其中激活的又可以分为三种状态，分别是 stalled(停滞的)，eligible（符合条件的），selected(被选中的)。只有激活的warp才能放入槽中。一个 warp scheduler 一个时钟周期可以 issue 1 个 warp

因此，虽然理论上 Occupancy可以用上述计算，但是实际上的Occupancy可能要通过 warp slots的占用率来计算。

造成 warp stalled 的原因可能如下：
* 在读取指令
* 依赖内存指令的访存结果
* 依赖之前指令的执行结果
* pipeline 正在忙
* 同步 barrier

现在看单纯的向量加法：

```c++
template <typename T>
__global__ void add_kernel(T* d_vecA, T* d_vecB, T* d_vecC)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    d_vecC[idx] = d_vecA[idx] + d_vecB[idx];
}
```

nsight compute 输出如下：

![](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250428114729.png)

可以看到，只实现了87.37的占用率。

根据建议可得：计算出的理论占有率（100.0%）与实际测得的占有率（84.2%）之间的差异可能是由于内核执行期间的 warp **调度开销或工作负载不平衡所致**。工作负载不平衡可能发生在同一个块内的 warp 之间，也可能发生在同一个内核的不同块之间。

即上述核函数的工作只是相加两个数，却要对全局内存做两次读，一次写，因此调度开销和工作负载不平衡。
 
再看 *Schedluer Statistics* 这一栏

![](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250428120353.png)

可以看到，平均一个scheduler只有10.37个活跃线程，因此实际占用率是 10.37/12=0.864.

调度器指令发布活动总结如下：
- 每个调度器维护一个可发布指令的线程束池。池中线程束上限（理论线程束数）受启动配置限制。
- 每个周期，调度器检查池中分配线程束的状态（活动线程束）。
- 未阻塞的活动线程束（可调度线程束）已准备好发布下一条指令。
- 调度器从可调度线程束中选择一个线程束来发布一条或多条指令（已发布线程束）。
- 若无可用线程束，发布时隙将被跳过，本周期内不会发布指令。频繁跳过发布时隙表明延迟隐藏效果不佳。

我们再看 *Warp State Statistics* 这一栏：

Warp 状态描述了 warp 是否准备好发出下一条指令或无法发出的原因。Warp 每条指令的周期数决定了两条连续指令之间的延迟。值越高，表明需要更多的 warp 并行性来隐藏延迟。对于每个 warp 状态，图表显示了每条已发出指令在该状态下平均花费的周期数。阻塞并不总是影响整体性能，也不是完全可以避免的。只有当调度程序未能在每个周期内发出指令时，才需要关注阻塞原因。当执行包含混合库代码和用户代码的内核时，这些指标显示的是组合值。

![](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250428115929.png)

由上图给的指示可以知道：*在执行该内核时，平均每个 warp 有 162.0 个周期因 L1TEX（本地、全局、表面、纹理）操作的记分板依赖而阻塞。找出产生被等待数据的指令以确定罪魁祸首。为了减少等待 L1TEX 数据访问的周期数，需验证内存访问模式是否针对目标架构进行了优化，可通过增加数据局部性（合并访问）来提高缓存命中率，或者更改缓存配置。考虑将频繁使用的数据移至共享内存。这种阻塞类型约占每次发出两条指令之间平均 170.3 个周期的 95.2%。*

因此，我们的占用率低是因为全局访存相对较大。

常见的 Stall Reason：

* **Long Scoreboard**
    * 长记分板阻塞,这是最常见的Stall之一，主要是由于等待全局内存（Global Memory）或L2缓存的数据返回。当Warp执行一个加载（Load）指令，从全局内存读取数据时，由于内存访问延迟很高（数百个时钟周期），该Warp必须等待数据到达寄存器后才能继续执行依赖该数据的后续指令。GPU使用记分板（Scoreboard）机制来跟踪这种依赖关系。"Long" 指的就是这种长延迟的操作。
    * 可以减少内存访问延迟/次数，使用共享内存来优化
* **Short Scoreboard branching**
    * Short Scoreboard Stall: 指的是等待短延迟操作的结果，比如等待来自共享内存的数据、等待寄存器文件的读写、或者等待前一条算术指令（如浮点乘加）的结果。这通常发生在指令之间存在数据依赖（Data Dependency）时。
    * Branching (Warp Divergence) Stall: 指的是由于Warp内线程执行了不同的条件分支（如 if-else），导致执行路径发散。硬件必须串行执行每个不同的路径，而未走该路径的线程则处于阻塞状态。这虽然不是典型的Scoreboard阻塞（等待数据），但它阻止了Warp作为一个整体前进，表现为Stall。
    * 优化关注点：调整代码顺序，将相互依赖的指令分离开，中间插入可以并行执行的独立指令。编译器会做优化，但手动调整有时也有帮助（例如循环展开）
* **LG Throttle**
    * 本地/全局内存队列节流：表示Warp尝试发出 **本地内存（Local Memory）或全局内存（Global Memory）的加载/存储指令时，发现负责处理这些请求的硬件单元（如Load/Store Unit, LSU）或其相关的请求队列（LGK - Load Global Queue）已满或正忙。这通常发生在大量Warp同时尝试访问内存，导致内存访问发起端（Initiation）** 出现瓶颈。它反映的是内存指令发射的吞吐量限制，而不是内存返回的延迟（那是Long Scoreboard）。
    * 同Long Scoreboard优化，减少对Local/Global内存的访问次数。
* **MIMO Throttle**
    * 内存输入输出节流：通常指SM和下一级内存（如L1缓存、纹理缓存、L2缓存）之间的数据传输带宽达到了瓶颈。当大量数据需要通过内存接口传输时（无论是读还是写），即使LSU能够发出请求，数据通路本身也可能变得拥塞。这表明内存子系统的带宽受限。
    * 优化建议：提高缓存命中率: 更好地利用L1/Texture缓存可以减少对L2和全局内存的带宽需求；减少数据传输量: 使用更紧凑的数据类型，只传输必要的数据；内存访问合并: 提高合并度可以更有效地利用可用带宽；数据压缩/解压: 如果计算开销可接受，可以考虑在GPU上进行数据压缩/解压，以减少传输量。
* **Math Pipe Throttle**
    * 数学流水线节流：指Warp等待特定的算术/数学执行单元（如FP32（单精度浮点）单元、FP64（双精度浮点）单元、INT32（整数）单元、SFU（特殊函数单元，如sin, cos, exp, rsqrt）、或者Tensor Core等）变得可用。当代码中密集地使用了某一种类型的计算指令，超出了SM上对应计算单元的吞吐能力时，就会发生这种阻塞。这表明计算受限于特定类型运算单元的处理能力。
    * 优化建议：平衡指令组合: 混合使用不同类型的计算指令（整数、浮点、特殊函数），以利用SM上所有类型的计算单元；降低计算强度:使用数学等价但计算量更小的公式；使用近似计算（如果精度允许）；如果精度允许，使用FP16（半精度）代替FP32，因为FP16单元通常有更高的吞吐量；对于符合条件的矩阵运算，使用Tensor Core，其吞吐量远超标准FP单元。

### 例子

**Long Scoreboard**：

```c++
template <typename T>
__global__ void stall_reason_lsb(T *data)
{
    int tid = threadIdx.x;
    int laneId = tid % 32;
    data[laneId] = laneId;

    __syncthreads();

    auto idx = laneId;

    for (int i = 0; i < 1000; ++i)
    {
        idx = data[idx];
    }
    data[laneId] = idx;
}
```

导致 Eligible Warps per scheduler 只有 0.45, 大部分都因为 Long Scoreboard 而被卡住。

**LG Throttle** 例子：

```c++
template <typename T>
__global__ void stall_reason_lgt(T *data, T *out)
{
    int tid = threadIdx.x;
    int offset = tid * 100;
#pragma unroll
    for (int i = 0; i < 200; ++i)
    {
        out[offset + i] = data[i + offset];
    }
}
```

导致 Eligible Warps per scheduler 只有 0.02, 而 issued warp per scheduler 接近 0.00， 是十分吓人的。

**Short Scoreboard Stall**

```cc
template <typename T>
__global__ void stall_reason_ssb(T *data)
{
    __shared__ T smm[32];

    int tid = threadIdx.x;
    int laneId = tid % 32;

    smm[laneId] = laneId;
    __syncthreads();

    int idx = laneId;
    for (int i = 0; i < 100; ++i)
    {
        idx = smm[idx];
    }
    data[laneId] = idx;
}
```

## 实例

回到在 [coalesced_practice](../coalesced_demo/coalesced_practice.cu) 中矩阵转置的例子，在探讨合并访存的影响的时候，我们得到的最优的优化方式为采用 `float4`, 在一个线程中利用4个 `float4` 取得一个 `4x4` 的矩阵，然后在寄存器中进行转置后，再写回到全局内存中，同时当时最优的线程块形状为 `8x32`.

观察采用 `float4` （蓝色）和 naive （绿色）的矩阵转置的 warp state （两者都是最优的线程块形状）可以看到：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250429112310.png)

`float4` 使得大量的线程因为全局存储而停滞，这是因为使用 `float4`，我们线程数是之前的 1/16，这就使得并行度降低了。

有没有办法在提升并行度的同时，尽可能保持较高的合并访存呢？我们可以在中间取一个 trade-offs，使用 `float2`, 对 `2x2` 的矩阵进行转置，其代码如下：

```c++
__global__ void transpose_global_2x2(const float *input, float *output, std::size_t m, std::size_t n)
{
    auto colIdx = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
    auto rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
    if (colIdx >= n || rowIdx >= m)
    {
        return;
    }
    float2 srcVec2[2];
    float2 dstVec2[2];
    srcVec2[0] = fetch_vec2(input + rowIdx * n + colIdx);
    srcVec2[1] = fetch_vec2(input + (rowIdx + 1) * n + colIdx);

    dstVec2[0] = make_float2(srcVec2[0].x, srcVec2[1].x);
    dstVec2[1] = make_float2(srcVec2[0].y, srcVec2[1].y);

    fetch_vec2(output + colIdx * m + rowIdx) = dstVec2[0];
    fetch_vec2(output + (colIdx + 1) * m + rowIdx) = dstVec2[1];
}
```

我们得到的结果对比：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250429112737.png)
![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250429112751.png)

其中蓝色是 2 x 2，紫色是 4 x 4，绿色是 naive，可以看到 2 x 2 ，其 Stall Long Scoreboard 下降了，而 Active Warps Per Scheduler 相差不大。

同时，虽然该方法的合并访存没有 4 x 4 那么好，但是速度稍微比 4 x 4 要快一些，4 x 4 是 24.03 ns，而 2 x 2 达到了 21.82 ns。

我们还可以跟激进一点，只读取 1 x 2 的数据，进行转置，其代码如下：

```c++
__global__ void transpose_global_1x2(const float *input, float *output, std::size_t m, std::size_t n)
{
    auto colIdx = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
    auto rowIdx = (blockIdx.y * blockDim.y + threadIdx.y);
    if (colIdx >= n || rowIdx >= m)
    {
        return;
    }
    float2 srcVec2;
    srcVec2 = fetch_vec2(input + rowIdx * n + colIdx);


    *(output + colIdx * m + rowIdx) = srcVec2.x;
    *(output + (colIdx + 1) * m + rowIdx) = srcVec2.y;
}
```
该实现的执行时间与 `2x2` 不相上下，但由于 Stall Long Scoreboard 导致的时间周期更小

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250429113241.png)
