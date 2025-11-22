## CUDA WMMA HGEMM 核函数实现

### **CUDA WMMA HGEMM 核函数优化分析笔记**

本文档分析了 [wmma_hgemm.cuh](wmma_hgemm.cuh) 文件中实现的几个使用 Tensor Core (WMMA \- Warp Matrix Multiply Accumulate) 的半精度矩阵乘法（HGEMM）核函数，并详细解释了它们逐步引入的优化技术及其原理。

#### **1\. wmma\_hgemm\_16x16\_kernel (基础版本)**

* **核函数签名:**  
  template \<const int WMMA\_M \= 16, const int WMMA\_N \= 16, const int WMMA\_K \= 16\>  
  \_\_global\_\_ void wmma\_hgemm\_16x16\_kernel(half \*a, half \*b, half \*c,  
                                          const std::size\_t m, const std::size\_t n, const std::size\_t k)

* **主要思想:**  
  * **Tensor Core (WMMA) 基本应用:** 这是最基础的 WMMA 使用示例。它直接利用 nvcuda::wmma 命名空间提供的 API 来执行 16x16x16 的矩阵乘法累加操作。  
  * **Warp 级别并行:** 每个 CUDA block 只包含一个 warp (32 个线程)，这个 warp 协同完成一个 16x16 输出子块的计算。Grid 的维度决定了总共计算多少个这样的子块来覆盖整个输出矩阵 C。  
* **优化点分析:**  
  * **WMMA 指令利用:**  
    * wmma::fragment: 定义了用于存储 A、B、C 矩阵子块在 warp 线程寄存器中分布的数据结构。每个线程只持有整个子块的一部分数据。  
    * wmma::load\_matrix\_sync: 从全局内存同步加载 A 和 B 矩阵的子块到寄存器 fragment 中。sync 表示 warp 内所有线程必须等待加载完成后才能继续。  
    * wmma::mma\_sync: 执行核心的 D \= A \* B \+ C 操作。利用 Tensor Core 进行高效的矩阵乘法累加。sync 同样表示需要等待计算完成。  
    * wmma::store\_matrix\_sync: 将计算结果（累加后的 C fragment）从寄存器同步写回到全局内存。  
  * **循环 K 维度分块 (Tiling):** 通过 NUM\_K\_TILES 循环，将 K 维度分成大小为 WMMA\_K (16) 的块，逐步累加结果到 C\_frag。这是矩阵乘法的标准分块策略。  
  * **\#pragma unroll:** 提示编译器展开 K 维度的循环，可能减少循环开销，但在此基础版本中效果有限，因为主要瓶颈是访存。  
* **局限性:**  
  * **全局内存访问:** 每次 K 循环迭代都需要直接从全局内存加载 A 和 B 的子块，访存延迟高，带宽利用率低。  
  * **无数据复用:** 没有利用共享内存（Shared Memory）来缓存数据，A 和 B 的子块在 K 维度上没有被复用。  
  * **低占用率:** 每个 block 只有一个 warp，可能无法充分利用 SM (Streaming Multiprocessor) 的计算资源，难以隐藏访存延迟。

#### **2\. hgemm\_wmma\_m16n16k16\_mma4x2\_kernel (引入共享内存)**

* **核函数签名:**  
  template \<const int WMMA\_M \= 16, const int WMMA\_N \= 16, const int WMMA\_K \= 16,  
            const int WMMA\_TILE\_M \= 4, const int WMMA\_TILE\_N \= 2\>  
  \_\_global\_\_ void hgemm\_wmma\_m16n16k16\_mma4x2\_kernel(half \*a, half \*b, half \*c,  
                                                     int m, int n, int k)

* **主要思想:**  
  * **共享内存 (Shared Memory) Tiling:** 引入共享内存 (s\_a, s\_b) 作为 L1 Cache 的软件管理替代，缓存从全局内存加载的 A 和 B 矩阵块。  
  * **Block 级别 Tiling:** 每个 CUDA block (包含多个 warp，这里是 256 线程/8 warp) 负责计算一个更大的输出子块 C，尺寸为 BM x BN (64 x 32)。这个大块由多个 WMMA tile (16x16) 组成。  
* **优化点分析:**  
  * **共享内存缓存:**  
    * \_\_shared\_\_ half s\_a\[BM\]\[BK\], s\_b\[BK\]\[BN\];: 定义了用于缓存 A 和 B 子块的共享内存数组。  
    * **数据复用:** 在 K 维度的循环 (NUM\_K\_TILES) 中，block 内的所有 warp 可以重复访问加载到共享内存中的 s\_a 和 s\_b 数据，显著减少了对高延迟全局内存的访问次数。  
  * **协同加载 (Cooperative Loading):**  
    * Block 内的所有线程 (256个) 协同将所需的大块数据从全局内存搬运到共享内存 (s\_a 和 s\_b)。  
    * **内存访问合并:** 通过精心计算每个线程负责加载的数据地址 (load\_gmem\_a\_addr, load\_gmem\_b\_addr)，并使用 LDST32BITS/LDST64BITS 宏（本质是 reinterpret\_cast 到 half2 或 float2）进行 32 位或 64 位宽度的加载，尝试合并访存，提高全局内存带宽利用率。  
  * **Warp 内部计算分工:**  
    * 每个 block 内有 WMMA\_TILE\_M \* WMMA\_TILE\_N (4 \* 2 \= 8\) 个 warp。  
    * 通过 warp\_id, warp\_m, warp\_n 计算，将 block 计算的 BM x BN 大块 C 分配给 8 个 warp，每个 warp 负责计算一个 WMMA\_M x WMMA\_N (16x16) 的子块。  
    * wmma::load\_matrix\_sync 现在从**共享内存**加载数据到寄存器 fragment。  
  * **同步:** \_\_syncthreads() 用于确保：  
    * 所有线程都完成了从全局内存到共享内存的数据加载后，才能开始从共享内存加载到寄存器和执行 WMMA 计算。  
    * 所有 warp 完成了当前 K 步的计算后，才能进入下一个 K 步（加载下一批数据到共享内存）。  
* **改进:** 相比基础版本，显著降低了全局内存访问频率，提高了数据复用率，性能通常会有较大提升。

#### **3\. hgemm\_wmma\_m16n16k16\_mma4x2\_warp2x4\_kernel (Warp 级 Tiling 和寄存器 Blocking)**

* **核函数签名:**  
  template \<const int WMMA\_M \= 16, const int WMMA\_N \= 16, const int WMMA\_K \= 16,  
            const int WMMA\_TILE\_M \= 4, const int WMMA\_TILE\_N \= 2,  
            const int WARP\_TILE\_M \= 2, const int WARP\_TILE\_N \= 4\>  
  \_\_global\_\_ void hgemm\_wmma\_m16n16k16\_mma4x2\_warp2x4\_kernel(half \*a, half \*b,  
                                                             half \*c, int m,  
                                                             int n, int k)

* **主要思想:**  
  * **更大的 Block Tile:** Block 负责的 C 子块尺寸进一步增大到 BM x BN (128 x 128)。  
  * **Warp 级 Tiling (Warp-Level Tiling / Register Blocking):** 每个 warp 不再只计算一个 16x16 的 WMMA tile，而是负责计算 WARP\_TILE\_M x WARP\_TILE\_N (2 x 4 \= 8\) 个 WMMA tile。这意味着每个 warp 需要处理更多的寄存器 fragment。  
* **优化点分析:**  
  * **增加数据复用:**  
    * 更大的 Block Tile (128x128) 意味着加载到共享内存 (s\_a, s\_b) 的数据会被 block 内的 warp 复用更多次，进一步摊销全局内存加载成本。共享内存大小也相应增加 (4KB \+ 4KB \= 8KB)。  
  * **提高计算密度和 ILP (Instruction Level Parallelism):**  
    * C\_frag\[WARP\_TILE\_M\]\[WARP\_TILE\_N\]: 每个 warp 使用二维数组存储多个累加器 fragment。  
    * A\_frag\[WARP\_TILE\_M\], B\_frag\[WARP\_TILE\_N\]: 每个 warp 也需要存储多个 A 和 B 的 fragment。  
    * **寄存器 Blocking:** 将更多的中间结果（多个 C fragment）和输入（多个 A/B fragment）保存在寄存器中。  
    * **循环展开 (\#pragma unroll):** 内层的 i, j 循环（遍历 warp 内的 tile）被展开。这使得 warp 在一次共享内存加载后，可以连续执行多次 wmma::load\_matrix\_sync (从共享内存到寄存器) 和 wmma::mma\_sync (计算)。这增加了指令级并行度，有助于隐藏计算指令本身的延迟，并减少了从共享内存加载数据的频率。  
  * **更宽的内存加载:**  
    * LDST128BITS: 使用 128 位（float4）加载，进一步尝试合并全局内存访问，提高带宽利用率。  
  * **计算与访存平衡:** 通过让每个 warp 做更多计算（8 次 MMA），提高了计算强度，更好地平衡了访存和计算的比例。  
* **改进:** 通过增加 block tile 大小和引入 warp 级 tiling/寄存器 blocking，进一步提高了数据复用，增加了计算密度和 ILP，通常能获得比上一个版本更好的性能。

#### **4\. hgemm\_wmma\_m16n16k16\_mma4x2\_warp2x4\_dbuf\_async\_kernel (双缓冲与异步拷贝)**

* **核函数签名:**  
  template \<const int WMMA\_M \= 16, const int WMMA\_N \= 16, const int WMMA\_K \= 16,  
            const int WMMA\_TILE\_M \= 4, const int WMMA\_TILE\_N \= 2,  
            const int WARP\_TILE\_M \= 2, const int WARP\_TILE\_N \= 4,  
            const int OFFSET \= 0\> // OFFSET for padding  
  \_\_global\_\_ void  
  hgemm\_wmma\_m16n16k16\_mma4x2\_warp2x4\_dbuf\_async\_kernel(half \*a, half \*b, half \*c,  
                                                        int m, int n, int k)

* **主要思想:**  
  * **双缓冲 (Double Buffering):** 使用两份共享内存 (s\_a\[2\], s\_b\[2\]) 来存储 A 和 B 的子块。  
  * **异步拷贝 (Asynchronous Copy):** 利用 cp.async 指令（通过 PTX 内联汇编实现）在计算当前 K 步的同时，**异步地**将下一个 K 步所需的数据从全局内存预取 (prefetch) 到另一份共享内存缓冲区中。  
* **优化点分析:**  
  * **隐藏内存延迟 (Latency Hiding):**  
    * **流水线操作:** 这是核心优化点。当 Tensor Core 正在使用 smem\_sel 指向的共享内存缓冲区进行计算时，cp.async 指令会启动数据传输，将下一个 K 步的数据加载到 smem\_sel\_next 指向的缓冲区。计算和数据传输并行进行，从而隐藏了大部分全局内存访问的延迟。  
    * CP\_ASYNC\_CG(dst, src, bytes): 发起一个异步拷贝命令，将 bytes 数据从全局内存 (src) 拷贝到共享内存 (dst)。cg (Copy Group) 可能表示拷贝组。L2::128B 可能暗示了拷贝经过 L2 缓存且以 128 字节为单位操作。  
    * CP\_ASYNC\_COMMIT\_GROUP(): 提交之前发起的 cp.async.cg 命令组，让硬件开始执行拷贝。  
    * CP\_ASYNC\_WAIT\_GROUP(0) / CP\_ASYNC\_WAIT\_ALL(): 等待之前提交的异步拷贝操作完成。这里等待是为了确保下一轮计算开始前，所需的数据已经在共享内存中就绪。  
  * **双缓冲管理:**  
    * smem\_sel \= (it \- 1\) & 1;: 选择当前计算使用的共享内存缓冲区 (0 或 1)。  
    * smem\_sel\_next \= it & 1;: 选择用于预取下一个 K 步数据的共享内存缓冲区。  
  * **共享内存填充 (Padding):**  
    * \_\_shared\_\_ half s\_a\[2\]\[BM\]\[BK \+ OFFSET\], s\_b\[2\]\[BK\]\[BN \+ OFFSET\];  
    * OFFSET 参数用于在共享内存数组的 K 维度（对 s\_a）或 N 维度（对 s\_b）末尾添加额外的空间。这是一种常见的避免**共享内存银行冲突 (Bank Conflict)** 的技术。通过错开访问地址，可以使得 warp 内的线程访问共享内存的不同 bank，从而实现并行访问，提高共享内存带宽。OFFSET=0 表示未使用填充。  
  * **地址转换:** \_\_cvta\_generic\_to\_shared() 用于将通用地址（可能指向全局内存或共享内存）转换为共享内存地址，这是 cp.async 指令需要的。  
* **改进:** 这是最高级的优化版本。通过异步拷贝和双缓冲，最大限度地隐藏了全局内存访问延迟，使得 SM 能够更专注于执行 WMMA 计算，有望达到接近硬件理论峰值的性能。

**总结:**

该文件展示了优化 CUDA WMMA HGEMM 的典型演进过程：

1. **基础 WMMA:** 直接使用 API，性能受限于全局内存。  
2. **共享内存 Tiling:** 利用共享内存缓存数据，减少全局内存访问，提高数据复用。  
3. **Warp 级 Tiling / 寄存器 Blocking:** 增加 Block/Warp 处理的数据量，提高寄存器和共享内存的数据复用，增加计算密度和 ILP。  
4. **双缓冲与异步拷贝:** 通过计算与访存的流水线并行，隐藏访存延迟，进一步逼近硬件性能极限。

这些优化技术是高性能 CUDA 编程，尤其是 GEMM 实现中的常用手段。

## CUDA MMA HGEMM 核函数实现

关于这部分内容详见 [mma_note](mma_note.md)

其实现过程如下，假设目标矩阵的大小就是`A (16 x 16), B (8 x 16)`

首先，加载全局内存到共享内存中，其操作如下：

每个线程读取连续的128个字节，也就是8个`half`

## Swizzle 实现

### Bank conflict 分析

**写入全局内存情况**

每个线程读取连续的 128 个 bits，也就是 8 个 `half` 到共享内存中，其中读取和写入的顺序如下：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250508201605.png)


注意，在 NVIDIA GPU 中一个 Cache line 的大小为 128 Bytes，即为 1024 bits，而每个 Cache line 分为多个 banks，但一个线程访问超过 4 bytes（32 bits）的时候，即一个 warp 超过 128 bytes，GPU 不会发出单个事务，因为每个事务的最大内存访问为 128 bytes。因此会该 warp 分成 4 个事务，每个事务包括 8 个线程，且 bank 的宽度为 128 bytess。

在这里, 由于每个线程都是按照顺序访问连续的地址（从全局内存中读取不是，但是从全局内存中写入是），而上图每八个线程刚好处理完 128 bytes 数据（如图黑色框线所示，为一个 bank），因此不会发生 bank 冲突。

**读取全局内存情况**

接下来是读取全局内存到寄存器的时候，按照 APi 声明，读取的顺序如下：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250508203033.png)

如图所示，由于我们的读取不是连续的，因此不能采用上述的理论，此时一个 warp 依然分成四个内存事务，第一个事务一起访问 32 个线程的 R0，第二个事务一起访问 32 个线程的 R1，以此类推。而此时一个 bank 大小依然是 128 bytes，因此可以看到线程 16 和线程 0 发生了冲突，四处事务，因此可以等效为 4 路冲突。

加上矩阵 B，该算法总共发生了 8 路的 bank conflict。

### 解决方案

### 填充法

最简单的想法就是使用 padding，在将贡献内存大小改为 `half[16][16+8]`, 使得读取的数据可以交错开来

### 地址重排法

另一个想法就是将共享内存的地址进行重排序，如图所示：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250508225531.png)


如图所示，左边是全局内存，右边是共享内存，thread 0-7 按照原来的顺序将数据从全局内存读取到共享内存，但是 thread 8-15，按照原来的顺序读取内存，但是交换写入的位置；然后 thread 16-23 按照原来的顺序，thread 24-31 交换位置，以此类推。可以按照上述方法分析，这种 store 共享内存的方式依然是没有 bank 冲突的。

这最终导致共享内存按照四行四行交替排序，如图所示：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250508225659.png)

然后，我们再从 Shared Mem 读取到寄存器的时候，由于共享内存的这种交替排序，为了保证寄存器内的矩阵排序正确，我们读取的时候也得交替，因此产生形如这样的读取方式：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250508230136.png)
![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250508230147.png)

前 16 个线程的 R0 读取的是左上 32 个数据，后 16 个线程的 R0 读取的上半部分右下的 32 个数据，这种交错的方式，使得 bank conflict 不存在了。

代码示例：

```c++
// i: row index; j: col index.
// e.g kColStride = 16, kStep = 8 -> load 8 half as 128 bits memory issue.
template <const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
  // for col_stride > 16, we have to permute it using col major ZigZag order.
  // e.g, A smem logical layout [Br,d]=[Br,64] -> store layout [4][Br][16].
  static_assert(kColStride <= 16, "kColStride must <= 16");
  // swizzle: ((int(j / kStep) ^ int(i / 4)) % int(kColStride / kStep)) * kStep;
  static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
  static_assert(kColStride % kStep == 0,
                "kColStride must be multiple of kStep.");
  if constexpr (kStep == 8) {
    return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
  } else {
    static_assert(kStep == 4);
    return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
  }
}
```

有两种方法，分别是针对地址和针对偏移量的，上面的代码是针对偏移量的。

假设我们现在读取 `16 x 16`, 而一个线程读取一行 8 个数据，因此行坐标的可取值为：`0 - 15`, 即 `0000 - 1111`。而对于列坐标，可取值只有两个 `0和8` `0000 - 1000`。上述 `kColStride = 16`，因此 `kColStride >> 3 = 2` 。

当 `0-3` 行时，`i>>2 = 0`, `(j >> 3) ^ 0 = (j >> 3)`. 由于 `j = 0000 or 1000`, 因此 `j >> 3` 要不是 0，要不是 1. 取模后不变，然后再移位回去，因此此时的 col 坐标不变。

当 `4-7` 行时，`i>>2=1`, `(j >> 3) ^ 1` 相当于如果是 1 则变为 0，否则变为 1. 通过这种方式，`4-7` 行读取列交换顺序。

当 `8-11` 行时，`i>>2=10`, `(0 or 1) ^ (10) = 10 or 11`, 再与 2 取模，得到 `(0 or 1)`，不变。

当 `12-15` 时，`i>>2=11` , `(0 or 1) ^ (11) = 11 or 10`, 再与 2 取模，得到 `(1 or 0)`，交换。






