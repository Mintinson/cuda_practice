# 一些 API 的解释

以下是对 mma 部分内容的处理解释：

### 函数声明部分

```c++
__device__ size_t __cvta_generic_to_shared(const void *ptr);：
```

*Returns the result of executing the PTX cvta. To. Shared instruction on the generic address denoted by ptr*.

该函数会执行 PTX（Parallel Thread Execution）指令集中的 `cvta.to.shared` 指令。`cvta.to.shared` 指令的作用是将通用地址（由 `ptr` 参数所表示）转换为共享地址，返回转换后的结果。

```c++
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)
                : "r"(addr));
```

​​* `.sync​​` 表示指令需要等待所有线程同步完成才能执行，确保 warp 内32个线程在数据加载时的协同性。
​​* `.aligned`​​要求共享内存地址必须满足128位对齐（16字节），否则会产生未定义行为。在代码中通过`__cvta_generic_to_shared`转换保证地址对齐。
​​* `.m8n8`​​定义加载的矩阵块大小为8×8，每个线程负责加载该矩阵的一个子块。结合.x4修饰符，实际会加载4个连续的8×8矩阵块。
​​* `.x4​`​表示重复执行4次加载操作，总共加载4×8×8=256个元素。每个线程最终会获得4个32位寄存器存储数据。
​​ `.shared.b16​​` 指定数据源为共享内存（shared memory），且元素为16位数据类型（如 half/bfloat16）。每个128位加载操作对应8个16位元素。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250506110755.png)

根据 [1. Introduction — PTX ISA 8.8 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ldmatrix%2520sync#warp-level-matrix-instructions-ldmatrix) 显式，一个 warp 中16 bits 数据的 8 x 8 矩阵的 load 格式如图所示，其中线程 0 读取 `[0,0],[0,1]`, 线程 2 读取 `[0,2],[0,3]`, 以此类推。

但是这里乘了 4，结果就不太一样：
![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250506111323.png)

可以看到是 `R1, R2, R3, R4` 是按列排布的，这是 NVIDIA 规定的。

```c++
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(R0), "=r"(R1)
                : "r"(addr));
```

同理，只不过这里只用了 `x2`，且做了转置：

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250506112212.png)

为什么要转置？这里要看接下来的乘法命令 $D = AB + C$：

```c++
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "
                "%4, %5}, {%6, %7}, {%8, %9};\n"
                : "=r"(RD0), "=r"(RD1)
                : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),
                  "r"(RC1));
```

* `.sync​​` 表示指令需要等待所有线程同步完成才能执行，确保 warp 内 32 个线程在数据加载时的协同性。
* `.aligned` ​​要求共享内存地址必须满足 128 位对齐（16 字节），否则会产生未定义行为。在代码中通过 `__cvta_generic_to_shared` 转换保证地址对齐。
*  `.m16n8k16` 定义矩阵乘法的形状
* `row` 矩阵 A 为行主序
* `col` 矩阵 B 为列主序
* `f16.f16.f16.f16`：D, A, B, C 都为 `f16` 类型

正是因为 B 为列主序，因此我们在加载 B 的时候，要转置一下。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250506113113.png)

上图展示了一个 warp 中，不同线程持有的 A，B，C/D 碎片布局。

根据 C 的布局，我们也就能理解下列代码了：

```c++
			// 将寄存器中的数据按照32bits读取（如上图中的[c0,c1]）
            Ldst_32_bits(s_c[lane_id / 4][(lane_id % 4) * 2]) = Ldst_32_bits(RC[0]);
            // 每个线程读取两组，并放置到正确的位置，注意这里`s_c`的索引计算方式
            Ldst_32_bits(s_c[lane_id / 4 + 8][(lane_id % 4) * 2]) = Ldst_32_bits(RC[1]);
```

```c++
			// // store s_c[16][8]
            if (lane_id < MMA_M)  // 一个线程读取一行（8 data），因此只用16个线程即可
            {
                // store 128 bits per memory issue.
                int store_gmem_c_m = by * BM + lane_id;
                int store_gmem_c_n = bx * BN;
                int store_gmem_c_addr = store_gmem_c_m * n + store_gmem_c_n;
                Ldst_128_bits(c[store_gmem_c_addr]) = (Ldst_128_bits(s_c[lane_id][0]));
            }
```