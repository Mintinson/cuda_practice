# Shared Memory 和 Bank Conflict

在CUDA编程中，共享内存（Shared Memory）的Bank Conflict是性能瓶颈的常见原因。其核心机制与共享内存的硬件设计密切相关，

共享内存被划分为32个Bank，每个Bank宽度为4字节（或8字节，依GPU架构而定）。当线程访问地址时，地址对Bank数量的取模结果决定其映射的Bank：

* **地址计算公式**：Bank_ID = Address % 32
* **冲突条件**：同一Warp内的多个线程访问同一Bank的不同地址时，触发串行化访问。

* 当多个 thread 访问同一个 bank 内的同一个 word，就会触发 broadcast 机制。这个 word 会同时发给对应的 thread；
* 当多个 thread 访问同一个 bank 内的不同 word 时，就会产生 bank conflict。于是请求会被拆分成多次 memory transaction，串行地被发射（issue）出去执行。（比如 2-way bank conflict，就拆分成 2 次 transaction）


## 64 位宽的访存指令

使用 LDS.64 指令（或者通过 float2、uint2 等类型）取数据时，每个 thread 请求 64 bits（即 8 bytes）数据，那么每 16 个 thread 就需要请求 128 bytes 的数据。

所以 CUDA 会默认将一个 warp 拆分为两个 half warp，每个 half warp 产生一次 memory transaction。即一共两次 transaction。

**_只有以下两个条件之一满足时，这两个 half warp 的访问才会合并成一次 memory transaction_**：

- 对于 Warp 内所有活跃的第 i 号线程，第 i xor 1 号线程不活跃或者访存地址和其一致；(`i.e. T0==T1, T2==T3, T4==T5, T6==T7, T8 == T9, ......`, `T30 == T31, etc.`)
- 对于 Warp 内所有活跃的第 i 号线程，第 i xor 2 号线程不活跃或者访存地址和其一致；(`i.e. T0==T2, T1==T3, T4==T6, T5==T7 etc.`)

简单理解一下，当上面两种情况发生时，硬件就可以判断（具体是硬件还是编译器的功劳，我也不确定，先归给硬件吧），单个 half warp 内，最多需要 64 bytes 的数据，那么两个 half warp 就可以合并起来，通过一次 memory transaction，拿回 128 bytes 的数据。然后线程之间怎么分都可以（broadcast 机制）。

### 例子

#### Case 1

每个线程依次访问连续的 uint2。即第 tid 个线程，访问第 tid 个 uint2。

这时，并没有触发合并的条件，每个 half warp 分别执行一次 memory transaction，**_一共两次_**。也没有产生 bank conflict。

看下第一个 half warp 访问的数据位置：（第一行中的 32 个 word，黄色部分）

![](https://picx.zhimg.com/v2-ad56619bdac19b3a749aecd50b6b5985_1440w.jpg)

上半部分的 Bank 表示数据排列，每个格子表示 1 个 word；下半部分表示线程排列；

第二个 half warp 则是访问第二行的 32 个 word。

  
**_注意！！_**

**_其实 bank conflict 是针对单次 memory transaction 而言的。如果单次 memory transaction 需要访问的 128 bytes 中有多个 word 属于同一个 bank，就产生了 bank conflict，从而需要拆分为多次 transaction_**。

比如这里，第一次访问了 0 - 31 个 word，第二次访问了 32 - 63 个 word，每次 transaction 内部并没有 bank conflict。

```c++
__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid];
}
```

Nsight Compute 计算结果：

![](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250501094343.png)

可以看到并没有发生Conflict。

#### Case 2

![](https://pic3.zhimg.com/v2-1bac8269b3bcb98bb19d3752c252a4ca_1440w.jpg)

这个模式就是符合了合并条件中的第一条。

所以两个 half warp 的访问合并，一共只有 1 次 memory transaction，没有 bank conflict。

```c++
__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid / 2];
}
```

## 128 位宽的访存指令

使用 LDS.128 指令（或者通过 `float4、uint4` 等类型）取数据时，每个 thread 请求 128 bits（即 16 bytes）数据，那么每 8 个 thread 就需要请求 128 bytes 的数据。

所以，CUDA 会默认把每个 half warp 进一步切分成两个 quarter warp，每个包含 8 个 thread。每个 quarter warp 产生一次 memory transaction。所以每个 warp 每次请求，默认会有 4 次 memory transaction。（没有 bank conflict 的情况下）。

类似 64 位宽的情况，当满足特定条件时，一个 half warp 内的两个 quarter warp 的访存请求会合并为 1 次 memory transaction。但是两个 half warp 不会再进一步合并了。（划重点！！！）

具体条件和 64 位宽一样：

* 对于 Warp 内所有活跃的第 i 号线程，第 i xor 1 号线程不活跃或者访存地址和其一致；(i.e. T0==T1, T2==T3, T4==T5, T6==T7, T8 == T9, ......, T30 == T31, etc.)
* 对于 Warp 内所有活跃的第 i 号线程，第 i xor 2 号线程不活跃或者访存地址和其一致；(i.e. T0==T2, T1==T3, T4==T6, T5==T7 etc.)

### Case 1

![](https://picx.zhimg.com/v2-f984fdca5f8e4e6ab36cff63972c7c55_1440w.jpg)



这时只有两个 quarter-warp 活跃，分别需要一次 memory transaction，一共 2 次。没有 bank conflict。

注：这两个 quarter-warp 属于两个不同的 half warp，所以不会合并访问。

```c++
__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 15 || tid == 16) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}
```

### Case 2

只激活第 0 和第 15 号线程，访问第 4 个 uint4：

![](https://pic4.zhimg.com/v2-62072d3cfb73ebd4370e6900429a8e25_1440w.jpg)

这是满足合并条件第一条，所以前两个 quarter warp 的访存请求合并成 1 次 memory transaction。没有 bank conflict。

```cpp
__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 0 || tid == 15) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}
```

### Case 3

![](https://pica.zhimg.com/v2-0686017ed02d7524ee9a661f574ac300_1440w.jpg)

满足合并条件第一条，前两个 quarter warp 和后两个 quarter warp 分别合并，分别需要 1 个 memory transaction（即每个 half warp 需要 1 个 transaction）。一共 2 个 transaction，没有 bank conflict。

```c++
__global__ void smem_3(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(
      smem)[(tid / 8) * 2 + ((tid % 8) / 2) % 2];
}
```

### Case 4

![](https://pic3.zhimg.com/v2-72bf3ddc2e1057e6e7b5c12e97615104_1440w.jpg)

这个排布有点意思，第一个 half warp 满足合并条件 1，第二个half warp 满足合并条件 2。但是需要整个 warp 都满足条件 1，或者条件 2，或者 1、2 同时满足，这样才可以合并。

所以这里仍然是每个 quarter warp 需要 1 次 memory transaction，一共 4 次。没有 bank conflict。

```c++
__global__ void smem_4(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr;
  if (tid < 16) {
    addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
  } else {
    addr = (tid / 8) * 2 + ((tid % 8) % 2);
  }
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
  // printf("tid: %d, addr: %d\n", tid, addr);
}
```

### Case 5

![](https://pic3.zhimg.com/v2-b00fd74abc9b9e55e493ea5ca4f588b0_1440w.jpg)

thread 0 - 3 访问第 0 个 uint4， thread 4 - 7 访问第 8 个 uint4（到了第二行）；

thread 8 - 11 访问第 1 个 uint4， thread 12 - 15 访问第 9 个 uint4（到了第二行）；

依次类推；（可以在 kernel 内通过 printf 打印 tid 和 addr）


这里符合合并条件 1，所以前两个和后两个 quarter warp 分别合并。但是每个 half warp 内，产生了 2-way bank conflict，所以需要拆成 2 次 transaction。

即一共 2 个 bank conflict， 4 次 transaction。

```c++
__global__ void smem_5(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = (tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8;
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
  printf("tid: %d, addr: %d\n", tid, addr);
}
```

![](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250501101408.png)

---

参考链接

* [搞懂 CUDA Shared Memory 上的 bank conflicts 和向量化指令（LDS.128 / float4）的访存特点 - 知乎](https://zhuanlan.zhihu.com/p/690052715)