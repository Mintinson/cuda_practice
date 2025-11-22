# GPU 的存储体系

## 一般解释

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250221155745.png)

| 内存类型            | 物理位置 | 访问权限（相对 GPU） | 可见范围      | 生命周期      |
|-----------------|------|--------------|-----------|-----------|
| 全局内存            | 芯片外  | 可读可写         | 所有线程和主机端  | 由主机分配与释放  |
| 常量内存            | 芯片外  | 可读           | 所有线程和主机端  | 由主机分配与释放  |
| texture memory  | 芯片外  | 一般可读         | 所有线程和主机端  | 由主机分配与释放  |
| register memory | 芯片内  | 可读可写（最快）     | 单个线程      | 所在线程      |
| local memory    | 芯片外  | 可读可写         | 单个线程      | 所在线程      |
| shared memory   | 芯片外  | 可读可写         | 单个线程**块** | 所在线程**块** |

可以在 [storage_time.cu](storage_time.cu) 中找到测试代码。

## 寄存器

* 寄存器内存是 on-chip 的，具有 GPU 上最快的访问速度，但是数量有限，属于 GPU 的稀缺资源。
* 寄存器仅可在线程内可见，生命周期也与所属线程一致。
* 核函数中定义的不加任何限定符的变量一般存放在寄存器中。（包括内建变量如 `gridDim` 和 `blockDim` 和 `blockIdx` 等）
* 核函数中定义的不加任何限定符的数组有可能存在于寄存器中，但也有可能存在于本地内存中；

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250221161146.png)

GPU 上的寄存器都是 32 位的，因此保存一个 double 类型数据需要两个寄存器，register 保存在 SM 的 register filer 中。

注意：每个线程的最大寄存器数量是 255 个，而 Fermi 架构是 63 个。

## 局部存储

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250221161414.png)

* 每个线程最多可使用 512 KB 的本地内存
* 本地内存从硬件角度看只是全局内存的一部分，延迟很高，因此如果过多使用本地内存，会降低程序的性能。
* 核函数所需的寄存器数量超出硬件设备支持，数据则会保存到本地内存 （local memory）中：
    * 一个 SM 运行并行运行多个线程块/线程束，总的需求寄存器容量大于64KB；
    * 单个线程运行所需寄存器数量个255个；
* 核函数中动态分配的数组保存在本地内存中，可能占用大量寄存器空间的较大本地结构体和数组也会保存在本地内存中；以及任何不满足核函数寄存器限定条件的变量也会保存在本地内存上。

另外，数组是否会保存到本地内存不仅仅取决于数组的大小，还取决于数组需要参与计算的复杂程度，比如一个数组如果要参与排序算法，则即使很小的数组，也会保存在局部内存中。

## 共享内存

* 片上，速度仅次于寄存器，且有限
* 在该线程块中所有线程可见，生命周期与线程块相同
* 使用 `__shared__` 修饰的变量存放于共享内存中，共享内存可定义*动态*与*静态*两种。
* 每个 SM 的共享内存数量是一定的，也就是说，如果在单个线程块中分配过渡的共享内存，将会限制活跃线程束的数量。
* 由于共享内存可以被多线程访问，因此需要 `__syncthreads()` 来显式同步。

![image.png](https://auto-imgs-1323334286.cos.ap-guangzhou.myqcloud.com/obsidian/20250221165319.png)

一般来说，对于经常访问的大数据，可以由全局内存移到共享内存中，提高访问的效率。能够改变全局内存访问内存的内存事务方式，提高数据访问的带宽。

**静态共享内存**：静态共享内存由 `__shared__`，定义，如果在核函数中声明，静态共享内存作用域局限在这个线程块中。在文件核函数外声明，
静态共享内存作用域对所有核函数有效。

另外，在 cuda 中，下列语句是有效的：

```c++
__shared__ float arr[2, 5];
```

即多维数组。

**静态共享内存在编译时就要确定内存大小**。

**动态共享内存**：动态共享内存定义如下:

```c++
extern __shared__ float arr[];
```

注意，不能加长度，也不能简单声明为指针，这是错误的；

其长度在创建 grid 的时候确定，即此时调用核函数有第三个参数：

```c++
kernel_function<<<grid_size, block_size, dynamic_size >>>();
```

即 `dynamic_size`（注意是以 bytes 为单位）

### bank conflict

共享存储由交替排列的bank构成，每个bank尺寸为32b，如果一个warp中多个线程同时访问同一个bank的不同地址，就会引发bank
conflict，导致 warp
中的命令是串行的。

见 [reduce/解决 bank 冲突](../reduce_demo/note.md#优化4-解决-bank-冲突)

在 [mat_transpose/优化3](../mat_transpose_demo/note.md#优化3-避免-bank-conflict) 也是一个例子。

### `volatile` 关键字

连续访问没有同步语句得到的数据可能是先前访问时缓存在寄存器中的数据，而不是共享存储获取最新更新的数据。

为了强行读取新值，从而避免由于缓存数据引发的错误，我们可以使用`volatile` 关键字来修饰：

```c++
__shared__ DataType tmp;

...

volatile auto* tmp_1 = tmp
```

## 常量内存

* 常量内存是有常量缓存的全局内存，数量有限，仅为 64 KB，但是由于有缓存（编译器知道设备不用写入该数据），因此访问速度比全局内存快；
* 常量内存中的数据对同一编译单元内所有线程可见
* 使用 `__constant__` 修饰的变量存放于常量内存中，其只能在函数外定义，不能再核函数或者主机函数中定义，且一定是静态的
* 常量内存可读不可写
* **给核函数传递数值参数时，这个变量就存放于常量内存**。

如果主机想要初始化或者读取常量内存，则需要：

写入：

```c++
_host__ cudaError_t  cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
```

其中 count 是字节数

读取：

```c++
__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
```

线程束中所有线程从相同内存地址中读取数据时，常量内存表现最好，例如数学公式中的系数，因为线程束中所有的线程都需要读取同一个地址空间的系数数据，
因此只需要读取一次，广播给线程束中的所有线程。

另外， 还有 `cudaGetSymbolAddress` 函数获取常量存储的地址，然后可以在 kernel 函数中修改常量存储的值（不推荐）

## 全局内存

* 全局内存在片外，容量最大，延迟最高，使用最多
* 全局内存中的数据所有线程可见，且可以直接访问；host 端可见，但是静态不能直接访问。
* 全局内存具有与程序相同的生命周期

全局内存又分为动态全局内存和静态全局内存：

* 动态全局内存，即用 `cudaMalloc` 等运行时 API 创建的
* 静态全局内存：使用 `__device__` 关键字声明的变量，静态全局变量必须在核函数或者主函数**外部**进行定义。且主机函数不能直接访问静态全局变量。

另外，`kernel` 中 用 `malloc` `free` 分配的内存也是全局内存。

主机函数可以用 `cudaMemcpyToSymbol` or `cudaMemcpyFromSymbol` 进行访问。

### 全局内存的合并优化

见 [coalesced_demo](../coalesced_demo/note.md)

### 利用纹理缓存通道访问全局内存

在不绑定全局存储到纹理的前提下，cuda 提供了两种纹理缓存通道访问全局存储的途径：

1. 使用 关键字 `const __restrict__` 修饰只读的全局存储
2. 利用 `__ldg()` 内置函数

例子见 [texture_demo](texture_demo.cu)

## 纹理内存

纹理内存是 CPU 重要特征之一，也是 GPU 编程优化的关键。

### CUDA 数组

CUDA 数组专为 纹理操作设计，位于显存池，且不能用指针访问，只能通过数组句柄 和 1D，2D或者3D坐标访问。

使用 `cudaMallocArray()` 和 `cudaFreeArray()` 函数创建和销毁数组

CUDA 数组无法直接访问，必须绑定到纹理才能读数据，利用 `cudaMemcpyToArray()` `cudaMemcpyFromArray()`
`cudaMemcpy2DToArray()` `cudaMemcpy2DFromArray()` 等函数传播数据，若要通过 kernel函数修改数据则需要绑定到 **表面存储**

### 纹理内存的操作和限制

纹理内存分为 1D 纹理，2D 纹理，3D 纹理，1D 分层纹理和 2D 分层纹理。（其实还有 ` Cubemap Textures`

见[【CUDA编程】纹理内存和纹理提取 - 知乎](https://zhuanlan.zhihu.com/p/688899761)

## 主机端内存

cuda 将主机端内存分成两种，分别是可分页内存 (pageable memory) 和页锁定内存 (pinned memory)

其中可分页内存即一般的主机内存。可分页指的是内存页可以被换出到磁盘。可分页内存无法使用 DMA。

页锁定内存不会被换出到磁盘，因此支持 DMA 访问，支持与 GPU 的异步通信。一般情况下，叶锁定内存传输效率是可分页内存的两倍左右。

而页锁定内存可以由两种方式获取：
* `cudaHostAlloc()` 和 `cudaFreeHost()` 直接分配和释放，该内存一直位于内存空间中。
* `cudaHostRegister() cudaHostUnregister()` 将可分页内存申请为页锁定内存。 

注意 `cudaHostAlloc()`分配开销是很大的，因此，使用 页锁定内存的最佳实践是用 `cudaHostRegister` 来将可分页内存申请为页锁定内存。

## 零拷贝操作

对于页锁定内存，可在 `cudaHostAlloc` 和 `cudaHostRegister` 的 flag 参数中传入 `cudaHostAllocMapped` 参数。

此时可以通过 `cudaHostGetDevicePointer()` 函数获得设备端指向主机端内存空间的指针，此时设备可以直接访问位于主机的页锁定内存，这被称为零拷贝操作。

```c++
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA内核函数：直接访问主机内存
__global__ void kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] *= 2.0f; // 将主机内存中的数据翻倍
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);

    // 1. 分配可分页主机内存
    float* h_data = (float*)malloc(size);
    if (!h_data) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // 2. 注册主机内存为固定内存（Pinned Memory）并映射到GPU地址空间
    cudaError_t err = cudaHostRegister(h_data, size, cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        printf("cudaHostRegister failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }

    // 3. 获取与主机内存关联的设备指针
    float* d_data;
    err = cudaHostGetDevicePointer(&d_data, h_data, 0);
    if (err != cudaSuccess) {
        printf("cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
        cudaHostUnregister(h_data);
        free(h_data);
        return 1;
    }

    // 4. 初始化主机内存数据
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // 5. 启动内核（直接使用设备指针访问主机内存）
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data);

    // 6. 同步设备并检查错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    } else {
        // 7. 验证结果（直接读取主机内存）
        printf("First 10 elements after kernel:\n");
        for (int i = 0; i < 10; i++) {
            printf("%.1f ", h_data[i]);
        }
        printf("\n");
    }

    // 8. 清理资源
    cudaHostUnregister(h_data);
    free(h_data);

    return 0;
}
```