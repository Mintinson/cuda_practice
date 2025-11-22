// 标准库头文件
#include <iostream> // 用于标准输入输出流操作

// CUTLASS库头文件包含
#include "cutlass/aligned_buffer.h" // 对齐内存缓冲区实现
#include "cutlass/gemm/gemm.h"      // GEMM（通用矩阵乘法）核心功能
#include "cutlass/layout/matrix.h"  // 矩阵内存布局定义
#include "cutlass/matrix_shape.h"   // 矩阵形状描述
#include "cutlass/numeric_types.h"  // 数值类型定义（如half_t）

#include "cutlass/core_io.h"             // 核心I/O操作
#include "cutlass/util/host_tensor.h"    // 主机端张量实现
#include "cutlass/util/tensor_view_io.h" // 张量视图I/O操作

#include "cutlass/util/reference/host/gemm.h"           // 主机端参考GEMM实现
#include "cutlass/util/reference/host/tensor_compare.h" // 张量比较操作
#include "cutlass/util/reference/host/tensor_fill.h"    // 张量填充操作

#include "cutlass/transform/pitch_linear_thread_map.h"                     // 线性间距线程映射
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"        // 带谓词的块迭代器
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h" // 规则张量操作迭代器

#include "cutlass/util/debug.h"       // 调试工具
#include "cutlass/util/device_dump.h" // 设备内存dump工具

// 定义示例矩阵维度
#define EXAMPLE_MATRIX_ROW 64 // 矩阵行数
#define EXAMPLE_MATRIX_COL 32 // 矩阵列数

// CUDA内核函数模板
template <typename Element, typename GmemIterator, typename SmemIterator>
__global__ void kernel_dump(
    typename GmemIterator::Params params, // 全局内存迭代器参数
    typename GmemIterator::TensorRef ref) // 全局内存张量引用
{
    // 声明动态共享内存，用于线程块内的数据共享
    extern __shared__ Element shared_storage[];

    // 计算线程块内的线性线程ID（将二维线程索引转换为一维）
    int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    // 初始化全局内存迭代器，指定参数、数据指针、矩阵维度和线程ID
    GmemIterator gmem_iterator(
        params,
        ref.data(),
        {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL},
        tb_thread_id);

    // 声明片段（Fragment）用于存储加载的数据
    typename GmemIterator::Fragment frag;
    frag.clear();             // 初始化片段
    gmem_iterator.load(frag); // 从全局内存加载数据到片段

    // 系列调试输出操作（仅在第一个线程块的第一个线程执行）
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nAll threads dump all the elements:\n");
    cutlass::debug::dump_fragment(frag); // 所有线程转储所有元素

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nFirst thread dumps all the elements:\n");
    cutlass::debug::dump_fragment(frag, /*N = */ 1); // 仅第一个线程转储所有元素

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nFirst thread dumps first 16 elements:\n");
    cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16); // 转储前16元素

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nFirst thread dumps first 16 elements with a stride of 8:\n");
    cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16, /*S = */ 8); // 步长8转储

    // 初始化共享内存迭代器，指定共享内存布局和线程ID
    SmemIterator smem_iterator(
        typename SmemIterator::TensorRef(
            {shared_storage,
             SmemIterator::Layout::packed({EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL})}),
        tb_thread_id);

    smem_iterator.store(frag); // 将片段数据存储到共享内存

    // 共享内存调试输出
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nDump all the elements:\n");
    cutlass::debug::dump_shmem(shared_storage,
                               EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL); // 转储全部共享内存

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\nDump all the elements with a stride of 8:\n");
    cutlass::debug::dump_shmem(
        shared_storage,
        EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL,
        /*S = */ 8); // 带步长的共享内存转储
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 主函数：dump_reg_shmem示例程序的入口
int main()
{
    // 初始化一个64x32的列主序矩阵，使用half精度类型
    using Element = cutlass::half_t;             // 使用半精度浮点类型
    using Layout = cutlass::layout::ColumnMajor; // 列主序内存布局

    // 创建主机端张量（矩阵）
    cutlass::HostTensor<Element, Layout> matrix({EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL});

    // 使用顺序值填充主机端矩阵（1,2,3...）
    cutlass::reference::host::BlockFillSequential(matrix.host_data(), matrix.capacity());

    // 打印矩阵内容
    std::cout << "Matrix:\n"
              << matrix.host_view() << "\n";

    // 同步数据到设备端（GPU显存）
    matrix.sync_device();

    // 定义线程映射和迭代器类型 --------------------------------------------------------

    // 使用线性间距线程映射，配置线程组织结构
    using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
        cutlass::layout::PitchLinearShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>,
        32,                                      // 线程数
        cutlass::layout::PitchLinearShape<8, 4>, // 线程块形状
        8>;                                      // 每个线程块处理的元素数

    // 全局内存迭代器定义（带谓词的块迭代器）
    using GmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, // 矩阵形状
        Element,                                                      // 元素类型
        Layout,                                                       // 内存布局
        1,                                                            // 每个迭代的向量数量
        ThreadMap>;                                                   // 使用的线程映射

    // 初始化全局内存迭代器参数（基于矩阵布局）
    typename GmemIterator::Params params(matrix.layout());

    // 共享内存迭代器定义（规则张量操作迭代器）
    using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, // 矩阵形状
        Element,                                                      // 元素类型
        // 16: 代表瓦片沿着列主序矩阵的连续内存维度（即 M 维度 / 行数）的大小。
        // 64: 代表瓦片沿着列主序矩阵的跨步内存维度（即 K 维度 / 列数）的大小。
        cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>, // 张量核心优化布局 (ElementSize,  Crosswise)
        1,                                                                 // 每个迭代的向量数量
        ThreadMap>;

    // 配置CUDA内核执行参数 -----------------------------------------------
    dim3 grid(1, 1);      // 网格维度（1x1 个线程块）
    dim3 block(32, 1, 1); // 线程块维度（32x1x1 个线程）

    // 计算共享内存需求（矩阵元素总数 × 元素大小）
    int smem_size = int(sizeof(Element) * EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL);

    // 启动内核函数 ------------------------------------------------------
    kernel_dump<Element, GmemIterator, SmemIterator>
        <<<grid, block, smem_size, 0>>>(params, matrix.device_ref());

    // 同步设备并检查结果 -------------------------------------------------
    cudaError_t result = cudaDeviceSynchronize(); // 等待内核执行完成

    if (result != cudaSuccess)
    {
        std::cout << "Failed" << std::endl; // 错误处理
    }

    return (result == cudaSuccess ? 0 : -1); // 返回执行状态
}