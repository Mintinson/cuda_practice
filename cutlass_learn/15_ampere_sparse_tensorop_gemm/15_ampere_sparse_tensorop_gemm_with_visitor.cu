//
// Created by asus on 2025/5/16.
//
/**
Please check example 07, 08 and 17 for the basics of dense tensor op gemm kernels.  NVIDIA Ampere
architecture also supports structured sparse tensor op for tf32, fp16, int8 and int4.
Sparse GEMM kernels needs to takes an additional E matrix which stores the meta data.  The format of
meta data is different for every data types.   CUTLASS templates can automatically infer it based on
input A and B.  Check code below.
Moreover, matrix E needs to be preprocessed so that it can use ldmatrix to load into the registers
efficiently.
*/
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse_with_visitor.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_uncompress.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "../cutlass_helper.cuh"
// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t; // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
using ElementInputA = int8_t; // <- data type of elements in input matrix A
using ElementInputB = int8_t; // <- data type of elements in input matrix B
using ElementOutput = int32_t; // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Row Major for
// Matrix A, Column Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// The number of elements per vectorized memory access.
constexpr int AlignmentInputA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
constexpr int AlignmentInputB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
constexpr int AlignmentComputeEpilogue = 128 / cutlass::sizeof_bits<ElementComputeEpilogue>::value;
constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;
// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
cutlass::gemm::GemmShape<128, 128, 128>; // <- threadblock tile M = 128, N = 128, K = 128
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>; // <- warp tile M = 64, N = 64, K = 128
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>; // <- MMA Op tile M = 16, N = 8, K = 64

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
// 定义具体的 Tensor Core 操作：带饱和的乘加
using Operator = cutlass::arch::OpMultiplyAddSaturate;

// Number of pipelines you want to use
constexpr int NumStages = 3;

// Epilogue（尾声）阶段是 GEMM 计算（C = A*B）完成之后，对累加结果 C 进行进一步处理的阶段。
// Visitor 模式允许我们像搭积木一样组合这些操作。
constexpr auto NumEVTEpilogueStages = 1; // Epilogue Visitor Tree (EVT) 的阶段数

// Accumulator Fetch: 定义如何从累加器（通常在寄存器中）获取数据
using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
// Bias Loading: 定义如何加载 Bias 数据 (通常是另一个向量或矩阵 C)
// 1. BiasTileThreadMap: 定义线程如何协作加载 Bias 瓦片
using BiasTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ShapeMMAThreadBlock, // Threadblock 瓦片形状
    ShapeMMAWarp, // Warp 瓦片形状
    ElementComputeEpilogue, // Bias 数据类型
    AlignmentComputeEpilogue, // Bias 对齐
    NumEVTEpilogueStages
>;
// 2. Bias Visitor: 定义加载 Bias 的具体行为
using Bias = cutlass::epilogue::threadblock::VisitorAuxLoad<
    BiasTileThreadMap,
    ElementComputeEpilogue,
    cute::Stride<int64_t, cute::_1, int64_t>// cute::Stride 定义了 Bias 数据的访问步长，
                                            // 这里表示 Bias 是一个向量 (stride M=N, stride N=1) 或与输出D形状相同的矩阵。
                                            // cute::_1{} 表示沿着N维度是单位步长，即Bias[i] = C[i] or Bias[i] = D[i,j]中的j维度。
>;
// Bias Application: 定义如何应用 Bias (例如 C_final = C_accum + Bias)
using ApplyBias = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementComputeEpilogue,
                                                                 ElementComputeEpilogue,
                                                                 cutlass::FloatRoundStyle::round_to_nearest
>;// 使用 cutlass::plus 进行加法，并指定舍入模式 (虽然这里是整数，但API是通用的)

// EVT (Epilogue Visitor Tree) Node for Bias: 将 Accumulator Fetch, Bias Load, Bias Apply 组合起来
using EVTApplyBias = cutlass::epilogue::threadblock::Sm80EVT<ApplyBias, Accum, Bias>;
// Sm80EVT<ComputeOp, InputA_EVT_Node, InputB_EVT_Node>
// 效果: ApplyBias(Accum_result, Bias_result)

// Output Storing: 定义如何存储最终结果到输出矩阵 D
// 1. OutputTileThreadMap: 定义线程如何协作存储输出瓦片
using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ElementOutput,// 输出数据类型
    AlignmentOutput, // 输出对齐
    NumEVTEpilogueStages
>;
// 2. Output Visitor: 定义存储输出的具体行为
using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,// 舍入模式
    cute::Stride<int64_t, cute::_1, int64_t>// 输出D的步长，同Bias

>;
// EVT Node for Output: 将上一步 Bias 应用的结果作为输入，进行存储
using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<Output, EVTApplyBias>;
// 效果: Output(EVTApplyBias_result)
// 这样就构成了 D = (A*B + Bias) 的融合操作链

// Use element type in EVT with the smallest bitwidth as ElementC.
using ElementC = ElementComputeEpilogue;
using LayoutC = LayoutOutput;

using Gemm = cutlass::gemm::device::SparseGemmWithVisitor<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementC, LayoutC,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ShapeMMAOp,
    EVTOutput,
    SwizzleThreadBlock,
    NumStages,
    AlignmentInputA,
    AlignmentInputB,
    Operator,
    NumEVTEpilogueStages>;

// Data type and layout of meta data matrix E can be inferred from template Gemm.
using ElementInputE = Gemm::GemmKernel::ElementE;
using LayoutInputE = cutlass::layout::RowMajor;
using ReorderedLayoutInputE = Gemm::GemmKernel::LayoutE;

// Blow property is defined in include/cutlass/arch/sp_mma_sm80.h
// 50% Sparsity on Ampere
constexpr int kSparse = Gemm::kSparse;
// How many elements of A are covered per ElementE
constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
// The size of individual meta data
constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;

int run()
{
    const int length_m = 512;
    const int length_n = 512;
    const int length_k = 1024;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse));
    // <- Create matrix A with dimensions M x (K / 2)
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a_uncompressed(
        problem_size.mk()); // <- Create uncompressed matrix A with dimensions M x K for reference computing

    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn()); // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementComputeEpilogue, LayoutOutput> tensor_c(
        problem_size.mn()); // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
    // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
    // reference kernel

    // Create matrix E with dimensions M x (K / 2 / kElementsPerElementE). This one is used by reference computing.
    cutlass::HostTensor<ElementInputE, LayoutInputE> tensor_e(
        cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
    // Same size as the above.  The above one needs to be reordered and stored in this one.
    cutlass::HostTensor<ElementInputE, ReorderedLayoutInputE> tensor_e_reordered(
        cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));

    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(8),
        ElementInputA(-8),
        0); // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(8),
        ElementInputB(-8),
        0); // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        ElementOutput(8),
        ElementOutput(-8),
        0); // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomSparseMeta(
        tensor_e.host_view(),
        1,
        kMetaSizeInBits); // <- Fill matrix E on host with uniform-distribution random meta data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view()); // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view()); // <- fill matrix D for reference on host with zeros

    // Reorder the meta data matrix so that we can use ldmatrix to load them to tensor core
    // instructions.
    cutlass::reorder_meta(tensor_e_reordered.host_ref(), tensor_e.host_ref(),
                          {
                              problem_size.m(), problem_size.n(),
                              problem_size.k() / kSparse / kElementsPerElementE
                          });

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_e_reordered.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);
    // 为 Epilogue Visitor Tree 中的 Bias 加载节点准备参数
    Bias::Arguments bias_arguments{
        tensor_c.device_data(),// Bias 数据的设备指针 (这里复用了 tensor_c)
        ElementComputeEpilogue(0),// 用于条件执行的阈值，这里是0表示无条件
        {problem_size.n(), cute::_1{}, problem_size.mn().product()}// Bias 的步长/布局信息
    };
    // 为 Epilogue Visitor Tree 中的 Output 存储节点准备参数
    Output::Arguments output_arguments{
        tensor_d.device_data(), // 输出矩阵 D 的设备指针
        {problem_size.n(), cute::_1{}, problem_size.mn().product()} // 输出矩阵 D 的步长/布局信息
    };
    // 组合成最终 Epilogue 操作树的参数
    EVTOutput::Arguments callback_arguments{
        {
            {}, // Accum
            bias_arguments, // Bias
            {} // ApplyBias
        },
        output_arguments // Output
    }; // EVTOutput

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
        problem_size, // <- problem size of matrix multiplication
        tensor_a.device_ref(), // <- reference to matrix A on device
        tensor_b.device_ref(), // <- reference to matrix B on device
        tensor_e_reordered.device_ref(), // <- reference to matrix E on device
        callback_arguments
    }; // <- epilogue arguments

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);

    // uncompress tensor_a based on meta data tensor_e. We need it for reference computing.
    cutlass::uncompress(tensor_a_uncompressed.host_ref(), tensor_a.host_ref(),
                        tensor_e.host_ref(), problem_size.m(), problem_size.k());

    // Create instantiation for host reference gemm kernel
    cutlass::reference::host::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue,
                                   typename Gemm::Operator>
        gemm_host;

    // Launch host reference gemm kernel
    gemm_host(problem_size,
              alpha,
              tensor_a_uncompressed.host_ref(),
              tensor_b.host_ref(),
              beta,
              tensor_c.host_ref(),
              tensor_ref_d.host_ref());

    // Copy output data from CUTLASS host for comparison
    tensor_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
        tensor_d.host_view(),
        tensor_ref_d.host_view());

    std::cout << (passed ? "Passed" : "Failed") << std::endl;

    return (passed ? 0 : -1);
}

int main()
{
    bool notSupported = false;

    // Ampere Sparse Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 11.1.
    //
    // CUTLASS must be compiled with CUDA 11.1 Toolkit to run these examples.

    if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 1)))
    {
        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.1 Toolkit or later." << std::endl;
        notSupported = true;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (props.major * 10 + props.minor < 80)
    {
        std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
            << std::endl;
        notSupported = true;
    }

    if (notSupported)
    {
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }

    return run();
}
