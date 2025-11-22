/**
This example shows how to run matrix multiplication kernels using functions and data structures
provided by CUTLASS using tensor cores; which we run on a NVIDIA Turing GPU.

Writing a single high performance matrix multiplication kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of gemm kernel. When used properly, the kernels can hit peak performance of GPU
easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set matrices will be used to compute
output of matrix multiplication.

First, we setup the data types of matrices A, B, C and D along with alpha, beta as the equation for
GEMM is D = alpha * A * B + beta * C. In CUTLASS, the kernels first compute A * B and leaves the
rest of the computation to end of the kernel as alpha * X + beta * C is a simple element-wise
operation on X (A * B) and C. We call this as epilogue of kernel. Hence, we setup data types for
alpha and beta to be equal to ElementComputeEpilogue = int32_t. As we want to use MMA instructions
on Turing and they support 8-bit signed integer (int8_t), we use data type for elements in input
matrix A and B as int8_t. Volta also supports accumulation of partial dot product to int32_t, which
can store wider range of numbers, we use it as data type of output matrix elements and accumulation.
We convey this to CUTLASS kernel by initializing template variables ElementAccumulator (int32_t),
ElementComputeEpilogue (int32_t), ElementInputA (int8_t), ElementInputB (int8_t), ElementOutput
(int32_t). Communicating just the data type is not enough. As the data is laid out linearly in
memory, we have to convey the layout of matrices. We do that by initializing template variable
LayoutInputA to column major cutlass variable, LayoutInputB to row major and LayoutOutput to row
major. Next, we setup rules to comptue alpha * X + beta * C which is called epilogue of the kernel.
We initialize template variable EpilogueOp, which takes the data type of output ElementOutput
(int32_t), the number of elements per vector memory access (16), data type of accumulator (int32_t)
and data type of computation of linear combination (alpha * X + beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x256x64,
64x64x16, 8x8x16 (MxNxK) respectively. When passed to instantiate CUTLASS GEMM kernel, it internally
deduce the amount of threads needed per thread-block, amount of shared memory, storing data in
bank-conflict free manner, and ton of other variables required to compose, initialize and launch a
high performance GEMM kernel. This is the beauty of CUTLASS, it relieves developer from
understanding and coding complicated hardware optimizations which can easily go wrong.

CUTLASS also supports multiple MMA pipelines in a threadblock. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma pipeline.

matrix in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
output to global memory

The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
until the previous finished executing. There are stages in the pipeline which do not have fixed
latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
Finally, the pipeline in a kernel looks like

(1) matrix in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) matrix in global
memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
(9) output to global memory

This way, you can hide the second global memoroy load latency by doing computation on already loaded
input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS GEMM kernel using
cutlass::gemm::device::Gemm template.

The next step is to initialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the matrices are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to initialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference gemm kernel (from CUTLASS utilities) to compare if
the output from CUTLASS kernel is same as reference GEMM kernel.
*/

#include <iostream>
#include <cutlass/util/command_line.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass_helper.cuh"

using ElementAccumulator = int32_t;
using ElementComputeEpilogue = ElementAccumulator; // data type of epilogue operations
using ElementInputA = int8_t;                      // data type of elements in input matrix A
using ElementInputB = int8_t;                      // data type of elements in input matrix B
using ElementOutput = int32_t;                     // data type of elements in output matrix D
//
// using ElementAccumulator = int32_t;
// using ElementComputeEpilogue = ElementAccumulator; // data type of epilogue operations
// using ElementInputA = int32_t;                      // data type of elements in input matrix A
// using ElementInputB = int32_t;                      // data type of elements in input matrix B
// using ElementOutput = int32_t;                     // data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Row Major for
// Matrix A, Column Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;
// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 64>; // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>; // <- warp tile M = 64, N = 64, K = 64
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>; // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                    // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value, // <- the number of elements per vectorized
                                                      // memory access. For a byte, it's 16
                                                      // elements. This becomes the vector width of
                                                      // math instructions in the epilogue too
    ElementAccumulator,                               // <- data type of accumulator
    ElementComputeEpilogue>;                          // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

struct Result
{
    double runtime_ms;
    double gflops;
    cutlass::Status status;
    cudaError_t error;
    bool passed;

    //
    // Methods
    //
    Result(double runtime_ms = 0, double gflops = 0,
           cutlass::Status status = cutlass::Status::kSuccess,
           cudaError_t error = cudaSuccess)
        : runtime_ms(runtime_ms), gflops(gflops),
          status(status), error(error), passed(true)
    {
    }
};

// Command line options parsing
struct Options
{
    bool help;

    cutlass::gemm::GemmCoord problem_size;
    int batch_count;
    float alpha;
    float beta;

    bool reference_check;
    int iterations;

    Options():
        help(false),
        problem_size({5120, 4096, 4096}),
        batch_count(1),
        reference_check(true),
        iterations(20),
        alpha(1),
        beta()
    {
    }

    bool valid()
    {
        return true;
    }

    // Parses the command line
    void parse(int argc, char const** args)
    {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help"))
        {
            help = true;
        }

        cmd.get_cmd_line_argument("m", problem_size.m());
        cmd.get_cmd_line_argument("n", problem_size.n());
        cmd.get_cmd_line_argument("k", problem_size.k());

        cmd.get_cmd_line_argument("alpha", alpha);
        cmd.get_cmd_line_argument("beta", beta);

        cmd.get_cmd_line_argument("iterations", iterations);
    }

    /// Prints the usage statement.
    std::ostream& print_usage(std::ostream& out) const
    {
        out << "14_ampere_tf32_tensorop_gemm example\n\n"
            << "  This example uses the CUTLASS Library to execute TF32 tensorop GEMM computations.\n\n"
            << "Options:\n\n"
            << "  --help                      If specified, displays this usage statement.\n\n"
            << "  --m=<int>                   GEMM M dimension\n"
            << "  --n=<int>                   GEMM N dimension\n"
            << "  --k=<int>                   GEMM K dimension\n"
            << "  --alpha=<f32>               Epilogue scalar alpha\n"
            << "  --beta=<f32>                Epilogue scalar beta\n\n"
            << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

        out << "\n\nExamples:\n\n"
            << "$ ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=1024 --n=512 --k=1024 \\\n"
            << "     --alpha=2 --beta=0.707 \n\n";

        return out;
    }

    /// Compute performance in GFLOP/s
    double gflops(double runtime_s) const
    {
        // Number of real-valued multiply-adds
        int64_t fmas = problem_size.product() * batch_count;

        // Two flops per multiply-add
        return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
    }
};


int run()
{
    const int length_m = 5120;
    const int length_n = 4096;
    const int length_k = 4096;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        problem_size.mk());                                                       // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn()); // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
        problem_size.mn()); // <- Create matrix C with dimensions M x N

    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                            // reference kernel

    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(4),
        ElementInputA(-4),
        0); // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(4),
        ElementInputB(-4),
        0); // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        ElementOutput(4),
        ElementOutput(-4),
        0); // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view()); // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view()); // <- fill matrix D for reference on host with zeros

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,          // <- problem size of matrix multiplication
                                       tensor_a.device_ref(), // <- reference to matrix A on device
                                       tensor_b.device_ref(), // <- reference to matrix B on device
                                       tensor_c.device_ref(), // <- reference to matrix C on device
                                       tensor_d.device_ref(), // <- reference to matrix D on device
                                       {alpha, beta},         // <- tuple of alpha and beta
                                       split_k_slices};       // <- k-dimension split factor
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

    // Create instantiation for device reference gemm kernel
    cutlass::reference::device::Gemm<ElementInputA,
                                     LayoutInputA,
                                     ElementInputB,
                                     LayoutInputB,
                                     ElementOutput,
                                     LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue>
        gemm_device;
    // launch device reference gemm kernel
    gemm_device(problem_size, alpha, tensor_a.device_ref(),
                tensor_b.device_ref(), beta, tensor_c.device_ref(),
                tensor_ref_d.device_ref());

    // wait for kernels to finish
    cudaDeviceSynchronize();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
        tensor_d.host_view(),
        tensor_ref_d.host_view());

    std::cout << (passed ? "Passed" : "Failed") << std::endl;

    return (passed ? 0 : -1);
}
int run(Options& options)
{
    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size = options.problem_size;

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        problem_size.mk()); // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn()); // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
        problem_size.mn()); // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
    // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
    // reference kernel

    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(4),
        ElementInputA(-4),
        0); // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(4),
        ElementInputB(-4),
        0); // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        ElementOutput(4),
        ElementOutput(-4),
        0); // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view()); // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view()); // <- fill matrix D for reference on host with zeros

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    auto alpha = static_cast<ElementComputeEpilogue>(options.alpha);
    auto beta = static_cast<ElementComputeEpilogue>(options.beta);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
        problem_size, // <- problem size of matrix multiplication
        tensor_a.device_ref(), // <- reference to matrix A on device
        tensor_b.device_ref(), // <- reference to matrix B on device
        tensor_c.device_ref(), // <- reference to matrix C on device
        tensor_d.device_ref(), // <- reference to matrix D on device
        {alpha, beta}, // <- tuple of alpha and beta
        split_k_slices
    }; // <- k-dimension split factor

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

    // Result structure
    Result result;

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto& event : events)
    {
        result.error = cudaEventCreate(&event);
        if (result.error != cudaSuccess)
        {
            std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
            return -1;
        }
    }

    // Record an event at the start of a series of GEMMs
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess)
    {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter)
    {
        // Launch initialized CUTLASS kernel
        status = gemm_op();
        CUTLASS_CHECK(status);
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMMs are complete
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess)
    {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess)
    {
        std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess)
    {
        std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    // Cleanup
    for (auto event : events)
    {
        (void)cudaEventDestroy(event);
    }

    // Create instantiation for device reference gemm kernel
    cutlass::reference::device::Gemm<ElementInputA,
                                     LayoutInputA,
                                     ElementInputB,
                                     LayoutInputB,
                                     ElementOutput,
                                     LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue>
        gemm_device;

    // Launch device reference gemm kernel
    gemm_device(problem_size,
                alpha,
                tensor_a.device_ref(),
                tensor_b.device_ref(),
                beta,
                tensor_c.device_ref(),
                tensor_ref_d.device_ref());

    // Wait for kernels to finish
    cudaDeviceSynchronize();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
        tensor_d.host_view(),
        tensor_ref_d.host_view());

    if (passed)
    {
        std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
        std::cout << " GFLOPs: " << result.gflops << std::endl;
    }

    std::cout << (passed ? "Passed" : "Failed") << std::endl;

    return (passed ? 0 : -1);
}

int main(int argc, const char **argv)
{
    bool notSupported = false;

    // Turing Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 10.2.
    //
    // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)))
    {
        std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
        notSupported = true;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (!((props.major * 10 + props.minor) >= 75))
    {
        std::cerr << "Turing Tensor Core operations must be run on a machine with compute capability at least 75."
                  << std::endl;

        notSupported = true;
    }

    if (notSupported)
    {
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }
    Options options;
    options.parse(argc, argv);

    if (options.help)
    {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    printf("%d x %d x %d int8 tensor op Matrix Multiply\n",
           options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

    if (!options.valid())
    {
        std::cerr << "Invalid problem." << std::endl;
        return -1;
    }

    return run(options);
    // return run();
}

/**
 *
 * 
这个示例展示了如何使用CUTLASS提供的函数和数据结构，通过张量核心运行矩阵乘法内核；我们将在NVIDIA Turing GPU上运行。

编写单个高性能矩阵乘法内核虽然困难，但仍可实现。然而，大规模编写高性能内核，使其适用于多种问题规模并具有良好的抽象，确实极具挑战性。
CUTLASS通过提供简化的抽象来组合通用矩阵乘法（GEMM）内核的多个部分，从而解决了这一问题。如果使用得当，内核可以轻松达到GPU的峰值性能。

CUTLASS将内核划分为分层可组合的部分。这意味着，在每个线程、线程束和线程块级别，它们都在各自的瓦片大小上进行计算，
较高层次的瓦片大小由较低层次的瓦片大小组合而成。多个线程瓦片（每个线程计算的瓦片大小）可用于形成线程束瓦片（每个线程束计算的瓦片大小），
多个线程束瓦片可用于计算线程块瓦片（一个线程块计算的瓦片大小）。

在本示例中，我们将变量初始化分为以下两部分：
1. 设置数据属性：描述矩阵在内存中的布局方式，以及内核如何查看它们（逻辑到物理的映射）。
2. 设置计算属性：描述上述设置的矩阵将如何用于计算矩阵乘法的输出。

首先，我们设置矩阵A、B、C和D的数据类型，以及alpha和beta，因为GEMM的公式为D = alpha * A * B + beta * C。
在CUTLASS中，内核首先计算A * B，并将其余计算留到内核末尾，因为alpha * X + beta * C是对X（A * B）和C进行的简单逐元素操作。
我们将此称为内核的尾声。因此，我们将alpha和beta的数据类型设置为与ElementComputeEpilogue = int32_t相同。
由于我们希望在Turing上使用MMA指令，且它们支持8位有符号整数（int8_t），所以我们将输入矩阵A和B中的元素数据类型设为int8_t。
Volta还支持将部分点积累加为int32_t，它可以存储更大范围的数字，因此我们将其用作输出矩阵元素和累加的数据类型。
我们通过初始化模板变量ElementAccumulator（int32_t）、ElementComputeEpilogue（int32_t）、ElementInputA（int8_t）、ElementInputB（int8_t）、ElementOutput（int32_t），
将这些信息传达给CUTLASS内核。仅传达数据类型是不够的。由于数据在内存中是线性布局的，我们必须传达矩阵的布局。
我们通过将模板变量LayoutInputA初始化为列主序的cutlass变量，LayoutInputB初始化为行主序，LayoutOutput初始化为行主序来实现这一点。
接下来，我们设置计算alpha * X + beta * C的规则，这被称为内核的尾声。我们初始化模板变量EpilogueOp，它采用输出ElementOutput（int32_t）的数据类型、
每个向量内存访问的元素数量（ 16 ）、累加器的数据类型（int32_t）以及线性组合（alpha * X + beta * C）的计算数据类型。

现在我们已经设置了数据属性，接下来必须设置计算属性。

其次，我们分别将线程块、线程束和mma - op的瓦片大小模板变量创建为128x256x64、64x64x16、8x8x16（MxNxK）。
当传递这些变量来实例化CUTLASS GEMM内核时，它会在内部推断每个线程块所需的线程数量、共享内存的数量、以无存储体冲突的方式存储数据，以及组合、初始化和启动高性能GEMM内核所需的大量其他变量。
这就是CUTLASS的美妙之处，它使开发人员无需理解和编写复杂的硬件优化代码，因为这些代码很容易出错。

CUTLASS还支持线程块中的多个MMA流水线。什么是MMA流水线？MMA流水线构成了将输入数据从全局内存加载到共享内存、从共享内存加载到寄存器、进行矩阵乘法、存储到全局内存的整个过程。
以下流程序列展示了一个典型的mma流水线。

全局内存中的矩阵 -> 寄存器 -> 共享内存中的瓦片 -> 寄存器 -> mma -> 寄存器 -> 输出到全局内存

单一流水线的问题在于，每个阶段都是同步的，这意味着每个阶段都必须等待前一个阶段完成执行。流水线中有些阶段的延迟不固定，例如，从全局内存和共享内存的加载。
因此，我们可以在mma内核中添加另一个具有相移的流水线，以隐藏来自全局内存和共享内存加载的延迟。最后，内核中的流水线看起来像这样：

(1) 全局内存中的矩阵 -> (2) 寄存器 -> (3) 共享内存中的瓦片 -> (4) 寄存器 -> (5) mma -> (6) 寄存器 -> (7) 输出到全局内存
(1) <空> -> (2) <空> -> (3) 全局内存中的矩阵 -> (4) 寄存器 -> (5) 共享内存中的瓦片 -> (6) 寄存器 -> (7) mma -> (8) 寄存器 -> (9) 输出到全局内存

通过这种方式，你可以通过对已加载的输入数据进行计算，来隐藏第二次全局内存加载的延迟。

还有一些其他模板变量需要初始化，例如，由在SM上启动的哪个线程块完成输出矩阵的哪个线程块瓦片，以及你希望在其上运行的GPU的CUDA SM架构。

所有这些都组合在一起，使用cutlass::gemm::device::Gemm模板创建一个描述CUTLASS GEMM内核的模板变量。

下一步是初始化物理数据、实例化并初始化CUTLASS内核，然后运行它。我们使用CUTLASS实用程序来初始化、填充和比较矩阵，因为它们简单易懂，且不会妨碍对CUTLASS的学习。

一旦所有矩阵都初始化并填充了数据，创建用于启动CUTLASS内核的参数元组，该元组包含问题大小（M = 5120，N = 4096，K = 4096）、矩阵、alpha、beta，以及重要的拆分k维因子。
与此同时，我们向CUTLASS查询实例化的内核是否需要任何临时空间内存。如果需要，我们创建它，并将其与其他创建的参数一起传递，以初始化CUTLASS内核，然后启动内核。

在本示例中，我们随后启动一个参考GEMM内核（来自CUTLASS实用程序），以比较CUTLASS内核的输出是否与参考GEMM内核的输出相同。
 */