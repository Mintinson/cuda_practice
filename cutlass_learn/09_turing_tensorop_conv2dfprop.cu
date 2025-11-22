/**


This example shows how to run convolution kernels using functions and data structures
provided by CUTLASS using tensor cores; which we run on a NVIDIA Turing GPU.

Writing a single high performance convolution kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of implicit gemm kernel. When used properly, the kernels can hit peak performance
of GPU easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how tensors are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set tensors will be used to compute
output of convolution.

First, we setup the data types of the input tensor A, weights' tensor B and output tensor C along
with alpha, beta as the equation for convolution is C = alpha * Conv(A, B) + beta * C. In CUTLASS,
the kernels first compute Conv(A, B) and leave the rest of the computation to end of the kernel as
alpha * X + beta * C is a simple element-wise operation on X (Conv(A, B)) and C. We call this as 
epilogue of kernel. Hence, we setup data types for alpha and beta to be equal to 
ElementComputeEpilogue = float. We want to use MMA instructions on Turing and they support 4-bit
signed integer. But int4b_t is not fully supported by Nvidia software stack, so CUTLASS introduces
cutlass::int4b_t. We use the data type for elements in input tensor A and B as cutlass::int4b_t. We
convey this to CUTLASS kernel by initializing template variables ElementAccumulator (int32_t),
ElementComputeEpilogue (float), ElementInputA (cutlass::int4b_t), ElementInputB (cutlass::int4b_t),
ElementOutput (int32_t). Communicating just the data type is not enough. As the data is laid out 
linearly in memory, we have to convey the layout of tensors. We do that by initializing template
variables LayoutInputA, LayoutInputB and LayoutOutput to TensorNHWC cutlass variable. Next, we setup
rules to comptue alpha * X + beta * C which is called epilogue of the kernel. We initialize template
variable EpilogueOp, which takes the data type of output ElementOutput (int32_t), the number of
elements per vector memory access (32), data type of accumulator (int32_t) and data type of
computation of linear combination (alpha * X + beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x128,
64x64x128, 8x8x32 (MxNxK) respectively. When passed to instantiate CUTLASS Implicit GEMM kernel, it
internally deduces the amount of threads needed per thread-block, amount of shared memory, storing
data in bank-conflict free manner, and ton of other variables required to compose, initialize and
launch a high performance Implicit GEMM kernel. This is the beauty of CUTLASS, it relieves developer
from understanding and coding complicated hardware optimizations which can easily go wrong.

CUTLASS also supports multiple MMA pipelines in a threadblock. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma pipeline.

tensor in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
output to global memory

The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
until the previous finished executing. There are stages in the pipeline which do not have fixed
latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
Finally, the pipeline in a kernel looks like

(1) tensor in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) tensor in global
memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
(9) output to global memory

This way, you can hide the second global memory load latency by doing computation on already loaded
input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS Implicit GEMM
kernel using cutlass::conv::device::ImplicitGemm template.

The next step is to initialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare tensors as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the tensors are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (N = 1, H = 64, W = 64, C = 128), filter size (K = 64,
R = 3, S = 3, C = 128 ), padding, strides, dilation, tensors, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to initialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference convolution kernel (from CUTLASS utilities) to
compare if the output from CUTLASS kernel is same as the reference implicit GEMM kernel.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <cutlass/util/command_line.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/host/convolution.h>

#include "checker.cuh"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass_helper.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
// The code section below describes datatype for input, output tensors and computation between
// elements C = α * Conv(A, B) + β * C
using ElementAccumulator = int32_t; // Data type of accumulator
using ElementComputeEpilogue = float; // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::int4b_t; // Data type of elements in input tensor
using ElementInputB = cutlass::int4b_t; // Data type of elements in input tensor
using ElementOutput = cutlass::int4b_t; // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size of a thread block will compute
using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 128>; // Thread Block tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>; // warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>; // TensorCore instruction shape

// This code section describes how thread blocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
// CUTLASS 还支持线程块中的多个 MMA 流水线
constexpr int NumStages = 2;

// This code section describes the epilogue part of the kernel, we use default value
// 接下来，我们设置计算 α * X + β * C的规则，这被称为内核的尾声。我们初始化模板变量 EpilogueOp，
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput, // Data type of output matrix
    8, // The number of elements per vectorized memory access.
    // This becomes the vector width of math instructions
    // in the epilogue too.
    ElementAccumulator, // Data type of accumulator
    ElementComputeEpilogue // Data type for alpha/beta in linear combination
>;
// 既然我们已经设置了数据的属性，接下来就必须设置计算的属性。
using Conv2DFpropKernel = cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadBlockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;
// 所有这些都组合在一起，使用 cutlass::conv::device::ImplicitGemm 模板创建一个描述 CUTLASS 隐式 GEMM 内核的模板变量。
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2DFpropKernel>;

struct Options
{
    bool help;
    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;

    bool reference_check;
    bool measure_performance;
    int iterations;
    bool save_workspace;
    ElementComputeEpilogue alpha;
    ElementComputeEpilogue beta;
    bool benchmark;
    std::string tag;
    // Verify the problem size is compatible with the CUTLASS Convolution implementation.
    bool valid()
    {
        //
        // CUTLASS attempts to load 128b vectors of int4b_t elements. Consequently,
        // all pointers, strides, and tensor extents must be divisible by 32 elements.
        //
        constexpr int kAlignment = 32;

        if ((input_size.c() % kAlignment) ||
            (filter_size.n() % kAlignment))
        {
            // misaligned tensors
            return false;
        }

        // Invalid padding
        if ((padding.h() != filter_size.h() / 2) ||
            (padding.w() != filter_size.w() / 2))
        {
            return false;
        }

        return true;
    }

    /// Updates input and filter sizes
    void update(
        cutlass::Tensor4DCoord input_size,
        cutlass::Tensor4DCoord filter_size)
    {
        this->input_size = input_size;
        this->filter_size = filter_size;

        padding.n() = filter_size.h() / 2;
        padding.h() = filter_size.h() / 2;
        padding.w() = filter_size.w() / 2;
        padding.c() = filter_size.w() / 2;
    }

    // Parses the command line
    void parse(int argc, char const** args)
    {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help"))
        {
            help = true;
        }

        if (cmd.check_cmd_line_flag("ref-check"))
        {
            reference_check = true;
        }

        if (cmd.check_cmd_line_flag("perf-check"))
        {
            measure_performance = true;
        }

        if (cmd.check_cmd_line_flag("save-workspace"))
        {
            save_workspace = true;
        }

        if (cmd.check_cmd_line_flag("benchmark"))
        {
            benchmark = true;
        }

        cmd.get_cmd_line_argument("n", input_size.n());
        cmd.get_cmd_line_argument("h", input_size.h());
        cmd.get_cmd_line_argument("w", input_size.w());
        cmd.get_cmd_line_argument("c", input_size.c());

        cmd.get_cmd_line_argument("k", filter_size.n());
        cmd.get_cmd_line_argument("r", filter_size.h());
        cmd.get_cmd_line_argument("s", filter_size.w());
        filter_size.c() = input_size.c();

        cmd.get_cmd_line_argument("alpha", alpha);
        cmd.get_cmd_line_argument("beta", beta);

        cmd.get_cmd_line_argument("iterations", iterations);
        cmd.get_cmd_line_argument("tag", tag);

        if (filter_size.h() == 3 && filter_size.w() == 3)
        {
            padding = {1, 1, 1, 1};
        }
        else
        {
            filter_size.h() = 1;
            filter_size.w() = 1;
            padding = {0, 0, 0, 0};
        }
    }

    /// Prints the usage statement.
    static std::ostream& print_usage(std::ostream& out)
    {
        out << "09_turing_tensorop_conv2dfprop example\n\n"
            << "  This example uses Turing's Tensor Core operators on int4 data types to compute\n"
            << "  forward convolution on tensors of layout NHWC.\n\n"
            << "Options:\n\n"
            << "  --help               If specified, displays this usage statement.\n\n"
            << "  --n=<int>            Input tensor extent N\n"
            << "  --h=<int>            Input tensor extent H\n"
            << "  --w=<int>            Input tensor extent W\n"
            << "  --c=<int>            Input tensor extent C\n"
            << "  --k=<int>            Filter extent K\n"
            << "  --r=<int>            Filter extent R\n"
            << "  --s=<int>            Filter extent S\n\n"
            << "  --alpha=<float>      Epilogue scalar alpha\n"
            << "  --beta=<float>       Epilogue scalar beta\n\n"
            << "  --ref-check          If set (true), reference check on the host is computed\n"
            << "  --perf-check         If set (true), performance is measured.\n"
            << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
            << "  --iterations=<int>   Number of profiling iterations to perform.\n"
            << "  --save-workspace     If set, workspace is written to a text file.\n"
            << "  --tag=<string>       String to replicate across the first column in the results table\n";

        out << "\n\nExamples:\n\n"
            << "$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
            << "$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

        return out;
    }

    Options()
        : help(false), input_size(1, 32, 32, 32),
          filter_size(32, 3, 3, 32), padding(1, 1, 1, 1),
          conv_stride(1, 1), dilation(1, 1),
          reference_check(false), measure_performance(true), iterations(20), save_workspace(false),
          alpha(1.0), beta(0), benchmark(false)
    {
    }


    cutlass::Tensor4DCoord output_size() const
    {
        return cutlass::Tensor4DCoord(input_size.n(),
                                      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row()
                                      + 1,
                                      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.
                                      column() + 1,
                                      filter_size.n()
        );
    }

    /// Compute performance in GFLOP/s
    double gflops(double x) const
    {
        // Number of multiply-adds = NPQK * CRS
        int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());

        // Two flops per multiply-add
        return 2.0 * static_cast<double>(fmas) / static_cast<double>(1.0e9) / x;
    }
};

struct Result
{
    double runtime_ms_;
    double gflops_;
    cutlass::Status status_;
    cutlass::Status reference_check_;
    cudaError_t error_;

    Result()
        : runtime_ms_{.0}, gflops_{.0}, status_{cutlass::Status::kSuccess},
          reference_check_{cutlass::Status::kInvalid}, error_{cudaSuccess}
    {
    }

    static std::ostream& print_header(std::ostream& out, const Options& options)
    {
        if (!options.tag.empty())
        {
            out << "Name,";
        }

        out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs";

        return out;
    }

    std::ostream& print(std::ostream& out, int idx, const Options& options)
    {
        if (!options.tag.empty())
        {
            out << options.tag << ",";
        }

        out
            << "conv_" << idx << ","
            << options.input_size.n() << ","
            << options.input_size.h() << ","
            << options.input_size.w() << ","
            << options.input_size.c() << ","
            << options.filter_size.n() << ","
            << options.filter_size.h() << ","
            << options.filter_size.w() << ","
            << runtime_ms_ << ","
            << gflops_;

        return out;
    }
};

// runs one benchmark
Result profile_convolution(const Options& options)
{
    Result result;

    // Allocate host-device tensors using the cutlass utilities
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.input_size);
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.filter_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.output_size());
    // Fill tensor C for reference on host with zeros
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_c(options.output_size());
    //
    // Initialize tensors
    //

    // fill tensor A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(7),
        ElementInputA(-8), 0
    );
    // Fill tensor B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(7),
        ElementInputB(-8),
        0);

    // Fill tensor C on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_c.host_view());
    // Fill tensor C for reference on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_c.host_view());

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_ref_c.sync_device();

    //
    // Define arguments for CUTLASS Convolution
    //
    // mode (kCrossCorrelation or kConvolution)
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Construct Conv2dProblesmSize with user defined output size
    cutlass::conv::Conv2dProblemSize problem_size(
        options.input_size,
        options.filter_size,
        options.padding,
        options.conv_stride,
        options.dilation,
        options.output_size(),
        mode,
        split_k_slices);

    // Construct ImplicitGemm::Argument structure with conv2d
    // problem size, data pointers, and epilogue values
    ImplicitGemm::Arguments arguments{
        problem_size,
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        tensor_c.device_ref(),
        tensor_c.device_ref(),
        {options.alpha, options.beta}
    };

    //
    // Initialize CUTLASS Convolution
    //
    ImplicitGemm implicit_gemm_op;
    // 与此同时，我们向CUTLASS查询实例化的内核是否需要任何临时空间内存。
    // 如果需要，我们创建它，并将其与其他创建的参数一起传递，以初始化CUTLASS内核，然后启动内核。
    auto workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    // allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    result.status_ = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(result.status_);

    result.status_ = implicit_gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(result.status_);

    //
    // Launch initialized CUTLASS kernel
    //
    result.status_ = implicit_gemm_op();

    CUTLASS_CHECK(result.status_);

    //
    // Optional reference check
    //
    if (options.reference_check)
    {
        std::cout << "Verification on host...\n";

        // Compute with reference implementation
        cutlass::reference::host::Conv2dFprop<
            ElementInputA,
            LayoutInputA,
            ElementInputB,
            LayoutInputB,
            ElementOutput,
            LayoutOutput,
            ElementComputeEpilogue,
            ElementAccumulator,
            ElementOutput,
            cutlass::NumericConverterClamp<ElementOutput, ElementComputeEpilogue>
        >(
            problem_size,
            tensor_a.host_ref(),
            tensor_b.host_ref(),
            tensor_c.host_ref(),
            tensor_ref_c.host_ref(),
            options.alpha,
            options.beta
        );

        // Check if output from CUTLASS kernel and reference kernel are equal or not
        tensor_c.sync_host();

        bool passed = cutlass::reference::host::TensorEquals(
            tensor_c.host_view(),
            tensor_ref_c.host_view());

        if (!passed)
        {
            result.reference_check_ = cutlass::Status::kErrorInternal;
            std::cout << "ERROR - results miscompared.\n";
        }
        else
        {
            result.reference_check_ = cutlass::Status::kSuccess;
            std::cout << "Passed.\n";
        }
    }
    else
    {
        result.reference_check_ = cutlass::Status::kInvalid;
    }

    if (options.save_workspace)
    {
        std::stringstream ss;

        ss << "09_tensor_conv_workspace_conv2dfprop_"
            << options.input_size.n() << "x" << options.input_size.h() << "x" << options.input_size.w() << "x" <<
            options.input_size.c()
            << "_"
            << options.filter_size.n() << "x" << options.filter_size.h() << "x" << options.filter_size.w() << "x" <<
            options.filter_size.c()
            << ".dat";

        std::ofstream output_workspace(ss.str());

        output_workspace
            << "Input = \n" << tensor_a.host_view() << "\n\n"
            << "Filters = \n" << tensor_b.host_view() << "\n\n";

        if (options.reference_check)
        {
            output_workspace << "Reference = \n" << tensor_ref_c.host_view() << "\n\n";
        }

        output_workspace << "Computed = \n" << tensor_c.host_view() << std::endl;

        std::cout << "Results written to '" << ss.str() << "'." << std::endl;
    }

    //
    // Performance measurement
    //

    if (options.measure_performance)
    {
        cudaEvent_t events[2];

        for (auto& event : events)
        {
            result.error_ = cudaEventCreate(&event);
            if (result.error_ != cudaSuccess)
            {
                std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error_) << std::endl;
                return result;
            }
        }

        // Record an event at the start of a series of convolution operations.
        result.error_ = cudaEventRecord(events[0]);
        if (result.error_ != cudaSuccess)
        {
            std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error_) << std::endl;
            return result;
        }

        // Launch a sequence of implicit GEMM operations on the device
        for (int iteration = 0; iteration < options.iterations; ++iteration)
        {
            result.status_ = implicit_gemm_op();
            CUTLASS_CHECK(result.status_);
        }

        // Record an event when the convolutions have been launched.
        result.error_ = cudaEventRecord(events[1]);
        if (result.error_ != cudaSuccess)
        {
            std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error_) << std::endl;
            return result;
        }

        // Wait for work on the device to complete.
        result.error_ = cudaEventSynchronize(events[1]);
        if (result.error_ != cudaSuccess)
        {
            std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error_) << std::endl;
            return result;
        }

        // Measure elapsed runtime
        float runtime_ms = 0;
        result.error_ = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
        if (result.error_ != cudaSuccess)
        {
            std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error_) << std::endl;
            return result;
        }

        // Print average runtime and GFLOPs.
        result.runtime_ms_ = static_cast<double>(runtime_ms) / double(options.iterations);
        result.gflops_ = options.gflops(result.runtime_ms_ / 1000.0);

        // Cleanup
        for (auto event : events)
        {
            (void)cudaEventDestroy(event);
        }
    }

    return result;
}


int main(int argc, char const** args)
{
    // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
    //
    // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)))
    {
        std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
        return 0;
    }

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, 0));

    if (!(props.major > 7 || (props.major == 7 && props.minor >= 5)))
    {
        std::cerr << "Turing Tensor Ops must be run on a machine with compute capability at least 75."
            << std::endl;
        return 0;
    }

    Options options;

    options.parse(argc, args);

    if (options.help)
    {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    if (options.benchmark)
    {
        // Benchmark several layers

        int batch_sizes[] = {1, 32, 64, 128, 256, 512};

        struct Benchmark
        {
            int h, w, c, k, r, s;
        } layers[] = {
                {56, 56, 64, 256, 1, 1},
                {56, 56, 64, 64, 1, 1},
                {56, 56, 64, 64, 3, 3},
                {56, 56, 256, 64, 1, 1},
                {56, 56, 256, 512, 1, 1},
                {56, 56, 256, 128, 1, 1},
                {28, 28, 128, 128, 3, 3},
                {28, 28, 128, 512, 1, 1},
                {28, 28, 512, 128, 1, 1},
                {28, 28, 512, 1024, 1, 1},
                {28, 28, 512, 256, 1, 1},
                {14, 14, 256, 256, 3, 3},
                {14, 14, 256, 1024, 1, 1},
                {14, 14, 1024, 256, 1, 1},
                {14, 14, 1024, 2048, 1, 1},
                {14, 14, 1024, 512, 1, 1},
                {7, 7, 512, 512, 3, 3},
            };

        Result::print_header(std::cout, options) << std::endl;

        int idx = 1;

        for (auto const& layer : layers)
        {
            for (auto N : batch_sizes)
            {
                options.update({N, layer.h, layer.w, layer.c}, {layer.k, layer.r, layer.s, layer.c});

                Result result = profile_convolution(options);
                result.print(std::cout, idx, options) << std::endl;
            }

            ++idx;
        }
    }
    else
    {
        // Execute one problem size
        if (!options.valid())
        {
            std::cerr << "Invalid problem." << std::endl;
            return -1;
        }

        Result result = profile_convolution(options);

        Result::print_header(std::cout, options) << std::endl;
        result.print(std::cout, 1, options) << std::endl;
    }
    return 0;
}

/**
 *这个示例展示了如何使用CUTLASS提供的函数和数据结构，通过张量核心运行卷积内核；我们将在NVIDIA Turing GPU上运行此内核。

编写单个高性能卷积内核虽然困难，但并非无法实现。然而，大规模编写高性能内核，使其适用于多种问题规模并具有良好的抽象，确实极具挑战性。
CUTLASS通过提供简化的抽象来组合隐式通用矩阵乘（GEMM）内核的多个部分，从而解决了这一问题。如果使用得当，这些内核能够轻松达到 GPU 的峰值性能。

CUTLASS将内核划分为分层可组合的部分。这意味着，在每个线程、线程束和线程块级别，它们以各自的瓦片大小进行计算，较高级别的瓦片大小由较低级别的瓦片大小组合而成。
多个线程瓦片（每个线程计算的瓦片大小）可用于形成线程束瓦片（每个线程束计算的瓦片大小），多个线程束瓦片可用于计算线程块瓦片（一个线程块计算的瓦片大小）。

在这个示例中，我们将变量初始化分为以下两部分：
1. 设置数据属性：描述张量在内存中的布局方式，以及内核如何查看它们（逻辑到物理的映射）。
2. 设置计算属性：描述上述设置的张量将如何用于计算卷积的输出。

首先，我们设置输入张量A、权重张量B和输出张量C的数据类型，同时设置 α 和 β，因为卷积的公式为C = α * Conv(A, B) + β * C。
在CUTLASS中，内核首先计算 Conv(A, B)，并将其余计算留到内核末尾，因为 α * X + β * C 是对 X（Conv(A, B)） 和 C 进行的简单逐元素操作。
我们将此称为内核的尾声。因此，我们将α和β的数据类型设置为与 ElementComputeEpilogue = float 相同。我们希望在 Turing上使用MMA指令，
它们支持4位有符号整数。但 int4b_t不完全受 NVIDIA软件栈支持，因此 CUTLASS 引入了cutlass::int4b_t。
我们将输入张量 A 和 B 中的元素数据类型设为 cutlass::int4b_t。我们通过初始化模板变量 ElementAccumulator（int32_t）、
ElementComputeEpilogue（float）、ElementInputA（cutlass::int4b_t）、ElementInputB（cutlass::int4b_t）、ElementOutput（int32_t），
将这些信息传达给CUTLASS内核。仅仅传达数据类型是不够的。由于数据在内存中是线性布局的，我们还必须传达张量的布局。
我们通过将模板变量 LayoutInputA、LayoutInputB和 LayoutOutput 初始化为 TensorNHWC cutlass变量来实现这一点。
接下来，我们设置计算 α * X + β * C的规则，这被称为内核的尾声。我们初始化模板变量 EpilogueOp，它接受输出数据类型 ElementOutput（int32_t）、
每个向量内存访问的元素数量（32）、累加器的数据类型（int32_t）以及线性组合（α * X + β * C）的计算数据类型。

既然我们已经设置了数据的属性，接下来就必须设置计算的属性。

其次，我们分别将线程块、线程束 和 mm -op的瓦片大小模板变量设置为 128x128x128、64x64x128、8x8x32（MxNxK）。
当传递这些变量来实例化 CUTLASS 隐式 GEMM 内核时，
它会在内部推断每个线程块所需的线程数量、共享内存的数量、以无存储体冲突的方式存储数据，以及组合、初始化和启动高性能隐式GEMM内核所需的大量其他变量。
这就是CUTLASS的美妙之处，它使开发人员无需理解和编写复杂的硬件优化代码，这些代码很容易出错。

CUTLASS 还支持线程块中的多个 MMA 流水线。什么是 MMA 流水线呢？
MMA流水线构成了从全局内存将输入数据加载到共享内存、从共享内存将数据加载到寄存器、进行矩阵乘法、存储到全局内存的整个过程。
下面的流程序列展示了一个典型的mma流水线。

全局内存中的张量 -> 寄存器 -> 共享内存中的瓦片 -> 寄存器 -> mma -> 寄存器 -> 输出到全局内存

单一流水线的问题在于，每个阶段都是同步的，这意味着每个阶段都必须等待前一个阶段执行完毕。流水线中有些阶段的延迟不是固定的，
例如，从全局内存和共享内存的加载操作。因此，我们可以在mma内核中添加另一个具有相移的流水线，以隐藏来自全局内存和共享内存加载的延迟。
最终，内核中的流水线看起来如下：

(1) 全局内存中的张量 -> (2) 寄存器 -> (3) 共享内存中的瓦片 -> (4) 寄存器 -> (5) mma -> (6) 寄存器 -> (7) 输出到全局内存
(1) <空> -> (2) <空> -> (3) 全局内存中的张量 -> (4) 寄存器 -> (5) 共享内存中的瓦片 -> (6) 寄存器 -> (7) mma -> (8) 寄存器 -> (9) 输出到全局内存

通过这种方式，你可以通过对已加载的输入数据进行计算，来隐藏第二次全局内存加载的延迟。

还有一些其他的模板变量需要初始化，例如，由在SM上启动的哪个线程块完成输出矩阵的哪个线程块瓦片，以及你希望在其上运行的GPU的CUDA SM架构。

所有这些都组合在一起，使用 cutlass::conv::device::ImplicitGemm 模板创建一个描述 CUTLASS 隐式 GEMM 内核的模板变量。

下一步是初始化物理数据、实例化并初始化 CUTLASS 内核，然后运行它。我们使用 CUTLASS 实用程序来初始化、填充和比较张量，因为它们简单易懂，
不会妨碍对CUTLASS的学习。

一旦所有张量都被初始化并填充了数据，就创建一个参数元组来启动 CUTLASS内核，
该参数元组包含问题大小（N = 1, H = 64, W = 64, C = 128）、滤波器大小（K = 64, R = 3, S = 3, C = 128 ）、
填充、步幅、扩张、张量、α、β以及重要的拆分k维因子。
与此同时，我们向CUTLASS查询实例化的内核是否需要任何临时空间内存。如果需要，我们创建它，并将其与其他创建的参数一起传递，以初始化CUTLASS内核，然后启动内核。

在这个示例中，我们随后启动一个参考卷积内核（来自CUTLASS实用程序），以比较CUTLASS内核的输出是否与参考隐式GEMM内核的输出相同。
 **/
