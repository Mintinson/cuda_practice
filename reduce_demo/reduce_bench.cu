//
// Created by asus on 2025/3/31.
//
#include "helper.cuh"
#include "reduce_v1.cuh"
#include "reduce_v2.cuh"
#include "reduce_v3.cuh"
#include "reduce_v4.cuh"
#include "reduce_v5.cuh"
#include "reduce_v6.cuh"
#include "reduce_v7.cuh"
#include "reduce_v9.cuh"
#include "reduce_v10.cuh"
#include "timer.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random_gen.hpp>
#include <thrust/async/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <utility>
#include <vector>
#include <execution>
GpuTimer timer;
constexpr size_t N = 1024 * 1024 * 32;
constexpr int BlockSize = 512;
constexpr int BenchmarkWarmupRuns = 5;
constexpr int BenchmarkMeasuredRuns = 20;

class Logger
{
    std::map<std::string, std::map<size_t, std::vector<double>>> records;
    std::string filename;

public:
    Logger(std::string name)
        : filename(std::move(name))
    {
    }
    void record(std::string name, size_t size, double time)
    {
        if (records.find(name) == records.end())
        {
            records[name] = std::map<size_t, std::vector<double>>();
        }
        if (records[name].find(size) == records[name].end())
        {
            records[name][size] = std::vector<double>();
        }
        records[name][size].push_back(time);
    }
    void save() const
    {
        std::ofstream ofs{filename, std::ios::out};
        if (!ofs)
        {
            std::cerr << "Failed to open file " << filename << std::endl;
            return;
        }
        ofs << "method,elements,mean_ms\n";
        ofs << std::setprecision(10);
        for (const auto &[name, records] : records)
        {
            for (const auto &[size, times] : records)
            {
                const auto mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                ofs << std::quoted(name) << ',' << size << ',' << mean << '\n';
            }
        }
    }
};
Logger logger("reduce_benchmark.csv");

// constexpr int temp = N % (2 * BlockSize);
// 获取当前 GPU 的理论峰值带宽 (GB/s)
double getTheoreticalPeakBandwidth()
{
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    // memoryClockRate 单位是 kHz，memoryBusWidth 单位是 bits
    // 乘以 2 是因为 GDDR 显存在时钟的上升沿和下降沿都传输数据 (Double Data Rate)
    // 除以 8 将 bits 转换为 bytes
    // 除以 1.0e6 将 kHz 转换为 GHz (等价于 10^9 转换 Bytes 为 GB)
    double peakBandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    return peakBandwidth;
}

double g_peak_bandwidth = 0.0;

template <typename Callable>
double benchmarkCuda(Callable &&callable)
{
    using Result = std::invoke_result_t<Callable &>;
    for (int i = 0; i < BenchmarkWarmupRuns; ++i)
    {
        if constexpr (std::is_void_v<Result>)
        {
            callable();
        }
        else
        {
            auto completion = callable();
            completion.wait();
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());

    GpuTimer benchmark_timer;
    double total_ms = 0.0;
    for (int i = 0; i < BenchmarkMeasuredRuns; ++i)
    {
        benchmark_timer.start();
        if constexpr (std::is_void_v<Result>)
        {
            callable();
            benchmark_timer.stop();
            total_ms += benchmark_timer.elapsed<double>();
        }
        else
        {
            // Keep asynchronous algorithm resources alive until the stop event
            // (recorded on the same stream) has completed.
            auto completion = callable();
            benchmark_timer.stop();
            total_ms += benchmark_timer.elapsed<double>();
            completion.wait();
        }
    }

    return total_ms / BenchmarkMeasuredRuns;
}

template <typename T, int ThreadsPerBlock>
__global__ void cub_block_reduce_atomic_kernel(const T *input, size_t size, T *output)
{
    using BlockReduce = cub::BlockReduce<T, ThreadsPerBlock>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T thread_sum{};
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        thread_sum += input[i];
    }

    const T block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(output, block_sum);
    }
}

void printGpuBenchmark(const char *name, size_t size, double time_ms, float answer)
{
    const double bandwidth = (size * sizeof(float) / 1.0e9) / (time_ms / 1000.0);
    const double utilization = bandwidth / g_peak_bandwidth * 100.0;
    logger.record(name, size, time_ms);
    std::cout << name << ": " << time_ms << " ms"
              << " | BW: " << bandwidth << " GB/s"
              << " | Util: " << utilization << "%"
              << " | answer = " << answer << std::endl;
}

void benchmarkCudaLibraryReductions(thrust::device_vector<float> &d_input)
{
    using T = float;
    const size_t size = d_input.size();
    T *input_ptr = thrust::raw_pointer_cast(d_input.data());

    // The synchronous thrust::reduce API returns a host value and necessarily
    // includes D2H. reduce_into keeps the scalar on-device, matching the
    // kernel-only timing used by the hand-written reductions.
    thrust::device_vector<T> d_thrust_output(1);
    const double thrust_ms = benchmarkCuda([&]
                                           { return thrust::async::reduce_into(
                                                 thrust::cuda::par.on(0),
                                                 d_input.cbegin(), d_input.cend(), d_thrust_output.begin(),
                                                 T{}, thrust::plus<T>{}); });
    T thrust_result{};
    checkCudaErrors(cudaMemcpy(
        &thrust_result, thrust::raw_pointer_cast(d_thrust_output.data()),
        sizeof(T), cudaMemcpyDeviceToHost));
    printGpuBenchmark("GPU (Thrust async::reduce_into)", size, thrust_ms, thrust_result);

    T *d_cub_output{};
    void *d_temp_storage{};
    size_t temp_storage_bytes = 0;
    checkCudaErrors(cudaMalloc(&d_cub_output, sizeof(T)));
    checkCudaErrors(cub::DeviceReduce::Sum(
        d_temp_storage, temp_storage_bytes, input_ptr, d_cub_output, size));
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    T cub_device_result{};
    const double cub_device_ms = benchmarkCuda([&]
                                               { checkCudaErrors(cub::DeviceReduce::Sum(
                                                     d_temp_storage, temp_storage_bytes, input_ptr, d_cub_output, size)); });
    checkCudaErrors(cudaMemcpy(
        &cub_device_result, d_cub_output, sizeof(T), cudaMemcpyDeviceToHost));
    printGpuBenchmark("GPU (CUB DeviceReduce)", size, cub_device_ms, cub_device_result);

    int number_of_sms = 0;
    checkCudaErrors(cudaDeviceGetAttribute(
        &number_of_sms, cudaDevAttrMultiProcessorCount, 0));
    constexpr int threads = 256;
    const int blocks = number_of_sms * 4;
    T cub_atomic_result{};
    const double cub_atomic_ms = benchmarkCuda([&]
                                               {
        checkCudaErrors(cudaMemsetAsync(d_cub_output, 0, sizeof(T)));
        cub_block_reduce_atomic_kernel<T, threads><<<blocks, threads>>>(
            input_ptr, size, d_cub_output);
        checkCudaErrors(cudaGetLastError()); });
    checkCudaErrors(cudaMemcpy(
        &cub_atomic_result, d_cub_output, sizeof(T), cudaMemcpyDeviceToHost));
    printGpuBenchmark("GPU (CUB BlockReduce + atomicAdd)", size, cub_atomic_ms, cub_atomic_result);

    checkCudaErrors(cudaFree(d_temp_storage));
    checkCudaErrors(cudaFree(d_cub_output));
}

template <typename T, typename ReduceOp>
__global__ void reduce_rest_kernel(T *input, T *output, T init, size_t sz, ReduceOp op)
{
    for (int i = 0; i < sz; ++i)
    {
        init = op(init, input[i]);
    }
    output[0] = init;
}

// template <typename T, typename ReduceOp>
// T reduce_v1(const T *input, size_t size, T init, ReduceOp op)
// {
//     helper::DeviceDataHandler d_input(input, size);
//     const size_t originalSize = size;
//     // helper::DeviceDataHandler<T> d_output(size);
//     timer.start();
//     while (size >= BlockSize)
//     {

//         reduce_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
//             d_input.data, d_input.data, size, op);
//         size = (size + BlockSize - 1) / BlockSize;
//     }
//     reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
//     timer.stop();
//     auto res = d_input.singleDataToHost(0);

//     double time_ms = timer.elapsed();

//     // ---------------- 带宽计算部分 ----------------
//     // 1. 计算处理的总字节数 (Effective Bytes)
//     // 规约的主要访存发生在地第一次读取全部数据，因此学术界通常用 N * sizeof(T) 作为基准
//     double bytes = originalSize * sizeof(T);

//     // 2. 计算有效带宽 (GB/s) = (Bytes / 1e9) / (time_ms / 1000)
//     double effective_bandwidth = (bytes / 1e9) / (time_ms / 1000.0);

//     // 3. 计算带宽利用率 (%)
//     double utilization = (effective_bandwidth / g_peak_bandwidth) * 100.0;
//     logger.record("GPU (reduce_v1)", originalSize, timer.elapsed());
//     std::cout << "GPU (reduce_v1): " << time_ms << " ms"
//               << " | BW: " << effective_bandwidth << " GB/s"
//               << " | Util: " << utilization << "%"
//               << " | answer = " << res << std::endl;
//     return res;
// }

// template <typename T, typename ReduceOp>
// T reduce_v2(const T *input, size_t size, T init, ReduceOp op)
// {
//     helper::DeviceDataHandler d_input(input, size);
//     const size_t originalSize = size;

//     // helper::DeviceDataHandler<T> d_output(size);
//     timer.start();
//     while (size > BlockSize)
//     {

//         reduce_kernel_v2<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
//             d_input.data, d_input.data, size, op);
//         size = (size + BlockSize - 1) / BlockSize;
//         // size /= BlockSize;
//     }
//     reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
//     timer.stop();
//     auto res = d_input.singleDataToHost(0);
//     double time_ms = timer.elapsed();

//     logger.record("GPU (reduce_v2)", originalSize, timer.elapsed());
//     double bytes = originalSize * sizeof(T);

//     // 2. 计算有效带宽 (GB/s) = (Bytes / 1e9) / (time_ms / 1000)
//     double effective_bandwidth = (bytes / 1e9) / (time_ms / 1000.0);

//     // 3. 计算带宽利用率 (%)
//     double utilization = (effective_bandwidth / g_peak_bandwidth) * 100.0;
//     logger.record("GPU (reduce_v2)", originalSize, timer.elapsed());
//     std::cout << "GPU (reduce_v2): " << time_ms << " ms"
//               << " | BW: " << effective_bandwidth << " GB/s"
//               << " | Util: " << utilization << "%"
//               << " | answer = " << res << std::endl;

//     return res;
// }

// template <typename T, typename ReduceOp>
// T reduce_v3(const T *input, size_t size, T init, ReduceOp op)
// {
//     helper::DeviceDataHandler d_input(input, size);
//     const size_t originalSize = size;
//     // helper::DeviceDataHandler<T> d_output(size);
//     timer.start();
//     while (size > BlockSize)
//     {

//         reduce_kernel_v3<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
//             d_input.data, d_input.data, size, op);
//         size = (size + BlockSize - 1) / BlockSize;
//     }
//     reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
//     timer.stop();
//     auto res = d_input.singleDataToHost(0);
//     double bytes = originalSize * sizeof(T);
//     double time_ms = timer.elapsed();

//     // 2. 计算有效带宽 (GB/s) = (Bytes / 1e9) / (time_ms / 1000)
//     double effective_bandwidth = (bytes / 1e9) / (time_ms / 1000.0);

//     // 3. 计算带宽利用率 (%)
//     double utilization = (effective_bandwidth / g_peak_bandwidth) * 100.0;
//     logger.record("GPU (reduce_v3)", originalSize, timer.elapsed());
//     std::cout << "GPU (reduce_v3): " << time_ms << " ms"
//               << " | BW: " << effective_bandwidth << " GB/s"
//               << " | Util: " << utilization << "%"
//               << " | answer = " << res << std::endl;

//     return res;
// }
// template <typename T, typename ReduceOp>
// T reduce_v4(const T *input, size_t size, T init, ReduceOp op)
// {
//     helper::DeviceDataHandler d_input(input, size);
//     const size_t originalSize = size;
//     // helper::DeviceDataHandler<T> d_output(size);
//     timer.start();
//     while (size > BlockSize)
//     {

//         reduce_kernel_v4<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
//             d_input.data, d_input.data, size, op);
//         size = (size + BlockSize - 1) / BlockSize;
//     }
//     reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
//     timer.stop();
//     double time_ms = timer.elapsed();

//     auto res = d_input.singleDataToHost(0);
//     double bytes = originalSize * sizeof(T);

//     // 2. 计算有效带宽 (GB/s) = (Bytes / 1e9) / (time_ms / 1000)
//     double effective_bandwidth = (bytes / 1e9) / (time_ms / 1000.0);

//     // 3. 计算带宽利用率 (%)
//     double utilization = (effective_bandwidth / g_peak_bandwidth) * 100.0;
//     logger.record("GPU (reduce_v4)", originalSize, timer.elapsed());
//     std::cout << "GPU (reduce_v4): " << time_ms << " ms"
//               << " | BW: " << effective_bandwidth << " GB/s"
//               << " | Util: " << utilization << "%"
//               << " | answer = " << res << std::endl;

//     return res;
// }
// template <typename T, typename ReduceOp>
// T reduce_v5(const T *input, size_t size, T init, ReduceOp op)
// {
//     helper::DeviceDataHandler d_input(input, size);
//     const size_t originalSize = size;
//     // helper::DeviceDataHandler<T> d_output(size);
//     timer.start();
//     while (size > BlockSize * 2)
//     {
//         reduce_kernel_v5<<<(size + 2 * BlockSize - 1) / (BlockSize * 2),
//                            BlockSize, BlockSize * sizeof(T)>>>(
//             d_input.data, d_input.data, size, op);
//         size = (size + 2 * BlockSize - 1) / (2 * BlockSize);
//     }
//     reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
//     timer.stop();
//     double time_ms = timer.elapsed();

//     auto res = d_input.singleDataToHost(0);
//     double bytes = originalSize * sizeof(T);

//     // 2. 计算有效带宽 (GB/s) = (Bytes / 1e9) / (time_ms / 1000)
//     double effective_bandwidth = (bytes / 1e9) / (time_ms / 1000.0);

//     // 3. 计算带宽利用率 (%)
//     double utilization = (effective_bandwidth / g_peak_bandwidth) * 100.0;
//     logger.record("GPU (reduce_v5)", originalSize, timer.elapsed());
//     std::cout << "GPU (reduce_v5): " << time_ms << " ms"
//               << " | BW: " << effective_bandwidth << " GB/s"
//               << " | Util: " << utilization << "%"
//               << " | answer = " << res << std::endl;

//     return res;
// }
// template <typename T, typename ReduceOp>
// T reduce_v6(const T *input, size_t size, T init, ReduceOp op)
// {
//     helper::DeviceDataHandler d_input(input, size);
//     const size_t originalSize = size;
//     // helper::DeviceDataHandler<T> d_output(size);
//     timer.start();
//     while (size > BlockSize * 2)
//     {
//         reduce_kernel_v6<<<(size + 2 * BlockSize - 1) / (BlockSize * 2),
//                            BlockSize, BlockSize * sizeof(T)>>>(
//             d_input.data, d_input.data, size, op);
//         size = (size + 2 * BlockSize - 1) / (2 * BlockSize);
//     }
//     reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
//     timer.stop();
//     double time_ms = timer.elapsed();

//     auto res = d_input.singleDataToHost(0);
//     double bytes = originalSize * sizeof(T);

//     // 2. 计算有效带宽 (GB/s) = (Bytes / 1e9) / (time_ms / 1000)
//     double effective_bandwidth = (bytes / 1e9) / (time_ms / 1000.0);

//     // 3. 计算带宽利用率 (%)
//     double utilization = (effective_bandwidth / g_peak_bandwidth) * 100.0;
//     logger.record("GPU (reduce_v6)", originalSize, timer.elapsed());
//     std::cout << "GPU (reduce_v6): " << time_ms << " ms"
//               << " | BW: " << effective_bandwidth << " GB/s"
//               << " | Util: " << utilization << "%"
//               << " | answer = " << res << std::endl;

//     return res;
// }
// template <typename T, typename ReduceOp>
// T reduce_v7(const T *input, size_t size, T init, ReduceOp op)
// {
//     helper::DeviceDataHandler d_input(input, size);
//     const size_t originalSize = size;
//     // helper::DeviceDataHandler<T> d_output(size);
//     timer.start();
//     while (size > BlockSize * 2)
//     {
//         reduce_kernel_v7<<<(size + 2 * BlockSize - 1) / (BlockSize * 2),
//                            BlockSize, BlockSize * sizeof(T)>>>(
//             d_input.data, d_input.data, size, op);
//         size = (size + 2 * BlockSize - 1) / (2 * BlockSize);
//     }
//     reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
//     timer.stop();
//     double time_ms = timer.elapsed();

//     auto res = d_input.singleDataToHost(0);
//     double bytes = originalSize * sizeof(T);

//     // 2. 计算有效带宽 (GB/s) = (Bytes / 1e9) / (time_ms / 1000)
//     double effective_bandwidth = (bytes / 1e9) / (time_ms / 1000.0);

//     // 3. 计算带宽利用率 (%)
//     double utilization = (effective_bandwidth / g_peak_bandwidth) * 100.0;
//     logger.record("GPU (reduce_v7)", originalSize, timer.elapsed());
//     std::cout << "GPU (reduce_v7): " << time_ms << " ms"
//               << " | BW: " << effective_bandwidth << " GB/s"
//               << " | Util: " << utilization << "%"
//               << " | answer = " << res << std::endl;

//     return res;
// }
// template <typename T, typename ReduceOp>
// T reduce_v9(const T *input, size_t size, T init, ReduceOp op)
// {
//     helper::DeviceDataHandler d_input(input, size);
//     const size_t originalSize = size;
//     // helper::DeviceDataHandler<T> d_output(size);
//     timer.start();
//     while (size > BlockSize * 2)
//     {
//         reduce_kernel_v9<<<(size + 2 * BlockSize - 1) / (BlockSize * 2),
//                            BlockSize, BlockSize * sizeof(T)>>>(
//             d_input.data, d_input.data, size, init, op);
//         size = (size + 2 * BlockSize - 1) / (2 * BlockSize);
//     }
//     reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
//     timer.stop();
//     double time_ms = timer.elapsed();

//     auto res = d_input.singleDataToHost(0);
//     double bytes = originalSize * sizeof(T);

//     // 2. 计算有效带宽 (GB/s) = (Bytes / 1e9) / (time_ms / 1000)
//     double effective_bandwidth = (bytes / 1e9) / (time_ms / 1000.0);

//     // 3. 计算带宽利用率 (%)
//     double utilization = (effective_bandwidth / g_peak_bandwidth) * 100.0;
//     logger.record("GPU (reduce_v9)", originalSize, timer.elapsed());
//     std::cout << "GPU (reduce_v9): " << time_ms << " ms"
//               << " | BW: " << effective_bandwidth << " GB/s"
//               << " | Util: " << utilization << "%"
//               << " | answer = " << res << std::endl;

//     return res;
// }

// template <typename T, typename ReduceOp>
// T reduce_v10(const T *input, size_t size, T init, ReduceOp op)
// {
//     helper::DeviceDataHandler d_input(input, size);
//     const size_t originalSize = size;
//     int num_sms;
//     cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
//     int grid_size = num_sms * 4; // 432 for A100
//     std::printf("Current Device has %d SMs, grid size set to %d\n", num_sms, grid_size);
//     T *d_temp;
//     cudaMalloc(&d_temp, grid_size * sizeof(T));

//     timer.start();
//     // 第一级：处理整个数组，输出 grid_size 个部分和
//     reduce_kernel_v10<<<grid_size, BlockSize>>>(d_input.data, d_temp, size, init, op);

//     // 第二级：对 grid_size 个部分和进行规约（假设 reduce_rest_kernel 能处理）
//     reduce_rest_kernel<<<1, 1>>>(d_temp, d_input.data, init, grid_size, op);
//     timer.stop();

//     cudaFree(d_temp);
//     double time_ms = timer.elapsed();

//     auto res = d_input.singleDataToHost(0);
//     double bytes = originalSize * sizeof(T);

//     // 2. 计算有效带宽 (GB/s) = (Bytes / 1e9) / (time_ms / 1000)
//     double effective_bandwidth = (bytes / 1e9) / (time_ms / 1000.0);

//     // 3. 计算带宽利用率 (%)
//     double utilization = (effective_bandwidth / g_peak_bandwidth) * 100.0;
//     logger.record("GPU (reduce_v10)", originalSize, timer.elapsed());
//     std::cout << "GPU (reduce_v10): " << time_ms << " ms"
//               << " | BW: " << effective_bandwidth << " GB/s"
//               << " | Util: " << utilization << "%"
//               << " | answer = " << res << std::endl;

//     return res;
// }

template <typename T, typename ReduceOp, typename PassLauncher>
void launchMultiPassReduction(
    const T *input,
    T *scratch_a,
    T *scratch_b,
    T *output,
    size_t size,
    size_t elements_per_block,
    bool include_threshold,
    T init,
    ReduceOp op,
    PassLauncher &&launch_pass)
{
    const T *current_input = input;
    T *current_output = scratch_a;
    size_t current_size = size;

    const auto should_reduce = [&]
    {
        return include_threshold
                   ? current_size >= elements_per_block
                   : current_size > elements_per_block;
    };

    while (should_reduce())
    {
        launch_pass(current_input, current_output, current_size);
        current_size = (current_size + elements_per_block - 1) / elements_per_block;
        current_input = current_output;
        current_output = current_output == scratch_a ? scratch_b : scratch_a;
    }

    reduce_rest_kernel<<<1, 1>>>(
        const_cast<T *>(current_input), output, init, current_size, op);
    checkCudaErrors(cudaGetLastError());
}

template <typename T, typename Launch>
void benchmarkReduction(const char *name, size_t size, T *device_output, Launch &&launch)
{
    const double time_ms = benchmarkCuda(std::forward<Launch>(launch));
    T result{};
    checkCudaErrors(cudaMemcpy(
        &result, device_output, sizeof(T), cudaMemcpyDeviceToHost));
    printGpuBenchmark(name, size, time_ms, static_cast<float>(result));
}

template <typename T, typename ReduceOp>
void benchmarkHandwrittenReductions(
    const T *input, size_t size, T init, ReduceOp op)
{
    // The largest first-pass output is produced by the one-element-per-thread
    // kernels. Two buffers allow every pass to remain out-of-place, so the
    // original input stays read-only across warm-up and measured runs.
    const size_t scratch_size = (size + BlockSize - 1) / BlockSize;
    helper::DeviceDataHandler<T> scratch_a(scratch_size);
    helper::DeviceDataHandler<T> scratch_b(scratch_size);
    helper::DeviceDataHandler<T> output(1);

    const auto run = [&](const char *name,
                         size_t elements_per_block,
                         bool include_threshold,
                         auto &&launch_pass)
    {
        benchmarkReduction(name, size, output.data, [&]
                           { launchMultiPassReduction(
                                 input, scratch_a.data, scratch_b.data, output.data, size,
                                 elements_per_block, include_threshold, init, op,
                                 launch_pass); });
    };

    run("GPU (reduce_v1)", BlockSize, true,
        [&](const T *src, T *dst, size_t count)
        {
            reduce_kernel_v1<<<
                (count + BlockSize - 1) / BlockSize,
                BlockSize, BlockSize * sizeof(T)>>>(src, dst, count, op);
        });

    run("GPU (reduce_v2)", BlockSize, false,
        [&](const T *src, T *dst, size_t count)
        {
            reduce_kernel_v2<<<
                (count + BlockSize - 1) / BlockSize,
                BlockSize, BlockSize * sizeof(T)>>>(src, dst, count, op);
        });

    run("GPU (reduce_v3)", BlockSize, false,
        [&](const T *src, T *dst, size_t count)
        {
            reduce_kernel_v3<<<
                (count + BlockSize - 1) / BlockSize,
                BlockSize, BlockSize * sizeof(T)>>>(src, dst, count, op);
        });

    run("GPU (reduce_v4)", BlockSize, false,
        [&](const T *src, T *dst, size_t count)
        {
            reduce_kernel_v4<<<
                (count + BlockSize - 1) / BlockSize,
                BlockSize, BlockSize * sizeof(T)>>>(src, dst, count, op);
        });

    run("GPU (reduce_v5)", BlockSize * 2, false,
        [&](const T *src, T *dst, size_t count)
        {
            reduce_kernel_v5<<<
                (count + BlockSize * 2 - 1) / (BlockSize * 2),
                BlockSize, BlockSize * sizeof(T)>>>(src, dst, count, op);
        });

    run("GPU (reduce_v6)", BlockSize * 2, false,
        [&](const T *src, T *dst, size_t count)
        {
            reduce_kernel_v6<<<
                (count + BlockSize * 2 - 1) / (BlockSize * 2),
                BlockSize, BlockSize * sizeof(T)>>>(src, dst, count, op);
        });

    run("GPU (reduce_v7)", BlockSize * 2, false,
        [&](const T *src, T *dst, size_t count)
        {
            reduce_kernel_v7<<<
                (count + BlockSize * 2 - 1) / (BlockSize * 2),
                BlockSize, BlockSize * sizeof(T)>>>(src, dst, count, op);
        });

    run("GPU (reduce_v9)", BlockSize * 2, false,
        [&](const T *src, T *dst, size_t count)
        {
            reduce_kernel_v9<<<
                (count + BlockSize * 2 - 1) / (BlockSize * 2),
                BlockSize, BlockSize * sizeof(T)>>>(src, dst, count, init, op);
        });

    int number_of_sms = 0;
    checkCudaErrors(cudaDeviceGetAttribute(
        &number_of_sms, cudaDevAttrMultiProcessorCount, 0));
    const int grid_size = number_of_sms * 4;
    benchmarkReduction("GPU (reduce_v10)", size, output.data, [&]
                       {
        reduce_kernel_v10<<<grid_size, BlockSize>>>(
            const_cast<T *>(input), scratch_a.data, size, init, op);
        reduce_rest_kernel<<<1, 1>>>(
            scratch_a.data, output.data, init, grid_size, op);
        checkCudaErrors(cudaGetLastError()); });
}

template <typename T, typename ReduceOp>
void benchmarkAllGpuReductions(
    const std::vector<T> &host_input, T init, ReduceOp op)
{
    thrust::device_vector<T> device_input(host_input);
    const T *input = thrust::raw_pointer_cast(device_input.data());

    benchmarkHandwrittenReductions(input, host_input.size(), init, op);
    benchmarkCudaLibraryReductions(device_input);
}

int main()
{
    g_peak_bandwidth = getTheoreticalPeakBandwidth();
    std::cout << "GPU Theoretical Peak Bandwidth: " << g_peak_bandwidth << " GB/s\n\n";

    for (int i = 1; i < 5 + 1; ++i)
    {
        size_t size = helper::generate_sequence_i<size_t>(1, N * i, N * (i + 1))[0];
        for (int k = 0; k < 1; ++k)
        {
            std::cout << "Reduce with elements : " << size << "\n";
            auto randomVec = helper::generate_sequence<float>(size);
            using ValueType = decltype(randomVec)::value_type;

            ValueType res{0};
            // auto op = std::plus<volatile ValueType> {};
            auto op = [] __device__ __host__(ValueType a, ValueType b)
            {
                return a + b;
            };
            StdTimer<> cpu_timer;
            cpu_timer.start();
            for (auto c : randomVec)
            {
                res = op(res, c);
            }
            cpu_timer.stop();
            logger.record("CPU (raw loop)", size, cpu_timer.elapsed());
            std::cout << "CPU (raw loop): " << cpu_timer.elapsed() << " " << cpu_timer.unit() << ", answer = " << res << std::endl;

            cpu_timer.start();
            auto res2 = std::accumulate(randomVec.cbegin(), randomVec.cend(), static_cast<ValueType>(0), op);
            cpu_timer.stop();
            logger.record("CPU (accumulate loop)", size, cpu_timer.elapsed());
            std::cout << "CPU (accumulate loop): " << cpu_timer.elapsed() << " " << cpu_timer.unit() << ", answer = " << res2 << std::endl;

            cpu_timer.start();
            auto res3 = std::reduce(std::execution::par_unseq, randomVec.cbegin(), randomVec.cend(), static_cast<ValueType>(0), op);
            cpu_timer.stop();
            logger.record("CPU (reduce par)", size, cpu_timer.elapsed());
            std::cout << "CPU (reduce par): " << cpu_timer.elapsed() << " " << cpu_timer.unit() << ", answer = " << res3 << std::endl;

            benchmarkAllGpuReductions(
                randomVec, static_cast<ValueType>(0), op);
            std::cout << "\n";
        }
    }
    logger.save();

    std::cout << "Finished!\n";
}
