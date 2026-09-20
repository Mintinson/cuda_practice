#include "checker.cuh"
#include "helper.cuh"
#include "naive.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include "vec2_optimize.cuh"
#include "vec4_optimize.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <type_traits>
#include <utility>
#include <vector>

constexpr size_t DefaultBaseSize = 1024ULL * 1024 * 64;
constexpr int DefaultCaseCount = 8;
constexpr int BlockDim = 256;
constexpr int WarmupRuns = 5;
constexpr int MeasuredRuns = 20;
constexpr const char *BenchmarkCsv = "element_wise_benchmark.csv";

struct BenchmarkRecord
{
    std::string method;
    size_t elements{};
    double mean_ms{};
    double effective_bandwidth_gbps{};
    double bandwidth_utilization_pct{};
};

class BenchmarkLogger
{
public:
    void record(BenchmarkRecord record)
    {
        records_.push_back(std::move(record));
    }

    void save(const std::string &filename) const
    {
        std::ofstream output(filename, std::ios::out);
        if (!output)
        {
            std::cerr << "Failed to open " << filename << '\n';
            return;
        }

        output << "method,elements,mean_ms,effective_bandwidth_gbps,"
                  "bandwidth_utilization_pct\n";
        output << std::setprecision(10);
        for (const auto &record : records_)
        {
            output << std::quoted(record.method) << ','
                   << record.elements << ','
                   << record.mean_ms << ','
                   << record.effective_bandwidth_gbps << ','
                   << record.bandwidth_utilization_pct << '\n';
        }
    }

private:
    std::vector<BenchmarkRecord> records_;
};

void checkCublas(cublasStatus_t status, const char *operation)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS error " << static_cast<int>(status)
                  << " while calling " << operation << '\n';
        std::exit(EXIT_FAILURE);
    }
}

double theoreticalPeakBandwidthGBps()
{
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    checkCudaErrors(cudaGetDeviceProperties(&properties, device));
    return 2.0 * properties.memoryClockRate *
           (properties.memoryBusWidth / 8.0) / 1.0e6;
}

template <typename Prepare, typename Operation>
double benchmarkCuda(Prepare &&prepare, Operation &&operation)
{
    for (int iteration = 0; iteration < WarmupRuns; ++iteration)
    {
        prepare();
        operation();
    }
    checkCudaErrors(cudaDeviceSynchronize());

    GpuTimer timer;
    double total_ms = 0.0;
    for (int iteration = 0; iteration < MeasuredRuns; ++iteration)
    {
        // Any setup queued by prepare executes before the start event and is
        // intentionally excluded from the measured kernel interval.
        prepare();
        timer.start();
        operation();
        timer.stop();
        total_ms += timer.elapsed<double>();
    }
    return total_ms / MeasuredRuns;
}

template <typename Operation>
double benchmarkCuda(Operation &&operation)
{
    return benchmarkCuda([] {}, std::forward<Operation>(operation));
}

template <typename Operation>
double benchmarkCpu(Operation &&operation)
{
    constexpr int runs = 3;
    StdTimer<> timer;
    double total_ms = 0.0;
    for (int iteration = 0; iteration < runs; ++iteration)
    {
        timer.start();
        operation();
        timer.stop();
        total_ms += timer.elapsed<double>();
    }
    return total_ms / runs;
}

BenchmarkRecord makeRecord(
    std::string method,
    size_t size,
    double time_ms,
    double peak_bandwidth_gbps,
    bool gpu)
{
    // Binary element-wise operation: read A + read B + write output.
    const double transferred_bytes = 3.0 * size * sizeof(float);
    const double bandwidth = (transferred_bytes / 1.0e9) / (time_ms / 1000.0);
    return {
        std::move(method),
        size,
        time_ms,
        bandwidth,
        gpu ? bandwidth / peak_bandwidth_gbps * 100.0 : 0.0};
}

void printRecord(const BenchmarkRecord &record)
{
    std::cout << record.method << ": " << record.mean_ms << " ms"
              << " | BW: " << record.effective_bandwidth_gbps << " GB/s";
    if (record.method.rfind("GPU", 0) == 0)
    {
        std::cout << " | Util: " << record.bandwidth_utilization_pct << '%';
    }
    std::cout << '\n';
}

template <typename T, typename Launch>
void runGpuCase(
    const char *name,
    size_t size,
    double peak_bandwidth_gbps,
    BenchmarkLogger &logger,
    T *device_output,
    std::vector<T> &host_output,
    const std::vector<T> &reference,
    Launch &&launch)
{
    const double time_ms = launch();
    checkCudaErrors(cudaMemcpy(
        host_output.data(), device_output, size * sizeof(T),
        cudaMemcpyDeviceToHost));
    helper::check_difference(
        const_cast<T *>(reference.data()), host_output.data(), size);

    auto record = makeRecord(name, size, time_ms, peak_bandwidth_gbps, true);
    printRecord(record);
    logger.record(std::move(record));
}

template <typename T, typename Operator>
void benchmarkGpuImplementations(
    const std::vector<T> &host_a,
    const std::vector<T> &host_b,
    const std::vector<T> &reference,
    Operator operation,
    double peak_bandwidth_gbps,
    BenchmarkLogger &logger)
{
    const size_t size = host_a.size();
    thrust::device_vector<T> input_a(host_a);
    thrust::device_vector<T> input_b(host_b);
    thrust::device_vector<T> output(size);
    std::vector<T> host_output(size);

    T *a = thrust::raw_pointer_cast(input_a.data());
    T *b = thrust::raw_pointer_cast(input_b.data());
    T *result = thrust::raw_pointer_cast(output.data());

    // Give every GPU implementation the same output contents and cache state.
    // This setup is ordered before the start event, so it is not timed.
    const auto prepare_output = [&]
    {
        checkCudaErrors(cudaMemcpyAsync(
            result, a, size * sizeof(T), cudaMemcpyDeviceToDevice));
    };
    const auto measure = [&](auto &&launch)
    {
        return benchmarkCuda(
            prepare_output, std::forward<decltype(launch)>(launch));
    };

    runGpuCase(
        "GPU (naive)", size, peak_bandwidth_gbps, logger,
        result, host_output, reference, [&]
        { return measure([&]
                         {
                element_wise_naive_kernel<<<
                    (size + BlockDim - 1) / BlockDim, BlockDim>>>(
                    a, b, result, size, operation);
                checkCudaErrors(cudaGetLastError()); }); });

    runGpuCase(
        "GPU (float2)", size, peak_bandwidth_gbps, logger,
        result, host_output, reference, [&]
        {
            const size_t values_per_block = static_cast<size_t>(BlockDim) * 2;
            return measure([&]
            {
                vec2_element_wise_kernel<<<
                    (size + values_per_block - 1) / values_per_block,
                    BlockDim>>>(a, b, result, size, operation);
                checkCudaErrors(cudaGetLastError());
            }); });

    runGpuCase(
        "GPU (float4)", size, peak_bandwidth_gbps, logger,
        result, host_output, reference, [&]
        {
            const size_t values_per_block = static_cast<size_t>(BlockDim) * 4;
            return measure([&]
            {
                vec4_element_wise_kernel<<<
                    (size + values_per_block - 1) / values_per_block,
                    BlockDim>>>(a, b, result, size, operation);
                checkCudaErrors(cudaGetLastError());
            }); });

    runGpuCase(
        "GPU (Thrust transform)", size, peak_bandwidth_gbps, logger,
        result, host_output, reference, [&]
        { return measure([&]
                         { thrust::transform(
                               thrust::device,
                               input_a.begin(), input_a.end(), input_b.begin(),
                               output.begin(), operation); }); });

    cublasHandle_t handle{};
    checkCublas(cublasCreate(&handle), "cublasCreate");
    const T alpha = static_cast<T>(-1);
    runGpuCase(
        "GPU (cuBLAS AXPY)", size, peak_bandwidth_gbps, logger,
        result, host_output, reference, [&]
        { return measure([&]
                         {
                if constexpr (std::is_same_v<T, float>)
                {
                    checkCublas(
                        cublasSaxpy_v2(
                            handle, static_cast<int>(size), &alpha,
                            b, 1, result, 1),
                        "cublasSaxpy_v2");
                }
                else
                {
                    checkCublas(
                        cublasDaxpy_v2(
                            handle, static_cast<int>(size), &alpha,
                            b, 1, result, 1),
                        "cublasDaxpy_v2");
                } }); });
    checkCublas(cublasDestroy(handle), "cublasDestroy");
}

int main(int argc, char **argv)
{
    const size_t base_size = argc > 1
                                 ? static_cast<size_t>(std::stoull(argv[1]))
                                 : DefaultBaseSize;
    const int case_count = argc > 2 ? std::stoi(argv[2]) : DefaultCaseCount;

    helper::print_device_info();
    const double peak_bandwidth_gbps = theoreticalPeakBandwidthGBps();
    std::cout << "Theoretical peak bandwidth: " << peak_bandwidth_gbps
              << " GB/s\n\n";

    BenchmarkLogger logger;
    for (int case_index = 1; case_index <= case_count; ++case_index)
    {
        using ValueType = float;
        const size_t size = base_size * case_index;
        std::cout << "Elements: " << size << '\n';

        auto input_a = helper::generate_sequence<ValueType>(size);
        auto input_b = helper::generate_sequence<ValueType>(size);
        std::vector<ValueType> reference(size);
        std::vector<ValueType> cpu_output(size);
        const auto operation = std::minus<ValueType>{};

        const auto run_cpu_case = [&](const char *name, auto &&callable)
        {
            const double time_ms = benchmarkCpu(
                std::forward<decltype(callable)>(callable));
            auto record = makeRecord(
                name, size, time_ms, peak_bandwidth_gbps, false);
            printRecord(record);
            logger.record(std::move(record));
        };

        run_cpu_case("CPU (raw loop)", [&]
                     {
            for (size_t index = 0; index < size; ++index)
            {
                reference[index] = operation(input_a[index], input_b[index]);
            } });

        run_cpu_case("CPU (std::transform)", [&]
                     { std::transform(
                           input_a.cbegin(), input_a.cend(), input_b.cbegin(),
                           cpu_output.begin(), operation); });
        helper::check_difference(
            reference.data(), cpu_output.data(), reference.size());

        run_cpu_case("CPU (std::transform par_unseq)", [&]
                     { std::transform(
                           std::execution::par_unseq,
                           input_a.cbegin(), input_a.cend(), input_b.cbegin(),
                           cpu_output.begin(), operation); });
        helper::check_difference(
            reference.data(), cpu_output.data(), reference.size());

        benchmarkGpuImplementations(
            input_a, input_b, reference, operation,
            peak_bandwidth_gbps, logger);
        std::cout << '\n';
    }

    const auto csv_path = std::filesystem::path(__FILE__).parent_path() / BenchmarkCsv;
    logger.save(csv_path.string());
    std::cout << "Saved benchmark data to " << csv_path.string() << '\n';
}
