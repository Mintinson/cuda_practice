#include "helper.cuh"
#include "random_gen.hpp"

#include "cpu_mat_trans.hpp"
#include "mat_trans_v0.cuh"
#include "mat_trans_v1.cuh"
#include "mat_trans_v2.cuh"
#include "mat_trans_v3.cuh"
#include "timer.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

constexpr int BenchmarkWarmupRuns = 5;
constexpr int BenchmarkMeasuredRuns = 20;

__global__ void flushL2CacheKernel(uint32_t *buffer, size_t words)
{
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < words)
    buffer[index] = buffer[index] * 1664525u + 1013904223u;
}

class CacheFlusher
{
public:
  explicit CacheFlusher(size_t l2_bytes)
      : words_(std::max<size_t>(2 * l2_bytes, 16 * 1024 * 1024) /
               sizeof(uint32_t)),
        buffer_(words_)
  {
    checkCudaErrors(cudaMemset(buffer_.data, 0, words_ * sizeof(uint32_t)));
  }

  void flush()
  {
    constexpr int threads = 256;
    flushL2CacheKernel<<<(words_ + threads - 1) / threads, threads>>>(
        buffer_.data, words_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

private:
  size_t words_{};
  helper::DeviceDataHandler<uint32_t> buffer_;
};

struct BenchmarkRecord
{
  std::string method;
  size_t rows{};
  size_t cols{};
  double mean_ms{};
  double effective_bandwidth_gbps{};
  double bandwidth_utilization_pct{};
};

class BenchmarkLogger
{
public:
  explicit BenchmarkLogger(std::string filename)
      : filename_(std::move(filename)) {}

  void record(BenchmarkRecord record) { records_.push_back(std::move(record)); }

  void save() const
  {
    std::ofstream output(filename_, std::ios::out);
    if (!output)
      throw std::runtime_error("Failed to open benchmark CSV: " + filename_);

    output << "method,rows,cols,elements,mean_ms,effective_bandwidth_gbps,"
              "bandwidth_utilization_pct\n";
    output << std::setprecision(10);
    for (const auto &record : records_)
    {
      output << std::quoted(record.method) << ',' << record.rows << ','
             << record.cols << ',' << record.rows * record.cols << ','
             << record.mean_ms << ',' << record.effective_bandwidth_gbps << ','
             << record.bandwidth_utilization_pct << '\n';
    }
    std::cout << "Saved benchmark data to " << filename_ << '\n';
  }

private:
  std::string filename_;
  std::vector<BenchmarkRecord> records_;
};

double getTheoreticalPeakBandwidth()
{
  int device = 0;
  checkCudaErrors(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  checkCudaErrors(cudaGetDeviceProperties(&properties, device));
  return 2.0 * properties.memoryClockRate * (properties.memoryBusWidth / 8.0) /
         1.0e6;
}

template <typename Launch>
double benchmarkCudaKernel(CacheFlusher &cache_flusher, Launch &&launch)
{
  for (int i = 0; i < BenchmarkWarmupRuns; ++i)
  {
    cache_flusher.flush();
    launch();
    checkCudaErrors(cudaGetLastError());
  }
  checkCudaErrors(cudaDeviceSynchronize());

  GpuTimer timer;
  double total_ms = 0.0;
  for (int i = 0; i < BenchmarkMeasuredRuns; ++i)
  {
    cache_flusher.flush();
    timer.start();
    launch();
    timer.stop();
    checkCudaErrors(cudaGetLastError());
    total_ms += timer.elapsed<double>();
  }
  return total_ms / BenchmarkMeasuredRuns;
}

template <typename Launch>
double benchmarkCPUFunction(Launch &&launch)
{
  using Clock = std::chrono::high_resolution_clock;
  double total_ms = 0.0;
  for (int i = 0; i < BenchmarkMeasuredRuns; ++i)
  {
    auto start = Clock::now();
    launch();
    auto end = Clock::now();
    total_ms +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0;
  }
  return total_ms / BenchmarkMeasuredRuns;
}

template <typename T>
void validateMatrix(const char *method, const std::vector<T> &reference,
                    const std::vector<T> &actual, T tolerance)
{
  T max_error{};
  size_t max_error_index = 0;
  for (size_t i = 0; i < reference.size(); ++i)
  {
    const T error = std::abs(reference[i] - actual[i]);
    if (!std::isfinite(actual[i]) || error > max_error)
    {
      max_error = error;
      max_error_index = i;
    }
  }
  if (!std::isfinite(actual[max_error_index]) || max_error > tolerance)
  {
    std::cerr << method << " failed validation at element " << max_error_index
              << ": reference=" << reference[max_error_index]
              << ", actual=" << actual[max_error_index]
              << ", max abs error=" << max_error << ", tolerance=" << tolerance
              << std::endl;
    throw std::runtime_error(std::string(method) +
                             " failed validation at element " +
                             std::to_string(max_error_index) +
                             ", max abs error = " + std::to_string(max_error));
  }
}

template <typename T, typename Launch>
void runGpuBenchmark(const char *method, size_t rows, size_t cols,
                     helper::DeviceDataHandler<T> &device_output,
                     std::vector<T> &host_output,
                     const std::vector<T> &reference, T tolerance,
                     double peak_bandwidth_gbps, CacheFlusher &cache_flusher,
                     BenchmarkLogger &logger, Launch &&launch)
{
  const double mean_ms =
      benchmarkCudaKernel(cache_flusher, std::forward<Launch>(launch));

  // D2H happens once after timing. H2D and allocations happen before timing.
  device_output.cpyToHost(host_output.data());
  validateMatrix(method, reference, host_output, tolerance);

  const size_t elements = rows * cols;
  // Common lower-bound traffic: one input read plus one output write.
  // Extra implementation-specific passes are omitted for fair comparison.
  const double effective_bytes = 2.0 * elements * sizeof(T);
  const double bandwidth_gbps = effective_bytes / (mean_ms * 1.0e6);
  const double utilization_pct = bandwidth_gbps / peak_bandwidth_gbps * 100.0;

  logger.record({method, rows, cols, mean_ms, bandwidth_gbps, utilization_pct});
  std::cout << std::left << std::setw(28) << method << " time: " << std::right
            << std::setw(9) << mean_ms << " ms"
            << " | Effective BW: " << std::setw(9) << bandwidth_gbps << " GB/s"
            << " | Util: " << std::setw(7) << utilization_pct << "%\n";
}

template <typename T, typename Launch>
void runCPUBenchmark(const char *method, size_t rows, size_t cols,
                     std::vector<T> &output, const std::vector<T> &reference,
                     T tolerance, BenchmarkLogger &logger, Launch &&launch)
{
  const double mean_ms = benchmarkCPUFunction(std::forward<Launch>(launch));

  validateMatrix(method, reference, output, tolerance);

  logger.record({method, rows, cols, mean_ms});
  std::cout << std::left << std::setw(28) << method << " time: " << std::right
            << std::setw(9) << mean_ms << " ms\n";
}

void benchmarkSize(size_t rows, size_t cols, double peak_bandwidth_gbps,
                   CacheFlusher &cache_flusher, BenchmarkLogger &logger)
{
  using T = float;
  const size_t elements = rows * cols;
  std::cout << "\n Transpose shape: " << rows << " x " << cols << " ("
            << elements << " elements)\n";

  const std::vector<T> host_input = helper::generate_sequence<T>(elements);
  std::vector<T> reference(elements);
  runCPUBenchmark("CPU Trans(con read)", rows, cols, reference, reference,
                  T{1e-5}, logger, [&]
                  { cpu_mat_trans(host_input.data(), reference.data(),
                                  static_cast<int>(rows),
                                  static_cast<int>(cols)); });

  std::vector<T> cpu_output(elements);

  runCPUBenchmark("CPU Trans(con write)", rows, cols, cpu_output, reference,
                  T{1e-5}, logger, [&]
                  { cpu_mat_trans<T, false>(
                        host_input.data(), cpu_output.data(),
                        static_cast<int>(rows), static_cast<int>(cols)); });

  helper::DeviceDataHandler<T> device_input(host_input.data(), elements);
  helper::DeviceDataHandler<T> device_output(elements);
  std::vector<T> host_output(elements);

  T *const input = device_input.data;
  T *const output = device_output.data;
  const int m = static_cast<int>(rows);
  const int n = static_cast<int>(cols);

  const auto run = [&](const char *method, T tolerance, auto &&launch)
  {
    runGpuBenchmark(method, rows, cols, device_output, host_output, reference,
                    tolerance, peak_bandwidth_gbps, cache_flusher, logger,
                    std::forward<decltype(launch)>(launch));
  };

  run("GPU Trnas v0", T{1e-5}, [&]
      {
    const dim3 block(32, 32);
    const dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x),
                    static_cast<unsigned>((m + block.y - 1) / block.y));
    mat_trans_kernel_v0<<<grid, block>>>(input, output, m, n); });
  run("GPU Trnas v1", T{1e-5}, [&]
      {
    const dim3 block(transv1::TILE_WIDTH, transv1::TILE_WIDTH);
    const dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x),
                    static_cast<unsigned>((m + block.y - 1) / block.y));
    mat_trans_kernel_v1<<<grid, block>>>(input, output, m, n); });
  run("GPU Trnas v2", T{1e-5}, [&]
      {
    const dim3 block(transv1::TILE_WIDTH, transv1::TILE_WIDTH);
    const dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x),
                    static_cast<unsigned>((m + block.y - 1) / block.y));
    mat_trans_kernel_v2<<<grid, block>>>(input, output, m, n); });
  run("GPU Trnas v3", T{1e-5}, [&]
      {
    const dim3 block(transv3::TILE_WIDTH, transv3::TILE_WIDTH);
    const unsigned tiles_x = static_cast<unsigned>(
        (n + transv3::TILE_WIDTH - 1) / transv3::TILE_WIDTH);
    const unsigned tiles_y = static_cast<unsigned>(
        (m + transv3::TILE_WIDTH - 1) / transv3::TILE_WIDTH);
    const dim3 grid(std::min(tiles_x, transv3::MAX_GRID_DIM),
                    std::min(tiles_y, transv3::MAX_GRID_DIM));
    mat_trans_kernel_v3<<<grid, block>>>(input, output, m, n); });

  cublasHandle_t cublas{};
  const cublasStatus_t status = cublasCreate(&cublas);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "warning: cublasCreate failed (" << static_cast<int>(status)
              << "); skipping the cuBLAS reference\n";
    return;
  }
  else
  {
    checkCudaErrors(cudaDeviceSynchronize());
  }

  std::vector<T> identity(elements, 0.0f);
  for (std::size_t i = 0; i < std::min(m, n); ++i)
  {
    identity[i * n + i] = 1.0f;
  }
  helper::DeviceDataHandler<T> device_identity(identity.data(), elements);
  float *const h_identity_ptr = device_identity.data;

  run("GPU Trnas cublas", T{1e-5}, [&]
      {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // const int m_int = static_cast<int>(m);
    // const int n_int = static_cast<int>(n);
    const cublasStatus_t status =
        cublasSgeam(cublas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, input, n,
                    &beta, h_identity_ptr, m, output, m); });
}

std::vector<std::pair<std::size_t, std::size_t>> fullSuite()
{
  return {
      {128, 128},
      {512, 512},
      {1024, 1024},
      {2048, 2048},
      {4096, 4096},
      {256, 1024},
      {1024, 256},
      {512, 4096},
      {4096, 512},
      {129, 129},
      {257, 129},
      {1000, 1000},
      {2047, 2049},
  };
}

struct DeviceInfo
{
  std::string name;
  int compute_major{};
  int compute_minor{};
  int memory_clock_khz{};
  int memory_bus_width_bits{};
  std::size_t l2_bytes{};
  int sm_count{};
  double theoretical_peak_gbps{0.0};
};

DeviceInfo queryDevice(int device_id)
{
  DeviceInfo info;
  cudaDeviceProp properties{};
  checkCudaErrors(cudaGetDeviceProperties(&properties, device_id));
  info.name = properties.name;
  info.compute_major = properties.major;
  info.compute_minor = properties.minor;
  info.memory_clock_khz = properties.memoryClockRate;
  info.memory_bus_width_bits = properties.memoryBusWidth;
  info.l2_bytes = static_cast<std::size_t>(properties.l2CacheSize);
  info.sm_count = properties.multiProcessorCount;
  // GDDR 在时钟上下沿都传输数据，故乘 2；bus width 是 bit，除以 8 换算成 byte。
  info.theoretical_peak_gbps = 2.0 *
                               static_cast<double>(info.memory_clock_khz) *
                               (info.memory_bus_width_bits / 8.0) / 1.0e6;
  return info;
}

int main()
{
  std::cout << std::unitbuf;
  std::string csv_path = "mat_transpose_benchmark.csv";

  checkCudaErrors(cudaSetDevice(0));
  const DeviceInfo device = queryDevice(0);
  DeviceInfo info;
  cudaDeviceProp properties{};
  checkCudaErrors(cudaGetDeviceProperties(&properties, 0));

  CacheFlusher cache_flusher(static_cast<size_t>(device.l2_bytes));
  std::cout << "GPU theoretical peak bandwidth: "
            << device.theoretical_peak_gbps << " GB/s\n";
  std::cout
      << "Timing policy: kernel launches only; 5 warm-ups + 20 measured runs\n";
  std::cout << "Cache policy: L2 is evicted before each run, outside the timed "
               "interval\n";
  std::cout
      << "Bandwidth policy: effective bytes = input read + output write\n";

  BenchmarkLogger logger(csv_path);
  const auto mat_sizes = fullSuite();
  for (const auto [row, col] : fullSuite())
    benchmarkSize(row, col, device.theoretical_peak_gbps, cache_flusher,
                  logger);
  logger.save();
  return 0;
}
