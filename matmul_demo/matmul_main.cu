// =====================================================================
// GEMM benchmark：从 naive 到 cp.async 流水线的全系列对比
//
// 计时策略（与 mat_transpose_demo 一致）：
//   - 只测 kernel：设备缓冲的分配与 H2D 上传在计时区间外一次性完成，
//     所有方法复用同一份设备缓冲；D2H 取回和正确性校验也在计时区间外
//   - 5 次 warmup + 20 次测量取平均；每次测量前先把 L2 缓存刷掉（刷不计时）
// 带宽策略：所有方法统一用"理论下限流量"计算有效带宽——
//   读 A 一次 + 读 B 一次 + 写 C 一次 = (M*K + K*N + M*N) * sizeof(T)
//   实际流量高于该下限的方法（数据复用差）会表现为有效带宽偏低，对比口径一致
// 计算强度：AI = 2*M*N*K / 流量字节（FLOP/Byte，只与形状有关，所有方法相同）；
//   实测算力 GFLOPS = 2*M*N*K / 时间（因方法而异）。立方矩阵（M=N=K=n、float）
//   的 AI = n/6，随 n 线性增长，正好演示 kernel 从"带宽受限"走向"计算受限"
//
// 注：v6 / v8 两个 kernel 存在正确性问题，未纳入对比（v6 只累加了前
//     sharedSize 项且会把同一段重复累加 depth/sharedSize 次；v8 未完成）。
// =====================================================================
#include "cpu_matmul.hpp"
#include "matmul_v0.cuh"
#include "matmul_v1.cuh"
#include "matmul_v2.cuh"
#include "matmul_v4.cuh"
#include "matmul_v5.cuh"
#include "matmul_v7.cuh"
#include "matmul_coal.cuh"
#include "matmul_reg.cuh"
#include "matmul_warp.cuh"
#include "matmul_vec.cuh"
#include "matmul_bank.cuh"
#include "matmul_buffer.cuh"
#include "matmul_cp_async.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include <helper.cuh>
#include <cublas_v2.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ValueType = float;

constexpr int BenchmarkWarmupRuns = 5;
constexpr int BenchmarkMeasuredRuns = 20;
// 立方尺寸 M=N=K=n：float 下计算强度 AI = n/6 FLOP/Byte，随 n 线性增长
constexpr size_t BenchmarkSizes[] = {256, 512, 1024, 2048, 4096};
constexpr size_t BenchmarkSizeCount = std::size(BenchmarkSizes);

// ---------------------------------------------------------------------
// 基准设施：L2 缓存刷新
// ---------------------------------------------------------------------
__global__ void flushL2CacheKernel(uint32_t *buffer, size_t words)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < words)
        buffer[index] = buffer[index] * 1664525u + 1013904223u;
}

class CacheFlusher
{
public:
    explicit CacheFlusher(size_t l2_bytes)
        : words_(std::max<size_t>(2 * l2_bytes, 16 * 1024 * 1024) / sizeof(uint32_t)),
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

struct DeviceInfo
{
    std::string name;
    std::size_t l2_bytes{};
    double theoretical_peak_gbps{0.0};
};

DeviceInfo queryDevice(int device_id)
{
    cudaDeviceProp properties{};
    checkCudaErrors(cudaGetDeviceProperties(&properties, device_id));
    DeviceInfo info;
    info.name = properties.name;
    info.l2_bytes = static_cast<std::size_t>(properties.l2CacheSize);
    // GDDR 在时钟上下沿都传输数据，故乘 2；bus width 是 bit，除以 8 换算成 byte
    info.theoretical_peak_gbps = 2.0 * properties.memoryClockRate *
                                 (properties.memoryBusWidth / 8.0) / 1.0e6;
    return info;
}

// ---------------------------------------------------------------------
// CSV 记录
// ---------------------------------------------------------------------
struct BenchmarkRecord
{
    std::string method;
    size_t m{};
    size_t n{};
    size_t k{};
    double mean_ms{};
    double gflops{};
    double arithmetic_intensity_flop_per_byte{};
    double effective_bandwidth_gbps{};
    double bandwidth_utilization_pct{};
    double device_peak_bandwidth_gbps{};
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

        output << "method,M,N,K,mean_ms,gflops,arithmetic_intensity_flop_per_byte,"
                  "effective_bandwidth_gbps,bandwidth_utilization_pct,"
                  "device_peak_bandwidth_gbps\n";
        output << std::setprecision(10);
        for (const auto &record : records_)
        {
            output << std::quoted(record.method) << ',' << record.m << ',' << record.n
                   << ',' << record.k << ',' << record.mean_ms << ',' << record.gflops
                   << ',' << record.arithmetic_intensity_flop_per_byte << ','
                   << record.effective_bandwidth_gbps << ','
                   << record.bandwidth_utilization_pct << ','
                   << record.device_peak_bandwidth_gbps << '\n';
        }
        std::cout << "Saved benchmark data to " << filename_ << '\n';
    }

private:
    std::string filename_;
    std::vector<BenchmarkRecord> records_;
};

// 所有方法共用同一套指标口径，CPU 基线也不例外（其带宽是主机内存吞吐，
// 与 GPU 指标不可直接比较，只作数量级对照）
BenchmarkRecord makeRecord(const char *method, size_t M, size_t N, size_t K,
                           double mean_ms, double peak_gbps)
{
    const double flops = 2.0 * M * N * K; // 一次乘加记 2 个浮点运算
    const double bytes = double(M * K + K * N + M * N) * sizeof(ValueType);
    BenchmarkRecord record;
    record.method = method;
    record.m = M;
    record.n = N;
    record.k = K;
    record.mean_ms = mean_ms;
    record.gflops = flops / (mean_ms * 1.0e6);
    record.arithmetic_intensity_flop_per_byte = flops / bytes;
    record.effective_bandwidth_gbps = bytes / (mean_ms * 1.0e6);
    record.bandwidth_utilization_pct =
        record.effective_bandwidth_gbps / peak_gbps * 100.0;
    record.device_peak_bandwidth_gbps = peak_gbps;
    return record;
}

void reportRecord(const BenchmarkRecord &record)
{
    std::cout << std::left << std::setw(22) << record.method << " time: " << std::right
              << std::setw(10) << record.mean_ms << " ms"
              << " | GFLOPS: " << std::setw(9) << record.gflops
              << " | Eff BW: " << std::setw(9) << record.effective_bandwidth_gbps
              << " GB/s"
              << " | Util: " << std::setw(6) << record.bandwidth_utilization_pct << "%"
              << " | AI: " << record.arithmetic_intensity_flop_per_byte << " F/B\n";
}

// ---------------------------------------------------------------------
// 计时与校验
// ---------------------------------------------------------------------
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
        throw std::runtime_error(std::string(method) +
                                 " failed validation at element " +
                                 std::to_string(max_error_index) +
                                 ": reference=" +
                                 std::to_string(reference[max_error_index]) +
                                 ", actual=" + std::to_string(actual[max_error_index]) +
                                 ", max abs error=" + std::to_string(max_error) +
                                 ", tolerance=" + std::to_string(tolerance));
    }
}

// 双精度累加的 CPU 参考结果：只用于正确性校验，不计时。
// 单精度 kernel 之间因累加顺序不同本就存在舍入差异，用双精度参考可以把
// 误差完全归因到被测方法的单精度舍入上。
void gemmReferenceDouble(const std::vector<ValueType> &a,
                         const std::vector<ValueType> &b,
                         std::vector<ValueType> &c, size_t M, size_t N, size_t K)
{
    std::vector<double> acc(M * N, 0.0);
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t k = 0; k < K; ++k)
        {
            const double av = a[i * K + k];
            const ValueType *b_row = &b[k * N];
            double *acc_row = &acc[i * N];
            for (size_t j = 0; j < N; ++j)
                acc_row[j] += av * static_cast<double>(b_row[j]);
        }
    }
    for (size_t i = 0; i < M * N; ++i)
        c[i] = static_cast<ValueType>(acc[i]);
}

// ---------------------------------------------------------------------
// v3 / v5 / v7 使用的 pitch 对齐矩阵（RAII）
// 上传发生在构造时、取回在显式 fetch()，都不落在任何计时区间内。
// 立方尺寸下 float 行宽恰好对齐（pitch 通常等于行宽），主要价值是
// 让 kernel 可以去掉边界检查 / 使用对齐访存。
// ---------------------------------------------------------------------
template <typename T>
class PitchedMatrix
{
public:
    PitchedMatrix(const std::vector<T> &host, size_t rows, size_t cols)
    {
        init(rows, cols);
        checkCudaErrors(cudaMemcpy2D(data_, pitch_, host.data(), cols * sizeof(T),
                                     cols * sizeof(T), rows, cudaMemcpyHostToDevice));
    }

    PitchedMatrix(size_t rows, size_t cols) { init(rows, cols); }

    ~PitchedMatrix()
    {
        if (data_)
            cudaFree(data_);
    }

    PitchedMatrix(const PitchedMatrix &) = delete;
    PitchedMatrix &operator=(const PitchedMatrix &) = delete;

    T *data() { return data_; }
    // pitch 换算成"每行多少个元素"
    size_t stride() const { return pitch_ / sizeof(T); }

    void fetch(std::vector<T> &host) const
    {
        checkCudaErrors(cudaMemcpy2D(host.data(), cols_ * sizeof(T), data_, pitch_,
                                     cols_ * sizeof(T), rows_, cudaMemcpyDeviceToHost));
    }

private:
    void init(size_t rows, size_t cols)
    {
        rows_ = rows;
        cols_ = cols;
        checkCudaErrors(cudaMallocPitch(&data_, &pitch_, cols * sizeof(T), rows));
    }

    T *data_{};
    size_t pitch_{};
    size_t rows_{};
    size_t cols_{};
};

// ---------------------------------------------------------------------
// 一个尺寸下所有 GPU 方法共享的基准设施：
//   - 设备缓冲一次性分配/上传，全部方法复用（计时区间内没有 H2D / 分配）
//   - run()：只计 kernel 时间 -> D2H 取回（不计时）-> 校验 -> 指标 -> CSV
// ---------------------------------------------------------------------
class GemmBench
{
public:
    GemmBench(size_t M, size_t N, size_t K,
              const std::vector<ValueType> &host_a,
              const std::vector<ValueType> &host_b,
              const std::vector<ValueType> &reference, double peak_gbps,
              CacheFlusher &flusher, BenchmarkLogger &logger)
        : m_(M), n_(N), k_(K),
          d_a_(host_a.data(), M * K),
          d_b_(host_b.data(), K * N),
          d_c_(M * N),
          host_output_(M * N),
          reference_(&reference),
          peak_gbps_(peak_gbps),
          flusher_(&flusher),
          logger_(&logger) {}

    ValueType *A() const { return d_a_.data; }
    ValueType *B() const { return d_b_.data; }
    ValueType *C() const { return d_c_.data; }
    std::vector<ValueType> &hostOutput() { return host_output_; }

    // 平坦布局：kernel 直接写 C，取回就是一次整块 D2H
    template <typename Launch>
    void run(const char *method, ValueType tolerance, Launch &&launch)
    {
        const double mean_ms =
            benchmarkCudaKernel(*flusher_, std::forward<Launch>(launch));
        checkCudaErrors(cudaMemcpy(host_output_.data(), d_c_.data,
                                   host_output_.size() * sizeof(ValueType),
                                   cudaMemcpyDeviceToHost));
        finish(method, tolerance, mean_ms);
    }

    // 非平坦布局（pitched 等）：fetch() 负责把结果转回 hostOutput_，同样不计时
    template <typename Launch, typename Fetch>
    void run(const char *method, ValueType tolerance, Launch &&launch, Fetch &&fetch)
    {
        const double mean_ms =
            benchmarkCudaKernel(*flusher_, std::forward<Launch>(launch));
        fetch();
        finish(method, tolerance, mean_ms);
    }

private:
    void finish(const char *method, ValueType tolerance, double mean_ms)
    {
        validateMatrix(method, *reference_, host_output_, tolerance);
        const BenchmarkRecord record =
            makeRecord(method, m_, n_, k_, mean_ms, peak_gbps_);
        logger_->record(record);
        reportRecord(record);
    }

    size_t m_{}, n_{}, k_{};
    helper::DeviceDataHandler<ValueType> d_a_;
    helper::DeviceDataHandler<ValueType> d_b_;
    helper::DeviceDataHandler<ValueType> d_c_;
    std::vector<ValueType> host_output_;
    const std::vector<ValueType> *reference_{};
    double peak_gbps_{};
    CacheFlusher *flusher_{};
    BenchmarkLogger *logger_{};
};

void checkCublas(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error: " + std::to_string(int(status)));
}

// 单精度累加误差随 K 增长，容差按 K 缩放（K=1024 时为 1e-1，与旧版一致）
ValueType toleranceOf(size_t K)
{
    return static_cast<ValueType>(1e-1 * std::max(1.0, double(K) / 1024.0));
}

// ---------------------------------------------------------------------
// 单个尺寸的全系列对比
// ---------------------------------------------------------------------
void benchmarkSize(size_t M, size_t N, size_t K, double peak_gbps,
                   CacheFlusher &cache_flusher, BenchmarkLogger &logger,
                   cublasHandle_t cublas)
{
    using T = ValueType;
    std::cout << "\n=== GEMM shape: M x N x K = " << M << " x " << N << " x " << K
              << " ===\n";

    const std::vector<T> host_a = helper::generate_sequence<T>(M * K);
    const std::vector<T> host_b = helper::generate_sequence<T>(K * N);

    // 正确性参考：双精度 CPU 乘法（不计时）
    std::vector<T> reference(M * N);
    gemmReferenceDouble(host_a, host_b, reference, M, N, K);

    // CPU 基线：cache 友好的 swap-loop，单次计时，只作数量级对照
    {
        std::vector<T> cpu_c(M * N);
        StdTimer<std::milli> cpu_timer;
        cpu_timer.start();
        matmul_swap_loop<T, false>(host_a.data(), host_b.data(), cpu_c.data(), M, N, K);
        cpu_timer.stop();
        validateMatrix("CPU swap-loop", reference, cpu_c, toleranceOf(K));
        const BenchmarkRecord record =
            makeRecord("CPU swap-loop", M, N, K, cpu_timer.elapsed<double>(), peak_gbps);
        logger.record(record);
        reportRecord(record);
    }

    GemmBench bench(M, N, K, host_a, host_b, reference, peak_gbps, cache_flusher,
                    logger);

    // v3 / v5 / v7 的 pitch 对齐布局：一次性上传（不计时）
    PitchedMatrix<T> p_a(host_a, M, K);
    PitchedMatrix<T> p_b(host_b, K, N);
    PitchedMatrix<T> p_c(M, N);
    const auto fetchPitchedC = [&]
    { p_c.fetch(bench.hostOutput()); };

    const T tolerance = toleranceOf(K);

    // v0：每线程计算一个 C 元素，A 广播 / B 合并访问，但零数据复用
    bench.run("GPU v0 naive", tolerance, [&]
              {
        const unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((M * N + threads - 1) / threads);
        matmul_v0_kernel<<<blocks, threads>>>(bench.A(), bench.B(), bench.C(), M, N, K); });

    // v1：一个 block 负责一行 C（无共享内存，A 靠广播、B 靠合并访问）
    bench.run("GPU v1 block-loop", tolerance, [&]
              {
        matmul_v1_kernel<<<static_cast<unsigned>(M), 256>>>(
            bench.A(), bench.B(), bench.C(), M, N, K); });

    // v2：整行 A 缓存进共享内存，B 合并访问
    bench.run("GPU v2 row-smem", tolerance, [&]
              {
        matmul_v2_kernel<<<static_cast<unsigned>(M), 256, K * sizeof(T)>>>(
            bench.A(), bench.B(), bench.C(), M, N, K); });

    // v3：v2 + pitch 对齐布局（上传/取回都不计时，只计 kernel）
    bench.run("GPU v3 pitch", tolerance, [&]
              {
        matmul_v2_kernel<<<static_cast<unsigned>(M), 256, p_a.stride() * sizeof(T)>>>(
            p_a.data(), p_b.data(), p_c.data(), M, p_c.stride(), p_a.stride()); },
              fetchPitchedC);

    // v4：棋盘阵列分块，每 block 缓存 A/B 的 32x32 子块
    bench.run("GPU v4 checkerboard", tolerance, [&]
              {
        const dim3 block(32, 32);
        const dim3 grid(static_cast<unsigned>((N + block.x - 1) / block.x),
                        static_cast<unsigned>((M + block.y - 1) / block.y));
        matmul_v4_kernel<<<grid, block, 2 * 32 * 32 * sizeof(T)>>>(
            bench.A(), bench.B(), bench.C(), M, N, K); });

    // v5：v4 去掉边界检查（依赖 pitch 补齐 + 尺寸对齐）
    bench.run("GPU v5 pitch-nocheck", tolerance, [&]
              {
        const dim3 block(32, 32);
        const dim3 grid(static_cast<unsigned>((p_c.stride() + block.x - 1) / block.x),
                        static_cast<unsigned>((M + block.y - 1) / block.y));
        matmul_v5_kernel<<<grid, block, 2 * 32 * 32 * sizeof(T)>>>(
            p_a.data(), p_b.data(), p_c.data(), M, p_c.stride(), p_a.stride()); },
              fetchPitchedC);

    // v7：pitch + 每线程负责 2x2 个子块，提高单线程计算量
    bench.run("GPU v7 pitch-subtile", tolerance, [&]
              {
        constexpr unsigned kSub = 2;
        const dim3 block(16, 16);
        const dim3 grid(static_cast<unsigned>((p_c.stride() + kSub * block.x - 1) / (kSub * block.x)),
                        static_cast<unsigned>((M + kSub * block.y - 1) / (kSub * block.y)));
        matmul_v7_kernel<kSub><<<grid, block, 2 * 16 * 16 * sizeof(T) * kSub>>>(
            p_a.data(), p_b.data(), p_c.data(), M, p_c.stride(), p_a.stride()); },
              fetchPitchedC);

    // cuBLAS：已通过 CUBLAS_PEDANTIC_MATH 关闭 TF32，保证是纯 FP32 路径
    bench.run("GPU cublas fp32", tolerance, [&]
              {
        const T alpha = 1.0f;
        const T beta = 0.0f;
        // 行主序 C = A*B 等价于列主序视角下 C^T = B^T * A^T，
        // 于是把 B 当列主序的 B^T、A 当列主序的 A^T 传入即可
        checkCublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(N), static_cast<int>(M),
                                static_cast<int>(K), &alpha, bench.B(),
                                static_cast<int>(N), bench.A(), static_cast<int>(K),
                                &beta, bench.C(), static_cast<int>(N))); });

    // block-tiling：每 block 一个 128x128 的 C 分块，A/B 子块进共享内存
    bench.run("GPU block-tiling", tolerance, [&]
              {
        const dim3 block(16, 16);
        const dim3 grid(static_cast<unsigned>((N + 127) / 128),
                        static_cast<unsigned>((M + 127) / 128));
        sgemm_block_tiling<128, 128, 8, 256><<<grid, block>>>(
            bench.A(), bench.B(), bench.C(), static_cast<int>(M),
            static_cast<int>(N), static_cast<int>(K)); });

    // thread-tiling：在 block-tiling 基础上让每线程负责 8x8 的寄存器子块
    bench.run("GPU thread-tiling", tolerance, [&]
              {
        const dim3 block(16, 16); // (128/8) * (128/8) = 256 线程
        const dim3 grid(static_cast<unsigned>((N + 127) / 128),
                        static_cast<unsigned>((M + 127) / 128));
        sgemm_thread_tiling<128, 128, 8, 8, 8><<<grid, block>>>(
            bench.A(), bench.B(), bench.C(), static_cast<int>(M),
            static_cast<int>(N), static_cast<int>(K)); });

    // warp-tiling：warp 内 4x8 个 lane 协作一个 32x64 的子块
    bench.run("GPU warp-tiling", tolerance, [&]
              {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((N + 127) / 128),
                        static_cast<unsigned>((M + 127) / 128));
        sgemm_warp_tiling<128, 128, 8, 4, 8, 8, 8><<<grid, block>>>(
            bench.A(), bench.B(), bench.C(), static_cast<int>(M),
            static_cast<int>(N), static_cast<int>(K)); });

    // float4：向量化全局访存 + A 转置进共享内存
    bench.run("GPU warp-float4", tolerance, [&]
              {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((N + 127) / 128),
                        static_cast<unsigned>((M + 127) / 128));
        sgemm_warp_tiling_vec4<128, 128, 8, 4, 8, 8, 8><<<grid, block>>>(
            bench.A(), bench.B(), bench.C(), static_cast<int>(M),
            static_cast<int>(N), static_cast<int>(K)); });

    // z-order：warp-float4 + Z 序线程映射消除共享内存 bank conflict
    bench.run("GPU swizzle-zorder", tolerance, [&]
              {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((N + 127) / 128),
                        static_cast<unsigned>((M + 127) / 128));
        sgemm_warp_tiling_vec4_z_order<128, 128, 8, 4, 8, 8, 8><<<grid, block>>>(
            bench.A(), bench.B(), bench.C(), static_cast<int>(M),
            static_cast<int>(N), static_cast<int>(K)); });

    // double-buffer：Global->Shared 双缓冲，搬运与计算重叠
    bench.run("GPU double-buffer", tolerance, [&]
              {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((N + 127) / 128),
                        static_cast<unsigned>((M + 127) / 128));
        sgemm_double_buffer<128, 128, 8, 4, 8, 8, 8><<<grid, block>>>(
            bench.A(), bench.B(), bench.C(), static_cast<int>(M),
            static_cast<int>(N), static_cast<int>(K)); });

    // cp.async：3 级硬件异步流水线，A/B 全异步直拷进 smem。
    // BLOCK_K=16 是 swizzle 和流水线粒度的甜点（每级 16KB，smem 恰好 48KB）
    bench.run("GPU cp.async-3stage", tolerance, [&]
              {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((N + 127) / 128),
                        static_cast<unsigned>((M + 127) / 128));
        sgemm_cp_async_pipeline<128, 128, 16, 4, 8, 8, 8, 3><<<grid, block>>>(
            bench.A(), bench.B(), bench.C(), static_cast<int>(M),
            static_cast<int>(N), static_cast<int>(K)); });

    // ---- 临时调参对照（跑完删除）----
}

int main()
{
    std::cout << std::unitbuf;
    checkCudaErrors(cudaSetDevice(0));
    const DeviceInfo device = queryDevice(0);

    CacheFlusher cache_flusher(device.l2_bytes);
    std::cout << "GPU: " << device.name << "\n";
    std::cout << "GPU theoretical peak bandwidth: " << device.theoretical_peak_gbps
              << " GB/s\n";
    std::cout << "Timing policy: kernel only; " << BenchmarkWarmupRuns
              << " warm-ups + " << BenchmarkMeasuredRuns
              << " measured runs; L2 flushed before each run (outside timing)\n";
    std::cout << "Bandwidth policy: effective bytes = read A + read B + write C "
                 "(lower bound)\n";

    BenchmarkLogger logger("matmul_benchmark.csv");
    cublasHandle_t cublas{};
    checkCublas(cublasCreate(&cublas));
    // cuBLAS 默认在 Ampere+ 上允许用 TF32 Tensor Core 做 FP32 GEMM，
    // 那不是纯 FP32；PEDANTIC_MATH 强制走 FP32 CUDA Core，和手写 kernel
    // 同一精度，对比才公平
    checkCublas(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));

    for (size_t i = 0; i < BenchmarkSizeCount; ++i)
    {
        const size_t n = BenchmarkSizes[i];
        benchmarkSize(n, n, n, device.theoretical_peak_gbps, cache_flusher, logger,
                      cublas);
    }

    checkCublas(cublasDestroy(cublas));
    logger.save();
    return 0;
}
