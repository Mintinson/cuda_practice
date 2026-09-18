#include "cpu_softmax.hpp"
#include <vector>
#include "random_gen.hpp"
#include "timer.cuh"
#include "softmax_v0.cuh"
#include "softmax_v1.cuh"
#include "softmax_v2.cuh"
#include "softmax_v3.cuh"
#include "softmax_v4.cuh"
#include "softmax_v5.cuh"
#include "helper.cuh"

GpuTimer timer;
constexpr size_t M = 4096;
constexpr size_t N = 4096;
// constexpr size_t L = 1024;
constexpr size_t BlockSize = 1024;

template <typename T>
void softmax_gpu_v0(const T *__restrict__ A, T *__restrict__ C, size_t row, size_t col)
{
    helper::DeviceDataHandler d_input_a(A, row * col);
    helper::DeviceDataHandler<T> d_output(row * col);
    cudda::softmax_v0_kernel<<<(row + BlockSize - 1) / BlockSize, BlockSize>>>(d_input_a.data, d_output.data, row, col);
    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}

template <typename T>
void softmax_gpu_v1(const T *__restrict__ A, T *__restrict__ C, size_t row, size_t col)
{
    helper::DeviceDataHandler d_input_a(A, row * col);
    helper::DeviceDataHandler<T> d_output(row * col);
    cudda::softmax_v1_kernel<<<(row + BlockSize - 1) / BlockSize, BlockSize>>>(d_input_a.data, d_output.data, row, col);
    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}

template <typename T>
void softmax_gpu_v2(const T *__restrict__ A, T *__restrict__ C, size_t row, size_t col)
{
    helper::DeviceDataHandler d_input_a(A, row * col);
    helper::DeviceDataHandler<T> d_output(row * col);
    // here the grid size is rows
    cudda::softmax_v2_kernel<<<row, BlockSize>>>(d_input_a.data, d_output.data, row, col);
    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}

template <typename T>
void softmax_gpu_v3(const T *__restrict__ A, T *__restrict__ C, size_t row, size_t col)
{
    helper::DeviceDataHandler d_input_a(A, row * col);
    helper::DeviceDataHandler<T> d_output(row * col);
    // here the grid size is rows
    cudda::softmax_v3_kernel<<<row, BlockSize>>>(d_input_a.data, d_output.data, row, col);
    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}

template <typename T>
void softmax_gpu_v4(const T *__restrict__ A, T *__restrict__ C, size_t row, size_t col)
{
    helper::DeviceDataHandler d_input_a(A, row * col);
    helper::DeviceDataHandler<T> d_output(row * col);
    dim3 block_size(BlockSize);
    int warp_size = 32;
    size_t smem_size = (block_size.x + warp_size - 1) / warp_size * sizeof(T);
    // here the grid size is rows
    cudda::softmax_v4_kernel<<<row, block_size, smem_size>>>(d_input_a.data, d_output.data, row, col);
    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}
template <typename T>
void softmax_gpu_v5(const T *__restrict__ A, T *__restrict__ C, size_t row, size_t col)
{
    helper::DeviceDataHandler d_input_a(A, row * col);
    helper::DeviceDataHandler<T> d_output(row * col);
    int num_threads_x = 32;

    dim3 block_size(cudda::TILE_SIZE, num_threads_x);
    dim3 grid_size((row + cudda::TILE_SIZE - 1) / cudda::TILE_SIZE, 1);
    // here the grid size is rows
    cudda::softmax_v5_kernel<<<grid_size, block_size>>>(d_input_a.data, d_output.data, row, col);
    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}

template <typename Func>
void average_check(std::string_view desc, Func &&func, int n)
{
    float total_time = 0.0f;
    for (int i = 0; i < n; i++)
    {
        timer.start();
        func();
        timer.stop();
        total_time += timer.elapsed() / n;
    }
    std::cout << desc << " Average Time: " << total_time << " " << timer.unit() << std::endl;
}

int main()
{
    size_t row = M;
    size_t col = N;
    size_t loop = 10;
    // size_t depth = L;
    using ValueType = float;
    auto matA = helper::generate_sequence<ValueType>(row * col);

    auto matCCpu1 = decltype(matA)(row * col);
    average_check("CPU Softmax Naive", [&]()
                  { softmax_naive(matA.data(), matCCpu1.data(), row, col); }, loop);

    auto matCCpu2 = decltype(matA)(row * col);
    average_check("CPU Softmax (Threads)", [&]()
                  { softmax_threads(matA.data(), matCCpu2.data(), row, col); }, loop);
    helper::check_difference(matCCpu1.data(), matCCpu2.data(), row * col, static_cast<ValueType>(1e-5));

    auto matCCpu3 = decltype(matA)(row * col);
    average_check("CPU Softmax (Concurrence)", [&]()
                  { softmax_concurrence(matA.data(), matCCpu3.data(), row, col); }, loop);
    helper::check_difference(matCCpu1.data(), matCCpu3.data(), row * col, static_cast<ValueType>(1e-5));

    auto matCCpu4 = decltype(matA)(row * col);
    average_check("CPU Softmax (Concurrence + Combine)", [&]()
                  { softmax_combine_concurrence(matA.data(), matCCpu4.data(), row, col); }, loop);
    helper::check_difference(matCCpu1.data(), matCCpu4.data(), row * col, static_cast<ValueType>(1e-5));


    auto matCGpu0 = decltype(matA)(row * col);
    average_check("GPU Softmax (v0)", [&]()
                  { softmax_gpu_v0(matA.data(), matCGpu0.data(), row, col); }, loop);

    helper::check_difference(matCCpu1.data(), matCGpu0.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu1 = decltype(matA)(row * col);
    average_check("GPU Softmax (v1)", [&]()
                  { softmax_gpu_v1(matA.data(), matCGpu1.data(), row, col); }, loop);
    helper::check_difference(matCCpu1.data(), matCGpu1.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu2 = decltype(matA)(row * col);
    average_check("GPU Softmax (v2)", [&]()
                  { softmax_gpu_v2(matA.data(), matCGpu2.data(), row, col); }, loop);
    helper::check_difference(matCCpu1.data(), matCGpu2.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu3 = decltype(matA)(row * col);

    average_check("GPU Softmax (v3)", [&]()
                  { softmax_gpu_v3(matA.data(), matCGpu3.data(), row, col); }, loop);
    helper::check_difference(matCCpu1.data(), matCGpu3.data(), row * col, static_cast<ValueType>(1e-5));

    auto matCGpu4 = decltype(matA)(row * col);
    average_check("GPU Softmax (v4)", [&]()
                  { softmax_gpu_v4(matA.data(), matCGpu4.data(), row, col); }, loop);
    helper::check_difference(matCCpu1.data(), matCGpu4.data(), row * col, static_cast<ValueType>(1e-5));

    auto matCGpu5 = decltype(matA)(row * col);
    average_check("GPU Softmax (v5)", [&]()
                  { softmax_gpu_v5(matA.data(), matCGpu5.data(), row, col); }, loop);
    helper::check_difference(matCCpu1.data(), matCGpu5.data(), row * col, static_cast<ValueType>(1e-4));

    // std::vector<float> a = helper::generate_sequence_f(N);
    // std::vector<float> c(N);
    // StdTimer timer;
    // timer.start();
    // softmax_naive(a.data(), c.data(), N);
    // timer.stop();
    // std::cout << "CPU Softmax Elapsed Time: " << timer.elapsed() << " " << timer.unit() << std::endl;
    return 0;
}