#include "checker.cuh"
#include "cpu_mat_trans.hpp"
#include "helper.cuh"
#include "mat_trans_v0.cuh"
#include "mat_trans_v1.cuh"
#include "mat_trans_v2.cuh"
#include "mat_trans_v3.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include <cstddef>
#include <iostream>
#include <string>

GpuTimer timer;

constexpr size_t M = 256;
constexpr size_t N = 256;

template <typename F>
void tictoc(const std::string& name, F&& f)
{
    timer.start();
    f();
    timer.stop();
    std::cout << name << " time: " << timer.elapsed() << " " << timer.unit() << std::endl;
}

template <typename T>
void mat_transpose_v0(const T* src, T* dst, size_t m, size_t n)
{
    helper::DeviceDataHandler<T> d_src(src, m * n);
    helper::DeviceDataHandler<T> d_dst(n * m);
    dim3 block(32, 32);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    mat_trans_kernel_v0<<<grid, block>>>(d_src.data, d_dst.data, m, n);

    d_dst.cpyToHost(dst);
}

template <typename T>
void mat_transpose_v1(const T* src, T* dst, size_t m, size_t n)
{
    helper::DeviceDataHandler<T> d_src(src, m * n);
    helper::DeviceDataHandler<T> d_dst(n * m);
    dim3 block(transv1::TILE_WIDTH, transv1::TILE_WIDTH);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    mat_trans_kernel_v1<<<grid, block>>>(d_src.data, d_dst.data, m, n);

    d_dst.cpyToHost(dst);
}
template <typename T>
void mat_transpose_v2(const T* src, T* dst, size_t m, size_t n)
{
    helper::DeviceDataHandler<T> d_src(src, m * n);
    helper::DeviceDataHandler<T> d_dst(n * m);
    dim3 block(transv2::TILE_WIDTH, transv2::TILE_WIDTH);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    mat_trans_kernel_v2<<<grid, block>>>(d_src.data, d_dst.data, m, n);

    d_dst.cpyToHost(dst);
}
template <typename T>
void mat_transpose_v3(const T* src, T* dst, size_t m, size_t n)
{
    helper::DeviceDataHandler<T> d_src(src, m * n);
    helper::DeviceDataHandler<T> d_dst(n * m);
    dim3 block(transv2::TILE_WIDTH, transv2::TILE_WIDTH);
    dim3 grid(std::min((n + block.x - 1) / block.x, 64UL), std::min((m + block.y - 1) / block.y, 64UL));
    // dim3 grid(, 1);
    mat_trans_kernel_v3<<<grid, block>>>(d_src.data, d_dst.data, m, n);

    d_dst.cpyToHost(dst);
}
int main()
{
    std::vector<size_t> MRate = { 1, 2, 4, 8, 16, 32 };
    std::vector<size_t> NRate = { 1, 2, 4, 8, 16, 32 };
    for (auto mr : MRate) {
        for (auto nr : NRate) {
            const size_t m = M * mr;
            const size_t n = N * nr;
            std::cout << "m: " << m << ", n: " << n << std::endl;
            auto origA = helper::generate_sequence<float>(m * n);
            auto cpuA1 = decltype(origA)(origA.size());
            auto cpuA2 = decltype(origA)(origA.size());

            tictoc("CPU (con read)", [&]() {
                cpu_mat_trans(origA.data(), cpuA1.data(), m, n);
            });
            tictoc("CPU (con write)", [&]() {
                cpu_mat_trans(origA.data(), cpuA2.data(), m, n);
            });
            helper::check_difference(cpuA1.data(), cpuA2.data(), m * n);

            auto gpuA0 = decltype(origA)(origA.size());
            tictoc("GPU (v0)", [&]() {
                mat_transpose_v0(origA.data(), gpuA0.data(), m, n);
            });
            helper::check_difference(cpuA1.data(), gpuA0.data(), m * n);

            auto gpuA1 = decltype(origA)(origA.size());
            tictoc("GPU (v1)", [&]() {
                mat_transpose_v1(origA.data(), gpuA1.data(), m, n);
            });
            helper::check_difference(cpuA1.data(), gpuA1.data(), m * n);

            auto gpuA2 = decltype(origA)(origA.size());
            tictoc("GPU (v2)", [&]() {
                mat_transpose_v1(origA.data(), gpuA2.data(), m, n);
            });
            helper::check_difference(cpuA1.data(), gpuA2.data(), m * n);

            auto gpuA3 = decltype(origA)(origA.size());
            tictoc("GPU (v3)", [&]() {
                mat_transpose_v3(origA.data(), gpuA3.data(), m, n);
            });
            helper::check_difference(cpuA1.data(), gpuA3.data(), m * n);

            std::cout << "\n";
        }
    }
    // const size_t m = M;
    // const size_t n = N;
}