#include "checker.cuh"
#include "cublas_hgemm.cuh"
// #include "helper.cuh"
#include "mma_hegemm.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include "wmma_hgemm.cuh"
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include "swizzle_hgemm.cuh"

GpuTimer timer;
std::size_t M = 512;
std::size_t N = 2048;
std::size_t K = 1024;
// std::size_t M = 16;
// std::size_t N = 16;
// std::size_t K = 16;

template <typename T, bool isInit = true>
void matmul_swap_loop(T *a, T *b, T *c, std::size_t m, std::size_t n, std::size_t l)
{
    if constexpr (!isInit)
    {

        std::fill(c, c + m * n, T{});
    }
    for (std::size_t i = 0; i < m; i++)
    {
        for (std::size_t k = 0; k < l; k++)
        {
            T tmp = a[i * l + k];
            for (std::size_t j = 0; j < n; j++)
            {
                c[i * n + j] += half(float(tmp) * float(b[k * n + j]));
            }
        }
    }
}
template <typename Func, typename... Args>
void tictoc(const std::string &text, std::size_t warmup, std::size_t loop, Func &&f, Args &&...args)
{
    std::cout << text << " start:\n";
    for (int i = 0; i < warmup; ++i)
    {
        f(std::forward<Args>(args)...);
    }
    double accumulateTime{};
    for (int i = 0; i < loop; ++i)
    {
        timer.start();
        f(std::forward<Args>(args)...);
        timer.stop();
        accumulateTime += timer.elapsed();
    }
    std::cout << text << " elapsed: " << accumulateTime / loop << " " << timer.unit() << "\n";
}
void print_matrix(half *mat, std::size_t m0, std::size_t m1, std::size_t n0, std::size_t n1, std::size_t ld)
{
    for (std::size_t i = m0; i < m1; ++i)
    {
        for (std::size_t j = n0; j < n1; ++j)
        {
            std::cout << double(mat[i * ld + j]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
void write_matrix_to_csv(const std::string &filename, half *mat, std::size_t m0, std::size_t m1, std::size_t n0, std::size_t n1, std::size_t ld)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for writing.");
    }
    for (std::size_t i = m0; i < m1; ++i)
    {
        for (std::size_t j = n0; j < n1; ++j)
        {
            file << double(mat[i * ld + j]);
            if (j != n1 - 1)
                file << ",";
        }
        file << "\n";
    }
    file.close();
}
int main()
{
    helper::print_device_info();
    for (int i = 1; i < 2; ++i)
    {
        const std::size_t m = i * M;
        const std::size_t n = i * N;
        const std::size_t k = i * K;
        std::cout << "m: " << m << ", n: " << n << ", k: " << k << "\n";
        auto inputA = helper::generate_sequence<float>(m * k, 0, 10);
        auto inputB = helper::generate_sequence<float>(n * k, 0, 10);
        std::vector<half> hinputA(inputA.begin(), inputA.end());
        std::vector<half> hinputB(inputB.begin(), inputB.end());

        const int loop = 1;
        // decltype(hinputA) houtCPU(m * n);
        // matmul_swap_loop(hinputA.data(), hinputB.data(), houtCPU.data(), m, n, k);
        // tictoc("cpu", 0, loop, [&]() {
        // });

        decltype(hinputA) houtCublas(m * n);
        tictoc("cublas_normal", 3, loop, [&]()
               { cublas::cublas_hgemm<false>(hinputA.data(), hinputB.data(), houtCublas.data(), m, n, k); });
        tictoc("cublas_tensor", 3, loop, [&]()
               { cublas::cublas_hgemm<true>(hinputA.data(), hinputB.data(), houtCublas.data(), m, n, k); });
        // helper::check_error(houtCPU.data(), houtCublas.data(), houtCPU.size(), true);

        // std::string rootpath = "/media/mintinson/新加卷/learningSomething/cuda_learn/cuda_practice/";
        // write_matrix_to_csv(rootpath + "a.csv", hinputA.data(), 0, m, 0, k, k);
        // write_matrix_to_csv(rootpath + "b.csv", hinputB.data(), 0, k, 0, n, n);
        // write_matrix_to_csv(rootpath + "c.csv", houtCublas.data(), 0, m, 0, n, n);

        decltype(hinputA) houtWMMA1(houtCublas.size());
        tictoc("wmma_naive", 0, loop, [&]()
               { cuda_wmma::hgemm_wmma_m16n16k16_naive(hinputA.data(), hinputB.data(), houtWMMA1.data(), m, n, k); });
        helper::check_error(houtCublas.data(), houtWMMA1.data(), houtCublas.size(), true);

        decltype(hinputA) houtWMMA2(houtCublas.size());
        tictoc("wmma_mma4x2", 0, loop, [&]()
               { cuda_wmma::hgemm_wmma_m16n16k16_mma4x2(hinputA.data(), hinputB.data(), houtWMMA2.data(), m, n, k); });
        helper::check_error(houtCublas.data(), houtWMMA2.data(), houtCublas.size(), true);
        // write_matrix_to_csv(rootpath + "c3.csv", houtWMMA2.data(), 0, m, 0, n, n);

        decltype(hinputA) houtWMMA3(houtCublas.size());
        tictoc("wmma_mma4x2_warp2x4", 0, loop, [&]()
               { cuda_wmma::hgemm_wmma_m16n16k16_mma4x2_warp2x4(hinputA.data(), hinputB.data(), houtWMMA3.data(), m, n, k); });
        helper::check_error(houtCublas.data(), houtWMMA3.data(), houtCublas.size(), true);
        // write_matrix_to_csv(rootpath + "c2.csv", houtWMMA3.data(), 0, m, 0, n, n);

        decltype(hinputA) houtWMMA4(houtCublas.size());
        tictoc("wmma_mma4x2_warp2x4_dbuf", 0, loop, [&]()
               { cuda_wmma::hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(hinputA.data(), hinputB.data(), houtWMMA4.data(), m, n, k); });
        helper::check_error(houtCublas.data(), houtWMMA4.data(), houtCublas.size(), true);

        decltype(hinputA) houtWMMA5(houtCublas.size());
        tictoc("wmma_mma4x2_warp2x4_dbuf_padding", 0, loop, [&]()
               { cuda_wmma::hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async<8>(hinputA.data(), hinputB.data(), houtWMMA5.data(), m, n, k); });
        helper::check_error(houtCublas.data(), houtWMMA5.data(), houtCublas.size(), true);

        decltype(hinputA) houtWMMA6(houtCublas.size());
        tictoc("mma_16x8x16_naive", 0, loop, [&]()
               { cuda_mma::hgemm_mma_m16n8k16_naive(hinputA.data(), hinputB.data(), houtWMMA6.data(), m, n, k); });
        helper::check_error(houtCublas.data(), houtWMMA6.data(), houtCublas.size(), true);

        // decltype(hinputA) houtWMMA7(houtCublas.size());
        // tictoc("mma_16x8x16_wmma", 0, loop, [&]()
        //        { cuda_mma::hgemm_mma_m16n8k16_wmma(hinputA.data(), hinputB.data(), houtWMMA7.data(), m, n, k); });
        // helper::check_error(houtCublas.data(), houtWMMA7.data(), houtCublas.size(), true);
        // std::cout << "\n";

        decltype(hinputA) houtWMMA7(houtCublas.size());
        tictoc("swizzle_16x8x16_dbuf", 0, loop, [&]()
               { swizzle::lanunch_hgemm_mma_m16n8k16_swizzle_nn(hinputA.data(), hinputB.data(), houtWMMA7.data(), m, n, k); });
        helper::check_error(houtCublas.data(), houtWMMA7.data(), houtCublas.size(), true);
    }
}