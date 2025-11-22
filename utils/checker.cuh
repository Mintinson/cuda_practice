#ifndef CHECKER_CUH_
#define CHECKER_CUH_
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

// #define FMT_UNICODE 0
// #include <fmt/core.h>
#define checkCudaErrors(val) helper::check((val), #val, __FILE__, __LINE__)
namespace helper {

template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

template <typename T>
void check_error(T* a, T* b, std::size_t size, bool avg = false)
{
    double diff = 0;
    double factor = avg ? 1.0 / size : 1.0;
    for (int i = 0; i < size; i++) {
        // std::cout << "diff : " << diff << "\n";
        diff += std::abs(double(a[i]) - double(b[i])) * factor;
        // if (i == 10)
        //     exit(0);
    }

    if (avg) {
        std::cout << "Average error: " << diff << std::endl;
    } else {
        std::cout << "Accumulated error: " << diff << std::endl;
    }
}

template <typename T>
void check_difference(T* a, T* b, size_t size, T eps = {})
{
    if constexpr (std::is_integral_v<T>) {
        for (size_t i = 0; i < size; i++) {
            if (a[i] != b[i]) {
                // if
                std::cout << "a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << "\n";
                return;
            }
        }
    } else {
        for (size_t i = 0; i < size; i++) {
            if (std::abs(a[i] - b[i]) > eps) {
                // if
                // std::cout << fmt::format("a[{0}] = {1}, b[{0}] = {2}\n", i, a[i], b[i]);
                std::cout << "a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << "\n";
                return;
            }
        }
    }
    std::cout << "Checking passed!\n";
}
} // namespace helper

#endif // CHECKER_CUH
