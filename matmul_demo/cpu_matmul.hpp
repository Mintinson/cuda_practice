#ifndef CPU_MATMUL_HPP_
#define CPU_MATMUL_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdlib>
template <typename T>
void matmul_general(const T* a, const T* b, T* c, int m, int n, int l)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            T tmp {};
            for (int k = 0; k < l; k++) {
                tmp += a[i * l + k] * b[k * n + j];
            }
            c[i * n + j] = tmp;
        }
    }
}

// if isInit is true, function assume that c is all zero, so we won't fill c with zero first
template <typename T, bool isInit = true>
void matmul_swap_loop(const T* a, const T* b, T* c, std::size_t m, std::size_t n, std::size_t l)
{
    if constexpr (!isInit) {

        std::fill(c, c + m * n, T {});
    }
    for (std::size_t i = 0; i < m; i++) {
        for (std::size_t k = 0; k < l; k++) {
            T tmp = a[i * l + k];
            for (std::size_t j = 0; j < n; j++) {
                c[i * n + j] += tmp * b[k * n + j];
            }
        }
    }
}

// if isTranspose is true, function assume that b is n x l, so we won't transpose b
template <typename T, bool isTranspose = true>
void matmul_transpose(const T* a, const T* b, T* c, std::size_t m, std::size_t n, std::size_t l)
{
    T* transB = const_cast<T*>(b);
    if constexpr (!isTranspose) {
        transB = static_cast<T*>(malloc(sizeof(T) * n * l));
        for (std::size_t i = 0; i < n; i++) {
            for (std::size_t j = 0; j < l; j++) {
                transB[i * l + j] = b[j * n + i];
            }
        }
    }
    for (std::size_t i = 0; i < m; i++) {
        for (std::size_t j = 0; j < n; j++) {
            T tmp {};
            for (std::size_t k = 0; k < l; k++) {
                tmp += a[i * l + k] * transB[j * l + k];
            }
            c[i * n + j] = tmp;
        }
    }
    if constexpr (isTranspose) {
        free(transB);
    }
}
#endif // CPU_MATMUL_HPP_