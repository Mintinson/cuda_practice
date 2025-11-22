#ifndef CPP_ONE_DIM_HPP
#define CPP_ONE_DIM_HPP

#include <cstddef>
namespace two_dim {

// assume kSize is odd
template <typename T>
void cpu_convolution(const T* input, T* output, std::size_t m, std::size_t n,
    const T* kernel, std::size_t kWidth)
{

    const size_t halfKSize = kWidth / 2;
    for (size_t i = 0; i < m; ++i) {
        size_t kiBeg = i < halfKSize ? halfKSize - i : 0;
        size_t kiEnd = i >= m - halfKSize ? (m + halfKSize) - i : kWidth;
        for (size_t j = 0; j < n; ++j) {
            T sum {};

            size_t kjBeg = j < halfKSize ? halfKSize - j : 0;
            size_t kjEnd = j >= n - halfKSize ? (n + halfKSize) - j : kWidth;

            for (size_t ki = kiBeg; ki < kiEnd; ++ki) {
                for (size_t kj = kjBeg; kj < kjEnd; ++kj) {
                    sum += input[((i + ki) - halfKSize) * n + (j + kj) - halfKSize] * kernel[ki * kWidth + kj];
                }
            }

            output[i * n + j] = sum;
        }
    }
}

template <typename T>
void cpu_convolution_pad(const T* input, T* output, std::size_t m, std::size_t n,
    const T* kernel, std::size_t kWidth)
{

    const size_t halfKSize = kWidth / 2;
    const size_t newN = n + kWidth - 1;
    T* newInput = new T[(m + kWidth - 1) * (n + kWidth - 1)] {};
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {

            newInput[(i + halfKSize) * newN + (j + halfKSize)] = input[i * n + j];
        }
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum {};
            for (size_t ki = 0; ki < kWidth; ++ki) {
                for (size_t kj = 0; kj < kWidth; ++kj) {
                    sum += kernel[ki * kWidth + kj] * newInput[(i + ki) * newN + (j + kj)];
                }
            }
            output[i * n + j] = sum;
        }
    }
    delete[] newInput;
}
}

#endif // CPP_ONE_DIM_HPP