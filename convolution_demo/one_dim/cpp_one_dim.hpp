#ifndef CPP_ONE_DIM_HPP
#define CPP_ONE_DIM_HPP

#include <cstddef>
namespace one_dim {

// assume kSize is odd
template <typename T>
void cpu_convolution(const T* input, T* output, std::size_t size, const T* kernel, std::size_t kSize)
{

    const size_t halfKSize = kSize / 2;
    for (size_t i = 0; i < size; ++i) {
        T sum {};
        if (i < halfKSize) {
            for (size_t j = halfKSize - i; j < kSize; ++j) {
                sum += input[(i + j) - halfKSize] * kernel[j];
            }
        } else if (i >= size - halfKSize) {
            for (size_t j = 0; j < size - i + halfKSize; ++j) {
                sum += input[i - halfKSize + j] * kernel[j];
            }
        } else {
            for (int j = 0; j < kSize; ++j) {
                sum += kernel[j] * input[i - halfKSize + j];
            }
        }
        output[i] = sum;
    }
}

template <typename T>
void cpu_convolution_pad(const T* input, T* output, std::size_t size, const T* kernel, std::size_t kSize)
{

    const size_t halfKSize = kSize / 2;
    T* newInput = new T[size + kSize - 1] {};
    for (size_t i = 0; i < size; ++i) {
        newInput[i + halfKSize] = input[i];
    }
    for (size_t i = 0; i < size; ++i) {
        T sum {};

        for (int j = 0; j < kSize; ++j) {
            sum += kernel[j] * newInput[i + j];
        }
        output[i] = sum;
    }
    delete[] newInput;
}
}

#endif // CPP_ONE_DIM_HPP