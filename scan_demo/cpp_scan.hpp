#ifndef CPP_SCAN_HPP
#define CPP_SCAN_HPP

#include <cstddef>
#include <functional>

template <typename T, typename Binary = std::plus<>>
void inclusive_prescan(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    output[0] = binary(init, input[0]);
    for (std::size_t i = 1; i < size; ++i) {
        output[i] = binary(output[i - 1], input[i]);
    }
}

template <typename T, typename Binary = std::plus<>>
void block_inclusive_prescan(const T* input, std::size_t size, T* output, std::size_t blockSize,
    T init = {}, Binary binary = {})
{
    for (std::size_t i = 0; i < size; i += blockSize) {
        output[i] = binary(init, input[i]);
        for (int k = 1; k < blockSize; ++k) {
            output[i + k] = binary(output[i + k - 1], input[i + k]);
        }
    }
}
template <typename T, typename Binary = std::plus<>>
void exclusive_prescan(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    output[0] = init;
    for (std::size_t i = 1; i < size; ++i) {
        output[i] = binary(output[i - 1], input[i - 1]);
    }
}

#endif // CPP_SCAN_HPP