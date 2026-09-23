#ifndef CPU_SOFTMAX_HPP_
#define CPU_SOFTMAX_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>
#include <numeric>
#include <execution>

template <typename T>
void softmax_naive_row(const T *a, T *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i];
    }
    T max_val = *std::max_element(c, c + n);
    T sum{};
    for (int i = 0; i < n; i++)
    {
        c[i] -= max_val;
        c[i] = std::exp(c[i]);
        sum += c[i];
    }
    for (int i = 0; i < n; i++)
    {
        c[i] /= sum;
    }
}

template <typename T>
void softmax_naive(const T *a, T *c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        softmax_naive_row(a + i * n, c + i * n, n);
    }
}

template <typename T>
void softmax_combine_row(const T *a, T *c, int n)
{

    T max_val = std::numeric_limits<T>::min();
    T sum{};
    for (int i = 0; i < n; i++)
    {

        T curr = a[i];
        if (curr > max_val)
        {
            sum = sum * std::exp(max_val - curr);
            max_val = curr;
        }
        sum += std::exp(curr - max_val);
    }
    for (int i = 0; i < n; i++)
    {
        c[i] = std::exp(a[i] - max_val) / sum;
    }
}

template <typename T>
void softmax_threads(const T *a, T *c, int m, int n)
{
    std::vector<std::thread> threads;
    threads.reserve(m);
    for (int i = 0; i < m; i++)
    {
        // auto thread = std::thread(softmax_naive_row<T>, a + i * n, c + i * n, n);
        threads.emplace_back(softmax_naive_row<T>, a + i * n, c + i * n, n);
    }
    for (auto &t : threads)
    {
        t.join();
    }
}
template <typename T>
void softmax_concurrence(const std::vector<std::size_t> &indices, const T *a, T *c, int m, int n)
{
    // std::vector<int> a_rows(m);
    // std::iota(a_rows.begin(), a_rows.end(), 0);
    std::for_each(
        std::execution::par_unseq,
        indices.begin(), indices.end(), [a, c, n](std::size_t i)
        { softmax_naive_row<T>(a + i * n, c + i * n, n); });
}

template <typename T>
void softmax_combine_concurrence(const std::vector<std::size_t> &indices, const T *a, T *c, int m, int n)
{
    // std::vector<int> a_rows(m);
    // std::iota(a_rows.begin(), a_rows.end(), 0);
    std::for_each(
        std::execution::par_unseq,
        indices.begin(), indices.end(), [a, c, n](std::size_t i)
        { softmax_combine_row<T>(a + i * n, c + i * n, n); });
}

#endif // CPU_SOFTMAX_HPP_