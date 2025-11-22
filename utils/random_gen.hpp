#ifndef RANDOM_GEN_HPP_
#define RANDOM_GEN_HPP_

#include <algorithm>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace helper {

template <typename T>
class UniformRandGenerator {

public:
    UniformRandGenerator(const T& min, const T& max, bool seed = true)
        : m_dist(min, max)
    {
        if (seed) {
            m_engine.seed(time(nullptr));
            // m_engine.seed(1);
        }
        // m_engine.seed(time(nullptr));
    }
    [[nodiscard]] T operator()() { return m_dist(m_engine); }
    void seed(long long newSeed) { m_engine.seed(newSeed); }

private:
    std::default_random_engine m_engine;
    std::conditional_t<std::is_integral_v<T>,
        typename std::uniform_int_distribution<T>,
        typename std::uniform_real_distribution<T>>
        m_dist;
};

template <typename T>
class NormalRandGenerator {

public:
    NormalRandGenerator(const T& mu, const T& sigma, bool seed = true)
        : m_dist(mu, sigma)
    {
        if (seed) {
            m_engine.seed(time(nullptr));
        }
        // m_engine.seed(time(nullptr));
    }
    [[nodiscard]] T operator()() { return m_dist(m_engine); }
    void seed(long long newSeed) { m_engine.seed(newSeed); }

private:
    std::default_random_engine m_engine;

    std::normal_distribution<T> m_dist;
};

template <typename T = float>
std::vector<T> generate_sequence_f(unsigned size, const T mu = static_cast<T>(0.0), const T sigma = static_cast<T>(1.5))
{
    std::vector<T> sequence(size);
    NormalRandGenerator<T> gen(mu, sigma);
    for (unsigned i = 0; i < size; ++i) {
        sequence[i] = gen();
    }
    return sequence;
}
// std::integral_constant<int,1>::value
template <typename T>
std::vector<T> generate_sequence_i(unsigned size,
    const T lower = std::conditional_t<std::is_signed_v<T>,
        std::integral_constant<T, T(-5)>,
        std::integral_constant<T, T(0)>>::value,
    const T upper = std::conditional_t<std::is_signed_v<T>,
        std::integral_constant<T, T(5)>,
        std::integral_constant<T, T(10)>>::value)
{
    std::vector<T> sequence(size);
    UniformRandGenerator<T> gen(lower, upper);
    for (unsigned i = 0; i < size; ++i) {
        sequence[i] = gen();
    }
    return sequence;
}

template <typename T>
std::vector<T> generate_sequence(unsigned size, T arg1 = {}, T arg2 = {})
{
    if constexpr (std::is_integral_v<T>) {
        if (arg2 == T {} && arg1 != T {})
            return generate_sequence_i<T>(size, arg1);
        if (arg1 == T {} && arg2 == T {})
            return generate_sequence_i<T>(size);
        return generate_sequence_i<T>(size, arg1, arg2);
    } else if (std::is_floating_point_v<T>) {
        if (arg2 == T {} && arg1 != T {})
            return generate_sequence_f<T>(size, arg1);
        if (arg1 == T {} && arg2 == T {})
            return generate_sequence_f<T>(size);
        return generate_sequence_f<T>(size, arg1, arg2);
    }
}

template <bool replace = false>
std::vector<std::size_t> generate_random_index(std::size_t max_index, std::size_t len)
{
    if constexpr (replace) {
        std::vector<std::size_t> res;
        UniformRandGenerator<size_t> randG(0, max_index - 1);
        for (int i = 0; i < len; ++i) {
            res.push_back(randG());
        }
        return res;
    } else {
        // std::iota()
        std::vector<std::size_t> res(max_index);
        std::iota(res.begin(), res.end(), 0);
        std::shuffle(res.begin(), res.end(), std::mt19937(std::random_device()()));
        return std::vector<size_t> { res.begin(), res.begin() + len };
    }
}
}

#endif