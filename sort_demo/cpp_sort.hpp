#ifndef CPP_SORT_HPP
#define CPP_SORT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <utility>

namespace cpp_sort
{
    template <typename Iterator, typename Compare = std::less<>>
    void select_sort(Iterator arr, std::size_t n, Compare comp = {})
    {
        using T = typename std::iterator_traits<Iterator>::value_type;

        for (std::size_t i = 0; i < n; ++i)
        {
            T tmp = *(arr + i);
            std::size_t id = i;
            for (std::size_t j = i + 1; j < n; ++j)
            {
                if (comp(*(arr + j), tmp))
                {
                    tmp = *(arr + j);
                    id = j;
                }
            }
            std::swap(*(arr + i), *(arr + id));
        }
    }

    template <typename Iterator, typename Compare = std::less<>>
    void bubble_sort(Iterator arr, std::size_t n, Compare comp = {})
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        for (std::size_t i = n - 1; i > 0; --i)
        {
            for (std::size_t j = 0; j < i; ++j)
            {
                if (comp(*(arr + j + 1), *(arr + j)))
                {
                    std::swap(*(arr + j), *(arr + j + 1));
                }
            }
        }
    }

    template <typename Iterator, typename Compare = std::less<>>
    void quick_sort(Iterator arr, std::size_t n, Compare comp = {})
    {
        if (n <= 1)
            return;

        // std::vector<std::pair<std::size_t, std::size_t>> ranges;
        using PairType = int;
        std::vector<std::pair<PairType, PairType>> ranges;
        ranges.reserve(64);
        ranges.emplace_back(0, static_cast<PairType>(n - 1));

        while (!ranges.empty())
        {
            auto [beg, end] = ranges.back();
            ranges.pop_back();

            if (beg >= end)
                continue;


            auto pivotal = arr[end];
            auto left = beg;
            auto right = end;


            while (left < right)
            {
                while (comp(arr[left], pivotal) && left < right)
                    left++;
                while (!comp(arr[right], pivotal) && left < right)
                    right--;
                if (left < right)
                    std::swap(arr[left], arr[right]);
            }


            if (!comp(arr[left], arr[end]))
                std::swap(arr[left], arr[end]);
            else
                left++;


            if (left - 1 > beg)
                ranges.emplace_back(beg, left - 1);
            if (end > left + 1)
                ranges.emplace_back(left + 1, end);
        }

        // if (n == 0)
        //     return;
        // std::vector<std::pair<int, int>> ranges;
        // ranges.reserve(64);
        // // ranges.reserve(n);
        // int ridx = 0;
        // ranges.emplace_back(0, n - 1);
        // ridx++;
        // // ranges[ridx++] = { 0, n - 1 };
        // while (ridx) {
        //     auto [beg, end] = ranges[--ridx];
        //     ranges.pop_back();
        //     if (beg >= end)
        //         continue;
        //     auto pivotal = arr[end];
        //     auto left = beg;
        //     auto right = end - 1;
        //     while (left < right) {
        //         while (comp(arr[left], pivotal) && left < right) {
        //             left++;
        //         }
        //         while (!comp(arr[right], pivotal) && left < right) {
        //             right--;
        //         }
        //         // if (left < right) {
        //         std::swap(arr[left], arr[right]);
        //         // }
        //     }
        //     if (!comp(arr[left], arr[end]))
        //         std::swap(arr[left], arr[end]);
        //     else
        //         left++;
        //     ranges.emplace_back(beg, left - 1);
        //     ranges.emplace_back(left + 1, end);
        //     ridx++;
        //     ridx++;
        //     // ranges[ridx++] = { beg, left - 1 };
        //     // ranges[ridx++] = { left + 1, end };
        // }
    }

    template <typename Iterator, typename Compare = std::less<>>
    void quick_sort_recursive(Iterator arr, std::size_t start, std::size_t end, Compare comp = {})
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        auto i = start;
        auto j = end;
        auto pivotal = arr[start];
        while (i < j)
        {
            while (!comp(arr[j], pivotal) && i < j)
            {
                --j;
            }
            arr[i] = arr[j];
            while ((comp(arr[i], pivotal) || arr[i] == pivotal) && i < j)
            {
                ++i;
            }
            arr[j] = arr[i];
        }
        arr[i] = pivotal;
        if (i - 1 > start)
        {
            quick_sort_recursive(arr, start, i - 1, comp);
        }
        if (i + 1 < end)
        {
            quick_sort_recursive(arr, i + 1, end, comp);
        }
    }

    template <typename Iterator, typename Compare = std::less<>>
    void quick_sort_recursive(Iterator arr, std::size_t n, Compare comp = {})
    {
        if (n <= 1)
            return;
        quick_sort_recursive(arr, 0, n - 1, comp);
    }
    namespace details
    {
        // UnsignedBits TwiddleIn(UnsignedBits key)
        // {
        //     static const UnsignedBits HIGH_BIT = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
        //     UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
        //     return key ^ mask;
        // };
    }

    template <typename Iterator, bool Ascend = true>
    void radix_sort(Iterator arr, std::size_t n)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        // static_assert(std::is_integral_v<T>, "Radix sort only works with integral types.");
        if constexpr (std::is_integral_v<T>)
        {
            if constexpr (std::is_unsigned_v<T>)
            {
                constexpr int num_bits = sizeof(T) * 8;
                // T* arr =
                for (int bit = 0; bit < num_bits; ++bit)
                {
                    auto second_part = std::stable_partition(arr, arr + n, [bit](T value)
                    {
                        return Ascend ^ ((value & (T(1) << bit)) != 0);
                    }); // Elements with the bit not set come first
                }
            }
            else
            {
                constexpr int num_bits = sizeof(T) * 8;
                for (int bit = 0; bit < num_bits - 1; ++bit)
                {
                    auto second_part = std::stable_partition(arr, arr + n, [bit](T value)
                    {
                        return Ascend ^ ((value & (T(1) << bit)) != 0);
                    }); // Elements with the bit not set come first
                }
                auto second_part = std::stable_partition(arr, arr + n, [bit = num_bits - 1](T value)
                {
                    return (!Ascend) ^ ((value & (T(1) << bit)) != 0);
                }); // Negative numbers come first
            }
        }
        else if constexpr (std::is_same_v<std::remove_reference_t<std::remove_cv_t<T>>, float>)
        {
            constexpr int num_bits = sizeof(T) * 8;
            for (int bit = 0; bit < num_bits - 1; ++bit)
            {
                auto second_part = std::stable_partition(arr, arr + n, [bit](T value)
                {
                    return Ascend ^ (((*reinterpret_cast<unsigned*>(&value)) & (unsigned(1) << bit)) != 0);
                }); // Elements with the bit not set come first
            }
            auto second_part = std::stable_partition(arr, arr + n, [bit = num_bits - 1](T value)
            {
                return (!Ascend) ^ (((*reinterpret_cast<unsigned*>(&value)) & (unsigned(1) << bit)) != 0);
            }); // Negative numbers come first
        }
    }

    template <typename Iterator, typename Compare = std::less<>>
    void bitonic_sort(Iterator arr, std::size_t n, Compare comp = {})
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        for (size_t k = 2; k <= n; k <<= 1)
        {
            for (size_t j = k >> 1; j > 0; j >>= 1)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    auto ixj = i ^ j;
                    if (ixj > i)
                    {
                        if ((i & k) == 0 && comp(*(arr + ixj), *(arr + i)))
                        {
                            std::swap(*(arr + i), *(arr + ixj));
                        }
                        if ((i & k) != 0 && comp(*(arr + i), *(arr + ixj)))
                        {
                            std::swap(*(arr + i), *(arr + ixj));
                        }
                    }
                }
            }
        }
    }

    namespace details
    {
        template <typename Iterator, typename Compare = std::less<>>
        void merge(Iterator arr, std::size_t low, std::size_t mid, std::size_t high, Compare comp = {})
        {
            using T = typename std::iterator_traits<Iterator>::value_type;
            T* tmp = static_cast<T*>(malloc(sizeof(T) * (high - low + 1)));
            size_t i = low;
            size_t j = mid + 1;
            size_t k = 0;
            while (i <= mid && j <= high)
            {
                tmp[k++] = comp(arr[j], arr[i]) ? arr[j++] : arr[i++];
            }
            while (i <= mid)
            {
                tmp[k++] = arr[i++];
            }
            while (j <= high)
            {
                tmp[k++] = arr[j++];
            }
            for (k = 0, i = low; i <= high; ++i)
            {
                arr[i] = tmp[k++];
            }
            free(tmp);
        }

        template <typename Iterator, typename Compare = std::less<>>
        void merge_sort(Iterator arr, std::size_t low, std::size_t high, Compare comp = {})
        {
            if (low < high)
            {
                auto mid = low + (high - low) / 2;
                merge_sort(arr, low, mid, comp);
                merge_sort(arr, mid + 1, high, comp);
                merge(arr, low, mid, high, comp);
            }
        }
    }

    template <typename Iterator, typename Compare = std::less<>>
    void merge_sort(Iterator arr, std::size_t n, Compare comp = {})
    {
        details::merge_sort(arr, 0, n - 1, comp);
    }
}

#endif // CPP_SORT_HPP
