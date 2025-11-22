#ifndef VEC2_OPTIMIZE_H_
#define VEC2_OPTIMIZE_H_

#include <type_traits>
template <typename T>
__device__ constexpr auto& fetch_vec2(T* ptr)
{
    using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<DecayType, float>) {
        return reinterpret_cast<float2*>((ptr))[0];
    } else if constexpr (std::is_same_v<DecayType, double>) {
        return reinterpret_cast<double2*>((ptr))[0];

    } else if constexpr (std::is_same_v<DecayType, int>) {
        return reinterpret_cast<int2*>((ptr))[0];
    } else if constexpr (std::is_same_v<DecayType, unsigned int>) {
        return reinterpret_cast<uint2*>((ptr))[0];
    }
}

template <typename T, typename Operator>
__global__ void vec2_element_wise_kernel(T* d_a, T* d_b, T* d_out, size_t n, Operator op)
{
    auto idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    // c[idx] = a[idx] + b[idx];
    auto reg_a = fetch_vec2(d_a + idx);
    auto reg_b = fetch_vec2(d_b + idx);
    std::remove_reference_t<decltype(reg_a)> reg_out;
    reg_out.x = op(reg_a.x, reg_b.x);
    reg_out.y = op(reg_a.y, reg_b.y);

    fetch_vec2(d_out + idx) = reg_out;
}

#endif // VEC2_OPTIMIZE_H_