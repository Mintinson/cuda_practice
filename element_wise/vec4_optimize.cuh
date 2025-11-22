#ifndef VEC4_OPTIMIZE_H_
#define VEC4_OPTIMIZE_H_

#include <type_traits>
template <typename T>
__device__ constexpr auto& fetch_vec4(T* ptr)
{
    using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<DecayType, float>) {
        return reinterpret_cast<float4*>((ptr))[0];
    } else if constexpr (std::is_same_v<DecayType, double>) {
        return reinterpret_cast<double4*>((ptr))[0];

    } else if constexpr (std::is_same_v<DecayType, int>) {
        return reinterpret_cast<int4*>((ptr))[0];
    } else if constexpr (std::is_same_v<DecayType, unsigned int>) {
        return reinterpret_cast<uint4*>((ptr))[0];
    }
}
template <typename T, typename Operator>
__global__ void vec4_element_wise_kernel(T* d_a, T* d_b, T* d_out, size_t n, Operator op)
{
    auto idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    // c[idx] = a[idx] + b[idx];
    auto reg_a = fetch_vec4(d_a + idx);
    auto reg_b = fetch_vec4(d_b + idx);
    decltype(reg_a) reg_out;
    reg_out.x = op(reg_a.x, reg_b.x);
    reg_out.y = op(reg_a.y, reg_b.y);
    reg_out.z = op(reg_a.z, reg_b.z);
    reg_out.w = op(reg_a.w, reg_b.w);
    fetch_vec4(d_out + idx) = reg_out;
}

#endif // VEC4_OPTIMIZE_H_