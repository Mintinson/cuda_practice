
#include "checker.cuh"
#include "helper.cuh"
#include "random_gen.hpp"
#include <cstdio>
#include <type_traits>

constexpr int N = 1024 * 1024 * 32;

template <typename T>
__global__ void add_kernel(T *d_vecA, T *d_vecB, T *d_vecC)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    d_vecC[idx] = d_vecA[idx] + d_vecB[idx];
}
template <typename T>
__device__ constexpr auto &fetch_vec2(T *ptr)
{
    using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<DecayType, float>)
    {
        return reinterpret_cast<float2 *>((ptr))[0];
    }
    else if constexpr (std::is_same_v<DecayType, double>)
    {
        return reinterpret_cast<double2 *>((ptr))[0];
    }
    else if constexpr (std::is_same_v<DecayType, int>)
    {
        return reinterpret_cast<int2 *>((ptr))[0];
    }
    else if constexpr (std::is_same_v<DecayType, unsigned int>)
    {
        return reinterpret_cast<uint2 *>((ptr))[0];
    }
}

template <typename T, typename Operator = std::plus<>>
__global__ void vec2_element_wise_kernel(T *d_a, T *d_b, T *d_out, size_t n, Operator op = {})
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
template <typename T>
__device__ constexpr auto &fetch_vec4(T *ptr)
{
    using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<DecayType, float>)
    {
        return reinterpret_cast<float4 *>((ptr))[0];
    }
    else if constexpr (std::is_same_v<DecayType, double>)
    {
        return reinterpret_cast<double4 *>((ptr))[0];
    }
    else if constexpr (std::is_same_v<DecayType, int>)
    {
        return reinterpret_cast<int4 *>((ptr))[0];
    }
    else if constexpr (std::is_same_v<DecayType, unsigned int>)
    {
        return reinterpret_cast<uint4 *>((ptr))[0];
    }
}
template <typename T, typename Operator = std::plus<>>
__global__ void vec4_element_wise_kernel(T *d_a, T *d_b, T *d_out, size_t n, Operator op = {})
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

template <typename T>
__global__ void stall_reason_lsb(T *data)
{
    int tid = threadIdx.x;
    int laneId = tid % 32;
    data[laneId] = laneId;

    __syncthreads();

    auto idx = laneId;

    for (int i = 0; i < 100; ++i)
    {
        idx = data[idx];
    }
    data[laneId] = idx;
}
template <typename T>
__global__ void stall_reason_lgt(T *data, T *out)
{
    int tid = threadIdx.x;
    int offset = tid * 100;
#pragma unroll
    for (int i = 0; i < 200; ++i)
    {
        out[offset + i] = data[i + offset];
    }
}
template <typename T>
__global__ void stall_reason_ssb(T *data)
{
    __shared__ T smm[32];

    int tid = threadIdx.x;
    int laneId = tid % 32;

    smm[laneId] = laneId;
    __syncthreads();

    int idx = laneId;
    for (int i = 0; i < 100; ++i)
    {
        idx = smm[idx];
    }
    data[laneId] = idx;
}
// bank conflict
template <typename T>
__global__ void stall_reason_mio_bad(T *data)
{
    __shared__ T smm[32][32];
    __shared__ T smm2[32][32];

    int tid = threadIdx.x;
    int laneId = tid % 32;

#pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        smm2[laneId][i] = smm[laneId][i];
    }
    __syncthreads();
}
template <typename T>
__global__ void stall_reason_mio_good(T *data)
{
    __shared__ T smm[32][32];
    __shared__ T smm2[32][32];

    int tid = threadIdx.x;
    int laneId = tid % 32;

#pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        smm2[i][laneId] = smm[i][laneId];
    }
    __syncthreads();
}
__global__ void warm_up_gpu()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

int main()
{
    auto vecA = helper::generate_sequence<float>(N);
    auto vecB = helper::generate_sequence<float>(N);

    auto vecC = decltype(vecA)(vecA.size());

    using ValueType = decltype(vecA)::value_type;

    ValueType *d_vecA, *d_vecB, *d_vecC;
    checkCudaErrors(cudaMalloc((void **)&d_vecA, vecA.size() * sizeof(ValueType)));
    checkCudaErrors(cudaMalloc((void **)&d_vecB, vecB.size() * sizeof(ValueType)));
    checkCudaErrors(cudaMalloc((void **)&d_vecC, vecC.size() * sizeof(ValueType)));

    checkCudaErrors(cudaMemcpy(d_vecA, vecA.data(), vecA.size() * sizeof(ValueType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vecB, vecB.data(), vecB.size() * sizeof(ValueType), cudaMemcpyHostToDevice));
    warm_up_gpu<<<512, 1024>>>();
    for (int i = 0; i < 1; ++i)
    {
        dim3 block(64);
        dim3 grid(N / (64 * 4));
        add_kernel<<<grid, block>>>(d_vecA, d_vecB, d_vecC);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < 1; ++i)
    {
        dim3 block(64);
        dim3 grid(N / (64 * 4) / 2);
        vec2_element_wise_kernel<<<grid, block>>>(d_vecA, d_vecB, d_vecC, N / 4);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < 1; ++i)
    {
        dim3 block(64);
        dim3 grid(N / (64 * 4) / 4);
        vec4_element_wise_kernel<<<grid, block>>>(d_vecA, d_vecB, d_vecC, N / 4);
        cudaDeviceSynchronize();
    }

    auto vecI = helper::generate_sequence<int>(N / 4, 0, N / 4 - 1);
    auto vecI2 = helper::generate_sequence<int>(N / 4, 0, N / 4 - 1);

    helper::DeviceDataHandler dvecI(vecI.data(), N / 4);
    helper::DeviceDataHandler dvecI2(vecI2.data(), N / 4);
    dim3 block(64);
    dim3 grid(N / (64) / 4);
    stall_reason_lsb<<<grid, block>>>(dvecI.data);
    dvecI.cpyToHost(vecI.data());

    stall_reason_lgt<<<grid, block>>>(dvecI.data, dvecI2.data);

    stall_reason_ssb<<<grid, block>>>(dvecI.data);

    stall_reason_mio_bad<<<grid, block>>>(dvecI.data);
    stall_reason_mio_good<<<grid, block>>>(dvecI.data);
}