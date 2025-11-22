#include "checker.cuh"
#include "random_gen.hpp"
// #include "timer.cuh"
#include "helper.cuh"
#include "naive.cuh"
#include "vec2_optimize.cuh"
#include "vec4_optimize.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
constexpr size_t N = 1024 * 1024 * 64;
constexpr int BlockDim = 128;

#ifdef _DEBUG
constexpr const char* const BuildType = "Debug";
#else
#ifndef NDEBUG
constexpr const char* const BuildType = "Debug";
#else
constexpr const char* const BuildType = "Release";
#endif
#endif

GpuTimer timer;

template <typename T, typename Operator>
void naive_element_wise(const T* input_a, const T* input_b, T* output, size_t size, Operator oper)
{
    helper::DeviceDataHandler d_input_a(input_a, size);
    helper::DeviceDataHandler d_input_b(input_b, size);
    helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    element_wise_naive_kernel<<<(size + BlockDim - 1) / BlockDim, BlockDim>>>(
        d_input_a.data, d_input_b.data, d_output.data, size, oper);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "GPU (naive): " << timer.elapsed() << "" << timer.unit() << std::endl;

    d_output.cpyToHost(output);
}
template <typename T, typename Operator>
void vec2_element_wise(const T* input_a, const T* input_b, T* output, size_t size, Operator oper)
{
    helper::DeviceDataHandler d_input_a(input_a, size);
    helper::DeviceDataHandler d_input_b(input_b, size);
    helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    vec2_element_wise_kernel<<<(size + BlockDim - 1) / BlockDim / 2, BlockDim>>>(
        d_input_a.data, d_input_b.data, d_output.data, size, oper);
    cudaDeviceSynchronize();

    timer.stop();
    std::cout << "GPU (vec2): " << timer.elapsed() << "" << timer.unit() << std::endl;

    d_output.cpyToHost(output);
}
template <typename T, typename Operator>
void vec4_element_wise(const T* input_a, const T* input_b, T* output, size_t size, Operator oper)
{
    helper::DeviceDataHandler d_input_a(input_a, size);
    helper::DeviceDataHandler d_input_b(input_b, size);
    helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    vec4_element_wise_kernel<<<(size + BlockDim - 1) / BlockDim / 4, BlockDim>>>(
        d_input_a.data, d_input_b.data, d_output.data, size, oper);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "GPU (vec4): " << timer.elapsed() << "" << timer.unit() << std::endl;
    d_output.cpyToHost(output);
}

enum class CUBLASOperationType {
    Aad,
    Subtract
};

template <typename T, CUBLASOperationType Operator>
void cublas_element_wise(const T* input_a, const T* input_b, T* output, size_t size)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "Cublas only supports float and double");
    cublasHandle_t handle;
    cublasCreate(&handle);
    T* d_input_a;
    T* d_input_b;
    // T* d_output;
    checkCudaErrors(cudaMalloc((void**)&d_input_a, size * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_input_b, size * sizeof(T)));
    cublasSetVector(size, sizeof(T), input_a, 1, d_input_a, 1);
    cublasSetVector(size, sizeof(T), input_b, 1, d_input_b, 1);
    if constexpr (Operator == CUBLASOperationType::Aad) {
        T alpha = static_cast<T>(1.0);
        timer.start();
        // y[i] = alpha * x[i] + y[i]
        if constexpr (std::is_same_v<T, float>) {
            cublasSaxpy_v2(handle, size, &alpha, d_input_a, 1, d_input_b, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cublasDaxpy_v2(handle, size, &alpha, d_input_a, 1, d_input_b, 1);
        }
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "GPU (cublas): " << timer.elapsed() << "" << timer.unit() << std::endl;
        cublasGetVector(size, sizeof(T), d_input_b, 1, output, 1);
    } else if constexpr (Operator == CUBLASOperationType::Subtract) {
        T alpha = static_cast<T>(-1.0);
        timer.start();
        // x[i] = -alpha * y[i] + x[i]
        if constexpr (std::is_same_v<T, float>) {
            cublasSaxpy_v2(handle, size, &alpha, d_input_b, 1, d_input_a, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cublasDaxpy_v2(handle, size, &alpha, d_input_b, 1, d_input_a, 1);
        }
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "GPU (cublas): " << timer.elapsed() << "" << timer.unit() << std::endl;
        cublasGetVector(size, sizeof(T), d_input_a, 1, output, 1);
    }
    checkCudaErrors(cudaFree(d_input_a));
    checkCudaErrors(cudaFree(d_input_b));
    cublasDestroy(handle);
}

int main()
{
    for (int i = 1; i < 8 + 1; ++i) {

        using ValueType = float;
        std::cout << "Build type: " << BuildType << " with element: " << N * i << "\n";
        auto randVec = helper::generate_sequence<float>(N * i);
        auto randVec2 = helper::generate_sequence<float>(randVec.size());
        auto cpuRes = decltype(randVec)(randVec.size());
        auto cpuRes2 = decltype(randVec)(randVec.size());
        auto gpuNaive = decltype(randVec)(randVec.size());
        auto gpuVec2 = decltype(randVec)(randVec.size());
        auto gpuVec4 = decltype(randVec)(randVec.size());
        auto gpuCublas = decltype(randVec)(randVec.size());

        auto oper = std::minus<ValueType> {};

        timer.start();
        for (size_t i = 0; i < randVec.size(); ++i) {
            cpuRes[i] = oper(randVec[i], randVec2[i]);
        }
        timer.stop();
        std::cout << "CPU (raw loop): " << timer.elapsed() << "" << timer.unit() << std::endl;

        timer.start();
        std::transform(randVec.cbegin(), randVec.cend(), randVec2.cbegin(), cpuRes2.begin(), oper);
        timer.stop();
        std::cout << "CPU (std::transform): " << timer.elapsed() << "" << timer.unit() << std::endl;
        helper::check_difference(cpuRes.data(), cpuRes2.data(), cpuRes.size());

        naive_element_wise(randVec.data(), randVec2.data(), gpuNaive.data(), randVec.size(), oper);
        helper::check_difference(cpuRes.data(), gpuNaive.data(), cpuRes.size());

        vec2_element_wise(randVec.data(), randVec2.data(), gpuVec2.data(), randVec.size(), oper);
        helper::check_difference(cpuRes.data(), gpuVec2.data(), cpuRes.size());

        vec4_element_wise(randVec.data(), randVec2.data(), gpuVec4.data(), randVec.size(), oper);

        helper::check_difference(cpuRes.data(), gpuVec4.data(), cpuRes.size());

        cublas_element_wise<ValueType, CUBLASOperationType::Subtract>(
            randVec.data(), randVec2.data(), gpuCublas.data(), randVec.size());
        helper::check_difference(cpuRes.data(), gpuCublas.data(), cpuRes.size());

        thrust::device_vector<ValueType> d_input_a(randVec);
        thrust::device_vector<ValueType> d_input_b(randVec2);
        thrust::device_vector<ValueType> d_output(randVec.size());
        timer.start();
        thrust::transform(
            d_input_a.begin(), d_input_a.end(), d_input_b.begin(), d_output.begin(), oper);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "GPU (thrust): " << timer.elapsed() << "" << timer.unit() << std::endl;
        thrust::host_vector<ValueType> gpuThrust = d_output;
        helper::check_difference(cpuRes.data(), gpuThrust.data(), cpuRes.size());

        std::cout << "\n";
    }
}