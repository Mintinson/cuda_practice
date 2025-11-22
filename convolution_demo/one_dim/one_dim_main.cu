#include "checker.cuh"
#include "conv_v1.cuh"
#include "conv_v2.cuh"
#include "conv_v3.cuh"
#include "conv_v4.cuh"
#include "cpp_one_dim.hpp"
#include "helper.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include <cstddef>
#include <cstdio>

GpuTimer timer;
constexpr size_t M = 1 << 25;
constexpr size_t K = 15;

constexpr size_t BlockSize = 256;
// namespace one_dim {
// template <typename T>
// __constant__ T dKernel[BlockSize];
// }

template <typename F>
void tictoc(const std::string& name, F&& f)
{
    timer.start();
    f();
    timer.stop();
    std::cout << name << " time: " << timer.elapsed() << " " << timer.unit() << std::endl;
}

template <typename T, bool pad = false>
void gpu_conv_v1(const T* input, T* output, std::size_t size, const T* kernel, std::size_t kSize)
{
    if constexpr (pad) {
        // helper::DeviceDataHandler<T> d_input(input, size + kSize - 1);
        helper::DeviceDataHandler<T> d_input(size + kSize - 1, [&](T* data) {
            checkCudaErrors(cudaMemcpy(data + kSize / 2, input, size * sizeof(T), cudaMemcpyHostToDevice));
        });
        helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v1_pad<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
            d_input.data, d_output.data, size, d_kernel.data, kSize);
        d_output.cpyToHost(output);
    } else {
        helper::DeviceDataHandler<T> d_input(input, size);
        helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
            d_input.data, d_output.data, size, d_kernel.data, kSize);
        d_output.cpyToHost(output);
    }
}

template <typename T, bool pad = false>
void gpu_conv_v2(const T* input, T* output, std::size_t size, const T* kernel, std::size_t kSize)
{
    if constexpr (pad) {
        helper::DeviceDataHandler<T> d_input(size + kSize - 1, [&](T* data) {
            checkCudaErrors(cudaMemcpy(data + kSize / 2, input, size * sizeof(T), cudaMemcpyHostToDevice));
        });
        checkCudaErrors(cudaMemcpyToSymbol(one_dim::dKernel<T>, kernel, kSize * sizeof(T)));

        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v2_pad<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
            d_input.data, d_output.data, size, kSize);
        d_output.cpyToHost(output);
    } else {
        helper::DeviceDataHandler<T> d_input(input, size);
        // helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        checkCudaErrors(cudaMemcpyToSymbol(one_dim::dKernel<T>, kernel, kSize * sizeof(T)));
        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v2<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
            d_input.data, d_output.data, size, kSize);
        d_output.cpyToHost(output);
    }
}

template <typename T, bool pad = false>
void gpu_conv_v3(const T* input, T* output, std::size_t size, const T* kernel, std::size_t kSize)
{
    if constexpr (pad) {
        helper::DeviceDataHandler<T> d_input(size + kSize - 1, [&](T* data) {
            checkCudaErrors(cudaMemcpy(data + kSize / 2, input, size * sizeof(T), cudaMemcpyHostToDevice));
        });
        helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v3_pad<<<(size + BlockSize - 1) / BlockSize, BlockSize, sizeof(T) * kSize>>>(
            d_input.data, d_output.data, size, d_kernel.data, kSize);
        d_output.cpyToHost(output);
    } else {
        helper::DeviceDataHandler<T> d_input(input, size);
        helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v3<<<(size + BlockSize - 1) / BlockSize, BlockSize, sizeof(T) * kSize>>>(
            d_input.data, d_output.data, size, d_kernel.data, kSize);
        d_output.cpyToHost(output);
    }
}
template <typename T, bool pad = false>
void gpu_conv_v4(const T* input, T* output, std::size_t size, const T* kernel, std::size_t kSize)
{
    if constexpr (pad) {
        helper::DeviceDataHandler<T> d_input(size + kSize - 1, [&](T* data) {
            checkCudaErrors(cudaMemcpy(data + kSize / 2, input, size * sizeof(T), cudaMemcpyHostToDevice));
        });
        checkCudaErrors(cudaMemcpyToSymbol(one_dim::v4::dKernel<T>, kernel, kSize * sizeof(T)));

        // helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v4_pad<<<(size + BlockSize - 1) / BlockSize, BlockSize,
            sizeof(T) * (kSize + BlockSize - 1)>>>(
            d_input.data, d_output.data, size, kSize);
        d_output.cpyToHost(output);
    } else {
        helper::DeviceDataHandler<T> d_input(input, size);
        // helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        checkCudaErrors(cudaMemcpyToSymbol(one_dim::v4::dKernel<T>, kernel, kSize * sizeof(T)));
        helper::DeviceDataHandler<T> d_output(size);

        one_dim::conv_kernel_v4<<<(size + BlockSize - 1) / BlockSize,
            BlockSize, sizeof(T) * (kSize + BlockSize - 1)>>>(
            d_input.data, d_output.data, size, kSize);
        d_output.cpyToHost(output);
    }
}

int main()
{
    for (int i = 1; i < 16; i <<= 1) {
        const size_t m = M * i;
        const size_t k = K;
        std::cout << "m: " << m << ", k: " << k << std::endl;
        using ValueType = float;
        auto input = helper::generate_sequence<ValueType>(m);
        auto kernel = helper::generate_sequence<ValueType>(k);

        auto outputCPU0 = decltype(input)(m);
        auto outputCPU1 = decltype(input)(m);

        tictoc("Conv (CPU) (if)", [&]() {
            one_dim::cpu_convolution(input.data(), outputCPU0.data(),
                input.size(), kernel.data(), kernel.size());
        });
        tictoc("Conv (CPU) (pad)", [&]() {
            one_dim::cpu_convolution_pad(input.data(), outputCPU1.data(),
                input.size(), kernel.data(), kernel.size());
        });
        helper::check_difference(outputCPU0.data(), outputCPU1.data(), outputCPU0.size());

        auto outputGPU1 = decltype(input)(m);
        tictoc("Conv (GPUv1) (if)", [&]() {
            gpu_conv_v1(input.data(), outputGPU1.data(),
                input.size(), kernel.data(), kernel.size());
        });
        helper::check_difference(outputCPU0.data(), outputGPU1.data(), outputCPU0.size());

        auto outputGPU1_2 = decltype(input)(m);
        tictoc("Conv (GPUv1) (pad)", [&]() {
            gpu_conv_v1<ValueType, true>(input.data(), outputGPU1_2.data(),
                input.size(), kernel.data(), kernel.size());
        });
        helper::check_difference(outputCPU0.data(), outputGPU1_2.data(), outputCPU0.size());

        auto outputGPU2 = decltype(input)(m);
        tictoc("Conv (GPUv2) (if)", [&]() {
            gpu_conv_v2(input.data(), outputGPU2.data(),
                input.size(), kernel.data(), kernel.size());
        });
        helper::check_difference(outputCPU0.data(), outputGPU2.data(), outputCPU0.size());

        auto outputGPU2_2 = decltype(input)(m);
        tictoc("Conv (GPUv2) (pad)", [&]() {
            gpu_conv_v2<ValueType, true>(input.data(), outputGPU2_2.data(),
                input.size(), kernel.data(), kernel.size());
        });
        helper::check_difference(outputCPU0.data(), outputGPU2_2.data(), outputCPU0.size());

        auto outputGPU3 = decltype(input)(m);
        tictoc("Conv (GPUv3) (if)", [&]() {
            gpu_conv_v3(input.data(), outputGPU3.data(),
                input.size(), kernel.data(), kernel.size());
        });
        helper::check_difference(outputCPU0.data(), outputGPU3.data(), outputCPU0.size());

        auto outputGPU3_2 = decltype(input)(m);
        tictoc("Conv (GPUv3) (pad)", [&]() {
            gpu_conv_v3<ValueType, true>(input.data(), outputGPU3_2.data(),
                input.size(), kernel.data(), kernel.size());
        });
        helper::check_difference(outputCPU0.data(), outputGPU3_2.data(), outputCPU0.size());

        auto outputGPU4 = decltype(input)(m);
        tictoc("Conv (GPUv4) (if)", [&]() {
            gpu_conv_v4(input.data(), outputGPU4.data(),
                input.size(), kernel.data(), kernel.size());
        });
        helper::check_difference(outputCPU0.data(), outputGPU4.data(), outputCPU0.size());

        auto outputGPU4_2 = decltype(input)(m);
        tictoc("Conv (GPUv4) (pad)", [&]() {
            gpu_conv_v4<ValueType, true>(input.data(), outputGPU4_2.data(),
                input.size(), kernel.data(), kernel.size());
        });
        helper::check_difference(outputCPU0.data(), outputGPU4_2.data(), outputCPU0.size());

        // for (int i = 0; i < 30; ++i) {
        //     std::cout << outputCPU0[i] << " " << outputGPU4[0] << "\n";
        // }

        std::cout << "\n";
    }
}