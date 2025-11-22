#include "checker.cuh"
#include "conv2d_v1.cuh"
#include "conv2d_v2.cuh"
#include "conv2d_v3.cuh"
// #include "conv2d_v4.cuh"
#include "cpp_two_dim.hpp"
#include "helper.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include <cstddef>
#include <cstdio>

GpuTimer timer;
constexpr size_t M = 4096;
constexpr size_t N = 4096 * 2;
constexpr size_t K = 25;

constexpr size_t BlockSize = 32;

template <typename F>
void tictoc(const std::string& name, F&& f)
{
    timer.start();
    f();
    timer.stop();
    std::cout << name << " time: " << timer.elapsed() << " " << timer.unit() << std::endl;
}
template <typename T>
void print_mat(T* mat, size_t startRow, size_t endRow, size_t startCol, size_t endCol, size_t stride)
{
    for (size_t i = startRow; i < endRow; ++i) {
        for (size_t j = startCol; j < endCol; ++j) {
            std::cout << mat[i * stride + j] << " ";
        }
        std::cout << "\n";
    }
}

template <typename T, bool pad = true>
void gpu_conv2d_v1(
    const T* input, T* output,
    std::size_t m, std::size_t n,
    const T* kernel, std::size_t kWidth)
{
    if constexpr (pad) {
        const size_t halfKSize = kWidth / 2;
        // helper::DeviceDataHandler<T> d_input(input, size + kSize - 1);
        helper::DeviceDataHandler<T> d_input(
            (m + kWidth - 1) * (n + kWidth - 1),
            [&](T* data) {
                for (int i = 0; i < m; ++i) {
                    checkCudaErrors(cudaMemcpy(data + (i + halfKSize) * (n + kWidth - 1) + halfKSize,
                        input + i * n, n * sizeof(T), cudaMemcpyHostToDevice));
                }
            });
        helper::DeviceDataHandler<T> d_kernel(kernel, kWidth * kWidth);
        // checkCudaErrors(cudaMemcpyToSymbol(two_dim::v2::dKernel<T>, kernel, kWidth * kWidth));
        helper::DeviceDataHandler<T> d_output(m * n);
        dim3 block(BlockSize, BlockSize);
        dim3 grid(((n + BlockSize - 1) / BlockSize), ((m + BlockSize - 1) / BlockSize));
        two_dim::conv_kernel_v1_pad<<<grid, block>>>(
            d_input.data, d_output.data, m, n, d_kernel.data, kWidth);
        d_output.cpyToHost(output);
    } else {
        // helper::DeviceDataHandler<T> d_input(input, size);
        // helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        // helper::DeviceDataHandler<T> d_output(size);

        // one_dim::conv_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
        //     d_input.data, d_output.data, size, d_kernel.data, kSize);
        // d_output.cpyToHost(output);
    }
}
template <typename T, bool pad = true>
void gpu_conv2d_v2(
    const T* input, T* output,
    std::size_t m, std::size_t n,
    const T* kernel, std::size_t kWidth)
{
    if constexpr (pad) {
        const size_t halfKSize = kWidth / 2;
        // helper::DeviceDataHandler<T> d_input(input, size + kSize - 1);
        helper::DeviceDataHandler<T> d_input(
            (m + kWidth - 1) * (n + kWidth - 1),
            [&](T* data) {
                for (int i = 0; i < m; ++i) {
                    checkCudaErrors(cudaMemcpy(data + (i + halfKSize) * (n + kWidth - 1) + halfKSize,
                        input + i * n, n * sizeof(T), cudaMemcpyHostToDevice));
                }
            });
        // helper::DeviceDataHandler<T> d_kernel(kernel, kWidth * kWidth);
        checkCudaErrors(cudaMemcpyToSymbol(two_dim::v2::dKernel<T>, kernel, kWidth * kWidth * sizeof(T)));
        helper::DeviceDataHandler<T> d_output(m * n);
        dim3 block(BlockSize, BlockSize);
        dim3 grid(((n + BlockSize - 1) / BlockSize), ((m + BlockSize - 1) / BlockSize));
        two_dim::conv_kernel_v2_pad<<<grid, block>>>(
            d_input.data, d_output.data, m, n, kWidth);
        d_output.cpyToHost(output);
    } else {
        // helper::DeviceDataHandler<T> d_input(input, size);
        // helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        // helper::DeviceDataHandler<T> d_output(size);

        // one_dim::conv_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
        //     d_input.data, d_output.data, size, d_kernel.data, kSize);
        // d_output.cpyToHost(output);
    }
}
template <typename T, bool pad = true>
void gpu_conv2d_v3(
    const T* input, T* output,
    std::size_t m, std::size_t n,
    const T* kernel, std::size_t kWidth)
{
    if constexpr (pad) {
        const size_t halfKSize = kWidth / 2;
        // helper::DeviceDataHandler<T> d_input(input, size + kSize - 1);
        helper::DeviceDataHandler<T> d_input(
            (m + kWidth - 1) * (n + kWidth - 1),
            [&](T* data) {
                for (int i = 0; i < m; ++i) {
                    checkCudaErrors(cudaMemcpy(data + (i + halfKSize) * (n + kWidth - 1) + halfKSize,
                        input + i * n, n * sizeof(T), cudaMemcpyHostToDevice));
                }
            });
        helper::DeviceDataHandler<T> d_kernel(kernel, kWidth * kWidth);
        helper::DeviceDataHandler<T> d_output(m * n);
        dim3 block(BlockSize, BlockSize);
        dim3 grid(((n + BlockSize - 1) / BlockSize), ((m + BlockSize - 1) / BlockSize));
        two_dim::conv_kernel_v3_pad<<<grid, block, kWidth * kWidth * sizeof(T)>>>(
            d_input.data, d_output.data, m, n, d_kernel.data, kWidth);
        d_output.cpyToHost(output);
    } else {
        // helper::DeviceDataHandler<T> d_input(input, size);
        // helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
        // helper::DeviceDataHandler<T> d_output(size);

        // one_dim::conv_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
        //     d_input.data, d_output.data, size, d_kernel.data, kSize);
        // d_output.cpyToHost(output);
    }
}
// template <typename T, bool pad = true>
// void gpu_conv2d_v4(
//     const T* input, T* output,
//     std::size_t m, std::size_t n,
//     const T* kernel, std::size_t kWidth)
// {
//     if constexpr (pad) {
//         const size_t halfKSize = kWidth / 2;
//         // helper::DeviceDataHandler<T> d_input(input, size + kSize - 1);
//         helper::DeviceDataHandler<T> d_input(
//             (m + kWidth - 1) * (n + kWidth - 1),
//             [&](T* data) {
//                 for (int i = 0; i < m; ++i) {
//                     checkCudaErrors(cudaMemcpy(data + (i + halfKSize) * (n + kWidth - 1) + halfKSize,
//                         input + i * n, n * sizeof(T), cudaMemcpyHostToDevice));
//                 }
//             });
//         checkCudaErrors(cudaMemcpyToSymbol(two_dim::v4::dKernel<T>, kernel, kWidth * kWidth * sizeof(T)));

//         // helper::DeviceDataHandler<T> d_kernel(kernel, kWidth * kWidth);
//         helper::DeviceDataHandler<T> d_output(m * n);
//         dim3 block(BlockSize, BlockSize);
//         dim3 grid(((n + BlockSize - 1) / BlockSize), ((m + BlockSize - 1) / BlockSize));
//         size_t sharedMem = (BlockSize + kWidth - 1) * (BlockSize + kWidth - 1) * sizeof(T);
//         two_dim::conv_kernel_v4_pad<<<grid, block, sharedMem>>>(
//             d_input.data, d_output.data, m, n, kWidth);
//         d_output.cpyToHost(output);
//     } else {
//         // helper::DeviceDataHandler<T> d_input(input, size);
//         // helper::DeviceDataHandler<T> d_kernel(kernel, kSize);
//         // helper::DeviceDataHandler<T> d_output(size);

//         // one_dim::conv_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
//         //     d_input.data, d_output.data, size, d_kernel.data, kSize);
//         // d_output.cpyToHost(output);
//     }
// }
int main()
{
    const size_t m = M;
    const size_t n = N;
    const size_t k = K;
    using ValueType = float;
    auto input = helper::generate_sequence<ValueType>(m * n);
    auto kernel = helper::generate_sequence<ValueType>(k * k);
    
    auto outputCPU0 = std::vector<ValueType>(m * n);
    tictoc("2D Conv (CPU) (if)", [&]() {
        two_dim::cpu_convolution(input.data(), outputCPU0.data(), m, n, kernel.data(), k);
    });

    auto outputCPU1 = std::vector<ValueType>(m * n);
    tictoc("2D Conv (CPU) (pad)", [&]() {
        two_dim::cpu_convolution_pad(input.data(), outputCPU1.data(), m, n, kernel.data(), k);
    });
    helper::check_difference(outputCPU0.data(), outputCPU1.data(), m * n);

    auto outputGPU1 = std::vector<ValueType>(m * n);
    tictoc("2D Conv (GPUv1) (pad)", [&]() {
        gpu_conv2d_v1(input.data(), outputGPU1.data(), m, n, kernel.data(), k);
    });
    helper::check_difference(outputCPU0.data(), outputGPU1.data(), m * n);

    auto outputGPU2 = std::vector<ValueType>(m * n);
    tictoc("2D Conv (GPUv2) (pad)", [&]() {
        gpu_conv2d_v2(input.data(), outputGPU2.data(), m, n, kernel.data(), k);
    });
    helper::check_difference(outputCPU0.data(), outputGPU2.data(), m * n);

    auto outputGPU3 = std::vector<ValueType>(m * n);
    tictoc("2D Conv (GPUv3) (pad)", [&]() {
        gpu_conv2d_v3(input.data(), outputGPU3.data(), m, n, kernel.data(), k);
    });
    helper::check_difference(outputCPU0.data(), outputGPU3.data(), m * n);

    // auto outputGPU4 = std::vector<ValueType>(m * n);
    // tictoc("2D Conv (GPUv4) (pad)", [&]() {
    //     gpu_conv2d_v4(input.data(), outputGPU4.data(), m, n, kernel.data(), k);
    // });
    // helper::check_difference(outputCPU0.data(), outputGPU4.data(), m * n);

    std::cout << "\n";
    // print_mat(outputCPU0.data(), 20, 30, 20, 30, n);
    // print_mat(outputCPU1.data(), 20, 30, 20, 30, n);
}