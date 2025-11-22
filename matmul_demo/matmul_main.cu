#include "checker.cuh"
#include "cpu_matmul.hpp"
#include "matmul_v0.cuh"
#include "matmul_v1.cuh"
#include "matmul_v2.cuh"
#include "matmul_v4.cuh"
#include "matmul_v5.cuh"
#include "matmul_v6.cuh"
#include "matmul_v7.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper.cuh>
#include <type_traits>
#include <vector>

GpuTimer timer;
constexpr size_t M = 2048;
constexpr size_t N = 1024;
constexpr size_t L = 1024;
constexpr size_t BlockSize = 256;

template <typename T>
void matmul_gpu_v0(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    helper::DeviceDataHandler d_input_a(A, row * depth);
    helper::DeviceDataHandler d_input_b(B, depth * col);
    helper::DeviceDataHandler<T> d_output(row * col);

    matmul_v0_kernel<<<(row * col + BlockSize - 1) / BlockSize, BlockSize>>>(
        d_input_a.data, d_input_b.data, d_output.data, row, col, depth);
    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}

template <typename T>
void matmul_gpu_v1(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    helper::DeviceDataHandler d_input_a(A, row * depth);
    helper::DeviceDataHandler d_input_b(B, depth * col);
    helper::DeviceDataHandler<T> d_output(row * col);

    matmul_v1_kernel<<<row, BlockSize>>>(
        d_input_a.data, d_input_b.data, d_output.data, row, col, depth);
    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}

template <typename T>
void matmul_gpu_v2(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    helper::DeviceDataHandler d_input_a(A, row * depth);
    helper::DeviceDataHandler d_input_b(B, depth * col);
    helper::DeviceDataHandler<T> d_output(row * col);

    matmul_v2_kernel<<<row, BlockSize, depth * sizeof(T)>>>(
        d_input_a.data, d_input_b.data, d_output.data, row, col, depth);
    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}

template <typename T>
void matmul_gpu_v3(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    size_t pitchA;
    size_t pitchB;
    size_t pitchC;
    T* d_a;
    T* d_b;
    T* d_c;
    checkCudaErrors(cudaMallocPitch(&d_a, &pitchA, depth * sizeof(T), row));
    checkCudaErrors(cudaMallocPitch(&d_b, &pitchB, col * sizeof(T), depth));
    checkCudaErrors(cudaMallocPitch(&d_c, &pitchC, col * sizeof(T), row));
    checkCudaErrors(cudaMemcpy2D(d_a, pitchA, A, depth * sizeof(T), depth * sizeof(T), row,
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_b, pitchB, B, col * sizeof(T), col * sizeof(T), depth,
        cudaMemcpyHostToDevice));
    matmul_v2_kernel<<<row, BlockSize, depth * sizeof(T)>>>(d_a, d_b, d_c,
                                                            row, pitchC / sizeof(T), pitchA / sizeof(T));
    checkCudaErrors(cudaMemcpy2D(C, col * sizeof(T), d_c, pitchC, col * sizeof(T), row,
        cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

template <typename T>
void matmul_gpu_v4(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    helper::DeviceDataHandler d_input_a(A, row * depth);
    helper::DeviceDataHandler d_input_b(B, depth * col);
    helper::DeviceDataHandler<T> d_output(row * col);
    const size_t blockSize = 32;
    dim3 block(blockSize, blockSize);
    dim3 grid((col + block.x - 1) / block.x, (row + block.y - 1) / block.y);
    matmul_v4_kernel<<<grid, block, 2 * (blockSize * blockSize) * sizeof(T)>>>(
        d_input_a.data, d_input_b.data, d_output.data, row, col, depth);

    cudaDeviceSynchronize();
    d_output.cpyToHost(C);
}

template <typename T>
void matmul_gpu_v5(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    size_t pitchA;
    size_t pitchB;
    size_t pitchC;
    T* d_a;
    T* d_b;
    T* d_c;
    checkCudaErrors(cudaMallocPitch(&d_a, &pitchA, depth * sizeof(T), row));
    checkCudaErrors(cudaMallocPitch(&d_b, &pitchB, col * sizeof(T), depth));
    checkCudaErrors(cudaMallocPitch(&d_c, &pitchC, col * sizeof(T), row));
    checkCudaErrors(cudaMemcpy2D(d_a, pitchA, A, depth * sizeof(T), depth * sizeof(T), row,
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_b, pitchB, B, col * sizeof(T), col * sizeof(T), depth,
        cudaMemcpyHostToDevice));

    const size_t blockSize = 32;
    col = pitchC / sizeof(T);
    depth = pitchA / sizeof(T);
    dim3 block(blockSize, blockSize);
    dim3 grid((col + block.x - 1) / block.x, (row + block.y - 1) / block.y);
    matmul_v5_kernel<<<grid, block, 2 * (blockSize * blockSize) * sizeof(T)>>>(
        d_a, d_b, d_c, row, col, depth);
    // cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy2D(C, col * sizeof(T), d_c, pitchC, col * sizeof(T), row,
        cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

template <typename T>
void matmul_gpu_cublas(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    T* d_a;
    T* d_b;
    T* d_c;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc(&d_a, row * depth * sizeof(T));
    cudaMalloc(&d_b, depth * col * sizeof(T));
    cudaMalloc(&d_c, row * col * sizeof(T));

    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);

    cublasSetVector(row * depth, sizeof(T), A, 1, d_a, 1);
    cublasSetVector(depth * col, sizeof(T), B, 1, d_b, 1);

    if constexpr (std::is_same_v<T, float>)
    {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, col, row, depth,
                    &alpha, d_b, col, d_a, depth,
                    &beta, d_c, col);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, col, row, depth,
                    &alpha, d_b, col, d_a, depth,
                    &beta, d_c, col);
    }
    cublasGetVector(row * col, sizeof(T), d_c, 1, C, 1);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
}

template <typename T>
void matmul_gpu_v6(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    helper::DeviceDataHandler d_input_a(A, row * depth);
    helper::DeviceDataHandler d_input_b(B, depth * col);
    helper::DeviceDataHandler<T> d_output(row * col);
    size_t sharedMemSize = std::min(depth, static_cast<size_t>(512));
    matmul_v6_kernel<<<row, BlockSize, sharedMemSize * sizeof(T)>>>(
        d_input_a.data, d_input_b.data, d_output.data, row, col, depth, sharedMemSize);
    cudaDeviceSynchronize();
    
    d_output.cpyToHost(C);
}

template <typename T>
void matmul_gpu_v7(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    size_t pitchA;
    size_t pitchB;
    size_t pitchC;
    T* d_a;
    T* d_b;
    T* d_c;
    checkCudaErrors(cudaMallocPitch(&d_a, &pitchA, depth * sizeof(T), row));
    checkCudaErrors(cudaMallocPitch(&d_b, &pitchB, col * sizeof(T), depth));
    checkCudaErrors(cudaMallocPitch(&d_c, &pitchC, col * sizeof(T), row));
    checkCudaErrors(cudaMemcpy2D(d_a, pitchA, A, depth * sizeof(T), depth * sizeof(T), row,
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_b, pitchB, B, col * sizeof(T), col * sizeof(T), depth,
        cudaMemcpyHostToDevice));

    constexpr size_t blockSize = 16;
    constexpr size_t N = 2;
    col = pitchC / sizeof(T);
    depth = pitchA / sizeof(T);
    dim3 block(blockSize, blockSize);
    dim3 grid((col + N * block.x - 1) / (block.x * N), (row + N * block.y - 1) / (N * block.y));
    matmul_v7_kernel<N><<<grid, block, 2 * (blockSize * blockSize) * sizeof(T) * N>>>(
        d_a, d_b, d_c, row, col, depth);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy2D(C, col * sizeof(T), d_c, pitchC, col * sizeof(T), row,
        cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main()
{
    size_t row = M;
    size_t col = N;
    size_t depth = L;
    using ValueType = float;
    auto matA = helper::generate_sequence<ValueType>(row * depth);
    auto matB = helper::generate_sequence<ValueType>(col * depth);

    auto matCCpu1 = decltype(matA)(row * col);
    timer.start();
    matmul_general(matA.data(), matB.data(), matCCpu1.data(), row, col, depth);
    timer.stop();
    std::cout << "CPU (general): " << timer.elapsed() << "" << timer.unit() << std::endl;

    auto matCCpu2 = helper::generate_sequence<ValueType>(row * col);
    timer.start();
    matmul_swap_loop<ValueType, false>(matA.data(), matB.data(), matCCpu2.data(), row, col, depth);
    timer.stop();
    std::cout << "CPU (swap loop): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_difference(matCCpu1.data(), matCCpu2.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCCpu3 = decltype(matA)(row * col);
    timer.start();
    matmul_transpose<ValueType, false>(matA.data(), matB.data(), matCCpu3.data(), row, col, depth);
    timer.stop();
    std::cout << "CPU (swap loop): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_difference(matCCpu1.data(), matCCpu3.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu0 = decltype(matA)(row * col);
    timer.start();
    matmul_gpu_v0(matA.data(), matB.data(), matCGpu0.data(), row, col, depth);
    timer.stop();
    std::cout << "GPU (v0): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_difference(matCCpu1.data(), matCGpu0.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu1 = decltype(matA)(row * col);
    timer.start();
    matmul_gpu_v1(matA.data(), matB.data(), matCGpu1.data(), row, col, depth);
    timer.stop();
    std::cout << "GPU (v1): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_difference(matCCpu1.data(), matCGpu1.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu2 = decltype(matA)(row * col);
    timer.start();
    matmul_gpu_v2(matA.data(), matB.data(), matCGpu2.data(), row, col, depth);
    timer.stop();
    std::cout << "GPU (v2): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_difference(matCCpu1.data(), matCGpu2.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu3 = decltype(matA)(row * col);
    timer.start();
    matmul_gpu_v3(matA.data(), matB.data(), matCGpu3.data(), row, col, depth);
    timer.stop();
    std::cout << "GPU (v3): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_difference(matCCpu1.data(), matCGpu3.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu4 = decltype(matA)(row * col);
    timer.start();
    matmul_gpu_v4(matA.data(), matB.data(), matCGpu4.data(), row, col, depth);
    timer.stop();
    std::cout << "GPU (v4): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_difference(matCCpu1.data(), matCGpu4.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu5 = decltype(matA)(row * col);
    timer.start();
    matmul_gpu_v5(matA.data(), matB.data(), matCGpu5.data(), row, col, depth);
    timer.stop();
    std::cout << "GPU (v5): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_error(matCCpu1.data(), matCGpu5.data(), row * col, true);

    auto matCGpu6 = decltype(matA)(row * col);
    timer.start();
    matmul_gpu_v6(matA.data(), matB.data(), matCGpu6.data(), row, col, depth);
    timer.stop();
    std::cout << "GPU (v6): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_difference(matCCpu1.data(), matCGpu5.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCGpu7 = decltype(matA)(row * col);
    timer.start();
    matmul_gpu_v7(matA.data(), matB.data(), matCGpu7.data(), row, col, depth);
    timer.stop();
    std::cout << "GPU (v7): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_difference(matCCpu1.data(), matCGpu7.data(), row * col, static_cast<ValueType>(1e-1));

    auto matCCublas = decltype(matA)(row * col);
    timer.start();
    matmul_gpu_cublas(matA.data(), matB.data(), matCCublas.data(), row, col, depth);
    timer.stop();
    std::cout << "GPU (cublas): " << timer.elapsed() << "" << timer.unit() << std::endl;
    helper::check_error(matCCpu1.data(), matCCublas.data(), row * col, true);
    return 0;
}
