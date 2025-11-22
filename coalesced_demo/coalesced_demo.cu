// #include "cuda_runtime.h"
#include "checker.cuh"
#include "helper.cuh"
#include "random_gen.hpp"
#include <cstdio>

constexpr int N = 1024 * 1024 * 32;

template <typename T>
__global__ void add_kernel(T* d_vecA, T* d_vecB, T* d_vecC)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    d_vecC[idx] = d_vecA[idx] + d_vecB[idx];
}

template <typename T>
__global__ void add_kernel2(T* d_vecA, T* d_vecB, T* d_vecC)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1;

    d_vecC[idx] = d_vecA[idx] + d_vecB[idx];
}

template <typename T>
__global__ void add_kernel3(T* d_vecA, T* d_vecB, T* d_vecC)
{
    auto idx = blockIdx.x * blockDim.x + (threadIdx.x ^ 0x1);

    d_vecC[idx] = d_vecA[idx] + d_vecB[idx];
}
template <typename T>
__global__ void add_kernel4(T* d_vecA, T* d_vecB, T* d_vecC)
{
    auto idx = blockIdx.x * blockDim.x + (threadIdx.x ^ 0x1);
    int warp_id = idx / 32;

    d_vecC[warp_id] = d_vecA[warp_id] + d_vecB[warp_id];
}
template <typename T>
__global__ void add_kernel5(T* d_vecA, T* d_vecB, T* d_vecC)
{
    auto idx = blockIdx.x * blockDim.x + (threadIdx.x ^ 0x1);

    d_vecC[idx * 4] = d_vecA[idx * 4] + d_vecB[idx * 4];
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
    checkCudaErrors(cudaMalloc((void**)&d_vecA, vecA.size() * sizeof(ValueType)));
    checkCudaErrors(cudaMalloc((void**)&d_vecB, vecB.size() * sizeof(ValueType)));
    checkCudaErrors(cudaMalloc((void**)&d_vecC, vecC.size() * sizeof(ValueType)));

    checkCudaErrors(cudaMemcpy(d_vecA, vecA.data(), vecA.size() * sizeof(ValueType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vecB, vecB.data(), vecB.size() * sizeof(ValueType), cudaMemcpyHostToDevice));
    warm_up_gpu<<<512, 1024>>>();
    for (int i = 0; i < 2; ++i) {
        dim3 block(64);
        dim3 grid(N / (64 * 4));
        add_kernel<<<grid, block>>>(d_vecA, d_vecB, d_vecC);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; ++i) {
        dim3 block(64);
        dim3 grid(N / (64 * 4));
        add_kernel2<<<grid, block>>>(d_vecA, d_vecB, d_vecC);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; ++i) {
        dim3 block(64);
        dim3 grid(N / (64 * 4));
        add_kernel3<<<grid, block>>>(d_vecA, d_vecB, d_vecC);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; ++i) {
        dim3 block(64);
        dim3 grid(N / (64 * 4));
        add_kernel4<<<grid, block>>>(d_vecA, d_vecB, d_vecC);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; ++i) {
        dim3 block(64);
        dim3 grid(N / (64 * 4));
        add_kernel5<<<grid, block>>>(d_vecA, d_vecB, d_vecC);
        cudaDeviceSynchronize();
    }
    checkCudaErrors(cudaMemcpy(vecC.data(), d_vecC, vecC.size() * sizeof(ValueType), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_vecA));
    checkCudaErrors(cudaFree(d_vecB));
    checkCudaErrors(cudaFree(d_vecC));
}