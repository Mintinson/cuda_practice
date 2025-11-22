//
// Created by asus on 2025/4/19.
//

#include <cuda_runtime.h>
#include <cstddef>

#include "helper.cuh"
#include "random_gen.hpp"
constexpr std::size_t N = 2048;
using DataType = int;

__constant__ DataType kData[N];

__global__ void test_register_latency(double* time, DataType* out, int its)
{
    int p = 0;
    double tmpTime{};
    unsigned startTime{};
    unsigned endTime{};
    DataType q = 1;
    for (int i = 0; i < its; i++)
    {
        __syncthreads();
        startTime = clock();
        for (int j = 1; j < N; ++j)
        {
            p += q * (i + 1);
        }
        endTime = clock();
        q = (q * 2) % 10;
        tmpTime += (endTime - startTime);
        p /= N;
    }
    tmpTime = tmpTime / its / N;
    time[1] = tmpTime;
    out[1] += p;
    printf("register latency :%.5lf\n", tmpTime);
}

__global__ void test_const_latency(double* time, DataType* out, int its)
{
    int p = 0;
    double tmpTime{};
    unsigned startTime{};
    unsigned endTime{};

    for (int i = 0; i < its; i++)
    {
        __syncthreads();
        startTime = clock();
        for (int j = 1; j < N; ++j)
        {
            p += kData[j] * (i + 1);
        }
        endTime = clock();
        tmpTime += (endTime - startTime);
        p /= N;
    }
    tmpTime = tmpTime / its / N;
    time[1] = tmpTime;
    out[1] += p;
    printf("const latency :%.5lf\n", tmpTime);
}

__global__ void test_shared_latency(double* time, DataType* out, int its)
{
    __shared__ DataType sharedArray[N];
    for (int i = 0; i < N; i++)
    {
        sharedArray[i] = kData[i];
    }
    double tmpTime{};
    unsigned startTime{};
    unsigned endTime{};
    int p = 0;
    for (int i = 0; i < its; i++)
    {
        __syncthreads();
        startTime = clock();
        for (int j = 1; j < N; ++j)
        {
            p += sharedArray[j] * (i + 1);
        }
        endTime = clock();
        tmpTime += (endTime - startTime);
        p /= N;
    }
    tmpTime = tmpTime / its / N;
    time[2] = tmpTime;
    out[2] += p;
    printf("shared latency :%.5lf\n", tmpTime);
}

__global__ void test_local_latency(double* time, DataType* out, int its)
{
    DataType localArray[N];
    for (int i = 0; i < N; i++)
    {
        localArray[i] = kData[i];
    }
    double tmpTime{};
    unsigned startTime{};
    unsigned endTime{};
    int p = 0;
    for (int i = 0; i < its; i++)
    {
        __syncthreads();
        startTime = clock();
        for (int j = 1; j < N; ++j)
        {
            p += localArray[j] * (i + 1);
        }
        endTime = clock();
        tmpTime += (endTime - startTime);
        p /= N;
    }
    tmpTime = tmpTime / its / N;
    time[2] = tmpTime;
    out[2] += p;
    printf("local latency :%.5lf\n", tmpTime);
}

__global__ void test_global_latency(double* time, DataType* input, DataType* out, int its)
{
    double tmpTime{};
    unsigned startTime{};
    unsigned endTime{};
    int p = 0;
    for (int i = 0; i < its; i++)
    {
        __syncthreads();
        startTime = clock();
        for (int j = 1; j < N; ++j)
        {
            p += input[j] * (i + 1);
        }
        endTime = clock();
        tmpTime += (endTime - startTime);
        p /= N;
    }
    tmpTime = tmpTime / its / N;
    time[2] = tmpTime;
    out[2] += p;
    printf("global latency :%.5lf\n", tmpTime);
}

__global__ void test_texture_latency(double* time, cudaTextureObject_t tex, DataType* out, int its)
{
    double tmpTime{};
    unsigned startTime{};
    unsigned endTime{};
    int p = 0;
    for (int i = 0; i < its; i++)
    {
        __syncthreads();
        startTime = clock();
        for (int j = 1; j < N; ++j)
        {
            p += tex1Dfetch<DataType>(tex, j) * (i + 1);
        }
        endTime = clock();
        tmpTime += (endTime - startTime);
        p /= N;
    }
    tmpTime = tmpTime / its / N;
    time[2] = tmpTime;
    out[2] += p;
    printf("texture latency :%.5lf\n", tmpTime);
}

int main()
{
    helper::print_device_info();
    auto vec = helper::generate_sequence<DataType>(N);
    decltype(vec) out(vec.size());

    helper::DeviceDataHandler<DataType> d_input(vec.data(), vec.size());
    helper::DeviceDataHandler<DataType> d_output(vec.size());

    checkCudaErrors(cudaMemcpyToSymbol(kData, vec.data(), sizeof(DataType) * N)); // const memory

    // texture object CUDA 12.0
    cudaTextureObject_t texObj;
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_input.data;
    resDesc.res.linear.sizeInBytes = N * sizeof(DataType);
    cudaTextureDesc texDesc = {};
    memset(&texDesc, 0, sizeof(cudaResourceDesc));

    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // time
    helper::DeviceDataHandler<double> d_times(5);

    int its = 128;
    test_register_latency<<<1,1>>>(d_times.data, d_output.data, its);
    test_const_latency<<<1,1>>>(d_times.data, d_output.data, its);
    test_shared_latency<<<1,1>>>(d_times.data, d_output.data, its);
    test_local_latency<<<1,1>>>(d_times.data, d_output.data, its);
    test_global_latency<<<1,1>>>(d_times.data, d_input.data, d_output.data, its);
    test_texture_latency<<<1,1>>>(d_times.data, texObj, d_output.data, its);

    d_output.cpyToHost(out.data());
    helper::do_not_optimize_away(out);
    cudaDestroyTextureObject(texObj);
}
