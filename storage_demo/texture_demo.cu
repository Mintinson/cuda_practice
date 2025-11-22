//
// Created by asus on 2025/4/19.
//
#include <cuda_runtime.h>
#include <cstddef>

#include "helper.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
GpuTimer timer;
using DataType = int;

__global__ void vector_add_gpu_0(DataType* a, DataType* b, DataType* c,
                                 std::size_t n)
{
    const auto tid = threadIdx.x;
    const auto stepSize = gridDim.x * blockDim.x;
    const auto bid = blockIdx.x;
    auto idx = bid * blockDim.x + tid;
    while (idx < n)
    {
        c[idx] = a[idx] + b[idx];
        idx += stepSize;
    }
}
__global__ void vector_add_gpu_1(const DataType* __restrict__ a, const DataType* __restrict__ b, DataType* c,
                                 std::size_t n)
{
    const auto tid = threadIdx.x;
    const auto stepSize = gridDim.x * blockDim.x;
    const auto bid = blockIdx.x;
    auto idx = bid * blockDim.x + tid;
    while (idx < n)
    {
        c[idx] = a[idx] + b[idx];
        idx += stepSize;
    }
}

__global__ void vector_add_gpu_2(DataType* a, DataType* b, DataType* c,
                                 std::size_t n)
{
    const auto tid = threadIdx.x;
    const auto stepSize = gridDim.x * blockDim.x;
    const auto bid = blockIdx.x;
    auto idx = bid * blockDim.x + tid;
    while (idx < n)
    {
        c[idx] = __ldg(&a[idx]) + __ldg(&b[idx]);
        idx += stepSize;
    }
}

int main()
{
    helper::print_device_info();
    const std::size_t n = 1024 * 1024 * 128;

    auto vecA = helper::generate_sequence<DataType>(n);
    auto vecB = helper::generate_sequence<DataType>(n);
    auto vecC = helper::generate_sequence<DataType>(n);

    helper::DeviceDataHandler<DataType> dA0(vecA.data(), vecA.size());
    helper::DeviceDataHandler<DataType> dB0(vecB.data(), vecB.size());
    helper::DeviceDataHandler<DataType> dC0(vecC.size());
    timer.start();
    vector_add_gpu_0<<<512,512>>>(dA0.data, dB0.data, dC0.data, n);
    timer.stop();
    std::cout << "naive vector_add_gpu_0: " << timer.elapsed() << " " << timer.unit() << std::endl;
    dC0.cpyToHost(vecC.data());
    helper::do_not_optimize_away(vecC);

    helper::DeviceDataHandler<DataType> dA1(vecA.data(), vecA.size());
    helper::DeviceDataHandler<DataType> dB1(vecB.data(), vecB.size());
    helper::DeviceDataHandler<DataType> dC1(vecC.size());
    timer.start();
    vector_add_gpu_1<<<512,512>>>(dA1.data, dB1.data, dC1.data, n);
    timer.stop();
    std::cout << "__restrict__ vector_add_gpu_1: " << timer.elapsed() << " " << timer.unit() << std::endl;
    dC1.cpyToHost(vecC.data());
    helper::do_not_optimize_away(vecC);

    helper::DeviceDataHandler<DataType> dA2(vecA.data(), vecA.size());
    helper::DeviceDataHandler<DataType> dB2(vecB.data(), vecB.size());
    helper::DeviceDataHandler<DataType> dC2(vecC.size());
    timer.start();
    vector_add_gpu_2<<<512,512>>>(dA2.data, dB2.data, dC2.data, n);
    timer.stop();
    std::cout << "__kdg vector_add_gpu_2: " << timer.elapsed() << " " << timer.unit() << std::endl;
    dC2.cpyToHost(vecC.data());

    helper::do_not_optimize_away(vecC);


}
