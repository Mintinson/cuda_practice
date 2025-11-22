//
// Created by asus on 2025/3/31.
//
#include "helper.cuh"
#include "reduce_v1.cuh"
#include "reduce_v2.cuh"
#include "reduce_v3.cuh"
#include "reduce_v4.cuh"
#include "reduce_v5.cuh"
#include "reduce_v6.cuh"
#include "reduce_v7.cuh"
#include "reduce_v9.cuh"
#include "timer.cuh"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random_gen.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <utility>
#include <vector>
GpuTimer timer;
constexpr size_t N = 1024 * 1024 * 32;
constexpr int BlockSize = 512;

class Logger {
    std::map<std::string, std::map<size_t, std::vector<double>>> records;
    std::string filename;

public:
    Logger(std::string name)
        : filename(std::move(name))
    {
    }
    void record(std::string name, size_t size, double time)
    {
        if (records.find(name) == records.end()) {
            records[name] = std::map<size_t, std::vector<double>>();
        }
        if (records[name].find(size) == records[name].end()) {
            records[name][size] = std::vector<double>();
        }
        records[name][size].push_back(time);
    }
    void save() const
    {
        std::ofstream ofs { filename, std::ios::out };
        if (!ofs) {
            std::cerr << "Failed to open file " << filename << std::endl;
            return;
        }
        for (const auto& [name, records] : records) {
            ofs << name << ": " << std::endl;
            for (const auto& [size, times] : records) {
                ofs << "----" << size << " ";
                auto res = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                ofs << "average time: " << res << " ms";
                ofs << std::endl;
            }
            ofs << std::endl;
        }
    }
};
Logger logger("GCC_Float_Release.log");

// constexpr int temp = N % (2 * BlockSize);

template <typename T, typename ReduceOp>
__global__ void reduce_rest_kernel(T* input, T* output, T init, size_t sz, ReduceOp op)
{
    for (int i = 0; i < sz; ++i) {
        init = op(init, input[i]);
    }
    output[0] = init;
}

template <typename T, typename ReduceOp>
T reduce_v1(const T* input, size_t size, T init, ReduceOp op)
{
    helper::DeviceDataHandler d_input(input, size);
    const size_t originalSize = size;
    // helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    while (size >= BlockSize) {

        reduce_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, size, op);
        size = (size + BlockSize - 1) / BlockSize;
    }
    reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
    timer.stop();
    auto res = d_input.singleDataToHost(0);
    logger.record("GPU (reduce_v1)", originalSize, timer.elapsed());
    std::cout << "GPU (reduce_v1): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res << std::endl;

    return res;
}

template <typename T, typename ReduceOp>
T reduce_v2(const T* input, size_t size, T init, ReduceOp op)
{
    helper::DeviceDataHandler d_input(input, size);
    const size_t originalSize = size;

    // helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    while (size > BlockSize) {

        reduce_kernel_v2<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, size, op);
        size = (size + BlockSize - 1) / BlockSize;
        // size /= BlockSize;
    }
    reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
    timer.stop();
    auto res = d_input.singleDataToHost(0);
    logger.record("GPU (reduce_v2)", originalSize, timer.elapsed());

    std::cout << "GPU (reduce_v2): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res << std::endl;

    return res;
}

template <typename T, typename ReduceOp>
T reduce_v3(const T* input, size_t size, T init, ReduceOp op)
{
    helper::DeviceDataHandler d_input(input, size);
    const size_t originalSize = size;
    // helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    while (size > BlockSize) {

        reduce_kernel_v3<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, size, op);
        size = (size + BlockSize - 1) / BlockSize;
    }
    reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
    timer.stop();
    auto res = d_input.singleDataToHost(0);
    logger.record("GPU (reduce_v3)", originalSize, timer.elapsed());

    std::cout << "GPU (reduce_v3): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res << std::endl;

    return res;
}
template <typename T, typename ReduceOp>
T reduce_v4(const T* input, size_t size, T init, ReduceOp op)
{
    helper::DeviceDataHandler d_input(input, size);
    const size_t originalSize = size;
    // helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    while (size > BlockSize) {

        reduce_kernel_v4<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, size, op);
        size = (size + BlockSize - 1) / BlockSize;
    }
    reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
    timer.stop();
    auto res = d_input.singleDataToHost(0);
    logger.record("GPU (reduce_v4)", originalSize, timer.elapsed());

    std::cout << "GPU (reduce_v4): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res << std::endl;

    return res;
}
template <typename T, typename ReduceOp>
T reduce_v5(const T* input, size_t size, T init, ReduceOp op)
{
    helper::DeviceDataHandler d_input(input, size);
    const size_t originalSize = size;
    // helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    while (size > BlockSize * 2) {
        reduce_kernel_v5<<<(size + 2 * BlockSize - 1) / (BlockSize * 2),
            BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, size, op);
        size = (size + 2 * BlockSize - 1) / (2 * BlockSize);
    }
    reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
    timer.stop();
    auto res = d_input.singleDataToHost(0);
    logger.record("GPU (reduce_v5)", originalSize, timer.elapsed());

    std::cout << "GPU (reduce_v5): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res << std::endl;

    return res;
}
template <typename T, typename ReduceOp>
T reduce_v6(const T* input, size_t size, T init, ReduceOp op)
{
    helper::DeviceDataHandler d_input(input, size);
    const size_t originalSize = size;
    // helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    while (size > BlockSize * 2) {
        reduce_kernel_v6<<<(size + 2 * BlockSize - 1) / (BlockSize * 2),
            BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, size, op);
        size = (size + 2 * BlockSize - 1) / (2 * BlockSize);
    }
    reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
    timer.stop();
    auto res = d_input.singleDataToHost(0);
    logger.record("GPU (reduce_v6)", originalSize, timer.elapsed());

    std::cout << "GPU (reduce_v6): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res << std::endl;

    return res;
}
template <typename T, typename ReduceOp>
T reduce_v7(const T* input, size_t size, T init, ReduceOp op)
{
    helper::DeviceDataHandler d_input(input, size);
    const size_t originalSize = size;
    // helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    while (size > BlockSize * 2) {
        reduce_kernel_v7<<<(size + 2 * BlockSize - 1) / (BlockSize * 2),
            BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, size, op);
        size = (size + 2 * BlockSize - 1) / (2 * BlockSize);
    }
    reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
    timer.stop();
    auto res = d_input.singleDataToHost(0);
    logger.record("GPU (reduce_v7)", originalSize, timer.elapsed());

    std::cout << "GPU (reduce_v7): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res << std::endl;

    return res;
}
template <typename T, typename ReduceOp>
T reduce_v9(const T* input, size_t size, T init, ReduceOp op)
{
    helper::DeviceDataHandler d_input(input, size);
    const size_t originalSize = size;
    // helper::DeviceDataHandler<T> d_output(size);
    timer.start();
    while (size > BlockSize * 2) {
        reduce_kernel_v9<<<(size + 2 * BlockSize - 1) / (BlockSize * 2),
            BlockSize, BlockSize * sizeof(T)>>>(
            d_input.data, d_input.data, size, init, op);
        size = (size + 2 * BlockSize - 1) / (2 * BlockSize);
    }
    reduce_rest_kernel<<<1, 1>>>(d_input.data, d_input.data, init, size, op);
    timer.stop();
    auto res = d_input.singleDataToHost(0);
    logger.record("GPU (reduce_v9)", originalSize, timer.elapsed());

    std::cout << "GPU (reduce_v9): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res << std::endl;

    return res;
}
int main()
{

    for (int i = 1; i < 5 + 1; ++i) {
        size_t size = helper::generate_sequence_i<size_t>(1, N * i, N * (i + 1))[0];
        for (int k = 0; k < 10; ++k) {
            std::cout << "Reduce with elements : " << size << "\n";
            auto randomVec = helper::generate_sequence<float>(size * i);
            using ValueType = decltype(randomVec)::value_type;

            ValueType res { 0 };
            // auto op = std::plus<volatile ValueType> {};
            auto op = [] __device__ __host__(ValueType a, ValueType b) {
                return a + b;
            };
            timer.start();
            for (auto c : randomVec) {
                res = op(res, c);
            }
            timer.stop();
            logger.record("CPU (raw loop)", size, timer.elapsed());
            std::cout << "CPU (raw loop): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res << std::endl;
            timer.start();
            auto res2 = std::accumulate(randomVec.cbegin(), randomVec.cend(), static_cast<ValueType>(0), op);
            timer.stop();
            logger.record("CPU (accumulate loop)", size, timer.elapsed());
            std::cout << "CPU (accumulate loop): " << timer.elapsed() << " " << timer.unit() << ", answer = " << res2 << std::endl;

            auto resGPU1 = reduce_v1(randomVec.data(), randomVec.size(), static_cast<ValueType>(0), op);

            auto resGPU2 = reduce_v2(randomVec.data(), randomVec.size(), static_cast<ValueType>(0), op);

            auto resGPU3 = reduce_v3(randomVec.data(), randomVec.size(), static_cast<ValueType>(0), op);

            auto resGPU4 = reduce_v4(randomVec.data(), randomVec.size(), static_cast<ValueType>(0), op);
            auto resGPU5 = reduce_v5(randomVec.data(), randomVec.size(), static_cast<ValueType>(0), op);
            auto resGPU6 = reduce_v6(randomVec.data(), randomVec.size(), static_cast<ValueType>(0), op);

            auto resGPU7 = reduce_v7(randomVec.data(), randomVec.size(), static_cast<ValueType>(0), op);
            auto resGPU9 = reduce_v9(randomVec.data(), randomVec.size(), static_cast<ValueType>(0), op);

            thrust::device_vector<ValueType> d_input(randomVec);
            // thrust::device_vector<ValueType> );
            timer.start();

            auto resThrust = thrust::reduce(thrust::device, d_input.cbegin(), d_input.cend(), static_cast<ValueType>(0), op);
            timer.stop();
            std::cout << "GPU (thrust): " << timer.elapsed() << " " << timer.unit() << ", answer = " << resThrust << std::endl;
            std::cout << "\n";
        }
    }
    logger.save();
    
    std::cout << "Finished!\n";
}