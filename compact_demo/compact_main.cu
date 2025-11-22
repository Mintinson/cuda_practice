//
// Created by asus on 2025/4/12.
//
#include <cstddef>
#include <cuda_runtime.h>
#include <string>

#include "compact_cuda_v0.cuh"
#include "helper.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

constexpr std::size_t N = 1024 * 1024 * 256;
constexpr std::size_t BlockSize = 512;
GpuTimer timer;

template <typename F>
void tictoc(const std::string& name, F&& f)
{
    timer.start();
    f();
    timer.stop();
    std::cout << name << " time: " << timer.elapsed() << " " << timer.unit() << std::endl;
}

template <size_t Size>
auto get_sum_size(size_t size)
{
    size_t res {};
    size_t cnt = 0;
    while (size > Size) {
        res += size;
        cnt++;
        size /= Size;
    }
    return std::make_pair(res + size, cnt);
}

template <typename T, typename Pred>
void cuda_compact_inplace_v0(T* data, std::size_t n, Pred pred)
{
    auto [sumSize, cnt] = get_sum_size<2 * BlockSize>(n);
    helper::DeviceDataHandler<T> input(data, n);
    helper::DeviceDataHandler<T> output(n);
    helper::DeviceDataHandler<char> predRes(n);
    helper::DeviceDataHandler<std::size_t> scanRes(sumSize + 1);

    compact_v0::pred_map<<<(n + BlockSize - 1) / BlockSize, BlockSize>>>(
        input.data, n, predRes.data, pred);

    // scan
    std::size_t* d_input_ptr = scanRes.data;
    std::size_t size = n;
    for (size_t i = 0; i < cnt; ++i) {
        if (i == 0) {
            compact_v0::brent_kung_kernel_block<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize,
                (2 * BlockSize + (2 * BlockSize) / 32) * sizeof(std::size_t)>>>(
                predRes.data, size, d_input_ptr);
            // cudaDeviceSynchronize();
            // checkCudaErrors(cudaGetLastError());
        } else {
            compact_v0::brent_kung_kernel_block<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize,
                (2 * BlockSize + (2 * BlockSize) / 32) * sizeof(std::size_t)>>>(
                d_input_ptr, size, d_input_ptr);
            // cudaDeviceSynchronize();
            // checkCudaErrors(cudaGetLastError());
        }
        compact_v0::brent_kung_collect_kernel<<<(size / (2 * BlockSize) + BlockSize - 1) / (BlockSize),
            BlockSize>>>(
            d_input_ptr, size / (2 * BlockSize), d_input_ptr + size, 2 * BlockSize);
        // cudaDeviceSynchronize();
        // checkCudaErrors(cudaGetLastError());
        d_input_ptr += size;
        size /= 2 * BlockSize;
    }
    compact_v0::scan_single_kernel<<<1, 1>>>(d_input_ptr, size, d_input_ptr);

    for (size_t i = 0; i < cnt; ++i) {
        size *= 2 * BlockSize;
        d_input_ptr -= size;
        if (i == cnt - 1)
            compact_v0::brent_kung_distribute_kernel<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr);
        else
            compact_v0::brent_kung_distribute_kernel<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr);
    }

    // distribute
    compact_v0::compact_distribute_kernel<<<(n + BlockSize - 1) / BlockSize, BlockSize>>>(
        scanRes.data, predRes.data, n, input.data, output.data);
    output.cpyToHost(data);
}

template <typename T, typename Pred>
void thrust_compact_inplace_v0(T* data, std::size_t n, Pred pred)
{
    thrust::device_vector<T> d_input(data, data + n);
    thrust::stable_partition(d_input.begin(), d_input.end(), pred);
    thrust::copy(d_input.begin(), d_input.end(), data);
}

template <typename T, typename Pred>
void thrust_compact_to_v0(const T* data, std::size_t n, T* src, Pred pred)
{
    thrust::device_vector<T> d_input(data, data + n);
    thrust::device_vector<T> d_out(n);
    thrust::copy_if(d_input.begin(), d_input.end(), d_out.begin(), pred);
    thrust::copy(d_out.begin(), d_out.end(), src);
}

int main()
{
    constexpr std::size_t n = N;
    using ValueType = int;
    auto input = helper::generate_sequence<ValueType>(n);

    auto stdVec = input;
    auto pred = [] __device__ __host__(ValueType x) -> bool {
        return x & (1 << 3);
    };
    tictoc("std (par)", [&]() {
        std::stable_partition(stdVec.begin(), stdVec.end(), pred);
    });
    auto cudaVec0 = input;
    tictoc("cuda_v0", [&]() {
        cuda_compact_inplace_v0(cudaVec0.data(), n, pred);
    });
    helper::check_difference(stdVec.data(), cudaVec0.data(), stdVec.size());

    auto thrustVec = input;
    tictoc("thrust(par)", [&]() {
        thrust_compact_inplace_v0(thrustVec.data(), n, pred);
    });
    helper::check_difference(stdVec.data(), thrustVec.data(), stdVec.size());

    auto stdVec1 = input;
    auto stdOut1 = input;
    tictoc("std(cp_if)", [&]() {
        std::copy_if(stdVec1.begin(), stdVec1.end(), stdOut1.begin(), pred);
    });

    auto thrustVec1 = input;
    auto thrustOut1 = input;
    tictoc("thrust(cp_if)", [&]() {
        thrust_compact_to_v0(thrustVec.data(), n, thrustOut1.data(), pred);
    });
    helper::check_difference(stdOut1.data(), thrustOut1.data(), stdOut1.size());
}
