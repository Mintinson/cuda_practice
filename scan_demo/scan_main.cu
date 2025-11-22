// #include "checker.cuh"
#include "brent_kung_v0.cuh"
#include "checker.cuh"
#include "cpp_scan.hpp"
#include "helper.cuh"
#include "kogge_stone_v0.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include <cstddef>
#include <cstdio>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <utility>
// #include <numeric>

GpuTimer timer;
constexpr std::size_t N = 1024 * 1024 * 512;
constexpr std::size_t BlockSize = 512;

template <typename F>
void tictoc(const std::string& name, F&& f)
{
    timer.start();
    f();
    timer.stop();
    std::cout << name << " time: " << timer.elapsed() << " " << timer.unit() << std::endl;
}

template <typename T, typename Binary = std::plus<>>
void kogge_stone_v0(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    helper::DeviceDataHandler d_input(input, size);
    helper::DeviceDataHandler<T> d_output(size);

    kogge_stone_kernel_v0<<<(size + BlockSize - 1) / BlockSize, BlockSize, BlockSize * sizeof(T)>>>(
        d_input.data, size, d_output.data, binary);
    // printf("yes\n");
    d_output.cpyToHost(output);
    // cudaDeviceSynchronize();
    // checkCudaErrors(cudaGetLastError());
    // printf("no\n");
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

template <typename T>
__global__ void print_kernel(const T* array, size_t start, size_t end)
{
    for (auto i = start; i < end; ++i) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

template <bool inclusive, typename T, typename Binary = std::plus<>>
void kogge_stone_v1(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    auto [sumSize, cnt] = get_sum_size<BlockSize>(size);
    helper::DeviceDataHandler<T> d_input(sumSize, [&](T* data) {
        checkCudaErrors(cudaMemcpy(data, input, sizeof(T) * size, cudaMemcpyHostToDevice));
    });
    T* d_input_ptr = d_input.data;
    for (size_t i = 0; i < cnt; ++i) {
        kogge_stone_kernel_v1<<<(size + BlockSize - 1) / BlockSize, BlockSize, 2 * BlockSize * sizeof(T)>>>(
            d_input_ptr, size, d_input_ptr, binary);
        kogge_stone_collect_kernel<<<(size / BlockSize + BlockSize - 1) / BlockSize, BlockSize>>>(
            d_input_ptr, size / BlockSize, d_input_ptr + size);
        d_input_ptr += size;
        size /= BlockSize;
    }
    kogge_stone_single_kernel<<<1, 1>>>(d_input_ptr, size, d_input_ptr, binary);

    for (size_t i = 0; i < cnt; ++i) {
        size *= BlockSize;
        d_input_ptr -= size;
        if (i == cnt - 1)
            kogge_stone_distribute_kernel<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr, init, binary);
        else
            kogge_stone_distribute_kernel<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr, {}, binary);
    }
    if constexpr (inclusive) {
        d_input.cpyToHost(output, 0, size);
    } else {
        d_input.cpyToHost(output + 1, 0, size - 1);
        output[0] = init;
    }
}

template <typename T, typename Binary = std::plus<>>
void brent_kung_v0(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    helper::DeviceDataHandler d_input(input, size);
    helper::DeviceDataHandler<T> d_output(size);

    brent_kung_kernel_v0<<<(size + BlockSize - 1) / BlockSize, BlockSize,
        BlockSize * sizeof(T)>>>(
        d_input.data, size, d_output.data, binary);
    d_output.cpyToHost(output);
}

template <typename T, typename Binary = std::plus<>>
void brent_kung_v1(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    helper::DeviceDataHandler d_input(input, size);
    helper::DeviceDataHandler<T> d_output(size);

    brent_kung_kernel_v1<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize,
        2 * BlockSize * sizeof(T)>>>(
        d_input.data, size, d_output.data, binary);
    d_output.cpyToHost(output);
}

template <typename T, typename Binary = std::plus<>>
void brent_kung_v2(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    helper::DeviceDataHandler d_input(input, size);
    helper::DeviceDataHandler<T> d_output(size);

    brent_kung_kernel_v2<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize,
        (2 * BlockSize + (2 * BlockSize) / 32) * sizeof(T)>>>(
        d_input.data, size, d_output.data, binary);
    d_output.cpyToHost(output);
}

template <bool inclusive, typename T, typename Binary = std::plus<>>
void brent_kung_v2_full(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    auto [sumSize, cnt] = get_sum_size<2 * BlockSize>(size);
    helper::DeviceDataHandler<T> d_input(sumSize, [&](T* data) {
        checkCudaErrors(cudaMemcpy(data, input, sizeof(T) * size, cudaMemcpyHostToDevice));
    });
    T* d_input_ptr = d_input.data;
    for (size_t i = 0; i < cnt; ++i) {
        brent_kung_kernel_v2<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize,
            (2 * BlockSize + (2 * BlockSize) / 32) * sizeof(T)>>>(
            d_input_ptr, size, d_input_ptr, binary);
        brent_kung_collect_kernel<<<(size / (2 * BlockSize) + BlockSize - 1) / (BlockSize),
            BlockSize>>>(
            d_input_ptr, size / (2 * BlockSize), d_input_ptr + size, 2 * BlockSize);
        d_input_ptr += size;
        size /= 2 * BlockSize;
    }
    kogge_stone_single_kernel<<<1, 1>>>(d_input_ptr, size, d_input_ptr, binary);

    for (size_t i = 0; i < cnt; ++i) {
        size *= 2 * BlockSize;
        d_input_ptr -= size;
        if (i == cnt - 1)
            brent_kung_distribute_kernel<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr, init, binary);
        else
            brent_kung_distribute_kernel<<<(size + 2 * BlockSize - 1) / (2 * BlockSize), BlockSize>>>(
                d_input_ptr + size, size, d_input_ptr, {}, binary);
    }
    if constexpr (inclusive) {
        d_input.cpyToHost(output, 0, size);
    } else {
        d_input.cpyToHost(output + 1, 0, size - 1);
        output[0] = init;
    }
}

template <bool inclusive, typename T, typename Binary = std::plus<>>
void brent_kung_warp_full(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    helper::DeviceDataHandler d_input(input, size);
    helper::DeviceDataHandler<T> d_output(size);

    warp_scan_fan(d_input.data, size, d_output.data, BlockSize, init, binary);
    d_output.cpyToHost(output);
}

template <bool inclusive, typename T, typename Binary = std::plus<>>
void thrust_scan(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    thrust::device_vector<T> d_input(input, input + size);
    if constexpr (inclusive) {
        thrust::device_vector<T> d_output(size, init);
        thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), binary);
        thrust::copy(d_output.begin(), d_output.end(), output);
    } else {
        thrust::device_vector<T> d_output(size);
        thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init, binary);
        thrust::copy(d_output.begin(), d_output.end(), output);
    }
}

template <bool inclusive, typename T, typename Binary = std::plus<>>
void cub_scan(const T* input, std::size_t size, T* output, T init = {}, Binary binary = {})
{
    helper::DeviceDataHandler<T> d_input(input, size);
    helper::DeviceDataHandler<T> d_ouput(size);
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    if constexpr (inclusive) {
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_input.data, d_ouput.data,
            binary, size);
        // allocate the temporary memory
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_input.data, d_ouput.data,
            binary, size);
    } else {
        cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_input.data, d_ouput.data,
            binary, init, size);
        // allocate the temporary memory
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_input.data, d_ouput.data,
            binary, init, size);
    }

    d_ouput.cpyToHost(output);
    checkCudaErrors(cudaFree(d_temp_storage));
}

int main()
{
    const std::size_t n = N;
    using ValueType = int;
    auto input = helper::generate_sequence<ValueType>(n);
    decltype(input) output(input.size());

    tictoc("inclusive(CPU)", [&]() {
        inclusive_prescan(input.data(), input.size(), output.data());
    });

    decltype(input) outputExc(input.size());

    tictoc("exclusive(CPU)", [&]() {
        exclusive_prescan(input.data(), input.size(), outputExc.data());
    });

    decltype(input) outputBlock(input.size());
    block_inclusive_prescan(input.data(), input.size(), outputBlock.data(), BlockSize);

    decltype(input) gpuOutBlock(input.size());
    kogge_stone_v0(input.data(), input.size(), gpuOutBlock.data());
    helper::check_difference(outputBlock.data(), gpuOutBlock.data(), input.size());

    decltype(input) gpuOutBlock2(input.size());
    tictoc(
        "kogge_stone_v1", [&]() {
            kogge_stone_v1<true>(input.data(), input.size(), gpuOutBlock2.data());
        });
    helper::check_error(output.data(), gpuOutBlock2.data(), input.size());

    decltype(input) gpuOutBlock2Exc(input.size());
    tictoc(
        "kogge_stone_v1", [&]() {
            kogge_stone_v1<false>(input.data(), input.size(), gpuOutBlock2Exc.data());
        });
    helper::check_error(outputExc.data(), gpuOutBlock2Exc.data(), input.size());

    decltype(input) brentKungOut0(input.size());
    tictoc(
        "brent_kung_v0", [&]() {
            brent_kung_v0(input.data(), input.size(), brentKungOut0.data());
        });
    helper::check_difference(outputBlock.data(), brentKungOut0.data(), input.size());

    block_inclusive_prescan(input.data(), input.size(), outputBlock.data(), 2 * BlockSize);
    decltype(input) brentKungOut1(input.size());
    tictoc(
        "brent_kung_v1", [&]() {
            brent_kung_v1(input.data(), input.size(), brentKungOut1.data());
        });
    helper::check_difference(outputBlock.data(), brentKungOut1.data(), input.size());

    decltype(input) brentKungOut2(input.size());
    tictoc(
        "brent_kung_v2", [&]() {
            brent_kung_v2(input.data(), input.size(), brentKungOut2.data());
        });
    helper::check_difference(outputBlock.data(), brentKungOut2.data(), input.size());

    decltype(input) brentKungOut3(input.size());
    tictoc(
        "brent_kung_v2(full)", [&]() {
            brent_kung_v2_full<true>(input.data(), input.size(), brentKungOut3.data());
        });
    helper::check_difference(output.data(), brentKungOut3.data(), input.size());

    decltype(input) brentKungOut4(input.size());
    tictoc(
        "brent_kung_warp(full)", [&]() {
            brent_kung_warp_full<true>(input.data(), input.size(), brentKungOut4.data());
        });
    helper::check_difference(output.data(), brentKungOut4.data(), input.size());

    decltype(input) thrustScanOut(input.size());
    tictoc(
        "thrust scan", [&]() {
            thrust_scan<true>(input.data(), input.size(), thrustScanOut.data());
        });
    helper::check_difference(output.data(), thrustScanOut.data(), input.size());

    decltype(input) cubScanOut(input.size());
    tictoc(
        "cub scan", [&]() {
            cub_scan<true>(input.data(), input.size(), cubScanOut.data());
        });
    helper::check_difference(output.data(), cubScanOut.data(), input.size());
    std::cout << "\n";
}
