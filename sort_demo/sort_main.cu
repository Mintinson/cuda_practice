#include "bitonic_sort.cuh"
#include "checker.cuh"
#include "cpp_sort.hpp"
#include "helper.cuh"
#include "merge_sort.cuh"
#include "quick_sort.cuh"
#include "random_gen.hpp"
#include "timer.cuh"
#include <algorithm>
// #include <bitset>
#include <cstddef>
#include <cstdio>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <utility>

#ifdef _MSC_VER
#include <Windows.h>
#endif

#ifdef __GNUC__
#include <pthread.h>
#endif

GpuTimer timer;
constexpr std::size_t N = 1024 * 1024 * 16;
// constexpr std::size_t N = 1024;
constexpr std::size_t BlockSize = 512;

template <typename F>
void tictoc(const std::string& name, F&& f)
{
    timer.start();
    f();
    timer.stop();
    std::cout << name << " time: " << timer.elapsed() << " " << timer.unit() << std::endl;
}

template <typename F, typename Dur>
void tictoc(const std::string& name, F&& f, Dur duration)
{
    bool done = false;
    auto finished = [&]() {
        tictoc(name, std::forward<F>(f));
        done = true;
    };
#ifdef _MSC_VER
    std::thread t(finished);
    auto start = std::chrono::high_resolution_clock::now();
    while (!done) {
        if (std::chrono::high_resolution_clock::now() - start > duration) {
            TerminateThread(t.native_handle(), 0);
            t.detach();

            std::cout << name << " executes too long! terminate it!\n";
            break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    if (t.joinable())
        t.join();
#endif
    // timer.start();
    // f();
    // timer.stop();
    // std::cout << name << " time: " << timer.elapsed() << " " << timer.unit() << std::endl;
}

template <typename T, bool Ascend = true>
void thrust_radix_sort(T* arr, std::size_t n)
{
    // thrust::host_vector<T> hostVector(arr, n);
    constexpr std::size_t numBites = sizeof(T) * 8;
    thrust::device_vector<T> deviceVector(arr, arr + n);

    for (std::size_t bit = 0; bit < numBites; ++bit) {
        thrust::stable_partition(deviceVector.begin(), deviceVector.end(), [bit] __device__(T x) {
            // return Ascend ^ ((x & (T(1) << bit)) != 0);
            return (Ascend) ^ (((*reinterpret_cast<unsigned*>(&x)) & (unsigned(1) << bit)) != 0);
        });
    }
    thrust::copy(deviceVector.begin(), deviceVector.end(), arr);
}

template <typename T, bool Ascend = true>
void cub_radix_sort(T* arr, std::size_t n)
{
    helper::DeviceDataHandler<T> d_keys_in(arr, n);
    helper::DeviceDataHandler<T> d_keys_out(n);

    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    if constexpr (Ascend) {
        cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in.data, d_keys_out.data, n);
    } else {
        cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys_in.data, d_keys_out.data, n);
    }
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if constexpr (Ascend) {
        cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in.data, d_keys_out.data, n);
    } else {
        cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys_in.data, d_keys_out.data, n);
    }
    d_keys_out.cpyToHost(arr);
    cudaFree(d_temp_storage);
}

template <typename T, typename Comp = std::less<>>
void cuda_merge_sort_v0(T* arr, std::size_t n, Comp comp = {})
{
    helper::DeviceDataHandler<T> d_input(arr, n);
    helper::DeviceDataHandler<T> d_output(n);
    for (std::size_t width = 1; width < n; width <<= 1) {
        int num_blocks = (n + (2 * BlockSize * width) - 1) / (2 * BlockSize * width);

        // merge sort sub blocks
        cuda_sort::merge_kernel<<<num_blocks, BlockSize>>>(d_input.data, d_output.data, n, width, comp);

        // swap the pointer to next sort
        std::swap(d_input.data, d_output.data);
    }
    d_input.cpyToHost(arr);
}
template <typename T, typename Comp = std::less<>>
void cuda_merge_sort_v1(T* arr, std::size_t n, Comp comp = {})
{
    helper::DeviceDataHandler<T> d_input(arr, n);
    helper::DeviceDataHandler<T> d_output(n);
    helper::DeviceDataHandler<T> d_buff(n);
    cuda_sort::run_merge_sort(d_input, d_output, d_buff, n, comp);
    d_output.cpyToHost(arr);
}
template <typename T>
__global__ void print_array(const T* arr, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

template <typename T, typename Comp = std::less<>>
void cuda_bitonic_sort_v0(T* arr, std::size_t n, Comp comp = {})
{
    helper::DeviceDataHandler<T> d_input(arr, n);
    helper::DeviceDataHandler<T> d_output(n);

    const auto grimSize = (n + 2 * BlockSize - 1) / (2 * BlockSize);
    auto sharedSize = 2 * BlockSize * sizeof(T);

    if (n <= 2 * BlockSize) {
        cuda_sort::bitonic_sort_shared<<<grimSize, BlockSize, sharedSize>>>(
            d_input.data, d_output.data, n, comp);
    } else {
        cuda_sort::bitonic_sort_sharedBlock<<<grimSize, BlockSize, sharedSize>>>(
            d_input.data, d_output.data, n, comp);

        // now every 4 * BlockSize sequence is bitonic
        for (auto size = 2 * 2 * BlockSize; size <= n; size <<= 1) {
            for (auto stride = size / 2; stride > 0; stride >>= 1) {
                if (stride >= 2 * BlockSize) {
                    cuda_sort::bitonic_merge_global<<<
                        grimSize, BlockSize>>>(d_output.data, d_output.data, n, size, stride, comp);
                } else {
                    cuda_sort::bitonic_merge_shared<<<grimSize, BlockSize, sharedSize>>>(
                        d_output.data, d_output.data, n, size, comp);

                    break;
                }
            }
        }
    }
    d_output.cpyToHost(arr);
}

template <typename T, typename Comp = std::less<>>
void cuda_quick_sort_v0(T* arr, std::size_t n, Comp comp = {})
{
    helper::DeviceDataHandler<T> d_input(arr, n);
    cuda_sort::cdp_simple_quicksort<<<1, 1>>>(d_input.data, 0, n - 1, 0);
    d_input.cpyToHost(arr);
}

template <typename T, typename Comp = std::less<>>
void cuda_quick_sort_v1(T* arr, std::size_t n, Comp comp = {})
{
    helper::DeviceDataHandler<T> d_input(arr, n);
    helper::DeviceDataHandler<T> d_buf(n);
    cuda_sort::run_quick_sort_cdp(d_input.data, d_buf.data, n, comp);
    d_input.cpyToHost(arr);
}
template <typename T, typename Comp = std::less<>>
void thrust_sort(T* arr, std::size_t n, Comp comp = {})
{
    thrust::device_vector<T> d_input(arr, arr + n);
    thrust::sort(d_input.begin(), d_input.end(), comp);
    thrust::copy(d_input.begin(), d_input.end(), arr);
}
int main()
{
    const std::size_t n = N;
    using ValueType = float;
    auto input = helper::generate_sequence<ValueType>(n, 0, n * 10);

    auto [minIt, maxIt] = std::minmax_element(input.begin(), input.end());
    std::cout << *minIt << "  " << *maxIt << '\n';

    decltype(input) stdSort(input);
    tictoc("std sort", [&] {
        std::sort(stdSort.begin(), stdSort.end());
    });
    // double* data{nullptr};

    /********** Too long Don't do it ********/
    // decltype(input) selectSort(input);
    // tictoc("select sort", [&] {
    //     cpp_sort::select_sort(selectSort.begin(), selectSort.size());
    // });
    // helper::check_difference(stdSort.data(), selectSort.data(), n);
    //
    // decltype(input) bubbleSort(input);
    // tictoc("bubble sort", [&] {
    //     cpp_sort::bubble_sort(bubbleSort.begin(), bubbleSort.size());
    // });
    // helper::check_difference(stdSort.data(), bubbleSort.data(), n);

    decltype(input) quickSort(input);
    tictoc("quick sort", [&] { cpp_sort::quick_sort(quickSort.data(), input.size()); }, std::chrono::seconds(10));
    helper::check_difference(stdSort.data(), quickSort.data(), n);

    decltype(input) radixSort(input);
    tictoc("radix sort", [&] {
        cpp_sort::radix_sort(radixSort.begin(), radixSort.size());
    });
    helper::check_difference(stdSort.data(), radixSort.data(), n);

    decltype(input) bitonicSort(input);
    tictoc("bitonic sort", [&] {
        cpp_sort::bitonic_sort(bitonicSort.begin(), bitonicSort.size());
    });
    helper::check_difference(stdSort.data(), bitonicSort.data(), n);

    decltype(input) mergeSort(input);
    tictoc("merge sort", [&] {
        cpp_sort::merge_sort(mergeSort.begin(), mergeSort.size());
    });
    helper::check_difference(stdSort.data(), mergeSort.data(), n);

    decltype(input) thrustRadixSort(input);
    tictoc("thrust radix sort", [&] {
        thrust_radix_sort(thrustRadixSort.data(), thrustRadixSort.size());
    });
    helper::check_difference(stdSort.data(), thrustRadixSort.data(), n);

    decltype(input) cubRadixSort(input);
    tictoc("cub radix sort", [&] {
        cub_radix_sort(cubRadixSort.data(), cubRadixSort.size());
    });
    helper::check_difference(stdSort.data(), cubRadixSort.data(), n);

    decltype(input) cudaMergeV0(input);
    tictoc("cuda merge(v0)", [&] {
        cuda_merge_sort_v0(cudaMergeV0.data(), cudaMergeV0.size());
    });
    helper::check_difference(stdSort.data(), cudaMergeV0.data(), n);

    decltype(input) cudaBitonicV0(input);
    tictoc("cuda bitonic(v0)", [&] {
        cuda_bitonic_sort_v0(cudaBitonicV0.data(), cudaBitonicV0.size());
    });
    helper::check_difference(stdSort.data(), cudaBitonicV0.data(), n);

    decltype(input) cudaQuickV0(input);
    tictoc("cuda quick(v0)", [&] {
        cuda_quick_sort_v0(cudaQuickV0.data(), cudaQuickV0.size());
    });
    helper::check_difference(stdSort.data(), cudaQuickV0.data(), n);

    decltype(input) cudaQuickV1(input);
    tictoc("cuda quick(v1)", [&] {
        cuda_quick_sort_v1(cudaQuickV1.data(), cudaQuickV1.size());
    });
    helper::check_difference(stdSort.data(), cudaQuickV1.data(), n);

    decltype(input) cudaMergeV1(input);
    tictoc("cuda merge(v1)", [&] {
        cuda_quick_sort_v1(cudaMergeV1.data(), cudaMergeV1.size());
    });
    helper::check_difference(stdSort.data(), cudaMergeV1.data(), n);

    decltype(input) thrustSort(input);
    tictoc("thrust sort", [&] {
        thrust_sort(thrustSort.data(), thrustSort.size());
    });
    helper::check_difference(stdSort.data(), thrustSort.data(), n);
}
