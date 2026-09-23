#ifndef HELPER_CUH_
#define HELPER_CUH_

#include "checker.cuh"
#include <cstddef>
#include <cuda_runtime.h>
#include <string>

#ifdef _MSC_VER
#define INLINE __forceinline
#elif defined(__GNUC__)
#define INLINE inline __attribute__((__always_inline__))
#else
#define INLINE inline
#endif

namespace helper
{
    namespace details
    {
        __global__ void print_compile_capacity()
        {
#if defined(__CUDA_ARCH__)
            printf("Compile Compute Capability: %d.%d\n",
                   __CUDA_ARCH__ / 100,
                   (__CUDA_ARCH__ % 100) / 10);
#endif
        }

        constexpr const char *get_compiler_name()
        {
#if defined(_MSC_VER)
            return "MSVC";
#elif defined(__clang__)
            return "Clang";
#elif defined(__GNUC__)
            return "GCC";
#else
            return "Unknown Compiler";
#endif
        }

        inline std::string get_compiler_version()
        {
            std::string version;
#if defined(_MSC_VER)
            version = std::to_string(_MSC_VER);
#if defined(_MSC_FULL_VER)
            version += " (Full: " + std::to_string(_MSC_FULL_VER) + ")";
#endif
#elif defined(__clang__)
            version = std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
            version = std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#endif
            return version;
        }

        constexpr const char *get_build_mode()
        {
#if defined(NDEBUG)
            return "Release";
#else
            return "Debug";
#endif
        }

        constexpr const char *get_cpp_standard()
        {
#if defined(_MSC_VER)
            constexpr long cpluplus = _MSVC_LANG;
#else
            constexpr long cpluplus = __cplusplus;
#endif

            switch (cpluplus)
            {
            case 199711L:
                return "C++98";
            case 201103L:
                return "C++11";
            case 201402L:
                return "C++14";
            case 201703L:
                return "C++17";
            case 202002L:
                return "C++20";
            case 202302L:
                return "C++23";
            default:
                return "Unknown";
            }
        }
    }

    template <typename T>
    struct DeviceDataHandler
    {
        explicit DeviceDataHandler(const std::size_t n)
            : size(n)
        {
            checkCudaErrors(cudaMalloc(&data, n * sizeof(T)));
        }

        DeviceDataHandler(std::size_t n, T init)
            : size(n)
        {
            checkCudaErrors(cudaMalloc(&data, n * sizeof(T)));
            checkCudaErrors(cudaMemset(data, init, n * sizeof(T)));
        }

        DeviceDataHandler(const T *src, std::size_t n, std::size_t start = 0, bool host = true)
            : size(n)
        {
            checkCudaErrors(cudaMalloc(&data, n * sizeof(T)));
            if (host)
            {
                checkCudaErrors(cudaMemcpy(data + start, src, (n - start) * sizeof(T),
                                           cudaMemcpyHostToDevice));
            }
            else
            {
                checkCudaErrors(cudaMemcpy(data + start, src, (n - start) * sizeof(T),
                                           cudaMemcpyDeviceToDevice));
            }
        }

        template <typename F>
        DeviceDataHandler(std::size_t n, F &&f)
            : size(n)
        {
            checkCudaErrors(cudaMalloc(&data, n * sizeof(T)));
            // checkCudaErrors(cudaMemset(data, 0, n * sizeof(T)));
            f(data);
        }

        DeviceDataHandler(const DeviceDataHandler &rhs)
            : DeviceDataHandler(rhs.size, rhs.data, false)
        {
        }

        T singleDataToHost(const std::size_t n) const
        {
            T hostData{};
            checkCudaErrors(cudaMemcpy(&hostData, data + n, 1 * sizeof(T), cudaMemcpyDeviceToHost));
            return hostData;
        }

        ~DeviceDataHandler()
        {
            checkCudaErrors(cudaFree(data));
        }

        void cpyToHost(T *hostData, const std::size_t start, const std::size_t n) const
        {
            checkCudaErrors(cudaMemcpy(hostData, data + start, n * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void cpyToHost(T *hostData) const
        {
            checkCudaErrors(cudaMemcpy(hostData, data, size * sizeof(T), cudaMemcpyDeviceToHost));
        }

        using value_type = T;

        T *data{nullptr};
        std::size_t size{0};
    };

    struct DevicePrintOptions
    {
        int deviceId{0};
        bool compilerInfo{true};
        bool driverAndRuntimeInfo{true};
        bool deviceInfo{true};

        bool memoryInfo{true};
        bool bandwidthInfo{true};
    };

    inline void print_device_info(DevicePrintOptions option = DevicePrintOptions{})
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount <= option.deviceId)
        {
            printf("find %d cuda device, but specify %d device\n", deviceCount, option.deviceId);
            return;
        }
        if (option.compilerInfo)
        {
            printf("######################## HOST INFO #############################\n");
            std::cout << "Compiler Info:\n"
                      << "  Name:    " << details::get_compiler_name() << "\n"
                      << "  Version: " << details::get_compiler_version() << "\n"
                      << "  Mode:    " << details::get_build_mode() << "\n"
                      << "  C++ Std: " << details::get_cpp_standard() << "\n"
                      << "  C++ Macro Value: " << __cplusplus << "\n";
        }
        cudaSetDevice(option.deviceId);
        printf("######################## CUDA INFO #############################\n");

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, option.deviceId);
        if (option.driverAndRuntimeInfo)
        {
            int driverVersion = 0, runtimeVersion = 0;
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);
            printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
        }
        if (option.deviceInfo)
        {

            printf("Device Name: %s\n", deviceProp.name);
            printf("Device Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);

            details::print_compile_capacity<<<1, 1>>>();
            cudaDeviceSynchronize();
            if (option.memoryInfo)
            {
                printf("    Total Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
                printf("    L2 Cache Size: %.2f MB\n", deviceProp.l2CacheSize / (1024.0 * 1024.0));
                printf("    Shared Mem per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
                printf("    Shared Mem per SM: %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
            }
            if (option.bandwidthInfo)
            {
                // memoryClockRate 单位是 kHz，memoryBusWidth 单位是 bits
                // 乘以 2 是因为 GDDR 显存在时钟的上升沿和下降沿都传输数据 (Double Data Rate)
                // 除以 8 将 bits 转换为 bytes
                // 除以 1.0e6 将 kHz 转换为 GHz (等价于 10^9 转换 Bytes 为 GB)
                double peakBandwidth = 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8.0) / 1.0e6;
                printf("    Device Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
                printf("    Device Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
                printf("    Device Peak Memory Bandwidth (GB/s): %.2f\n", peakBandwidth);
            }
        }

        printf("######################## END #############################\n\n");
        // printf("CUDA Toolkit version: %d.%d.%d\n",
        //     __CUDACC_VER_MAJOR__,
        //     __CUDACC_VER_MINOR__,
        //     __CUDACC_VER_BUILD__);
    }

    namespace details
    {
#pragma optimize("", off)

        inline void compiler_must_force_sink(void const *)
        {
        }

#pragma optimize("", on)

        struct compiler_must_not_elide_fn
        {
            template <typename T>
            INLINE void operator()(T const &t) const noexcept
            {
                compiler_must_force_sink(&t);
            }
        };

        inline constexpr compiler_must_not_elide_fn compiler_must_not_elide{};
    }

    template <class T>
    INLINE void do_not_optimize_away(const T &datum)
    {
        details::compiler_must_not_elide(datum);
    }
} // namespace helper

#endif // HELPER_CUH_
