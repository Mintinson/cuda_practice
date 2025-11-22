#ifndef TIMER_CUH_
#define TIMER_CUH_

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <ratio>
#include <type_traits>

struct GpuTimer {
    cudaEvent_t start_;
    cudaEvent_t stop_;

    GpuTimer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start()
    {
        cudaEventRecord(start_, 0);
    }

    void stop()
    {
        cudaEventRecord(stop_, 0);
    }
    template <typename Res = float>
    Res elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return static_cast<Res>(elapsed);
    }
    constexpr static const char* unit()
    {
        return "ms";
    }
};

template <typename Unit = std::milli, typename Clock = std::chrono::high_resolution_clock>
class StdTimer {
public:
    // template <std::invocable Call, typename... Args>
    //     requires std::invocable<Call, Args...>
    // void tictoc(const Call& call, Args&&... args)
    // {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     call(std::forward<Args>(args)...);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     // std::cout << std::format("{}", std::chrono::duration<double, std::milli> { end - start }) << std::endl;
    // }
    void start()
    {
        start_ = Clock::now();
    }
    void stop()
    {
        end_ = Clock::now();
    }
    template <typename Res = float>
    Res elapsed()
    {
        return std::chrono::duration<Res, Unit>(end_ - start_).count();
    }
    constexpr static const char* unit()
    {
        if constexpr (std::is_same_v<Unit, std::milli>) {
            return "ms";
        } else if constexpr (std::is_same_v<Unit, std::micro>) {
            return "us";
        } else if constexpr (std::is_same_v<Unit, std::nano>) {
            return "ns";
        } else {
            return "s";
        }
    }

private:
    std::chrono::time_point<Clock> start_;
    std::chrono::time_point<Clock> end_;
};
#endif // TIMER_CUH_
