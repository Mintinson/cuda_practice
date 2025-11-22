#include "helper.cuh"
#include "timer.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_types.h>

#include "checker.cuh"
#include "random_gen.hpp"

// #include <fmt/base.h>

__global__ void kernel()
{
    printf("Hello CUDA\n");
}

int main()
{

    printf("Hello World\n");
    // kernel<<<1, 10>>>();
    // cudaDeviceSynchronize();
    StdTimer timer;
    timer.start();
    int res {};
    for (int i = 0; i < 1000; ++i) {
        // int res = 10;
        res += i;
        // helper::do_not_optimize_away(res);
        // printf("res: %d\n", res);
    }
    timer.stop();
    auto elapsed = timer.elapsed();
    std::cout << "elapsed: " << elapsed << timer.unit() << std::endl;
    auto somthing = helper::generate_sequence<float>(1024 * 1024 * 32);
    decltype(somthing) res1(somthing.size());
    decltype(somthing) res2(somthing.size());
    std::inclusive_scan(somthing.begin(), somthing.end(), res1.begin());
    std::partial_sum(somthing.begin(), somthing.end(), res2.begin());
    std::cout << res2[0] << " " << res2[1] << std::endl;
    std::cout << res1[0] << " " << res1[1] << std::endl;
    helper::check_difference(res1.data(), res2.data(), res1.size());
    // fmt::println("{} {}", res, timer.unit());
}