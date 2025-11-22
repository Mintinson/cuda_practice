#include <cuda_runtime.h>
#include "helper.cuh"
#include <cstddef>
#include "timer.cuh"
#include "random_gen.hpp"

GpuTimer timer;
constexpr std::size_t M = 2048;
constexpr std::size_t N = 512;

void transpose_cpu_naive(const float *input, float *output, std::size_t m, std::size_t n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            output[j * m + i] = input[i * n + j];
        }
    }
}

__global__ void transpose_global_32_8(const float *input, float *output, std::size_t m, std::size_t n)
{
    auto rId = threadIdx.y + blockIdx.y * blockDim.y;
    auto cId = threadIdx.x + blockIdx.x * blockDim.x;
    if (rId < m && cId < n)
        output[cId * m + rId] = input[rId * n + cId];
}

template <typename T>
__device__ constexpr auto &fetch_vec4(T *ptr)
{
    using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
    return reinterpret_cast<std::conditional_t<std::is_const_v<T>, const float4, float4> *>((ptr))[0];
}
template <typename T>
__device__ constexpr auto &fetch_vec2(T *ptr)
{
    using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
    return reinterpret_cast<std::conditional_t<std::is_const_v<T>, const float2, float2> *>((ptr))[0];
}
__global__ void transpose_global_4x4(const float *input, float *output, std::size_t m, std::size_t n)
{
    auto colIdx = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    auto rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) << 2;
    if (colIdx >= n || rowIdx >= m)
    {
        return;
    }
    float4 srcVec4[4];
    float4 dstVec4[4];
    srcVec4[0] = fetch_vec4(input + rowIdx * n + colIdx);
    srcVec4[1] = fetch_vec4(input + (rowIdx + 1) * n + colIdx);
    srcVec4[2] = fetch_vec4(input + (rowIdx + 2) * n + colIdx);
    srcVec4[3] = fetch_vec4(input + (rowIdx + 3) * n + colIdx);

    dstVec4[0] = make_float4(srcVec4[0].x, srcVec4[1].x, srcVec4[2].x, srcVec4[3].x);
    dstVec4[1] = make_float4(srcVec4[0].y, srcVec4[1].y, srcVec4[2].y, srcVec4[3].y);
    dstVec4[2] = make_float4(srcVec4[0].z, srcVec4[1].z, srcVec4[2].z, srcVec4[3].z);
    dstVec4[3] = make_float4(srcVec4[0].w, srcVec4[1].w, srcVec4[2].w, srcVec4[3].w);

    fetch_vec4(output + colIdx * m + rowIdx) = dstVec4[0];
    fetch_vec4(output + (colIdx + 1) * m + rowIdx) = dstVec4[1];
    fetch_vec4(output + (colIdx + 2) * m + rowIdx) = dstVec4[2];
    fetch_vec4(output + (colIdx + 3) * m + rowIdx) = dstVec4[3];
}
__global__ void transpose_global_2x2(const float *input, float *output, std::size_t m, std::size_t n)
{
    auto colIdx = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
    auto rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
    if (colIdx >= n || rowIdx >= m)
    {
        return;
    }
    float2 srcVec2[2];
    float2 dstVec2[2];
    srcVec2[0] = fetch_vec2(input + rowIdx * n + colIdx);
    srcVec2[1] = fetch_vec2(input + (rowIdx + 1) * n + colIdx);

    dstVec2[0] = make_float2(srcVec2[0].x, srcVec2[1].x);
    dstVec2[1] = make_float2(srcVec2[0].y, srcVec2[1].y);

    fetch_vec2(output + colIdx * m + rowIdx) = dstVec2[0];
    fetch_vec2(output + (colIdx + 1) * m + rowIdx) = dstVec2[1];
}
__global__ void transpose_global_1x2(const float *input, float *output, std::size_t m, std::size_t n)
{
    auto colIdx = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
    auto rowIdx = (blockIdx.y * blockDim.y + threadIdx.y);
    if (colIdx >= n || rowIdx >= m)
    {
        return;
    }
    float2 srcVec2;
    srcVec2 = fetch_vec2(input + rowIdx * n + colIdx);


    *(output + colIdx * m + rowIdx) = srcVec2.x;
    *(output + (colIdx + 1) * m + rowIdx) = srcVec2.y;
}
__global__ void transpose_perfect(const float *input,
                                  float *__restrict__ output, std::size_t m, std::size_t n)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx >= (n >> 2) || row_idx >= (m >> 2))
    {
        return;
    }

    int offset = (row_idx * n + col_idx) << 2; // * 4
    const float4 *input_v4 = reinterpret_cast<const float4 *>(input + offset);

    float4 src_row0 = input_v4[0];
    float4 src_row1 = input_v4[n >> 2];
    float4 src_row2 = input_v4[n >> 1];
    float4 src_row3 = input_v4[(n >> 2) * 3];

    float4 dst_row0 = make_float4(src_row0.x, src_row1.x, src_row2.x, src_row3.x);
    float4 dst_row1 = make_float4(src_row0.y, src_row1.y, src_row2.y, src_row3.y);
    float4 dst_row2 = make_float4(src_row0.z, src_row1.z, src_row2.z, src_row3.z);
    float4 dst_row3 = make_float4(src_row0.w, src_row1.w, src_row2.w, src_row3.w);

    offset = (col_idx * m + row_idx) << 2;
    float4 *dst_v4 = reinterpret_cast<float4 *>(output + offset);
    dst_v4[0] = dst_row0;
    dst_v4[m >> 2] = dst_row1;
    dst_v4[m >> 1] = dst_row2;
    dst_v4[(m >> 2) * 3] = dst_row3;
}

void cuda_transpose_gloabl_32_8(const float *input, float *output, std::size_t m, std::size_t n)
{
    helper::DeviceDataHandler d_input(input, m * n);
    helper::DeviceDataHandler<float> d_output(m * n);

    dim3 block(32, 8);
    dim3 grid((n + block.x) / block.x, (m + block.y) / block.y);
    for (int i = 0; i < 5; ++i)
    {
        timer.start();
        transpose_global_32_8<<<grid, block>>>(d_input.data, d_output.data, m, n);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "32 x 8: " << timer.elapsed() << " " << timer.unit() << "\n";
    }

    d_output.cpyToHost(output);
}
void cuda_transpose_gloabl_16_16(const float *input, float *output, std::size_t m, std::size_t n)
{
    helper::DeviceDataHandler d_input(input, m * n);
    helper::DeviceDataHandler<float> d_output(m * n);

    dim3 block(16, 16);
    dim3 grid((n + block.x) / block.x, (m + block.y) / block.y);
    for (int i = 0; i < 5; ++i)
    {
        timer.start();
        transpose_global_32_8<<<grid, block>>>(d_input.data, d_output.data, m, n);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "16 x 16: " << timer.elapsed() << " " << timer.unit() << "\n";
    }

    d_output.cpyToHost(output);
}
void cuda_transpose_gloabl_8_32(const float *input, float *output, std::size_t m, std::size_t n)
{
    helper::DeviceDataHandler d_input(input, m * n);
    helper::DeviceDataHandler<float> d_output(m * n);

    dim3 block(8, 32);
    dim3 grid((n + block.x) / block.x, (m + block.y) / block.y);
    for (int i = 0; i < 5; ++i)
    {
        timer.start();
        transpose_global_32_8<<<grid, block>>>(d_input.data, d_output.data, m, n);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "8 x 32: " << timer.elapsed() << " " << timer.unit() << "\n";
    }

    d_output.cpyToHost(output);
}
void cuda_transpose_gloabl_32_8_4x4(const float *input, float *output, std::size_t m, std::size_t n)
{
    helper::DeviceDataHandler d_input(input, m * n);
    helper::DeviceDataHandler<float> d_output(m * n);

    dim3 block(32, 8);
    const std::size_t TildeSize = 4;
    dim3 grid((n + block.x * TildeSize - 1) / (block.x * TildeSize), (m + block.y * TildeSize - 1) / (block.y * TildeSize));
    for (int i = 0; i < 1; ++i)
    {
        timer.start();
        transpose_global_4x4<<<grid, block>>>(d_input.data, d_output.data, m, n);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "32 x 8(4x4): " << timer.elapsed() << " " << timer.unit() << "\n";
    }

    d_output.cpyToHost(output);
}
void cuda_transpose_gloabl_16_16_4x4(const float *input, float *output, std::size_t m, std::size_t n)
{
    helper::DeviceDataHandler d_input(input, m * n);
    helper::DeviceDataHandler<float> d_output(m * n);

    dim3 block(16, 16);
    const std::size_t TildeSize = 4;
    dim3 grid((n + block.x * TildeSize - 1) / (block.x * TildeSize), (m + block.y * TildeSize - 1) / (block.y * TildeSize));

    for (int i = 0; i < 1; ++i)
    {
        timer.start();
        transpose_perfect<<<grid, block>>>(d_input.data, d_output.data, m, n);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "16 x 16(4x4): " << timer.elapsed() << " " << timer.unit() << "\n";
    }

    d_output.cpyToHost(output);
}
void cuda_transpose_gloabl_8_32_4x4(const float *input, float *output, std::size_t m, std::size_t n)
{
    helper::DeviceDataHandler d_input(input, m * n);
    helper::DeviceDataHandler<float> d_output(m * n);

    dim3 block(8, 32);
    const std::size_t TildeSize = 4;
    dim3 grid((n + block.x * TildeSize - 1) / (block.x * TildeSize), (m + block.y * TildeSize - 1) / (block.y * TildeSize));
    for (int i = 0; i < 1; ++i)
    {
        timer.start();
        transpose_global_4x4<<<grid, block>>>(d_input.data, d_output.data, m, n);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "8 x 32(4x4): " << timer.elapsed() << " " << timer.unit() << "\n";
    }

    d_output.cpyToHost(output);
}
void cuda_transpose_gloabl_8_32_2x2(const float *input, float *output, std::size_t m, std::size_t n)
{
    helper::DeviceDataHandler d_input(input, m * n);
    helper::DeviceDataHandler<float> d_output(m * n);

    dim3 block(8, 32);
    const std::size_t TildeSize = 2;
    dim3 grid((n + block.x * TildeSize - 1) / (block.x * TildeSize), (m + block.y * TildeSize - 1) / (block.y * TildeSize));
    for (int i = 0; i < 1; ++i)
    {
        timer.start();
        transpose_global_2x2<<<grid, block>>>(d_input.data, d_output.data, m, n);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "8 x 32(2x2): " << timer.elapsed() << " " << timer.unit() << "\n";
    }

    d_output.cpyToHost(output);
}
void cuda_transpose_gloabl_8_32_1x2(const float *input, float *output, std::size_t m, std::size_t n)
{
    helper::DeviceDataHandler d_input(input, m * n);
    helper::DeviceDataHandler<float> d_output(m * n);

    dim3 block(8, 32);
    const std::size_t TildeSize = 2;
    dim3 grid((n + block.x * 1 - 1) / (block.x * 1), (m + block.y * TildeSize - 1) / (block.y * TildeSize));
    for (int i = 0; i < 1; ++i)
    {
        timer.start();
        transpose_global_2x2<<<grid, block>>>(d_input.data, d_output.data, m, n);
        cudaDeviceSynchronize();
        timer.stop();
        std::cout << "8 x 32(1x2): " << timer.elapsed() << " " << timer.unit() << "\n";
    }

    d_output.cpyToHost(output);
}

void print_matrix(float *data, int startR, int endR, int startC, int endC, int depth)
{
    for (int i = startR; i != endR; ++i)
    {
        for (int j = startC; j != endC; ++j)
        {
            std::cout << data[i * depth + j] << " ";
        }
        std::cout << "\n";
    }
}
int main()
{
    auto inputA = helper::generate_sequence<float>(M * N);
    decltype(inputA) outputB(inputA.size());

    transpose_cpu_naive(inputA.data(), outputB.data(), M, N);

    decltype(inputA) gpuOut1(inputA.size());
    cuda_transpose_gloabl_32_8(inputA.data(), gpuOut1.data(), M, N);
    helper::check_difference(outputB.data(), gpuOut1.data(), M * N);

    decltype(inputA) gpuOut2(inputA.size());
    cuda_transpose_gloabl_16_16(inputA.data(), gpuOut2.data(), M, N);
    helper::check_difference(outputB.data(), gpuOut2.data(), M * N);

    decltype(inputA) gpuOut3(inputA.size());
    cuda_transpose_gloabl_8_32(inputA.data(), gpuOut3.data(), M, N);
    helper::check_difference(outputB.data(), gpuOut3.data(), M * N);

    decltype(inputA) gpuOut4(inputA.size());
    cuda_transpose_gloabl_32_8_4x4(inputA.data(), gpuOut4.data(), M, N);
    helper::check_difference(outputB.data(), gpuOut4.data(), M * N);

    decltype(inputA) gpuOut5(inputA.size());
    cuda_transpose_gloabl_16_16_4x4(inputA.data(), gpuOut5.data(), M, N);
    helper::check_difference(outputB.data(), gpuOut5.data(), M * N);

    decltype(inputA) gpuOut6(inputA.size());
    cuda_transpose_gloabl_8_32_4x4(inputA.data(), gpuOut6.data(), M, N);
    helper::check_difference(outputB.data(), gpuOut6.data(), M * N);

    decltype(inputA) gpuOut7(inputA.size());
    cuda_transpose_gloabl_8_32_2x2(inputA.data(), gpuOut7.data(), M, N);
    helper::check_difference(outputB.data(), gpuOut7.data(), M * N);

    decltype(inputA) gpuOut8(inputA.size());
    cuda_transpose_gloabl_8_32_1x2(inputA.data(), gpuOut8.data(), M, N);
    helper::check_difference(outputB.data(), gpuOut8.data(), M * N);
}