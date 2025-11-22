//
// Created by asus on 2025/4/21.
//

#ifndef MATMUL_V8_CUH
#define MATMUL_V8_CUH
template <typename T>
__device__ constexpr auto& fetch_vec4(T* ptr)
{
    using DecayType = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<DecayType, float>) {
        return reinterpret_cast<float4*>((ptr))[0];
    } else if constexpr (std::is_same_v<DecayType, double>) {
        return reinterpret_cast<double4*>((ptr))[0];

    } else if constexpr (std::is_same_v<DecayType, int>) {
        return reinterpret_cast<int4*>((ptr))[0];
    } else if constexpr (std::is_same_v<DecayType, unsigned int>) {
        return reinterpret_cast<uint4*>((ptr))[0];
    }
}
template <typename T>
__global__ void matmul_v8_kernel(const T* A, const T* B, T* C, size_t row, size_t col, size_t depth)
{
    extern __shared__ T sharedMem[];
    T* sharedA = sharedMem;
    T* sharedB = sharedMem + blockDim.x * 2;
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    T* fillPtr = nullptr;
    if (tid < blockDim.x / 2)
    {
        fillPtr = sharedA;
    }
    else
    {
        fillPtr   = sharedB;
    }
    const auto blockRow = blockIdx.y * blockDim.y;
    const auto blockCol = blockIdx.x * blockDim.x;
    auto rowId = threadIdx.y;
    auto colId = threadIdx.x;
    T sum{};
    for (size_t j = 0; j < depth; j += blockDim.x)
    {
        sharedA[rowId * blockDim.x + colId] = A[(rowId + blockRow) * depth + j + colId];
        // sharedA[rowId * blockDim.x + colId] = static_cast<T>(0);
        sharedB[rowId * blockDim.x + colId] = B[(j + rowId) * col + colId + blockCol];
        // sharedB[rowId * blockDim.x + colId] = static_cast<T>(0);
        __syncthreads();
        for (size_t k = 0; k < blockDim.x; ++k)
        {
            sum += sharedA[rowId * blockDim.x + k] * sharedB[k * blockDim.x + colId];
        }
        __syncthreads();
    }
    // printf("blockRow: %d, blockCol: %d, rowId: %d, colId: %d, sum: %f\n", blockRow, blockCol, rowId, colId, sum);
    if (rowId + blockRow < row && colId + blockCol < col)
    {
        C[(rowId + blockRow) * col + colId + blockCol]
            = sum;
    }
}

#endif //MATMUL_V8_CUH
