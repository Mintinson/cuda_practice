/*
This example demonstrates how to use cutlass to compute a batched strided gemm in two different ways:
  1. By specifying pointers to the first matrices of the batch and the stride between the consecutive
     matrices of the batch (this is called a strided batched gemm).
  2. By copying pointers to all matrices of the batch to the device memory (this is called an array gemm).
In this example, both A and B matrix are non-transpose and column major matrix
batched_C = batched_A x batched_B
As an example, matrix C can be seen as
-----------------------------------------------------------
(0,0,0) | (0,0,1) | (0,0,2) | (1,0,0) | (1,0,1) | (1,0,2) |
-----------------------------------------------------------
(0,1,0) | (0,1,1) | (0,1,2) | (1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------------------------------------
(0,2,0) | (0,2,1) | (0,2,2) | (1,2,0) | (1,2,1) | (1,2,2) |
-----------------------------------------------------------
(0,3,0) | (0,3,1) | (0,3,2) | (1,3,0) | (1,3,1) | (1,3,2) |
-----------------------------------------------------------
(0,4,0) | (0,4,1) | (0,4,2) | (1,4,0) | (1,4,1) | (1,4,2) |
-----------------------------------------------------------
(0,5,0) | (0,5,1) | (0,5,2) | (1,5,0) | (1,5,1) | (1,5,2) |
-----------------------------------------------------------
           batch 0          |           batch 1
where we denote each element with (batch_idx, row_idx, column_idx)
In this example, batch size is 2, M is 6 and N is 3
The stride (batch_stride_C) between the first element of two batches is ldc * n

matrix A can be seen as
---------------------------------------
(0,0,0) | (0,0,1) | (1,0,0) | (1,0,1) |
---------------------------------------
(0,1,0) | (0,1,1) | (1,1,0) | (1,1,1) |
---------------------------------------
(0,2,0) | (0,2,1) | (1,2,0) | (1,2,1) |
---------------------------------------
(0,3,0) | (0,3,1) | (1,3,0) | (1,3,1) |
---------------------------------------
(0,4,0) | (0,4,1) | (1,4,0) | (1,4,1) |
---------------------------------------
(0,5,0) | (0,5,1) | (1,5,0) | (1,5,1) |
---------------------------------------
     batch 0      |      batch 1
, where batch size is 2, M is 6 and K is 2
The stride (batch_stride_A) between the first element of two batches is lda * k

matrix B can be seen as
-----------------------------
(0,0,0) | (0,0,1) | (0,0,2) |
----------------------------- batch 0
(0,1,0) | (0,1,1) | (0,1,2) |
-------------------------------------
(1,0,0) | (1,0,1) | (1,0,2) |
----------------------------- batch 1
(1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------
, where the batch size is 2, N is 3 and K is 2
The stride (batch_stride_B) between the first element of two batches is k


*/

#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"

#pragma warning(disable : 4503)

/**
 * 支持批量GEMM计算的模板类，每个批次的矩阵地址通过指针数组独立指定。
    参数：
    float：矩阵数据类型（单精度浮点）。
    cutlass::layout::ColumnMajor：列主序布局（内存中列优先存储）。
    适用场景：批量矩阵的地址不连续（如动态分配的独立内存块）。
 */
cudaError_t cutlass_array_sgemm(int m, int n, int k,
                                float alpha, float const *const *A, int lda, float const *const *B,
                                int ldb, float *const *C, int ldc, float beta, int batchCount)
{
    // 定义CUTLASS的批量GEMM模板类，支持指针数组形式的批量计算
    using Gemm = cutlass::gemm::device::GemmArray<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor>;
    Gemm gemm_op;

    // 构造参数对象，封装所有计算参数
    cutlass::Status status = gemm_op({
        {m, n, k}, // 问题维度(M, N, K)
        A,
        lda, // A矩阵指针数组及主维度
        B,
        ldb, // B矩阵指针数组及主维度
        C,
        ldc, // C矩阵指针数组及主维度（输入）
        C,
        ldc,           // D矩阵指针数组及主维度（输出，与C共用内存）
        {alpha, beta}, // 标量系数
        batchCount     // 批量数量
    });
    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;
    return cudaSuccess;
}

cudaError_t cutlass_strided_batched_sgemm(int m,
                                          int n,
                                          int k,
                                          float alpha,
                                          float const *A,
                                          int lda,
                                          long long int batchStrideA,
                                          float const *B,
                                          int ldb,
                                          long long int batchStrideB,
                                          float *C,
                                          int ldc,
                                          long long int batchStrideC,
                                          float beta,
                                          int batchCount)
{
    using Gemm = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor>;

    Gemm gemm_op;

    // 构造参数对象
    cutlass::Status status = gemm_op({
        {m, n, k},     // 问题维度
        {A, lda},      // A矩阵基地址及主维度
        batchStrideA,  // A批次步长
        {B, ldb},      // B矩阵基地址及主维度
        batchStrideB,  // B批次步长
        {C, ldc},      // C矩阵基地址及主维度（输入）
        batchStrideC,  // C批次步长
        {C, ldc},      // D矩阵基地址及主维度（输出，与C共用内存）
        batchStrideC,  // D批次步长
        {alpha, beta}, // 标量系数
        batchCount     // 批量数量
    });
    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}
template <typename T>
cudaError_t strided_batched_gemm_nn_reference(
    int m,
    int n,
    int k,
    T alpha,
    std::vector<T> const &A,
    int lda,
    long long int batch_stride_A,
    std::vector<T> const &B,
    int ldb,
    long long int batch_stride_B,
    std::vector<T> &C,
    int ldc,
    long long int batch_stride_C,
    T beta,
    int batch_count)
{
    /*
    strided batched gemm NN
    */

    cudaError_t result = cudaSuccess;

    if (A.size() < size_t(lda * k * batch_count))
    {
        std::cout << "the size of A is too small" << std::endl;
        return cudaErrorInvalidValue;
    }
    if (B.size() < size_t(ldb * n))
    {
        std::cout << "the size of B is too small" << std::endl;
        return cudaErrorInvalidValue;
    }
    if (C.size() < size_t(ldc * n * batch_count))
    {
        std::cout << "the size of C is too small" << std::endl;
        return cudaErrorInvalidValue;
    }

    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++)
    {
        for (int n_idx = 0; n_idx < n; n_idx++)
        {
            for (int m_idx = 0; m_idx < m; m_idx++)
            {
                T accum = beta * C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
                for (int k_idx = 0; k_idx < k; k_idx++)
                {
                    accum += alpha * A[batch_idx * batch_stride_A + k_idx * lda + m_idx] * B[batch_idx * batch_stride_B + n_idx * ldb + k_idx];
                }
                C[batch_idx * batch_stride_C + n_idx * ldc + m_idx] = accum;
            }
        }
    }

    return result;
}
cudaError_t run_batched_gemm(bool use_array)
{
    const char *gemm_desc = use_array ? "array" : "strided batched";
    std::cout << "Running " << gemm_desc << " gemm" << std::endl;

    // Arbitrary problem size
    int const M = 520;
    int const N = 219;
    int const K = 129;
    int const batchCount = 17;

    // A, B are non-transpose, column major
    int const lda = M;
    int const ldb = K * batchCount;
    int const ldc = M;

    int const countA = batchCount * lda * K;
    int const countB = ldb * N;
    int const countC = batchCount * ldc * N;

    // the memory is batched along K dimension
    long long int batchStrideA = static_cast<long long int>(lda) * static_cast<long long int>(K);
    long long int batchStrideB = static_cast<long long int>(K);
    long long int batchStrideC = static_cast<long long int>(ldc) * static_cast<long long int>(N);

    // alpha and beta
    float alpha = 1.0f;
    float beta = 2.0f;

    cudaError_t result = cudaSuccess;

    // allocate the host memory
    std::vector<float> host_A(countA);
    std::vector<float> host_B(countB);
    std::vector<float> host_C(countC);
    std::vector<float> result_C(countC);

    // allocate the device memory
    float *d_A;
    float *d_B;
    float *d_C;

    result = cudaMalloc(&d_A, countA * sizeof(float));
    if (result != cudaSuccess)
    {
        std::cerr << "cudaMalloc result = " << result << std::endl;
        return result;
    }
    result = cudaMalloc(&d_B, countB * sizeof(float));
    if (result != cudaSuccess)
    {
        std::cerr << "cudaMalloc result = " << result << std::endl;
        return result;
    }
    result = cudaMalloc(&d_C, countC * sizeof(float));
    if (result != cudaSuccess)
    {
        std::cerr << "cudaMalloc result = " << result << std::endl;
        return result;
    }

    // Limit range to avoid floating-point errors
    int const kRange = 8;

    // fill A
    for (int b_idx = 0; b_idx < batchCount; b_idx++)
    {
        for (int col_idx = 0; col_idx < K; col_idx++)
        {
            for (int row_idx = 0; row_idx < M; row_idx++)
            {
                host_A[row_idx + col_idx * lda + b_idx * lda * K] = static_cast<float>((row_idx + col_idx * lda + b_idx * lda * K) % kRange);
            }
        }
    }
    // fill B
    for (int b_idx = 0; b_idx < batchCount; b_idx++)
    {
        for (int col_idx = 0; col_idx < N; col_idx++)
        {
            for (int row_idx = 0; row_idx < K; row_idx++)
            {
                host_B[row_idx + col_idx * ldb + b_idx * K] = static_cast<float>(((N + K * ldb + batchCount * K) - (row_idx + col_idx * ldb + b_idx * K)) % kRange);
            }
        }
    }
    // fill C
    for (int b_idx = 0; b_idx < batchCount; b_idx++)
    {
        for (int col_idx = 0; col_idx < N; col_idx++)
        {
            for (int row_idx = 0; row_idx < M; row_idx++)
            {
                host_C[row_idx + col_idx * ldc + b_idx * ldc * N] = 1.f;
            }
        }
    }

    // ref memory
    std::vector<float> ref_A(host_A);
    std::vector<float> ref_B(host_B);
    std::vector<float> ref_C(host_C);
    // copy host memory to device
    cudaMemcpy(d_A, host_A.data(), countA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B.data(), countB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, host_C.data(), countC * sizeof(float), cudaMemcpyHostToDevice);

    // run cutlass
    if (use_array)
    {
        // allocate the host memory for the pointers to the matrices of the batch
        std::vector<float *> hostPtr_A(batchCount);
        std::vector<float *> hostPtr_B(batchCount);
        std::vector<float *> hostPtr_C(batchCount);

        // permute the batch elements to emphasize that GemmArray does not depend on matrices being separated by a fixed stride
        std::vector<size_t> permutation = {14, 11, 3, 10, 1, 13, 9, 4, 6, 16, 8, 15, 7, 12, 0, 2, 5};
        for (std::size_t b_idx = 0; b_idx < batchCount; ++b_idx)
        {
            hostPtr_A[b_idx] = d_A + permutation[b_idx] * batchStrideA;
            hostPtr_B[b_idx] = d_B + permutation[b_idx] * batchStrideB;
            hostPtr_C[b_idx] = d_C + permutation[b_idx] * batchStrideC;
        }

        // allocate the corresponding device memory
        const float **ptrA;
        const float **ptrB;
        float **ptrC;

        cudaMalloc(&ptrA, batchCount * sizeof(float *));
        cudaMalloc(&ptrB, batchCount * sizeof(float *));
        cudaMalloc(&ptrC, batchCount * sizeof(float *));

        cudaMemcpy(ptrA, hostPtr_A.data(), batchCount * sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(ptrB, hostPtr_B.data(), batchCount * sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(ptrC, hostPtr_C.data(), batchCount * sizeof(float *), cudaMemcpyHostToDevice);

        result = cutlass_array_sgemm(M, N, K, alpha, ptrA, lda, ptrB, ldb, ptrC, ldc, beta, batchCount);

        if (result != cudaSuccess)
            return result;
    }
    else
    {
        result = cutlass_strided_batched_sgemm(
            M, N, K, alpha, d_A, lda, batchStrideA, d_B, ldb, batchStrideB,
            d_C, ldc, batchStrideC, beta, batchCount);
        if (result != cudaSuccess)
            return result;
    }
    // copy device memory to host
    result = cudaMemcpy(result_C.data(), d_C, countC * sizeof(float), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        std::cerr << "cudaMemcpy result = " << result << std::endl;
        return result;
    }
    // compare with reference code
    result = strided_batched_gemm_nn_reference(M, N, K, alpha, ref_A, lda, batchStrideA,
                                               ref_B, ldb, batchStrideB, ref_C, ldc, batchStrideC, beta, batchCount);
    if (result != 0)
        return result;
    // Expect bit-level accuracy for this simple example
    if (ref_C != result_C)
    {
        std::cout << "CUTLASS " << gemm_desc << " gemm does not run correctly" << std::endl;
        return cudaErrorUnknown;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}

int main()
{

    cudaError_t result = cudaSuccess;
    for (bool use_array : {false, true})
    {
        result = run_batched_gemm(use_array);
        if (result == cudaSuccess)
        {
            std::cout << "Passed." << std::endl;
        }
        else
        {
            break;
        }
    }

    // Exit.
    return result == cudaSuccess ? 0 : -1;
}