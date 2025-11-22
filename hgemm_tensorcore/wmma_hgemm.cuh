#ifndef WMMA_HTEMM_CUH
#define WMMA_HTEMM_CUH
#include <cstddef>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

// #include <cmath>
#include "helper.cuh"

#define LDST32BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

#define CP_ASYNC_CG(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))

namespace cuda_wmma {
using namespace nvcuda;
namespace kernels {

    // only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
    template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
    __global__ void wmma_hgemm_16x16_kernel(half* a, half* b, half* c,
        const std::size_t m, const std::size_t n, const std::size_t k)
    {
        const int NUM_K_TILES = (k + WMMA_K - 1) / WMMA_K;
        const int rowIdx = blockIdx.y * WMMA_M;
        const int colIdx = blockIdx.x * WMMA_N;
        if (rowIdx >= m || colIdx >= n)
            return;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
        wmma::fill_fragment(C_frag, 0.0); // initialize
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
            wmma::row_major>
            A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
            wmma::row_major>
            B_frag;
#pragma unroll
        for (int it = 0; it < NUM_K_TILES; ++it) {
            // Load the inputs
            wmma::load_matrix_sync(A_frag, a + rowIdx * k + it * WMMA_K, k);
            wmma::load_matrix_sync(B_frag, b + (it * WMMA_K) * n + colIdx, n);
            // perform the matrix multiplication
            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }
        // store the output
        wmma::store_matrix_sync(c + rowIdx * n + colIdx, C_frag, n, wmma::mem_row_major);
    }
} // namespace kernels

void hgemm_wmma_m16n16k16_naive(half* a, half* b, half* c,
    const std::size_t m, const std::size_t n, const std::size_t k)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    helper::DeviceDataHandler d_a(a, m * k);
    helper::DeviceDataHandler d_b(b, k * n);
    helper::DeviceDataHandler<half> d_c(n * m);
    dim3 block(32);
    dim3 grid((n + WMMA_N - 1) / WMMA_N, (m + WMMA_M - 1) / WMMA_M);

    kernels::wmma_hgemm_16x16_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
        d_a.data, d_b.data, d_c.data, m, n, k);
    d_c.cpyToHost(c);
}

namespace kernels {
    __device__ void print_arr(half* arr, int m, int n)
    {
        for (int i = m; i < n; i++)
            printf("%.2f ", float(arr[i]));
        printf("\n");
    }
    // m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
    template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
        const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
    __global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(half* a, half* b, half* c,
        int m, int n, int k)
    {
        // 256 threads(8 warps) per block.
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int NUM_K_TILES = (k + WMMA_M - 1) / (WMMA_K);
        constexpr int BM = WMMA_M * WMMA_TILE_M; // 16x4=64
        constexpr int BN = WMMA_N * WMMA_TILE_N; // 16x2=32
        constexpr int BK = WMMA_K; // 16
        __shared__ half s_a[BM][BK], s_b[BK][BN]; // 64x16x2=2KB, 16x32x2=1KB

        // Ensure every thread within a warp executes the same instruction
        // warp_id 0 -> warp_m 0, warp_n 0
        // warp_id 1 -> warp_m 0, warp_n 1
        // warp_id 2 -> warp_m 1, warp_n 0
        // warp_id 3 -> warp_m 1, warp_n 1
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = tid / warpSize; // 0~7 warp_id within block
        // const int lane_id = tid % warpSize; // 0~31
        const int warp_m = warp_id / 2; // 0,1,2,3
        const int warp_n = warp_id % 2; // 0,1

        // 256 thread load s_a=64x16, s_b=16x32
        // 64*16/256=4, half4, 16x32/256=2, half2
        // s_a, 64*16, every thread loads 4 half, 4 threads for each row，64 rows，256 threads in total.
        const int load_smem_a_m = tid / 4; // 0~63
        const int load_smem_a_k = (tid % 4) * 4; // 0,4,12,...
        // s_b, 16x32, every thread loads 2 half, 8 threads for each row，32 rows，256 threads in total.
        const int load_smem_b_k = tid / 16; // 0~16
        const int load_smem_b_n = (tid % 16) * 2; // 0,2,4,...,32
        const int load_gmem_a_m = by * BM + load_smem_a_m; // global m
        const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n

        if (load_gmem_a_m >= m && load_gmem_b_n >= n)
            return;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
        wmma::fill_fragment(C_frag, 0.0);

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
            wmma::row_major>
            A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
            wmma::row_major>
            B_frag;

#pragma unroll
        for (int it = 0; it < NUM_K_TILES; ++it) {
            int load_gmem_a_k = it * WMMA_K + load_smem_a_k; // global col of a
            int load_gmem_a_addr = load_gmem_a_m * k + load_gmem_a_k;
            int load_gmem_b_k = it * WMMA_K + load_smem_b_k; // global row of b
            int load_gmem_b_addr = load_gmem_b_k * n + load_gmem_b_n;
            // 64 bits sync memory issues gmem_a -> smem_a.
            LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST64BITS(a[load_gmem_a_addr]));
            // 32 bits sync memory issues gmem_b -> smem_b.
            LDST32BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST32BITS(b[load_gmem_b_addr]));
            __syncthreads();

            wmma::load_matrix_sync(A_frag, &s_a[warp_m * WMMA_M][0],
                BK); // BM*BK, BK=WMMA_K
            wmma::load_matrix_sync(B_frag, &s_b[0][warp_n * WMMA_N],
                BN); // BK=BN, BK=WMMA_K

            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

            __syncthreads();
        }

        const int store_gmem_a_m = by * BM + warp_m * WMMA_M;
        const int store_gmem_a_n = bx * BN + warp_n * WMMA_N;
        wmma::store_matrix_sync(c + store_gmem_a_m * n + store_gmem_a_n, C_frag, n,
            wmma::mem_row_major);
    }

} // namespace kernels

void hgemm_wmma_m16n16k16_mma4x2(half* a, half* b, half* c,
    const std::size_t m, const std::size_t n, const std::size_t k)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int BlockSize = 256;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    helper::DeviceDataHandler d_a(a, m * k);
    helper::DeviceDataHandler d_b(b, k * n);
    helper::DeviceDataHandler<half> d_c(n * m);
    dim3 block(BlockSize);
    dim3 grid((n + WMMA_TILE_N * WMMA_N - 1) / (WMMA_TILE_N * WMMA_N),
        (m + WMMA_TILE_M * WMMA_M - 1) / (WMMA_TILE_M * WMMA_M));

    kernels::hgemm_wmma_m16n16k16_mma4x2_kernel<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N><<<grid, block>>>(
        d_a.data, d_b.data, d_c.data, m, n, k);
    d_c.cpyToHost(c);
}

namespace kernels {

    // m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
    template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
        const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
        const int WARP_TILE_M = 2, const int WARP_TILE_N = 4>
    __global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel(half* a, half* b,
        half* c, int m,
        int n, int k)
    {
        // 256 threads(8 warps) per block.
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int NUM_K_TILES = (k + WMMA_K - 1) / (WMMA_K);
        constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
        constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
        constexpr int BK = WMMA_K; // 16
        __shared__ half s_a[BM][BK], s_b[BK][BN]; // 16x128x2=4KB

        // Ensure every thread within a warp executes the same instruction
        // warp_id 0 -> warp_m 0, warp_n 0
        // warp_id 1 -> warp_m 0, warp_n 1
        // warp_id 2 -> warp_m 1, warp_n 0
        // warp_id 3 -> warp_m 1, warp_n 1
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = tid / warpSize; // 0~7 warp_id within block
        // const int lane_id = tid % warpSize; // 0~31
        const int warp_m = warp_id / 2; // 0,1,2,3
        const int warp_n = warp_id % 2; // 0,1

        // 0. compute the indices of shared memory first
        // for s_a, 16 data each row，every thread reads 8 data，2 threads for a row；
        // 128 rows，128x2 = 256 threads in total
        int load_smem_a_m = tid / 2; // row 0~127
        int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
        // for s_b 128 data each row，every thread reads 8 data，16 threads for a row
        // 16 rows，16x16=256 threads in total
        int load_smem_b_k = tid / 16; // row 0~15
        int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
        // 1. compute the indices of global memory
        // every block needs to write BM*BN in C
        int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
        int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
        if (load_gmem_a_m >= m || load_gmem_b_n >= n)
            return;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
            C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::fill_fragment(C_frag[i][j], 0.0);
            }
        }
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
            wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
            wmma::row_major>
            B_frag[WARP_TILE_N];
#pragma unroll
        for (int it = 0; it < NUM_K_TILES; ++it) {
            int load_gmem_a_k = it * WMMA_K + load_smem_a_k; // global col of a
            int load_gmem_a_addr = load_gmem_a_m * k + load_gmem_a_k;
            int load_gmem_b_k = it * WMMA_K + load_smem_b_k; // global row of b
            int load_gmem_b_addr = load_gmem_b_k * n + load_gmem_b_n;
            LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST128BITS(b[load_gmem_b_addr]));
            LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST128BITS(a[load_gmem_a_addr]));
            __syncthreads();

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
                const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                wmma::load_matrix_sync(A_frag[i], &s_a[warp_smem_a_m][0],
                    BK); // BM*BK, BK=WMMA_K
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
                const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                wmma::load_matrix_sync(B_frag[j], &s_b[0][warp_smem_b_n],
                    BN); // BM*BK, BK=WMMA_K
            }

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                wmma::store_matrix_sync(c + store_gmem_a_m * n + store_gmem_a_n,
                    C_frag[i][j], n, wmma::mem_row_major);
            }
        }
    }
} // namespace kernels

void hgemm_wmma_m16n16k16_mma4x2_warp2x4(half* a, half* b, half* c,
    const std::size_t m, const std::size_t n, const std::size_t k)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int BlockSize = 256;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;

    helper::DeviceDataHandler d_a(a, m * k);
    helper::DeviceDataHandler d_b(b, k * n);
    helper::DeviceDataHandler<half> d_c(n * m);
    dim3 block(BlockSize);
    dim3 grid((n + WARP_TILE_N * WMMA_TILE_N * WMMA_N - 1) / (WARP_TILE_N * WMMA_TILE_N * WMMA_N),
        (m + WARP_TILE_M * WMMA_TILE_M * WMMA_M - 1) / (WARP_TILE_M * WMMA_TILE_M * WMMA_M));

    kernels::hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel<WMMA_M, WMMA_N, WMMA_K,
        WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M, WARP_TILE_N>
        <<<grid, block>>>(
            d_a.data, d_b.data, d_c.data, m, n, k);
    d_c.cpyToHost(c);
}

namespace kernels {

    // Double buffers
    template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
        const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
        const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
        const int OFFSET = 0>
    __global__ void
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel(half* a, half* b, half* c,
        int m, int n, int k)
    {
        // 256 threads(8 warps) per block.
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int NUM_K_TILES = (k + WMMA_K - 1) / WMMA_K;
        constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
        constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
        constexpr int BK = WMMA_K; // 16
        // 16x128x2=4KB, 4+4=8KB, padding to reduce bank conflicts.
        __shared__ half s_a[2][BM][BK + OFFSET], s_b[2][BK][BN + OFFSET];

        // ensure all threads under the same warp execute the same instructions
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = tid / warpSize; // 0~7 warp_id within block
        // const int lane_id = tid % warpSize; // 0~31
        const int warp_m = warp_id / 2; // 0,1,2,3
        const int warp_n = warp_id % 2; // 0,1

        // 0. compute the indices of shared memory first
        // for s_a, 16 data each row，every thread reads 8 data，2 threads for a row；
        // 128 rows，128x2 = 256 threads in total
        int load_smem_a_m = tid / 2; // row 0~127
        int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
        // for s_b 128 data each row，every thread reads 8 data，16 threads for a row
        // 16 rows，16x16=256 threads in total
        int load_smem_b_k = tid / 16; // row 0~15
        int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
        // 1. compute the indices of global memory
        // every block needs to write BM*BN in C
        int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
        int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
        if (load_gmem_a_m >= m || load_gmem_b_n >= n)
            return;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
            C_frag[WARP_TILE_M][WARP_TILE_N];
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
            wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
            wmma::row_major>
            B_frag[WARP_TILE_N];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::fill_fragment(C_frag[i][j], 0.0);
            }
        }

        // k = 0 is loading here, buffer 0
        {
            int load_gmem_a_k = load_smem_a_k; // global col of a
            int load_gmem_a_addr = load_gmem_a_m * k + load_gmem_a_k;
            int load_gmem_b_k = load_smem_b_k; // global row of b
            int load_gmem_b_addr = load_gmem_b_k * n + load_gmem_b_n;

            uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&s_a[0][load_smem_a_m][load_smem_a_k]);
            CP_ASYNC_CG(load_smem_a_ptr, &a[load_gmem_a_addr], 16);

            uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[0][load_smem_b_k][load_smem_b_n]);
            CP_ASYNC_CG(load_smem_b_ptr, &b[load_gmem_b_addr], 16);

            CP_ASYNC_COMMIT_GROUP();
            CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();

#pragma unroll
        for (int it = 1; it < NUM_K_TILES; ++it) { // start from 1
            int smem_sel = (it - 1) & 1; // k 1->0, k 2->1, k 3->0, ...
            int smem_sel_next = it & 1; // k 1->1, k 2->0, k 3->1, ...

            int load_gmem_a_k = it * WMMA_K + load_smem_a_k; // global col of a
            int load_gmem_a_addr = load_gmem_a_m * k + load_gmem_a_k;
            int load_gmem_b_k = it * WMMA_K + load_smem_b_k; // global row of b
            int load_gmem_b_addr = load_gmem_b_k * n + load_gmem_b_n;

            uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
                &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
            CP_ASYNC_CG(load_smem_a_ptr, &a[load_gmem_a_addr], 16);

            uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
                &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
            CP_ASYNC_CG(load_smem_b_ptr, &b[load_gmem_b_addr], 16);
            CP_ASYNC_COMMIT_GROUP();

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
                const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0],
                    BK + OFFSET);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
                const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n],
                    BN + OFFSET);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }

            CP_ASYNC_WAIT_GROUP(0);

            __syncthreads();
        }

        // processing last k tile
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                wmma::row_major>
                A_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                wmma::row_major>
                B_frag[WARP_TILE_N];

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
                const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][0], BK + OFFSET);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
                const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                wmma::load_matrix_sync(B_frag[j], &s_b[1][0][warp_smem_b_n], BN + OFFSET);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }
        }

// finally, store back to C matrix.
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                wmma::store_matrix_sync(c + store_gmem_a_m * n + store_gmem_a_n,
                    C_frag[i][j], n, wmma::mem_row_major);
            }
        }
    }
} // namespace kernels
template <std::size_t Padding = 0>
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(half* a, half* b, half* c,
    const std::size_t m, const std::size_t n, const std::size_t k)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int BlockSize = 256;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;

    helper::DeviceDataHandler d_a(a, m * k);
    helper::DeviceDataHandler d_b(b, k * n);
    helper::DeviceDataHandler<half> d_c(n * m);
    dim3 block(BlockSize);
    dim3 grid((n + WARP_TILE_N * WMMA_TILE_N * WMMA_N - 1) / (WARP_TILE_N * WMMA_TILE_N * WMMA_N),
        (m + WARP_TILE_M * WMMA_TILE_M * WMMA_M - 1) / (WARP_TILE_M * WMMA_TILE_M * WMMA_M));

    kernels::hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel<WMMA_M, WMMA_N, WMMA_K,
        WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M, WARP_TILE_N, Padding>
        <<<grid, block>>>(
            d_a.data, d_b.data, d_c.data, m, n, k);
    d_c.cpyToHost(c);
}

}

#endif // WMMA_HTEMM_CUH