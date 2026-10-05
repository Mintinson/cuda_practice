#ifndef MATMUL_VEC_CUH_
#define MATMUL_VEC_CUH_

// #include "matmul_reg.cuh"
#include <cuda_runtime.h>

// =====================================================================
// Vectorized Warp-Tiled SGEMM
//
// 相比 sgemm_warp_tiling 的增量优化：
//   1. 全局内存 → 共享内存：用 float4 向量化加载，减少 4× 指令数
//   2. A 在共享内存中转置存储（smem_A_T[BK][BM]），使 reg_A 的读取
//      地址连续，从而也能向量化
//   3. B 保持行主序（smem_B[BK][BN]），读写两端都可向量化
//
// 共享内存布局：
//   smem_A_T : [BLOCK_K][BLOCK_M]   ← 转置，为向量化读取服务
//   smem_B   : [BLOCK_K][BLOCK_N]   ← 原始行主序
// =====================================================================
template <int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int WARP_M, int WARP_N,
          int THREAD_TILE_M, int THREAD_TILE_N,
          typename T>
__global__ void sgemm_warp_tiling_vec4(const T *__restrict__ A,
                                       const T *__restrict__ B,
                                       T *__restrict__ C,
                                       int M, int N, int K)
{
    // -------- 编译期检查：float4 要求 THREAD_TILE 是 4 的倍数 --------
    static_assert(THREAD_TILE_M % 4 == 0,
                  "reg_A 用 float4 读取时，THREAD_TILE_M 必须是 4 的倍数");
    static_assert(THREAD_TILE_N % 4 == 0,
                  "reg_B 用 float4 读取时，THREAD_TILE_N 必须是 4 的倍数");
    static_assert(BLOCK_K % 4 == 0,
                  "转置加载 A 时，BLOCK_K 必须是 4 的倍数");

    constexpr int WARP_SIZE = 32;
    constexpr int WARP_TILE_M = WARP_M * THREAD_TILE_M;
    constexpr int WARP_TILE_N = WARP_N * THREAD_TILE_N;
    constexpr int WARPS_M = BLOCK_M / WARP_TILE_M;
    constexpr int WARPS_N = BLOCK_N / WARP_TILE_N;
    constexpr int THREADS = WARPS_M * WARPS_N * WARP_SIZE;

    // -------- 共享内存：A 转置，B 保持 --------
    __shared__ T smem_A_T[BLOCK_K][BLOCK_M]; // 转置存储
    __shared__ T smem_B[BLOCK_K][BLOCK_N];   // 原始布局

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int global_row0 = block_row * BLOCK_M;
    const int global_col0 = block_col * BLOCK_N;

    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    // -------- 线程层次分解（与 sgemm_warp_tiling 相同）--------
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    const int warp_grid_row = warp_id / WARPS_N;
    const int warp_grid_col = warp_id % WARPS_N;
    const int lane_grid_row = lane_id / WARP_N;
    const int lane_grid_col = lane_id % WARP_N;

    const int tile_row0 = (warp_grid_row * WARP_M + lane_grid_row) * THREAD_TILE_M;
    const int tile_col0 = (warp_grid_col * WARP_N + lane_grid_col) * THREAD_TILE_N;

    T reg_A[THREAD_TILE_M];
    T reg_B[THREAD_TILE_N];
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {T(0)};

    // =================================================================
    // 主循环
    // =================================================================
    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
    {
        // -------------------------------------------------------------
        // 1. 向量化加载 A，并转置存入 smem_A_T
        //
        //    线程映射：每线程负责一行中的若干个 float4
        //      load_col : 该线程负责的第一个 float4 的列偏移
        //      load_row : 该线程负责的行
        //    要求 BLOCK_K 是 4 的倍数，所以列方向可以整片地按 float4 覆盖
        // -------------------------------------------------------------
        {
            constexpr int A_VEC_COLS = BLOCK_K / 4;               // BLOCK_K 方向 float4 个数
            constexpr int A_ROWS_PER_PASS = THREADS / A_VEC_COLS; // 每轮覆盖的行数
            const int a_col4 = thread_id % A_VEC_COLS;            // float4 列索引
            const int a_row = thread_id / A_VEC_COLS;             // 行索引

#pragma unroll
            for (int r = a_row; r < BLOCK_M; r += A_ROWS_PER_PASS)
            {
                const int global_row = global_row0 + r;
                const int global_col = k0 + a_col4 * 4;
                if (global_row < M && global_col + 3 < K)
                {
                    // 一次 LDG.128
                    float4 v = *reinterpret_cast<const float4 *>(
                        &A[global_row * K + global_col]);
                    // 转置：4 个元素分别写入 smem_A_T 的 4 行
                    smem_A_T[a_col4 * 4 + 0][r] = v.x;
                    smem_A_T[a_col4 * 4 + 1][r] = v.y;
                    smem_A_T[a_col4 * 4 + 2][r] = v.z;
                    smem_A_T[a_col4 * 4 + 3][r] = v.w;
                }
                else
                {
                    // 边界 fallback：标量读 + 转置写
                    for (int t = 0; t < 4; ++t)
                    {
                        const int gr = global_row;
                        const int gc = global_col + t;
                        smem_A_T[a_col4 * 4 + t][r] =
                            (gr < M && gc < K) ? A[gr * K + gc] : T(0);
                    }
                }
            }
        }

        // -------------------------------------------------------------
        // 2. 向量化加载 B（布局不变，读写都可向量化）
        // -------------------------------------------------------------
        {
            constexpr int B_VEC_COLS = BLOCK_N / 4;
            constexpr int B_ROWS_PER_PASS = THREADS / B_VEC_COLS;
            const int b_col4 = thread_id % B_VEC_COLS;
            const int b_row = thread_id / B_VEC_COLS;

#pragma unroll
            for (int r = b_row; r < BLOCK_K; r += B_ROWS_PER_PASS)
            {
                const int global_row = k0 + r;
                const int global_col = global_col0 + b_col4 * 4;
                if (global_row < K && global_col + 3 < N)
                {
                    float4 v = *reinterpret_cast<const float4 *>(
                        &B[global_row * N + global_col]);
                    *reinterpret_cast<float4 *>(&smem_B[r][b_col4 * 4]) = v;
                }
                else
                {
                    for (int t = 0; t < 4; ++t)
                    {
                        const int gr = global_row;
                        const int gc = global_col + t;
                        smem_B[r][b_col4 * 4 + t] =
                            (gr < K && gc < N) ? B[gr * N + gc] : T(0);
                    }
                }
            }
        }

        __syncthreads();

        // -------------------------------------------------------------
        // 3. 外积累加（内层循环用 float4 从共享内存批量读寄存器）
        // -------------------------------------------------------------
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk)
        {
            // ---- 3a. 向量化加载 A（从转置共享内存读连续地址）----
            // smem_A_T[kk][tile_row0 .. tile_row0 + THREAD_TILE_M - 1] 连续
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i += 4)
            {
                float4 v = *reinterpret_cast<const float4 *>(
                    &smem_A_T[kk][tile_row0 + i]);
                reg_A[i + 0] = v.x;
                reg_A[i + 1] = v.y;
                reg_A[i + 2] = v.z;
                reg_A[i + 3] = v.w;
            }

            // ---- 3b. 向量化加载 B（本身行主序，连续）----
#pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j += 4)
            {
                float4 v = *reinterpret_cast<const float4 *>(
                    &smem_B[kk][tile_col0 + j]);
                reg_B[j + 0] = v.x;
                reg_B[j + 1] = v.y;
                reg_B[j + 2] = v.z;
                reg_B[j + 3] = v.w;
            }

            // ---- 3c. 寄存器外积 ----
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j)
                    accum[i][j] += reg_A[i] * reg_B[j];
        }

        __syncthreads();
    }

    // =================================================================
    // 写回（与 sgemm_warp_tiling 相同）
    // =================================================================
#pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i)
    {
        const int global_row = global_row0 + tile_row0 + i;
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j)
        {
            const int global_col = global_col0 + tile_col0 + j;
            if (global_row < M && global_col < N)
                C[global_row * N + global_col] = accum[i][j];
        }
    }
}

#endif // MATMUL_VEC_CUH_