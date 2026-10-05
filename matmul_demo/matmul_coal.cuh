#ifndef MATMUL_COAL_CUH_
#define MATMUL_COAL_CUH_

#include <cuda_runtime.h>

// =====================================================================
// Block-Tiled SGEMM:  C[M,N] = A[M,K] * B[K,N]
//
// 每个 block 计算 C 的一个 BLOCK_M × BLOCK_N 分块；
// 沿 K 维度按 BLOCK_K 步长迭代，每次加载一对 A/B 分块到共享内存，
// 然后做外积累加。
//
// 坐标约定：
//   - "全局坐标"  : 指向 A/B/C 原始矩阵，写作 global_row / global_col
//   - "分块坐标"  : 指向共享内存 tile 或线程寄存器 tile，写作 local_*
// =====================================================================
template <int BLOCK_M, // 每个 block 覆盖的 C 行数
          int BLOCK_N, // 每个 block 覆盖的 C 列数
          int BLOCK_K, // K 维度每次迭代的深度
          int THREADS, // 每 block 的线程数
          typename T>
__global__ void sgemm_block_tiling(const T *A, const T *B, T *C,
                                   int M, int N, int K)
{
    // -----------------------------------------------------------------
    // 共享内存：每个 K-迭代加载的 A / B 分块
    //   smem_A : [BLOCK_M][BLOCK_K]
    //   smem_B : [BLOCK_K][BLOCK_N]
    // -----------------------------------------------------------------
    __shared__ T smem_A[BLOCK_M][BLOCK_K];
    __shared__ T smem_B[BLOCK_K][BLOCK_N];

    // -----------------------------------------------------------------
    // 当前 block 在 C 中的左上角全局坐标
    // -----------------------------------------------------------------
    const int global_row0 = blockIdx.y * BLOCK_M;
    const int global_col0 = blockIdx.x * BLOCK_N;

    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    // =================================================================
    // 线程重排 (thread mapping)
    //
    // 为了让每个线程都能用合并访存的方式读写，需要把三种任务
    // 分别按各自最优的形状摊平到同一个线程网格上。
    // =================================================================

    // ---- A 分块加载：形状 [BLOCK_M][BLOCK_K] 摊平为网格 ----
    constexpr int A_COLS_PER_PASS = BLOCK_K;                   // 8
    constexpr int A_ROWS_PER_PASS = THREADS / A_COLS_PER_PASS; // 256/8 = 32
    const int a_col = thread_id % A_COLS_PER_PASS;             // A 分块内的列
    const int a_row = thread_id / A_COLS_PER_PASS;             // A 分块内的行（步长起点）

    // ---- B 分块加载：形状 [BLOCK_K][BLOCK_N] 摊平为网格 ----
    constexpr int B_COLS_PER_PASS = 32;
    [[maybe_unused]] constexpr int B_ROWS_PER_PASS = THREADS / B_COLS_PER_PASS;
    const int b_col = thread_id % B_COLS_PER_PASS;
    const int b_row = thread_id / B_COLS_PER_PASS;

    // ---- C 分块计算：形状 [BLOCK_M][BLOCK_N] 摊平为网格 ----
    constexpr int C_COLS_PER_PASS = 16;
    constexpr int C_ROWS_PER_PASS = THREADS / C_COLS_PER_PASS; // 16
    const int c_col = thread_id % C_COLS_PER_PASS;
    const int c_row = thread_id / C_COLS_PER_PASS;

    // 每线程在 C 中负责的寄存器 tile 大小
    constexpr int TILE_M = BLOCK_M / C_ROWS_PER_PASS; // 8
    constexpr int TILE_N = BLOCK_N / C_COLS_PER_PASS; // 8

    // 每线程私有的输出累加器
    T accum[TILE_M][TILE_N] = {T(0)};

    // =================================================================
    // 主循环：沿 K 维度迭代
    // =================================================================
    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
    {
        // -------- 1. 协作加载 A 分块 [BLOCK_M][BLOCK_K] --------
#pragma unroll
        for (int i = a_row; i < BLOCK_M; i += A_ROWS_PER_PASS)
        {
            const int global_row = global_row0 + i;
            const int global_col = k0 + a_col;
            smem_A[i][a_col] =
                (global_row < M && global_col < K)
                    ? A[global_row * K + global_col]
                    : T(0);
        }

        // -------- 2. 协作加载 B 分块 [BLOCK_K][BLOCK_N] --------
#pragma unroll
        for (int j = b_col; j < BLOCK_N; j += B_COLS_PER_PASS)
        {
            const int global_row = k0 + b_row;
            const int global_col = global_col0 + j;
            smem_B[b_row][j] =
                (global_row < K && global_col < N)
                    ? B[global_row * N + global_col]
                    : T(0);
        }

        __syncthreads();

        // -------- 3. 外积累加：accum += smem_A[:, kk] * smem_B[kk, :] --------
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk)
        {
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
                const int local_row = c_row + i * C_ROWS_PER_PASS;
                const T a_val = smem_A[local_row][kk]; // 提升到寄存器，避免重复读

#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    const int local_col = c_col + j * C_COLS_PER_PASS;
                    accum[i][j] += a_val * smem_B[kk][local_col];
                }
            }
        }

        __syncthreads();
    }

    // =================================================================
    // 写回：把 accum 里的 TILE_M × TILE_N 个结果写回 C
    // =================================================================
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
        const int global_row = global_row0 + c_row + i * C_ROWS_PER_PASS;
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            const int global_col = global_col0 + c_col + j * C_COLS_PER_PASS;
            if (global_row < M && global_col < N)
                C[global_row * N + global_col] = accum[i][j];
        }
    }
}

#endif // MATMUL_COAL_CUH_