#ifndef MATMUL_WARP_CUH_
#define MATMUL_WARP_CUH_

#include <cuda_runtime.h>
#include "matmul_reg.cuh"

// =====================================================================
// Warp-Tiled SGEMM:  C[M,N] = A[M,K] * B[K,N]
//
// 在 Thread Tiling 的基础上，进一步优化 warp 内的 lane 排布：
//   - 让 warp 内 32 个 lane 排列成 WARP_M × WARP_N 的二维网格
//   - 每个 lane 负责 THREAD_TILE_M × THREAD_TILE_N 的输出子块
//   - 同一 M 位置的 WARP_N 个 lane 共享 reg_A 的读取 → 硬件广播
//   - 同一 N 位置的 WARP_M 个 lane 共享 reg_B 的读取 → 硬件广播
//   - 实测 4×8 或 8×4 的 warp 形状计算访存比最优
//
// 线程层级（以 BLOCK_M=BLOCK_N=128、THREAD_TILE=8、WARP=4×8 为例）：
//   Block (128 × 128)
//     └── 8 个 warp，排成 4 × 2 网格
//           └── 每个 warp (32 × 64)
//                 └── 32 个 lane，排成 4 × 8 网格
//                       └── 每个 lane (8 × 8)
// =====================================================================
template <int BLOCK_M,       // block 覆盖的 C 行数（如 128）
          int BLOCK_N,       // block 覆盖的 C 列数（如 128）
          int BLOCK_K,       // K 维度每次迭代的深度（如 8）
          int WARP_M,        // warp 内 M 方向的 lane 数（如 4）
          int WARP_N,        // warp 内 N 方向的 lane 数（如 8）
          int THREAD_TILE_M, // 每 lane 负责的 C 行数（如 8）
          int THREAD_TILE_N, // 每 lane 负责的 C 列数（如 8）
          typename T>
__global__ void sgemm_warp_tiling(const T *__restrict__ A,
                                  const T *__restrict__ B,
                                  T *__restrict__ C,
                                  int M, int N, int K)
{
    // -----------------------------------------------------------------
    // 编译期推导
    // -----------------------------------------------------------------
    constexpr int WARP_SIZE = 32;
    static_assert(WARP_M * WARP_N == WARP_SIZE,
                  "WARP_M × WARP_N 必须恰好等于 32");
    static_assert(BLOCK_M % (WARP_M * THREAD_TILE_M) == 0,
                  "BLOCK_M 必须能被 WARP_M × THREAD_TILE_M 整除");
    static_assert(BLOCK_N % (WARP_N * THREAD_TILE_N) == 0,
                  "BLOCK_N 必须能被 WARP_N × THREAD_TILE_N 整除");

    // 每个 warp 覆盖的 C 子块尺寸
    constexpr int WARP_TILE_M = WARP_M * THREAD_TILE_M; // 如 4*8 = 32
    constexpr int WARP_TILE_N = WARP_N * THREAD_TILE_N; // 如 8*8 = 64

    // block 内 warp 的网格形状
    constexpr int WARPS_M = BLOCK_M / WARP_TILE_M;         // 如 128/32 = 4
    constexpr int WARPS_N = BLOCK_N / WARP_TILE_N;         // 如 128/64 = 2
    constexpr int THREADS = WARPS_M * WARPS_N * WARP_SIZE; // 如 256

    // -----------------------------------------------------------------
    // 共享内存
    // -----------------------------------------------------------------
    __shared__ T smem_A[BLOCK_M][BLOCK_K];
    __shared__ T smem_B[BLOCK_K][BLOCK_N];

    // -----------------------------------------------------------------
    // block 在 C 中的左上角
    // -----------------------------------------------------------------
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int global_row0 = block_row * BLOCK_M;
    const int global_col0 = block_col * BLOCK_N;

    // -----------------------------------------------------------------
    // 线程层次分解
    //   thread_id  →  warp_id  →  lane_id
    //              →  warp_grid_(row, col)   ：warp 在 block 中的位置
    //              →  lane_grid_(row, col)   ：lane 在 warp 中的位置
    // -----------------------------------------------------------------
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    const int warp_grid_row = warp_id / WARPS_N;
    const int warp_grid_col = warp_id % WARPS_N;

    const int lane_grid_row = lane_id / WARP_N;
    const int lane_grid_col = lane_id % WARP_N;

    // -----------------------------------------------------------------
    // 该线程负责的 THREAD_TILE_M × THREAD_TILE_N 子块在 block 内的起点
    //   行：warp_grid_row 个 warp 行 + lane_grid_row 个 lane 行
    //   列：warp_grid_col 个 warp 列 + lane_grid_col 个 lane 列
    // -----------------------------------------------------------------
    const int tile_row0 = (warp_grid_row * WARP_M + lane_grid_row) * THREAD_TILE_M;
    const int tile_col0 = (warp_grid_col * WARP_N + lane_grid_col) * THREAD_TILE_N;

    // -----------------------------------------------------------------
    // 寄存器数组
    // -----------------------------------------------------------------
    T reg_A[THREAD_TILE_M];
    T reg_B[THREAD_TILE_N];
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {T(0)};

    // =================================================================
    // 主循环
    // =================================================================
    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
    {
        load_tile_A<BLOCK_M, BLOCK_K, THREADS>(
            A, smem_A, global_row0, k0, thread_id, M, K);
        load_tile_B<BLOCK_K, BLOCK_N, THREADS>(
            B, smem_B, k0, global_col0, thread_id, K, N);
        __syncthreads();

        // -------------------------------------------------------------
        // 外积累加
        //
        // 与 thread_tiling 的核心区别在于线程的 tile 排布：
        //   - thread_tiling 用 "thread_id / THREAD_COLS" 线性摊平，
        //     相邻 thread_id 的 tile 列相邻，warp 内 32 个线程散布在
        //     整个 block 上 → 几乎无广播机会
        //   - warp_tiling 让 warp 内的 32 个 lane 紧凑排成 WARP_M × WARP_N，
        //     使同一 M 位置的 WARP_N 个 lane 读取相同 smem_A → 广播
        //     使同一 N 位置的 WARP_M 个 lane 读取相同 smem_B → 广播
        // -------------------------------------------------------------
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk)
        {
            // 从共享内存批量加载到寄存器
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
                reg_A[i] = smem_A[tile_row0 + i][kk];

#pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j)
                reg_B[j] = smem_B[kk][tile_col0 + j];

            // 寄存器外积
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j)
                    accum[i][j] += reg_A[i] * reg_B[j];
        }

        __syncthreads();
    }

    // =================================================================
    // 写回
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

#endif // MATMUL_WARP_CUH_