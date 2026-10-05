#ifndef MATMUL_REG_CUH_
#define MATMUL_REG_CUH_

#include <cuda_runtime.h>

// ----------------------------------------------------------------
// 把 A 的 [BM × BK] 分块从全局内存加载到共享内存 As[BM][BK]
//
// 线程映射：
//   col = tid % BK   → 同 warp 内相邻线程读相邻列 → 合并访存
//   row = tid / BK   → 沿 BM 方向以 rows_per_iter 为步长跨步
//
// 举例（BM=128, BK=8, THREADS=256）：
//   rows_per_iter = 256 / 8 = 32
//   每线程处理 128 / 32 = 4 行
//   一个 warp（32 线程）覆盖 4 行 × 8 列 = 4 个 32 字节扇区（全部有效）
// ----------------------------------------------------------------
template <int BM, int BK, int THREADS, typename T>
__device__ __forceinline__ void load_tile_A(
    const T *__restrict__ A, T As[BM][BK],
    int global_row0, int global_col0, // tile 左上角的全局坐标 (by*BM, bk)
    int tid, int M, int K)
{
    constexpr int rows_per_iter = THREADS / BK;

    const int col = tid % BK; // 分块内的列
    const int row = tid / BK; // 分块内的起始行

#pragma unroll
    for (int i = row; i < BM; i += rows_per_iter)
    {
        const int global_row = global_row0 + i;
        const int global_col = global_col0 + col;
        As[i][col] = (global_row < M && global_col < K)
                         ? A[global_row * K + global_col]
                         : T(0);
    }
}

// ----------------------------------------------------------------
// 把 B 的 [BK × BN] 分块从全局内存加载到共享内存 Bs[BK][BN]
//
// 线程映射：
//   col = tid % BN   → 同 warp 内相邻线程读相邻列 → 完美合并
//   row = tid / BN   → 沿 BK 方向以 rows_per_iter 为步长跨步
//
// 举例（BK=8, BN=128, THREADS=256）：
//   rows_per_iter = 256 / 128 = 2
//   每线程处理 8 / 2 = 4 行
//   一个 warp（32 线程）完全落在同一行，读取 128 字节连续数据
// ----------------------------------------------------------------
template <int BK, int BN, int THREADS, typename T>
__device__ __forceinline__ void load_tile_B(
    const T *__restrict__ B, T Bs[BK][BN],
    int global_row0, int global_col0, // tile 左上角的全局坐标 (bk, bx*BN)
    int tid, int K, int N)
{
    constexpr int rows_per_iter = THREADS / BN;

    const int col = tid % BN;
    const int row = tid / BN;

#pragma unroll
    for (int i = row; i < BK; i += rows_per_iter)
    {
        const int global_row = global_row0 + i;
        const int global_col = global_col0 + col;
        Bs[i][col] = (global_row < K && global_col < N)
                         ? B[global_row * N + global_col]
                         : T(0);
    }
}

// =====================================================================
// Thread-Tiled SGEMM:  C[M,N] = A[M,K] * B[K,N]
//
// 每个 block 计算 C 的一个 BLOCK_M × BLOCK_N 分块。
// 与 block_tiling 版本的区别：
//   - 每线程在 C 中负责一块 THREAD_TILE_M × THREAD_TILE_N 的寄存器 tile
//   - K 维度内层循环中，先把 A/B 的片段加载到寄存器数组 reg_A/reg_B，
//     再在寄存器上做外积累加，避免每个 FMA 都访问共享内存
//
// 坐标约定（与 block_tiling 版本保持一致）：
//   - "全局坐标" : 指向 A/B/C 原始矩阵，写作 global_row / global_col
//   - "分块坐标" : 指向共享内存 tile 或寄存器 tile，写作 local_*
// =====================================================================
template <int BLOCK_M,       // block 覆盖的 C 行数（如 128）
          int BLOCK_N,       // block 覆盖的 C 列数（如 128）
          int BLOCK_K,       // K 维度每次迭代的深度（如 8）
          int THREAD_TILE_M, // 每线程负责的 C 行数（如 8）
          int THREAD_TILE_N, // 每线程负责的 C 列数（如 8）
          typename T>
__global__ void sgemm_thread_tiling(const T *__restrict__ A,
                                    const T *__restrict__ B,
                                    T *__restrict__ C,
                                    int M, int N, int K)
{
    // -----------------------------------------------------------------
    // 编译期推导：每 block 需要多少线程
    //   BLOCK_M × BLOCK_N 的 C 分块，每线程负责 THREAD_TILE_M × THREAD_TILE_N
    //   → 线程网格形状: (BLOCK_M / THREAD_TILE_M) × (BLOCK_N / THREAD_TILE_N)
    // -----------------------------------------------------------------
    constexpr int THREADS = (BLOCK_M / THREAD_TILE_M) *
                            (BLOCK_N / THREAD_TILE_N); // 例: (128/8)*(128/8) = 256

    [[maybe_unused]] constexpr int THREAD_ROWS = BLOCK_M / THREAD_TILE_M; // 线程网格行数（如 16）
    constexpr int THREAD_COLS = BLOCK_N / THREAD_TILE_N;                  // 线程网格列数（如 16）

    // -----------------------------------------------------------------
    // 共享内存：每次 K 迭代加载的 A / B 分块
    // -----------------------------------------------------------------
    __shared__ T smem_A[BLOCK_M][BLOCK_K];
    __shared__ T smem_B[BLOCK_K][BLOCK_N];

    // -----------------------------------------------------------------
    // 当前 block 在 C 中的左上角全局坐标
    // -----------------------------------------------------------------
    // const int block_row = blockIdx.y; // M 方向的块索引
    // const int block_col = blockIdx.x; // N 方向的块索引
    const int global_row0 = blockIdx.y * BLOCK_M;
    const int global_col0 = blockIdx.x * BLOCK_N;

    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    // =================================================================
    // 线程映射：把 256 个线程摊平到 C 分块上的 16×16 网格
    //
    //   每个线程负责一块 THREAD_TILE_M × THREAD_TILE_N 的子块
    //
    //   thread_id  →  (row_in_grid, col_in_grid)
    //       0      →   (0, 0)
    //       1      →   (0, 1)
    //      ...
    //      15      →   (0, 15)
    //      16      →   (1, 0)
    //      ...
    //
    //   线程网格布局:
    //      grid_row = thread_id / THREAD_COLS
    //      grid_col = thread_id % THREAD_COLS
    //
    //   tile 内起始坐标:
    //      tile_row0 = grid_row * THREAD_TILE_M
    //      tile_col0 = grid_col * THREAD_TILE_N
    // =================================================================
    const int tile_row0 = (thread_id / THREAD_COLS) * THREAD_TILE_M;
    const int tile_col0 = (thread_id % THREAD_COLS) * THREAD_TILE_N;

    // -----------------------------------------------------------------
    // 寄存器数组
    //   reg_A[i] : A 分块中第 k 列的第 i 个元素（当前线程负责的行）
    //   reg_B[j] : B 分块中第 k 行的第 j 个元素（当前线程负责的列）
    //   accum[i][j] : 该线程负责的 THREAD_TILE_M × THREAD_TILE_N 个输出
    // -----------------------------------------------------------------
    T reg_A[THREAD_TILE_M];
    T reg_B[THREAD_TILE_N];
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {T(0)};

    // =================================================================
    // 主循环：沿 K 维度迭代
    // =================================================================
    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
    {
        // -------- 1. 协作加载 A 分块 [BLOCK_M][BLOCK_K] --------
        // -------- 2. 协作加载 B 分块 [BLOCK_K][BLOCK_N] --------
        load_tile_A<BLOCK_M, BLOCK_K, THREADS>(
            A, smem_A, global_row0, k0, thread_id, M, K);
        load_tile_B<BLOCK_K, BLOCK_N, THREADS>(
            B, smem_B, k0, global_col0, thread_id, K, N);

        __syncthreads();

        // -------- 3. 外积累加：accum += smem_A[:, kk] * smem_B[kk, :] --------
        //
        //   与 block_tiling 版本的关键区别：
        //   这里先把 A 的一列、B 的一行加载到寄存器数组 reg_A / reg_B，
        //   之后 THREAD_TILE_M × THREAD_TILE_N 次 FMA 全在寄存器上完成，
        //   共享内存读取次数从 TM×TN 降为 TM+TN。
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk)
        {
            // --- 3a. 从共享内存批量加载到寄存器 ---
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
                reg_A[i] = smem_A[tile_row0 + i][kk];

#pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j)
                reg_B[j] = smem_B[kk][tile_col0 + j];

            // --- 3b. 寄存器外积 ---
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j)
                    accum[i][j] += reg_A[i] * reg_B[j];
        }

        __syncthreads();
    }

    // =================================================================
    // 写回：把 accum 里的 THREAD_TILE_M × THREAD_TILE_N 个结果写回 C
    //
    // 每个 (i, j) 对应 C 中一个唯一位置：
    //   global_row = block_row * BLOCK_M + tile_row0 + i
    //   global_col = block_col * BLOCK_N + tile_col0 + j
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

#endif // MATMUL_REG_CUH_