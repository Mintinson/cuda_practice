#ifndef MATMUL_CP_ASYNC_CUH_
#define MATMUL_CP_ASYNC_CUH_

#include <cuda_runtime.h>
#include <type_traits>

// =====================================================================
// cp.async 内联 PTX 封装
//
// .cg 变体：只走 L2，不分配 L1（tile 数据在本 block 内只读一次，
//           占 L1 没有收益）；16B 是 .cg 唯一合法宽度
// =====================================================================
__device__ __forceinline__ void cp_async_cg_16(void *smem_dst,
                                               const void *gmem_src)
{
    unsigned smem_addr = static_cast<unsigned>(
        __cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// =====================================================================
// A tile 的 XOR swizzle
//
// cp.async 只能做 16B 连续直拷，无法在搬运时转置，所以 A tile 只能按
// 全局的行主序存进 smem_A[m][k]。代价是：reg_A 每步读的是"固定 k、跨
// THREAD_TILE_M 行"的列片段。行距 = BLOCK_K 个 float：
//   bank(row) = (row * BLOCK_K + k) % 32
// warp 内 4 个行组（Z-order 映射，行距 8 行）在 BLOCK_K=8 时全部落进
// 同一个 bank，读 reg_A 会产生 4 路 conflict。
//
// 解决：把每行的 16B 块按行号做 XOR 旋转：
//   chunk' = chunk ^ ((row >> 3) & 3)
// 同一 warp 读到的 4 个行组 (row>>3) 互不相同，4 个行组被旋转到 4 个
// 不同的 16B 块上 -> bank 互不相同，reg_A 读取零 conflict。
// 存储侧：每个 warp 恰好覆盖 8 行的 16B 块，(row>>3) 在 warp 内为常数，
// 旋转不改变"8 个 lane 覆盖 32 个 bank 各一次"的性质，store 同样零 conflict。
//
// 这要求每行至少有 4 个 16B 块，即 BLOCK_K >= 16。
// =====================================================================
template <int BLOCK_K>
__device__ __forceinline__ int a_chunk_swizzle(int row, int chunk)
{
    static_assert(BLOCK_K % 16 == 0,
                  "A XOR swizzle needs BLOCK_K >= 16 (four 16B chunks per row)");
    return chunk ^ ((row >> 3) & 3);
}

// =====================================================================
// 用 cp.async 加载 A tile 到 smem_A[BLOCK_M][BLOCK_K]（行主序 + swizzle）
//
//   每个 16B 拷贝覆盖一行的 4 个连续 k 值。以 BLOCK_K=16、THREADS=256
//   为例：128 行 x 4 块 / 256 线程 = 每线程恰好 1 次拷贝。
// =====================================================================
template <int BLOCK_M, int BLOCK_K, int THREADS, typename T>
__device__ __forceinline__ void cp_async_load_A(
    const T *__restrict__ A,
    T (&smem_A)[BLOCK_M][BLOCK_K],
    int global_row0, int k0, int thread_id, int M, int K)
{
    static_assert(BLOCK_K % 4 == 0, "cp.async 要求 BLOCK_K 是 4 的倍数");
    static_assert(THREADS % (BLOCK_K / 4) == 0,
                  "THREADS 必须能被 BLOCK_K/4 整除");

    constexpr int A_VEC_COLS = BLOCK_K / 4; // 每行的 16B 块数
    constexpr int A_ROWS_PER_PASS = THREADS / A_VEC_COLS;

    const int a_col4 = thread_id % A_VEC_COLS;
    const int a_row = thread_id / A_VEC_COLS;

#pragma unroll
    for (int m = a_row; m < BLOCK_M; m += A_ROWS_PER_PASS)
    {
        const int global_row = global_row0 + m;
        const int global_col = k0 + a_col4 * 4;
        // 写入位置要经过与读取端一致的 swizzle
        const int chunk = a_chunk_swizzle<BLOCK_K>(m, a_col4);

        if (global_row < M && global_col + 3 < K)
        {
            cp_async_cg_16(&smem_A[m][chunk * 4],
                           &A[global_row * K + global_col]);
        }
        else
        {
            // 边界：cp.async 不负责边界处理，逐元素回退（含补 0）
#pragma unroll
            for (int t = 0; t < 4; ++t)
            {
                const int gc = global_col + t;
                smem_A[m][chunk * 4 + t] =
                    (global_row < M && gc < K) ? A[global_row * K + gc] : T(0);
            }
        }
    }
}

// =====================================================================
// 用 cp.async 加载 B tile 到 smem_B[BLOCK_K][BLOCK_N]（行主序，无需转置）
// =====================================================================
template <int BLOCK_K, int BLOCK_N, int THREADS, typename T>
__device__ __forceinline__ void cp_async_load_B(
    const T *__restrict__ B,
    T (&smem_B)[BLOCK_K][BLOCK_N],
    int k0, int global_col0, int thread_id, int K, int N)
{
    static_assert(BLOCK_N % 4 == 0, "cp.async 要求 BLOCK_N 是 4 的倍数");
    static_assert(THREADS % (BLOCK_N / 4) == 0,
                  "THREADS 必须能被 BLOCK_N/4 整除");

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
            cp_async_cg_16(&smem_B[r][b_col4 * 4],
                           &B[global_row * N + global_col]);
        }
        else
        {
#pragma unroll
            for (int t = 0; t < 4; ++t)
            {
                const int gc = global_col + t;
                smem_B[r][b_col4 * 4 + t] =
                    (global_row < K && gc < N) ? B[global_row * N + gc] : T(0);
            }
        }
    }
}

// =====================================================================
// 从行主序 + swizzle 的 smem_A 读 reg_A
//
//   reg_A[i] 对应 C 的行 tile_row0 + i，本步 k = kk。
//   逐标量读取（4B）；swizzle 保证 warp 内零 bank conflict。
// =====================================================================
template <int BLOCK_M, int BLOCK_K, int THREAD_TILE_M, typename T>
__device__ __forceinline__ void load_reg_A_rowmajor(
    const T (&smem_A)[BLOCK_M][BLOCK_K],
    T (&reg_A)[THREAD_TILE_M],
    int tile_row0, int kk)
{
#pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i)
    {
        const int row = tile_row0 + i;
        const int col = a_chunk_swizzle<BLOCK_K>(row, kk >> 2) * 4 + (kk & 3);
        reg_A[i] = smem_A[row][col];
    }
}

// =====================================================================
// 从 smem_B[BLOCK_K][BLOCK_N] 读 reg_B（行主序，float4 向量化）
// =====================================================================
template <int BLOCK_K, int BLOCK_N, int THREAD_TILE_N, typename T>
__device__ __forceinline__ void load_reg_B_rowmajor(
    const T (&smem_B)[BLOCK_K][BLOCK_N],
    T (&reg_B)[THREAD_TILE_N],
    int tile_col0, int kk)
{
    static_assert(THREAD_TILE_N % 4 == 0, "THREAD_TILE_N 必须是 4 的倍数");
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4)
    {
        const float4 v = *reinterpret_cast<const float4 *>(
            &smem_B[kk][tile_col0 + j]);
        reg_B[j + 0] = v.x;
        reg_B[j + 1] = v.y;
        reg_B[j + 2] = v.z;
        reg_B[j + 3] = v.w;
    }
}

// =====================================================================
// 主 kernel: 多级 cp.async 流水线 SGEMM（A/B 全异步）
//
// 与第 9 节双缓冲版本（LDG+STS）的区别：
//   1. A 和 B 都用 cp.async 直拷进 smem，稳态下线程内没有
//      "LDG 等数据 -> STS" 的同步链，全局内存延迟完全被
//      NUM_STAGES-1 级预取覆盖，线程只需发拷贝、算外积、等组。
//   2. A 的行主序 smem + XOR swizzle 代替"STS 时转置"。
//   3. BLOCK_K=16：每级流水线 16KB，同步粒度减半；也正好让 A 的
//      swizzle 有 4 个 16B 块可旋转。NUM_STAGES=3 时 smem 恰好 48KB。
//
// 同步协议（与第 10 节一致）：
//   prologue 预取 NUM_STAGES-1 个 tile，每个 tile 一个 commit_group；
//   主循环每迭代发射 1 个 tile（1 个 group），wait_group<NUM_STAGES-2>
//   保证"下一迭代要算的 tile 已就绪、最多还有 NUM_STAGES-2 个在飞"。
// =====================================================================
template <int BLOCK_M,       // block 覆盖的 C 行数（如 128）
          int BLOCK_N,       // block 覆盖的 C 列数（如 128）
          int BLOCK_K,       // K 维度每次迭代的深度（推荐 16）
          int WARP_M,        // warp 内 M 方向的 lane 数（如 4）
          int WARP_N,        // warp 内 N 方向的 lane 数（如 8）
          int THREAD_TILE_M, // 每 lane 负责的 C 行数（如 8）
          int THREAD_TILE_N, // 每 lane 负责的 C 列数（如 8）
          int NUM_STAGES,    // 流水线级数（推荐 3）
          typename T>
__global__ void sgemm_cp_async_pipeline(const T *__restrict__ A,
                                        const T *__restrict__ B,
                                        T *__restrict__ C,
                                        int M, int N, int K)
{
    // -------- 编译期检查 --------
    static_assert(NUM_STAGES >= 2, "pipeline at least 2 stages");
    static_assert(BLOCK_K % 16 == 0, "A 的 XOR swizzle 要求 BLOCK_K >= 16");
    static_assert(BLOCK_M % 4 == 0 && BLOCK_N % 4 == 0,
                  "cp.async 要求 BLOCK_M/N 是 4 的倍数");
    static_assert(THREAD_TILE_M % 4 == 0 && THREAD_TILE_N % 4 == 0,
                  "向量化寄存器加载要求 THREAD_TILE 是 4 的倍数");
    // 静态共享内存上限 48KB（全局形状需为 4 的倍数才能安全做 16B 拷贝，
    // 基准形状 M=N=K 均满足）
    static_assert(NUM_STAGES * (BLOCK_M + BLOCK_N) * BLOCK_K *
                      static_cast<int>(sizeof(T)) <=
                  48 * 1024,
                  "NUM_STAGES * tile size exceeds 48KB static smem limit");

    // -------- 编译期推导 --------
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_TILE_M = WARP_M * THREAD_TILE_M;
    constexpr int WARP_TILE_N = WARP_N * THREAD_TILE_N;
    constexpr int WARPS_M = BLOCK_M / WARP_TILE_M;
    constexpr int WARPS_N = BLOCK_N / WARP_TILE_N;
    constexpr int THREADS = WARPS_M * WARPS_N * WARP_SIZE;

    // -------- 多级 Shared Memory buffer --------
    __shared__ T smem_A[NUM_STAGES][BLOCK_M][BLOCK_K]; // A 行主序 + swizzle
    __shared__ T smem_B[NUM_STAGES][BLOCK_K][BLOCK_N]; // B 行主序

    // -------- block 坐标 --------
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int global_row0 = block_row * BLOCK_M;
    const int global_col0 = block_col * BLOCK_N;

    // -------- 线程层次分解 --------
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    const int warp_grid_row = warp_id / WARPS_N;
    const int warp_grid_col = warp_id % WARPS_N;

    // Z-order lane 映射（第 8 节优化）
    const int lane_row = lane_id % 2 + (lane_id / 16) * 2;
    const int lane_col = (lane_id % 16) / 2;

    const int tile_row0 = (warp_grid_row * WARP_M + lane_row) * THREAD_TILE_M;
    const int tile_col0 = (warp_grid_col * WARP_N + lane_col) * THREAD_TILE_N;

    // -------- 寄存器数组（p-Loop 双缓冲）--------
    T reg_A[2][THREAD_TILE_M];
    T reg_B[2][THREAD_TILE_N];
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {T(0)};

    // -------- 总 K 迭代次数 --------
    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    // =================================================================
    // Prologue：预取前 NUM_STAGES - 1 个 tile
    // =================================================================
#pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; ++s)
    {
        if (s < num_k_tiles)
        {
            const int k0 = s * BLOCK_K;
            cp_async_load_A<BLOCK_M, BLOCK_K, THREADS>(
                A, smem_A[s], global_row0, k0, thread_id, M, K);
            cp_async_load_B<BLOCK_K, BLOCK_N, THREADS>(
                B, smem_B[s], k0, global_col0, thread_id, K, N);
        }
        cp_async_commit_group();
    }

    // 等待第一个 tile 就绪（允许 NUM_STAGES - 2 个 tile 还在飞）
    cp_async_wait_group<NUM_STAGES - 2>();
    __syncthreads();

    // 预取第一个 tile 的寄存器
    load_reg_A_rowmajor<BLOCK_M, BLOCK_K, THREAD_TILE_M>(
        smem_A[0], reg_A[0], tile_row0, 0);
    load_reg_B_rowmajor<BLOCK_K, BLOCK_N, THREAD_TILE_N>(
        smem_B[0], reg_B[0], tile_col0, 0);

    // =================================================================
    // 主循环
    // =================================================================
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile)
    {
        const int compute_stage = k_tile % NUM_STAGES;
        const int next_fetch = k_tile + NUM_STAGES - 1;

        // ---- 发射下一 tile 的 cp.async（异步，立即返回）----
        if (next_fetch < num_k_tiles)
        {
            const int load_stage = next_fetch % NUM_STAGES;
            const int k0 = next_fetch * BLOCK_K;
            cp_async_load_A<BLOCK_M, BLOCK_K, THREADS>(
                A, smem_A[load_stage], global_row0, k0, thread_id, M, K);
            cp_async_load_B<BLOCK_K, BLOCK_N, THREADS>(
                B, smem_B[load_stage], k0, global_col0, thread_id, K, N);
        }
        cp_async_commit_group();

        // ---- p-Loop：BLOCK_K 步外积 ----
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk)
        {
            // 预取 kk+1 步的寄存器
            if (kk + 1 < BLOCK_K)
            {
                load_reg_A_rowmajor<BLOCK_M, BLOCK_K, THREAD_TILE_M>(
                    smem_A[compute_stage], reg_A[(kk + 1) & 1], tile_row0,
                    kk + 1);
                load_reg_B_rowmajor<BLOCK_K, BLOCK_N, THREAD_TILE_N>(
                    smem_B[compute_stage], reg_B[(kk + 1) & 1], tile_col0,
                    kk + 1);
            }

            // 外积
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j)
                    accum[i][j] += reg_A[kk & 1][i] * reg_B[kk & 1][j];
        }

        // ---- 等待下一 tile 就绪（允许 NUM_STAGES - 2 个 tile 在飞）----
        cp_async_wait_group<NUM_STAGES - 2>();
        __syncthreads();

        // ---- 预取新 tile 的 kk=0 寄存器 ----
        if (k_tile + 1 < num_k_tiles)
        {
            const int next_compute = (k_tile + 1) % NUM_STAGES;
            load_reg_A_rowmajor<BLOCK_M, BLOCK_K, THREAD_TILE_M>(
                smem_A[next_compute], reg_A[0], tile_row0, 0);
            load_reg_B_rowmajor<BLOCK_K, BLOCK_N, THREAD_TILE_N>(
                smem_B[next_compute], reg_B[0], tile_col0, 0);
        }
    }

    // =================================================================
    // 写回 C
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

#endif // MATMUL_CP_ASYNC_CUH_
