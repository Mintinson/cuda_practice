#ifndef MATMUL_BUFFER_CUH_
#define MATMUL_BUFFER_CUH_

#include <cuda_runtime.h>

// =====================================================================
// 辅助函数 1: Global → Shared 直接加载 A 并转置（Prologue 使用）
//
//   把 A[global_row0 : global_row0+BLOCK_M, k0 : k0+BLOCK_K]
//   加载到 smem_A_T[BLOCK_K][BLOCK_M]（转置）
//
//   线程映射（按 float4 处理）：
//     a_col4 = thread_id % (BLOCK_K / 4)   → 沿 BLOCK_K 方向的 float4 列索引
//     a_row  = thread_id / (BLOCK_K / 4)   → 沿 BLOCK_M 方向的行索引起点
// =====================================================================
template <int BLOCK_M, int BLOCK_K, int THREADS, typename T>
__device__ __forceinline__ void load_tile_A_vec4(
    const T *__restrict__ A,
    T (&smem_A_T)[BLOCK_K][BLOCK_M],
    int global_row0, int k0, int thread_id, int M, int K)
{
    static_assert(BLOCK_K % 4 == 0, "load_tile_A_vec4 要求 BLOCK_K 是 4 的倍数");
    static_assert(THREADS % (BLOCK_K / 4) == 0,
                  "THREADS 必须能被 BLOCK_K/4 整除");

    constexpr int A_VEC_COLS = BLOCK_K / 4;
    constexpr int A_ROWS_PER_PASS = THREADS / A_VEC_COLS;

    const int a_col4 = thread_id % A_VEC_COLS;
    const int a_row = thread_id / A_VEC_COLS;

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
            // 边界 fallback：逐元素检查 + 转置写
#pragma unroll
            for (int t = 0; t < 4; ++t)
            {
                const int gc = global_col + t;
                smem_A_T[a_col4 * 4 + t][r] =
                    (global_row < M && gc < K)
                        ? A[global_row * K + gc]
                        : T(0);
            }
        }
    }
}

// =====================================================================
// 辅助函数 2: Global → Shared 直接加载 B（Prologue 使用）
//
//   把 B[k0 : k0+BLOCK_K, global_col0 : global_col0+BLOCK_N]
//   加载到 smem_B[BLOCK_K][BLOCK_N]（保持行主序）
// =====================================================================
template <int BLOCK_K, int BLOCK_N, int THREADS, typename T>
__device__ __forceinline__ void load_tile_B_vec4(
    const T *__restrict__ B,
    T (&smem_B)[BLOCK_K][BLOCK_N],
    int k0, int global_col0, int thread_id, int K, int N)
{
    static_assert(BLOCK_N % 4 == 0, "load_tile_B_vec4 要求 BLOCK_N 是 4 的倍数");
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
            float4 v = *reinterpret_cast<const float4 *>(
                &B[global_row * N + global_col]);
            *reinterpret_cast<float4 *>(&smem_B[r][b_col4 * 4]) = v;
        }
        else
        {
#pragma unroll
            for (int t = 0; t < 4; ++t)
            {
                const int gc = global_col + t;
                smem_B[r][b_col4 * 4 + t] =
                    (global_row < K && gc < N)
                        ? B[global_row * N + gc]
                        : T(0);
            }
        }
    }
}

// =====================================================================
// 辅助函数 3: Shared → Register 批量加载
//
//   从 smem_A_T[kk][tile_row0 .. tile_row0+THREAD_TILE_M-1]（连续）
//   和 smem_B[kk][tile_col0 .. tile_col0+THREAD_TILE_N-1]（连续）
//   以 float4 粒度加载到寄存器数组
// =====================================================================
template <int BLOCK_K, int BLOCK_M, int BLOCK_N,
          int THREAD_TILE_M, int THREAD_TILE_N, typename T>
__device__ __forceinline__ void load_shared_to_reg(
    const T (&smem_A_T)[BLOCK_K][BLOCK_M],
    const T (&smem_B)[BLOCK_K][BLOCK_N],
    T (&reg_A)[THREAD_TILE_M],
    T (&reg_B)[THREAD_TILE_N],
    int tile_row0, int tile_col0, int kk)
{
    static_assert(THREAD_TILE_M % 4 == 0, "THREAD_TILE_M 必须是 4 的倍数");
    static_assert(THREAD_TILE_N % 4 == 0, "THREAD_TILE_N 必须是 4 的倍数");

    // A：从转置 smem 读连续地址，向量化
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
    // B：行主序，向量化
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
}

// =====================================================================
// 辅助函数 4: 发射 A 的 LDG（异步）到寄存器暂存
//
//   关键：只是"发射"，不写 smem；让 LDG 数据在后台传输
// =====================================================================
template <int BLOCK_M, int BLOCK_K, int THREADS, int NUM_F4, typename T>
__device__ __forceinline__ void issue_ldg_A(
    const T *__restrict__ A,
    int global_row0, int k0,
    int thread_id, int M, int K,
    float4 (&ldg_stash)[NUM_F4])
{
    constexpr int A_VEC_COLS = BLOCK_K / 4;
    constexpr int A_ROWS_PER_PASS = THREADS / A_VEC_COLS;

    const int a_col4 = thread_id % A_VEC_COLS;
    const int a_row = thread_id / A_VEC_COLS;

#pragma unroll
    for (int idx = 0; idx < NUM_F4; ++idx)
    {
        const int r = a_row + idx * A_ROWS_PER_PASS;
        const int global_row = global_row0 + r;
        const int global_col = k0 + a_col4 * 4;

        if (global_row < M && global_col + 3 < K)
        {
            ldg_stash[idx] = *reinterpret_cast<const float4 *>(
                &A[global_row * K + global_col]);
        }
        else
        {
            float4 v;
            v.x = (global_row < M && global_col + 0 < K) ? A[global_row * K + global_col + 0] : T(0);
            v.y = (global_row < M && global_col + 1 < K) ? A[global_row * K + global_col + 1] : T(0);
            v.z = (global_row < M && global_col + 2 < K) ? A[global_row * K + global_col + 2] : T(0);
            v.w = (global_row < M && global_col + 3 < K) ? A[global_row * K + global_col + 3] : T(0);
            ldg_stash[idx] = v;
        }
    }
}

// =====================================================================
// 辅助函数 5: 发射 B 的 LDG（异步）到寄存器暂存
// =====================================================================
template <int BLOCK_K, int BLOCK_N, int THREADS, int NUM_F4, typename T>
__device__ __forceinline__ void issue_ldg_B(
    const T *__restrict__ B,
    int k0, int global_col0,
    int thread_id, int K, int N,
    float4 (&ldg_stash)[NUM_F4])
{
    constexpr int B_VEC_COLS = BLOCK_N / 4;
    constexpr int B_ROWS_PER_PASS = THREADS / B_VEC_COLS;

    const int b_col4 = thread_id % B_VEC_COLS;
    const int b_row = thread_id / B_VEC_COLS;

#pragma unroll
    for (int idx = 0; idx < NUM_F4; ++idx)
    {
        const int r = b_row + idx * B_ROWS_PER_PASS;
        const int global_row = k0 + r;
        const int global_col = global_col0 + b_col4 * 4;

        if (global_row < K && global_col + 3 < N)
        {
            ldg_stash[idx] = *reinterpret_cast<const float4 *>(
                &B[global_row * N + global_col]);
        }
        else
        {
            float4 v;
            v.x = (global_row < K && global_col + 0 < N) ? B[global_row * N + global_col + 0] : T(0);
            v.y = (global_row < K && global_col + 1 < N) ? B[global_row * N + global_col + 1] : T(0);
            v.z = (global_row < K && global_col + 2 < N) ? B[global_row * N + global_col + 2] : T(0);
            v.w = (global_row < K && global_col + 3 < N) ? B[global_row * N + global_col + 3] : T(0);
            ldg_stash[idx] = v;
        }
    }
}

// =====================================================================
// 辅助函数 6: 把 LDG 暂存写入下一块 smem_A_T（转置写入）
// =====================================================================
template <int BLOCK_M, int BLOCK_K, int THREADS, int NUM_F4, typename T>
__device__ __forceinline__ void store_ldg_A_to_shared(
    T (&smem_A_T)[BLOCK_K][BLOCK_M],
    int thread_id,
    const float4 (&ldg_stash)[NUM_F4])
{
    constexpr int A_VEC_COLS = BLOCK_K / 4;
    constexpr int A_ROWS_PER_PASS = THREADS / A_VEC_COLS;

    const int a_col4 = thread_id % A_VEC_COLS;
    const int a_row = thread_id / A_VEC_COLS;

#pragma unroll
    for (int idx = 0; idx < NUM_F4; ++idx)
    {
        const int r = a_row + idx * A_ROWS_PER_PASS;
        smem_A_T[a_col4 * 4 + 0][r] = ldg_stash[idx].x;
        smem_A_T[a_col4 * 4 + 1][r] = ldg_stash[idx].y;
        smem_A_T[a_col4 * 4 + 2][r] = ldg_stash[idx].z;
        smem_A_T[a_col4 * 4 + 3][r] = ldg_stash[idx].w;
    }
}

// =====================================================================
// 辅助函数 7: 把 LDG 暂存写入下一块 smem_B（直接写入）
// =====================================================================
template <int BLOCK_K, int BLOCK_N, int THREADS, int NUM_F4, typename T>
__device__ __forceinline__ void store_ldg_B_to_shared(
    T (&smem_B)[BLOCK_K][BLOCK_N],
    int thread_id,
    const float4 (&ldg_stash)[NUM_F4])
{
    constexpr int B_VEC_COLS = BLOCK_N / 4;
    constexpr int B_ROWS_PER_PASS = THREADS / B_VEC_COLS;

    const int b_col4 = thread_id % B_VEC_COLS;
    const int b_row = thread_id / B_VEC_COLS;

#pragma unroll
    for (int idx = 0; idx < NUM_F4; ++idx)
    {
        const int r = b_row + idx * B_ROWS_PER_PASS;
        *reinterpret_cast<float4 *>(&smem_B[r][b_col4 * 4]) = ldg_stash[idx];
    }
}

// =====================================================================
// 主 kernel: 双缓冲 + 流水线 SGEMM
//
// 两级流水线：
//   Level 1（K-Loop 级）: Global → Shared 双缓冲
//      两套 smem_A_T / smem_B，一份计算、一份预加载
//   Level 2（p-Loop 级）: Shared → Register 双缓冲
//      两组 reg_A / reg_B，一份用于当前外积，一份用于预取下一步
//
// 主循环时序（每次迭代）：
//   1. 对 kk=0，发射下一 tile 的 LDG（异步）
//   2. 循环 kk=0..BLOCK_K-1：
//        - 预取 kk+1 步的寄存器
//        - 用 kk 步的寄存器做外积
//   3. 把 LDG 暂存写入下一块 smem
//   4. __syncthreads() 切换 buffer
//   5. 预取新 buffer 的 kk=0 寄存器
// =====================================================================
template <int BLOCK_M,       // block 覆盖的 C 行数（如 128）
          int BLOCK_N,       // block 覆盖的 C 列数（如 128）
          int BLOCK_K,       // K 维度每次迭代的深度（如 8）
          int WARP_M,        // warp 内 M 方向的 lane 数（如 4）
          int WARP_N,        // warp 内 N 方向的 lane 数（如 8）
          int THREAD_TILE_M, // 每 lane 负责的 C 行数（如 8）
          int THREAD_TILE_N, // 每 lane 负责的 C 列数（如 8）
          typename T>
__global__ void sgemm_double_buffer(const T *__restrict__ A,
                                    const T *__restrict__ B,
                                    T *__restrict__ C,
                                    int M, int N, int K)
{
    // -------- 编译期参数推导 --------
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_TILE_M = WARP_M * THREAD_TILE_M;
    constexpr int WARP_TILE_N = WARP_N * THREAD_TILE_N;
    constexpr int WARPS_M = BLOCK_M / WARP_TILE_M;
    constexpr int WARPS_N = BLOCK_N / WARP_TILE_N;
    constexpr int THREADS = WARPS_M * WARPS_N * WARP_SIZE;

    // 每线程暂存的 float4 数量（A/B 各自独立计算）
    constexpr int A_F4_PER_THREAD = (BLOCK_M * BLOCK_K / 4) / THREADS;
    constexpr int B_F4_PER_THREAD = (BLOCK_K * BLOCK_N / 4) / THREADS;

    static_assert(A_F4_PER_THREAD >= 1,
                  "配置不当：每线程至少需要加载 1 个 A float4");
    static_assert(B_F4_PER_THREAD >= 1,
                  "配置不当：每线程至少需要加载 1 个 B float4");
    static_assert(THREAD_TILE_M % 4 == 0 && THREAD_TILE_N % 4 == 0,
                  "向量化加载要求 THREAD_TILE_M/N 是 4 的倍数");
    static_assert(BLOCK_K % 4 == 0,
                  "float4 加载 + A 转置要求 BLOCK_K 是 4 的倍数");

    // -------- 双缓冲 Shared Memory --------
    __shared__ T smem_A_T[2][BLOCK_K][BLOCK_M]; // A 转置存储，两份
    __shared__ T smem_B[2][BLOCK_K][BLOCK_N];   // B 行主序，两份

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

    // -------- 寄存器数组 --------
    T reg_A[2][THREAD_TILE_M]; // p-Loop 双缓冲
    T reg_B[2][THREAD_TILE_N];
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {T(0)};

    // -------- LDG 暂存寄存器 --------
    float4 ldg_A_stash[A_F4_PER_THREAD];
    float4 ldg_B_stash[B_F4_PER_THREAD];

    // =================================================================
    // Prologue：加载第一个 tile 到 Shared Memory
    // =================================================================
    load_tile_A_vec4<BLOCK_M, BLOCK_K, THREADS>(
        A, smem_A_T[0], global_row0, /*k0=*/0, thread_id, M, K);
    load_tile_B_vec4<BLOCK_K, BLOCK_N, THREADS>(
        B, smem_B[0], /*k0=*/0, global_col0, thread_id, K, N);
    __syncthreads();

    // 预取 p-Loop 的第一组寄存器（kk=0）
    load_shared_to_reg<BLOCK_K, BLOCK_M, BLOCK_N, THREAD_TILE_M, THREAD_TILE_N>(
        smem_A_T[0], smem_B[0],
        reg_A[0], reg_B[0],
        tile_row0, tile_col0, /*kk=*/0);

    // =================================================================
    // 主循环：沿 K 维度迭代
    // =================================================================
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
    {
        const int next_buf = 1 - buf;
        const bool has_next = (k0 + BLOCK_K < K);

        // -------------------------------------------------------------
        // p-Loop：BLOCK_K 步外积
        // -------------------------------------------------------------
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk)
        {
            // ---- 预取 kk+1 步的寄存器（写入另一组 reg_*）----
            if (kk + 1 < BLOCK_K)
            {
                load_shared_to_reg<BLOCK_K, BLOCK_M, BLOCK_N,
                                   THREAD_TILE_M, THREAD_TILE_N>(
                    smem_A_T[buf], smem_B[buf],
                    reg_A[(kk + 1) & 1], reg_B[(kk + 1) & 1],
                    tile_row0, tile_col0, kk + 1);
            }

            // ---- 在 kk=0 时，发射下一 tile 的 LDG（异步）----
            // LDG 是异步的，发射后立即继续执行后续 FFMA
            if (kk == 0 && has_next)
            {
                issue_ldg_A<BLOCK_M, BLOCK_K, THREADS, A_F4_PER_THREAD>(
                    A, global_row0, k0 + BLOCK_K, thread_id, M, K,
                    ldg_A_stash);
                issue_ldg_B<BLOCK_K, BLOCK_N, THREADS, B_F4_PER_THREAD>(
                    B, k0 + BLOCK_K, global_col0, thread_id, K, N,
                    ldg_B_stash);
            }

            // ---- 寄存器外积（使用当前 kk 步的寄存器）----
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j)
                    accum[i][j] += reg_A[kk & 1][i] * reg_B[kk & 1][j];
        }

        // -------------------------------------------------------------
        // 把 LDG 暂存写入下一块 Shared Memory
        // 此时 LDG 数据通常已到达（被上面的 FFMA 隐藏了延迟）
        // -------------------------------------------------------------
        if (has_next)
        {
            store_ldg_A_to_shared<BLOCK_M, BLOCK_K, THREADS, A_F4_PER_THREAD>(
                smem_A_T[next_buf], thread_id, ldg_A_stash);
            store_ldg_B_to_shared<BLOCK_K, BLOCK_N, THREADS, B_F4_PER_THREAD>(
                smem_B[next_buf], thread_id, ldg_B_stash);
        }

        // -------------------------------------------------------------
        // 唯一同步点：切换 buffer
        // -------------------------------------------------------------
        __syncthreads();

        // -------------------------------------------------------------
        // 预取新 buffer 的 kk=0 寄存器
        // -------------------------------------------------------------
        if (has_next)
        {
            load_shared_to_reg<BLOCK_K, BLOCK_M, BLOCK_N,
                               THREAD_TILE_M, THREAD_TILE_N>(
                smem_A_T[next_buf], smem_B[next_buf],
                reg_A[0], reg_B[0],
                tile_row0, tile_col0, /*kk=*/0);
        }

        buf = next_buf;
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

#endif // MATMUL_BUFFER_CUH_