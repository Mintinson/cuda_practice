#ifndef MMA_HTEMM_CUH
#define MMA_HTEMM_CUH

#include "helper.cuh"
#include <cstddef>
#include <cuda_runtime.h>
#include <mma.h>
#include <type_traits>

#define REG(val) (*reinterpret_cast<uint32_t *>(&(val)))
#define HALF2(val) (*reinterpret_cast<half2 *>(&val))

namespace cuda_mma
{
    namespace kernels
    {

        template <typename T>
        __device__ void ldmatrix_x4(T &R0, T &R1, T &R2, T &R3, const T &addr)
        {
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)
                : "r"(addr));
        }
        template <typename T>
        __device__ inline void ldmatrix_x2_t(T &R0, T &R1, const T &addr)
        {
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(R0), "=r"(R1)
                : "r"(addr));
        }
        template <typename T>
        __device__ inline void hmma16816(T &RD0, T &RD1, T &RA0, T &RA1, T &RA2, T &RA3, T &RB0, T &RB1, T &RC0, T &RC1)
        {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "
                "%4, %5}, {%6, %7}, {%8, %9};\n"
                : "=r"(RD0), "=r"(RD1)
                : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),
                  "r"(RC1));
        }
        template <typename T>
        __device__ inline auto &Ldst_128_bits(T &value)
        {
            using ReturnType = std::conditional_t<std::is_const_v<T>, const float4, float4>;
            return reinterpret_cast<ReturnType *>(&(value))[0];
        }
        template <typename T>
        __device__ inline auto &Ldst_32_bits(T &value)
        {
            using ReturnType = std::conditional_t<std::is_const_v<T>, const half2, half2>;
            return reinterpret_cast<ReturnType *>(&(value))[0];
        }
        // only 1 warp per block(32 threads), m16n8k16. A, B, C (mma support m16n8k16 only): all row_major.
        template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16>
        __global__ void hgemm_mma_m16n8k16_naive_kernel(half *a, half *b, half *c,
                                                        int m, int n, int k)
        {
            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            const int NUM_K_TILES = (k + MMA_K - 1) / MMA_K;
            constexpr int BM = MMA_M; // 16
            constexpr int BN = MMA_N; // 8
            constexpr int BK = MMA_K; // 16

            __shared__ half s_a[MMA_M][MMA_K]; // 16x16
            __shared__ half s_b[MMA_K][MMA_N]; // 16x8
            __shared__ half s_c[MMA_M][MMA_N]; // 16x16

            const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
            const int lane_id = tid % warpSize;                     // 0~31

            // s_a[16][16], 16 data each row，each thread loads 8 data, each row for 2 threads
            // 16 rows, 16*2=32 threads in total
            const int load_smem_a_m = tid / 2;       // row 0~15
            const int load_smem_a_k = (tid % 2) * 8; // col 0,8
            // s_b[16][8], 8 data each row，each thread loads 8 data, each row for 1 threads
            // 16 rows, 16*1=16 threads in total. Only half threads need to load data
            const int load_smem_b_k = tid;                     // row 0~31, but only use 0~15
            const int load_smem_b_n = 0;                       // col 0
            const int load_gmem_a_m = by * BM + load_smem_a_m; // global m
            const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n
            if (load_gmem_a_m >= m && load_gmem_b_n >= n)
                return;

            uint32_t RC[2] = {0, 0};

#pragma unroll
            for (int it = 0; it < NUM_K_TILES; ++it)
            {
                // gmem_a -> smem_a
                int load_gmem_a_k = it * BK + load_smem_a_k; // global col of a
                int load_gmem_a_addr = load_gmem_a_m * k + load_gmem_a_k;
                Ldst_128_bits(s_a[load_smem_a_m][load_smem_a_k]) = (Ldst_128_bits(a[load_gmem_a_addr]));

                // gmem_b -> smem_b
                if (lane_id < MMA_K)
                {
                    int load_gmem_b_k = it * MMA_K + load_smem_b_k; // global row of b
                    int load_gmem_b_addr = load_gmem_b_k * n + load_gmem_b_n;
                    Ldst_128_bits(s_b[load_smem_b_k][load_smem_b_n]) = (Ldst_128_bits(b[load_gmem_b_addr]));
                }
                __syncthreads();

                uint32_t RA[4];
                uint32_t RB[2];

                // ldmatrix for s_a, ldmatrix.trans for s_b.
                // s_a: (0,1)*8 -> 0,8 -> [(0~15),(0,8)]
                uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&s_a[lane_id % 16][(lane_id / 16) * 8]);
                ldmatrix_x4(RA[0], RA[1], RA[2], RA[3], load_smem_a_ptr);
                uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[lane_id % 16][0]);
                ldmatrix_x2_t(RB[0], RB[1], load_smem_b_ptr);

                hmma16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
                          RC[1]);

                __syncthreads();
            }

            // s_c[16][8],
            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
            // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
            // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
            Ldst_32_bits(s_c[lane_id / 4][(lane_id % 4) * 2]) = Ldst_32_bits(RC[0]);
            Ldst_32_bits(s_c[lane_id / 4 + 8][(lane_id % 4) * 2]) = Ldst_32_bits(RC[1]);

            __syncthreads();

            // store s_c[16][8]
            if (lane_id < MMA_M)
            {
                // store 128 bits per memory issue.
                int store_gmem_c_m = by * BM + lane_id;
                int store_gmem_c_n = bx * BN;
                int store_gmem_c_addr = store_gmem_c_m * n + store_gmem_c_n;
                Ldst_128_bits(c[store_gmem_c_addr]) = (Ldst_128_bits(s_c[lane_id][0]));
            }
        }
        namespace details
        {
            // vector load 128 bit, will be compiled to .128 instructions
            __device__ __forceinline__ void ld_st_128bit(void *dst, void *src)
            {
                *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
            }

            using fp16 = half;

            __device__ __forceinline__ void ldmatrix_sync(fp16 *dst, void *addr)
            {
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
                    : "=r"(REG(dst[0])),
                      "=r"(REG(dst[2])),
                      "=r"(REG(dst[4])),
                      "=r"(REG(dst[6]))
                    : "l"(__cvta_generic_to_shared(addr)));
            }

            __device__ __forceinline__ void ldmatrix_trans_sync(fp16 *dst, void *addr)
            {
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.trans.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(REG(dst[0])),
                               "=r"(REG(dst[2])),
                               "=r"(REG(dst[4])),
                               "=r"(REG(dst[6]))
                             : "l"(__cvta_generic_to_shared(addr)));
            }

            // C = A * B^T
            __device__ __forceinline__ void mma_sync_m16n8k16(fp16 *c, fp16 *a, fp16 *b)
            {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                             "{%0, %1}, "
                             "{%2, %3, %4, %5}, "
                             "{%6, %7}, "
                             "{%8, %9};"
                             : "=r"(REG(c[0])), "=r"(REG(c[2]))
                             : "r"(REG(a[0])),
                               "r"(REG(a[2])),
                               "r"(REG(a[4])),
                               "r"(REG(a[6])),
                               "r"(REG(b[0])),
                               "r"(REG(b[2])),
                               "r"(0),
                               "r"(0));
            }

            __device__ __forceinline__ void stmatrix_sync(fp16 *dst, fp16 *src)
            {
                // ! Ampere doesn't have stmatrix.sync, we should simulate it
                uint64_t private_addr = (uint64_t)dst;
                uint64_t shared_addr[4];
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    shared_addr[i] =
                        __shfl_sync(0xFFFFFFFF, private_addr, i * 8 + threadIdx.x / 4);
                }
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    *(reinterpret_cast<half2 *>(shared_addr[i]) + threadIdx.x % 4) =
                        HALF2(src[2 * i]);
                }
            }
        }
        /**
         * \brief C = A * B^T using wmma API
         * \note Launch 1 block with 32 threads only, and 16x16 each matrix, other
         * parameters settings will cause UB
         */
        __global__ void mma16x16(half *a, half *b, half *c)
        {
            __shared__ half smem_a[16 * 16];
            __shared__ half smem_b[16 * 16];
            __shared__ half smem_c[16 * 16];

            int tx = threadIdx.x;
            details::ld_st_128bit(smem_a + 8 * tx, a + 8 * tx);
            details::ld_st_128bit(smem_b + 8 * tx, b + 8 * tx);

            __syncthreads();
            using namespace nvcuda::wmma;
            fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
            fragment<accumulator, 16, 16, 16, half> c_frag;

            load_matrix_sync(a_frag, smem_a, 16);
            load_matrix_sync(b_frag, smem_b, 16);

            fill_fragment(c_frag, 0.0f);

            mma_sync(c_frag, a_frag, b_frag, c_frag);

            store_matrix_sync(smem_c, c_frag, 16, mem_row_major);

            // sync threads not necessary when only 1 warp, but we will generalize it in
            // the future, so just keep it here
            __syncthreads();
            details::ld_st_128bit(c + 8 * tx, smem_c + 8 * tx);
        }

        /**
         * \brief C = A * B^T using wmma API with PTX ISA mma instructions, this kernel
         * illustrates how to use PTX ISA mma wrappers.
         */
        __global__ void mma16x16_ptx(half *c, half *a, half *b)
        {
            __shared__ half smem_a[16 * 16];
            __shared__ half smem_b[16 * 16];
            __shared__ half smem_c[16 * 16];

            // a = a + blockDim.y * blockIdx.y * k + blockDim.x * blockIdx.x;
            // b = b + blockDim.y * blockIdx.y * n + blockDim.x * blockIdx.x;
            // c = c + blockDim.y * blockIdx.y * n + blockDim.x * blockIdx.x;

            int tx = threadIdx.x;
            details::ld_st_128bit(smem_a + 8 * tx, a + 8 * tx);
            details::ld_st_128bit(smem_b + 8 * tx, b + 8 * tx);
            __syncthreads();

            uint32_t row = tx % 16;
            uint32_t col = tx / 16;

            using namespace nvcuda::wmma;
            fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
            fragment<accumulator, 16, 16, 16, half> c_frag;

            fill_fragment(c_frag, 0.0f);
            // you can also manually set the register to 0 like:
            // for (int i = 0; i < 8; i++) {
            //     c_frag.x[i] = 0.0f;
            // }

            details::ldmatrix_sync(a_frag.x, smem_a + row * 16 + col * 8);
            details::ldmatrix_sync(b_frag.x, smem_b + row * 16 + col * 8);

            // swap R1 and R2 of B, this is required by B's layout, more info see PTX
            // ISA mma instruction
            half2 tmp = HALF2(b_frag.x[2]);
            HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
            HALF2(b_frag.x[4]) = tmp;
            // 2 m16n8k16 HMMA to achieve m16n16k16 matrix multiplication
            details::mma_sync_m16n8k16(c_frag.x, a_frag.x, b_frag.x);
            details::mma_sync_m16n8k16(c_frag.x + 4, a_frag.x, b_frag.x + 4);
            // store the result back to shared memory, this can be hand coded, but we
            // are interested in LDSM now
            store_matrix_sync(smem_c, c_frag, 16, mem_row_major);

            // details::stmatrix_sync(smem_c + row * 16 + col * 8, c_frag.x);

            __syncthreads();
            details::ld_st_128bit(c + 8 * tx, smem_c + 8 * tx);
        }

    }

    void hgemm_mma_m16n8k16_naive(half *a, half *b, half *c,
                                  const std::size_t m, const std::size_t n, const std::size_t k)
    {
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 8;
        constexpr int WMMA_K = 16;
        helper::DeviceDataHandler d_a(a, m * k);
        helper::DeviceDataHandler d_b(b, k * n);
        helper::DeviceDataHandler<half> d_c(n * m);
        dim3 block(32);
        dim3 grid((n + WMMA_N - 1) / WMMA_N, (m + WMMA_M - 1) / WMMA_M);

        kernels::hgemm_mma_m16n8k16_naive_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
            d_a.data, d_b.data, d_c.data, m, n, k);
        d_c.cpyToHost(c);
    }

    void hgemm_mma_m16n8k16_wmma(half *a, half *b, half *c,
                                 const std::size_t m, const std::size_t n, const std::size_t k)
    {
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;
        helper::DeviceDataHandler d_a(a, m * k);
        helper::DeviceDataHandler d_b(b, k * n);
        helper::DeviceDataHandler<half> d_c(n * m);
        dim3 block(32);
        // dim3 grid((n + WMMA_N - 1) / WMMA_N, (m + WMMA_M - 1) / WMMA_M);
        dim3 grid(1);

        kernels::mma16x16_ptx<<<grid, block>>>(
            d_a.data, d_b.data, d_c.data);
        d_c.cpyToHost(c);
    }
}

#endif // MMA_HTEMM_CUH