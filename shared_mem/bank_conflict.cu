#include <cuda_runtime.h>

__global__ void smem_1(uint32_t *a)
{
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++)
    {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    reinterpret_cast<uint2 *>(a)[tid] =
        reinterpret_cast<const uint2 *>(smem)[tid];
}
__global__ void smem_2(uint32_t *a)
{
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++)
    {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    reinterpret_cast<uint2 *>(a)[tid] =
        reinterpret_cast<const uint2 *>(smem)[tid / 2];
}

__global__ void smem_1_4(uint32_t *a)
{
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++)
    {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    if (tid == 15 || tid == 16)
    {
        reinterpret_cast<uint4 *>(a)[tid] =
            reinterpret_cast<const uint4 *>(smem)[4];
    }
}
__global__ void smem_2_4(uint32_t *a)
{
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++)
    {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    if (tid == 0 || tid == 15)
    {
        reinterpret_cast<uint4 *>(a)[tid] =
            reinterpret_cast<const uint4 *>(smem)[4];
    }
}

__global__ void smem_3_4(uint32_t *a)
{
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++)
    {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(
        smem)[(tid / 8) * 2 + ((tid % 8) / 2) % 2];
}
__global__ void smem_4_4(uint32_t *a)
{
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++)
    {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    uint32_t addr;
    if (tid < 16)
    {
        addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
    }
    else
    {
        addr = (tid / 8) * 2 + ((tid % 8) % 2);
    }
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[addr];
    // printf("tid: %d, addr: %d\n", tid, addr);
}
__global__ void smem_5_4(uint32_t *a)
{
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++)
    {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    uint32_t addr = (tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8;
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[addr];
    // printf("tid: %d, addr: %d\n", tid, addr);
}
int main()
{
    uint32_t *d_input{};
    std::size_t sz = 32;
    cudaMalloc(&d_input, sz * sizeof(uint32_t));
    smem_1<<<1, 32>>>(d_input);
    smem_2<<<1, 32>>>(d_input);
    cudaFree(d_input);

    cudaMalloc(&d_input, 128 * sizeof(uint32_t));
    smem_1_4<<<1, 32>>>(d_input);
    smem_2_4<<<1, 32>>>(d_input);
    smem_3_4<<<1, 32>>>(d_input);
    smem_4_4<<<1, 32>>>(d_input);
    smem_5_4<<<1, 32>>>(d_input);
    cudaFree(d_input);
}