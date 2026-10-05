add_executable(matmul_bench matmul_main.cu)

cuda_practice_base(matmul_bench)
cuda_practice_extend(matmul_bench)
target_link_libraries(matmul_bench PRIVATE CUDA::cublas)