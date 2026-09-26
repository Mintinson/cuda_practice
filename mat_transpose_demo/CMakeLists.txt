add_executable(mat_transpose_bench mat_transpose_main.cu)

cuda_practice_base(mat_transpose_bench)
# cuBLAS Sgeam 作为官方参考实现（与 v0-v3 使用相同的输入/输出与计时范围）
target_link_libraries(mat_transpose_bench PRIVATE CUDA::cublas)
