add_executable(matmul
)
target_sources(matmul PRIVATE matmul_main.cu matmul_v4.cuh matmul_v7.cuh matmul_v8.cuh)
set_property(TARGET matmul PROPERTY CUDA_ARCHITECTURES 80)
target_link_libraries(matmul PRIVATE CUDA::cudart CUDA::cudart_static CUDA::cublas)
target_compile_options(reduce_0 PRIVATE "--expt-relaxed-constexpr")
target_compile_options(reduce_0 PRIVATE "--extended-lambda")