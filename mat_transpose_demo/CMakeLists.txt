add_executable(mat_transpose)
target_sources(mat_transpose PRIVATE mat_transpose_main.cu)
set_property(TARGET mat_transpose PROPERTY CUDA_ARCHITECTURES 80)
target_link_libraries(mat_transpose PRIVATE CUDA::cudart CUDA::cudart_static CUDA::cublas)