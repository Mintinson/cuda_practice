add_executable(matmul
)
target_sources(matmul PRIVATE matmul_main.cu matmul_v4.cuh matmul_v7.cuh matmul_v8.cuh)
set_property(TARGET matmul PROPERTY CUDA_ARCHITECTURES 80)
target_link_libraries(matmul PRIVATE CUDA::cudart CUDA::cudart_static CUDA::cublas)
# nvcc-only flags; clang's CUDA frontend rejects them
target_compile_options(matmul PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>"
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>")