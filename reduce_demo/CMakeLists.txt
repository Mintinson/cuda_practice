add_executable(reduce_bench)
target_sources(reduce_bench PRIVATE reduce_bench.cu)
set_property(TARGET reduce_bench PROPERTY CUDA_ARCHITECTURES 80)
target_link_directories(reduce_bench PRIVATE cuda::cudart)
# nvcc-only flags; clang's CUDA frontend rejects them
target_compile_options(reduce_bench PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>"
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>")
