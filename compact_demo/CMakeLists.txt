add_executable(compact_demo
        compact_cuda_v0.cuh)
target_sources(compact_demo PRIVATE compact_main.cu)
set_property(TARGET compact_demo PROPERTY CUDA_ARCHITECTURES 80)
target_link_directories(compact_demo PRIVATE cuda::cudart)
target_compile_options(compact_demo PRIVATE "--expt-relaxed-constexpr")
target_compile_options(compact_demo PRIVATE "--extended-lambda")