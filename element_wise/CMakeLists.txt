add_executable(element_wise_demo)
target_sources(element_wise_demo PRIVATE element_wise.cu naive.cuh vec2_optimize.cuh vec4_optimize.cuh)
target_link_libraries(element_wise_demo PRIVATE CUDA::cudart CUDA::cudart_static CUDA::cublas)
target_compile_options(element_wise_demo PRIVATE "--expt-relaxed-constexpr")
