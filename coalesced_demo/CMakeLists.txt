add_executable(coalesced_demo coalesced_demo.cu)
target_compile_options(coalesced_demo PRIVATE -lineinfo)
set_property(TARGET coalesced_demo PROPERTY CUDA_ARCHITECTURES 80)

add_executable(coalesced_practice coalesced_practice.cu)
target_compile_options(coalesced_practice PRIVATE -lineinfo)
set_property(TARGET coalesced_practice PROPERTY CUDA_ARCHITECTURES 80)
target_link_libraries(coalesced_practice PRIVATE CUDA::cudart CUDA::cudart_static CUDA::cublas)
