add_executable(storage_time storage_time.cu)

target_link_directories(storage_time PRIVATE cuda::cudart)
set_property(TARGET storage_time PROPERTY CUDA_ARCHITECTURES 80)

add_executable(texture_demo texture_demo.cu)

target_link_directories(texture_demo PRIVATE cuda::cudart)
set_property(TARGET texture_demo PROPERTY CUDA_ARCHITECTURES 80)