cmake_minimum_required(VERSION 3.20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)



project(cuda_practice VERSION 1.0 LANGUAGES C CXX CUDA)
if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>")
endif()
if(WIN32)
    # add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    # add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
     add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/utf-8>")

endif()


include(FindCUDAToolkit REQUIRED)

# find_package(fmt CONFIG REQUIRED)
# link_libraries(fmt::fmt)
add_compile_definitions(FMT_UNICODE=0)

include_directories(utils)
file(GLOB Util utils/*)
add_executable(demo demo.cu
    ${Util})
target_link_libraries(demo PRIVATE CUDA::cudart CUDA::cudart_static)

# target_include_directories(demo PRIVATE utils)
# target_link_libraries(demo PRIVATE fmt::fmt)
add_subdirectory(element_wise)
add_subdirectory(reduce_demo)
add_subdirectory(coalesced_demo)
add_subdirectory(matmul_demo)
add_subdirectory(mat_transpose_demo)
add_subdirectory(convolution_demo)
add_subdirectory(scan_demo)
add_subdirectory(sort_demo)
add_subdirectory(compact_demo)
add_subdirectory(storage_demo)
add_subdirectory(occupancy)
add_subdirectory(shared_mem)
add_subdirectory(hgemm_tensorcore)
add_subdirectory(cutlass_learn)
add_subdirectory(softmax_demo)
