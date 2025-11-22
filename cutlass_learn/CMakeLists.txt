add_executable(cutlass_1 test.cu)
set_target_properties(cutlass_1 PROPERTIES CUDA_STANDARD 17)
set_property(TARGET cutlass_1 PROPERTY CUDA_ARCHITECTURES 80)
target_link_libraries(cutlass_1 PRIVATE CUDA::cudart CUDA::cudart_static CUDA::cublas)
# target_include_directories(hgemm_tensorcore PRIVATE ${CUDA_CUBLAS_DIRS})
target_compile_options(cutlass_1 PRIVATE "--expt-relaxed-constexpr")
target_compile_options(cutlass_1 PRIVATE "--extended-lambda")
target_compile_options(cutlass_1 PRIVATE -lineinfo)
if (WIN32)
    # add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/Zc:__cplusplus>")
    target_compile_options(cutlass_1 PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/Zc:__cplusplus>")
    set(CUTLASS_DIR "D:/cppPackages/cutlass/include")
endif ()
target_include_directories(cutlass_1 PRIVATE ${CUTLASS_DIR} "${CUTLASS_DIR}/../tools/util/include"
        "${CUTLASS_DIR}/../tools/library/include")

# 封装为可重用的宏
macro(add_cutlass_executable TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})

    # 公共目标属性
    set_target_properties(${TARGET_NAME} PROPERTIES
            CUDA_STANDARD 17
            CUDA_ARCHITECTURES 80
    )

    # 公共编译选项
    target_compile_options(${TARGET_NAME} PRIVATE
            "--expt-relaxed-constexpr"
            "--extended-lambda"
            -lineinfo
    )

    # 平台特定设置
    if (WIN32)
        target_compile_options(${TARGET_NAME} PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/Zc:__cplusplus>"
        )
        set(CUTLASS_DIR "D:/cppPackages/cutlass/include")  # Windows路径
    else ()
        set(CUTLASS_DIR "path/to/linux/cutlass/include")   # 非Windows路径
    endif ()

    # 公共链接库和包含目录
    target_link_libraries(${TARGET_NAME} PRIVATE
            CUDA::cudart
            CUDA::cudart_static
            CUDA::cublas
    )

    target_include_directories(${TARGET_NAME} PRIVATE
            ${CUTLASS_DIR}
            "${CUTLASS_DIR}/../tools/util/include"
            "${CUTLASS_DIR}/../tools/library/include"
    )
    #    file(GLOB LIBRARY_SOURCE_FILES "${CUTLASS_DIR}/../tools/library/src/*.cu" "${CUTLASS_DIR}/../tools/library/src/*.h*")
    #    target_sources(${TARGET_NAME} PRIVATE "${LIBRARY_SOURCE_FILES}")
endmacro()


add_cutlass_executable(02_dump_reg_shmem 02_dump_reg_shmem.cu)
add_cutlass_executable(03_visualized_layout 03_visualized_layout.cu)
add_cutlass_executable(04_tile_iterator 04_tile_iterator.cu)
add_cutlass_executable(05_batched_gemm 05_batched_gemm.cu)
add_cutlass_executable(06_splitK_gemm 06_splitK_gemm.cu)
set_target_properties(06_splitK_gemm PROPERTIES CUDA_ARCHITECTURES 75)
add_cutlass_executable(08_turing_tensorop_gemm 08_turing_tensorop_gemm.cu)
add_cutlass_executable(09_turing_tensorop_conv2dfprop 09_turing_tensorop_conv2dfprop.cu)
add_cutlass_executable(10_planar_complex 10_planar_complex.cu)
add_cutlass_executable(12_gemm_bias_relu 12_gemm_bias_relu.cu)
add_cutlass_executable(14_ampere_tf32_tensorop_gemm 14_ampere_tf32_tensorop_gemm.cu)
add_cutlass_executable(15_ampere_sparse_tensorop_gemm "15_ampere_sparse_tensorop_gemm/15_ampere_sparse_tensorop_gemm.cu")
add_cutlass_executable(15_ampere_sparse_tensorop_gemm_universal
        "15_ampere_sparse_tensorop_gemm/15_ampere_sparse_tensorop_gemm_universal.cu")
add_cutlass_executable(15_ampere_sparse_tensorop_gemm_with_visitor
        "15_ampere_sparse_tensorop_gemm/15_ampere_sparse_tensorop_gemm_with_visitor.cu")
add_cutlass_executable(16_ampere_tensorop_conv2dfprop 16_ampere_tensorop_conv2dfprop.cu)
add_cutlass_executable(17_fprop_per_channel_bias 17_fprop_per_channel_bias.cu)
add_cutlass_executable(18_ampere_fp64_tensorop_affine2_gemm 18_ampere_fp64_tensorop_affine2_gemm.cu)
add_cutlass_executable(19_tensorop_canonical 19_tensorop_canonical.cu)
add_cutlass_executable(20_simt_canonical 20_simt_canonical.cu)
add_cutlass_executable(23_ampere_gemm_operand_reduction_fusion 23_ampere_gemm_operand_reduction_fusion.cu)
add_cutlass_executable(24_gemm_grouped 24_gemm_grouped.cu)
add_cutlass_executable(35_gemm_softmax "35_gemm_softmax/35_gemm_softmax.cu" "35_gemm_softmax/35_gemm_with_softmax.h")
