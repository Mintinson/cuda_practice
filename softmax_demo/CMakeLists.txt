add_executable(softmax_demo softmax_main.cu)
target_link_libraries(softmax_demo PRIVATE CUDA::cudart CUDA::cudart)
    target_compile_options(softmax_demo PRIVATE
            "--expt-relaxed-constexpr"
            "--extended-lambda"
            -lineinfo
    )