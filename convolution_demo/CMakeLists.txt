add_executable(onn_dim_conv "one_dim/one_dim_main.cu")
set_property(TARGET onn_dim_conv PROPERTY CUDA_ARCHITECTURES 80)

add_executable(two_dim_conv "two_dim/two_dim_main.cu")
set_property(TARGET two_dim_conv PROPERTY CUDA_ARCHITECTURES 80)