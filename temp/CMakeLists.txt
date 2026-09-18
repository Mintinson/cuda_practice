include(FindCUDAToolkit REQUIRED)

add_executable(final_project_cuda "")
target_include_directories(final_project_cuda PRIVATE
    "C:/Program\\ Files/NVIDIA\\ GPU\\ Computing\\ Toolkit/CUDA/v12.5/include"
)

target_compile_options(final_project_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-DNDEBUG>
    $<$<COMPILE_LANGUAGE:CXX>:-DNDEBUG>
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
    $<$<COMPILE_LANGUAGE:CUDA>:-m64>
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_89,code=sm_89>
    $<$<COMPILE_LANGUAGE:CUDA>:-DNDEBUG>
)

if(CURRENT_COMPILER_ID STREQUAL "MSVC")
target_compile_options(final_project_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:C>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:C>:/arch:AVX2>
    $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
)
endif()
if(CURRENT_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(final_project_cuda PRIVATE /EHsc)
elseif(CURRENT_COMPILER_ID STREQUAL "Clang")
    target_compile_options(final_project_cuda PRIVATE -fexceptions)
    target_compile_options(final_project_cuda PRIVATE -fcxx-exceptions)
elseif(CURRENT_COMPILER_ID STREQUAL "GNU")
    target_compile_options(final_project_cuda PRIVATE -fexceptions)
endif()
set_target_properties(final_project_cuda PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(final_project_cuda PRIVATE cxx_std_23)
if(CURRENT_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(final_project_cuda PRIVATE $<$<CONFIG:Release>:-O2>)
else()
    target_compile_options(final_project_cuda PRIVATE -O3)
endif()
if(CURRENT_COMPILER_ID STREQUAL "MSVC")
else()
    target_compile_options(final_project_cuda PRIVATE -fvisibility=hidden)
endif()
if(CURRENT_COMPILER_ID STREQUAL "MSVC")
    set_property(TARGET final_project_cuda PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()
target_link_libraries(final_project_cuda PRIVATE
    opencv_phase_unwrapping4110
    opencv_surface_matching4110
    opencv_saliency4110
    opencv_wechat_qrcode4110
    opencv_mcc4110
    opencv_face4110
    opencv_img_hash4110
    opencv_videostab4110
    opencv_structured_light4110
    opencv_intensity_transform4110
    opencv_ccalib4110
    opencv_line_descriptor4110
    opencv_stereo4110
    opencv_dnn_objdetect4110
    opencv_dnn_superres4110
    opencv_fuzzy4110
    opencv_hfs4110
    opencv_rapid4110
    opencv_bgsegm4110
    opencv_bioinspired4110
    opencv_rgbd4110
    opencv_dpm4110
    opencv_aruco4110
    opencv_reg4110
    opencv_tracking4110
    opencv_datasets4110
    opencv_xfeatures2d4110
    opencv_shape4110
    opencv_superres4110
    opencv_plot4110
    opencv_quality4110
    opencv_text4110
    opencv_optflow4110
    opencv_ximgproc4110
    opencv_xobjdetect4110
    opencv_xphoto4110
    opencv_stitching4110
    opencv_ml4110
    opencv_photo4110
    opencv_gapi4110
    opencv_objdetect4110
    opencv_highgui4110
    opencv_videoio4110
    opencv_video4110
    opencv_calib3d4110
    opencv_dnn4110
    opencv_features2d4110
    opencv_flann4110
    opencv_imgcodecs4110
    opencv_imgproc4110
    opencv_core4110
    ade
    libjpeg-turbo
    libpng
    libprotobuf
    libtiff
    libwebp
    quirc
    zlib
    avfilter
    avdevice
    avformat
    avcodec
    swscale
    swresample
    avutil
    postproc
    cudadevrt
    cudart_static
    gdi32
    user32
    glu32
    opengl32
    advapi32
    comdlg32
    ws2_32
    Bcrypt
    Mfplat
    mfuuid
    Ole32
    Secur32
    Strmiids
    User32
)
target_link_directories(final_project_cuda PRIVATE
    "C:/Program\\ Files/NVIDIA\\ GPU\\ Computing\\ Toolkit/CUDA/v12.5/lib/x64"
    C:/Users/asus/AppData/Local/.xmake/packages/o/opencv/4.11.0/c4bf395a2bc447ccb891fadb03cdd4dc/x64/vc17/staticlib
    C:/Users/asus/AppData/Local/.xmake/packages/f/ffmpeg/7.1/c53cf462b5814beab5d0fe9fa1ca71b4/lib
)
# private sourcefiles from sourcebatch c++.build for target final_project_cuda
target_sources(final_project_cuda PRIVATE 
    src/final_project_cuda.cpp
)
# private sourcefiles from sourcebatch cuda.build for target final_project_cuda
target_sources(final_project_cuda PRIVATE 
    src/sml_kernels.cu
)
set_target_properties(final_project_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)