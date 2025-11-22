# CUDA 学习与实践 (CUDA Practice)

本项目记录了 CUDA 编程的学习过程与各种高性能算子的优化实践。涵盖了从基础的元素级操作到复杂的矩阵乘法、卷积以及 Tensor Core 的使用。

## 项目结构与内容

本项目包含多个子目录，每个目录对应一个特定的 CUDA 优化主题或算子实现。

### 1. 基础算子与优化
*   **Element Wise (`element_wise/`)**: 
    *   基础的逐元素操作实现。
    *   向量化读写优化 (`vec2`, `vec4`)。
*   **Reduction (`reduce_demo/`)**: 
    *   归约算法的多种实现与优化（Warp Shuffle, Shared Memory, 展开循环等）。
*   **Scan (`scan_demo/`)**: 
    *   前缀和（Scan）算法实现。
    *   包含 Kogge-Stone 和 Brent-Kung 等经典并行 Scan 算法。
*   **Compact (`compact_demo/`)**: 
    *   流压缩（Stream Compaction）操作，用于移除无效数据。
*   **Sort (`sort_demo/`)**: 
    *   排序算法实现，如双调排序 (Bitonic Sort)。

### 2. 访存优化
*   **Coalesced Access (`coalesced_demo/`)**: 
    *   全局内存合并访问（Coalesced Access）的演示与性能对比。
*   **Shared Memory (`shared_mem/`)**: 
    *   共享内存的使用与 Bank Conflict（存储体冲突）的避免。
*   **Matrix Transpose (`mat_transpose_demo/`)**: 
    *   矩阵转置的优化，重点在于解决写回全局内存时的非合并访问问题（利用 Shared Memory 避免 Bank Conflict）。

### 3. 矩阵乘法 (GEMM)
*   **SGEMM (`matmul_demo/`)**: 
    *   单精度矩阵乘法的逐步优化。
    *   从 Naive 实现到利用 Shared Memory 分块、寄存器分块、向量化访存等技术。
*   **HGEMM & Tensor Core (`hgemm_tensorcore/`)**: 
    *   使用 Tensor Core (WMMA API) 进行半精度矩阵乘法 (HGEMM) 的实现。
    *   包含 Warp 级矩阵乘法累加操作的实践。

### 4. 卷积 (Convolution)
*   **Convolution (`convolution_demo/`)**: 
    *   一维和二维卷积的 CUDA 实现。

### 5. 高级库学习
*   **CUTLASS (`cutlass_learn/`)**: 
    *   深入学习 NVIDIA CUTLASS 库。
    *   包含 Layout、Tile Iterator、Gemm、TensorOp 等核心概念的测试与示例代码。

### 6. 性能分析理论
*   **Occupancy (`occupancy/`)**: 
    *   关于占用率（Occupancy）和 Warp 调度器的笔记与实验。
*   **Roofline Model (`roofling_model/`)**: 
    *   Roofline 模型相关的性能分析笔记。

## 构建与运行

本项目使用 CMake 进行构建，并依赖 vcpkg 管理第三方库（如适用）。

### 环境要求
*   CUDA Toolkit
*   CMake
*   C++ 编译器 (MSVC/GCC/Clang)

### 编译步骤
本项目使用 CMake Presets 进行构建管理。（可能得根据不同的平台配置 [CMakePresets.json](CMakePresets.json)。

```bash
# 1. 查看所有可用预设
cmake --list-presets

# 2. 配置项目 (以 x64-debug 为例)
cmake --preset x64-debug

# 3. 构建项目
# 注意：构建目录取决于预设配置，通常为 out/build/<preset-name>
cmake --build out/build/x64-debug
```

## 笔记
每个子目录下通常包含 `note.md`，详细记录了该模块的优化思路、原理分析及性能测试结果。
