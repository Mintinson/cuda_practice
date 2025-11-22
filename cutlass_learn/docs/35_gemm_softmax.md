好的，我们来详细讲解 `35_gemm_softmax.cu` (主程序) 和 `35_gemm_with_softmax.h` (定义 `GemmSoftmax` 类)
中涉及的核函数调用步骤以及每个步骤产生的结果。

这个示例的目标是计算 `Softmax(alpha * A * B + beta * C)`，其中 Softmax 是按行进行的。为了高效实现，它将这个过程分解为三个主要的
CUDA 核函数启动。

**核心类: `cutlass::GemmSoftmax` (在 `35_gemm_with_softmax.h` 中定义)**

这个类封装了执行 GEMM + Softmax 操作所需的多个核函数。它的 `run()` 方法（或者等效的 `operator()`）会按顺序启动这些核函数。

**核函数调用步骤及结果分析:**

在 `GemmSoftmax::run(cudaStream_t stream)` 方法中，会依次启动以下三个核函数：

1. **`cutlass::Kernel<GemmKernel>`**: GEMM + Epilogue Visitor (计算部分 Softmax 统计量)
2. **`Kernel<ApplyFinalReductionKernel>`**: 对部分统计量进行最终规约
3. **`Kernel<SoftmaxApplyKernel>`**: 应用 Softmax 计算

下面我们详细分析每一步：

---

**步骤 1: 启动 `GemmKernel` (即 `cutlass::gemm::kernel::GemmWithEpilogueVisitor`)**

* **核函数类型**: `cutlass::gemm::kernel::GemmWithEpilogueVisitor`
    * 这个核函数由 `GemmSoftmax` 类中的 `GemmKernel` typedef 定义。
    * 它内部组合了一个标准的 GEMM 计算 (`DefaultGemmKernel::Mma`) 和一个自定义的 Epilogue（尾声）处理阶段。
    * Epilogue 部分由 `EpilogueVisitorSoftmax` 实现，它被包装在 `EpilogueWithVisitorFromExistingEpilogue` 中。

* **目的**:
    1. 执行主要的矩阵乘法和线性组合：`P_mn = alpha * (A*B)_mn + beta * C_input_mn`。
    2. 在计算 `P_mn` 的同时，通过 `EpilogueVisitorSoftmax`，**针对每个线程块 (Threadblock/CTA) 处理的行**，计算出**部分的**
       统计数据：
        * 该行内元素的最大值 (partial `max_m`)。
        * 该行内元素减去其对应 partial `max_m` 后再取指数，然后求和 (partial
          `sum_exp_m = sum(exp(P_mn - partial_max_m))`)。

* **输入**:
    * `params_.gemm.ref_A` (来自 `block_A`): 输入矩阵 A 的设备指针和布局。
    * `params_.gemm.ref_B` (来自 `block_B`): 输入矩阵 B 的设备指针和布局。
    * `params_.gemm.ref_C` (来自 `block_C`): 输入矩阵 C (用于 `beta*C`) 的设备指针和布局。
    * `params_.gemm.epilogue_visitor.linear_scaling.alpha` (来自 `options.alpha`): alpha 缩放因子。
    * `params_.gemm.epilogue_visitor.linear_scaling.beta` (来自 `options.beta`): beta 缩放因子。

* **计算过程 (简化版)**:
    1. **主循环 (Main Loop - MMA)**:
        * 线程块从全局内存加载 A 和 B 的瓦片 (tile) 到共享内存 (Shared Memory)。
        * Warp 使用 Tensor Core 指令从共享内存加载数据到寄存器，并执行矩阵乘法累加操作，得到累加器中的 `(A*B)_mn` 结果。
    2. **尾声 (Epilogue)**:
        * **线性组合**: `EpilogueFunctorOp` (`cutlass::epilogue::thread::LinearCombination`) 被调用，它从全局内存加载 C
          的对应元素，计算 `P_mn = alpha * (A*B)_mn + beta * C_input_mn`。
        * **Epilogue Visitor (`EpilogueVisitorSoftmax`)**:
            * `visit()` 方法被调用，接收 `P_mn` 的元素。
            * 对于每个线程块处理的每一行，该 Visitor：
                * 在其内部累加器中追踪遇到的行内最大值 (`partial_max_m_for_this_tb_and_row`)。
                * 计算 `exp_val = exp(P_mn - partial_max_m_for_this_tb_and_row)`
                  。（注意：这里为了数值稳定性，通常会减去当前已知的行内最大值。如果一个元素导致行内最大值更新，之前的
                  `sum_exp` 需要相应调整）。
                * 累加 `exp_val` 到 `partial_sum_exp_m_for_this_tb_and_row`。
        * **写回主要输出**: 标准的 Epilogue 输出迭代器将计算得到的 `P_mn` 写回到全局内存的目标矩阵 D (
          `params_.gemm.ref_D.data()`，即 `block_D`)。
        * **写回部分统计数据**: `EpilogueVisitorSoftmax` 将其计算得到的**每个线程块内每行的部分最大值**写入到
          `params_.gemm.ptr_Max` (即 `block_Norm`)，并将**每个线程块内每行的部分指数和**写入到 `params_.gemm.ptr_Sum` (即
          `block_Sum`)。

* **输出 (结果)**:
    * **`block_D` (全局内存)**: 存储了 GEMM 和线性组合的结果，即 `D_mn = alpha * (A*B)_mn + beta * C_input_mn`。**这个矩阵是后续
      Softmax 计算的输入。**
    * **`block_Norm` (全局内存)**: 存储了**部分最大值**。对于矩阵 D 的每一行，如果这一行的数据由多个线程块共同计算完成（当
      N 维度大于一个线程块的 N 瓦片大小时），那么 `block_Norm` 中会包含由不同线程块计算出的多个针对该行的部分最大值。它的维度通常是
      `(batch_count, problem_m, num_blocks_along_n_dim)`。
    * **`block_Sum` (全局内存)**: 存储了**部分指数和** (`sum(exp(D_mn - partial_max_m))`)
      。同样，这也是基于每个线程块处理的行和它们各自找到的部分最大值计算的。

**为什么要这么做？**
Softmax 需要计算每一行的最大值和指数和。由于 GEMM 是分块进行的，单个线程块只能看到它所处理的输出瓦片（D
矩阵的一部分）中的数据。因此，它只能计算出这些数据里的“部分”最大值和“部分”指数和。这些部分结果需要后续步骤进行合并（规约）。

---

**步骤 2: 启动 `ApplyFinalReductionKernel` (即 `cutlass::reduction::kernel::ApplySoftmaxFinalReduction`)**

* **核函数类型**: `cutlass::reduction::kernel::ApplySoftmaxFinalReduction`

* **目的**:
    1. 对于矩阵 D 的每一行，读取由前一个核函数（`GemmKernel`）计算出的所有**部分最大值** (`block_Norm` 中的相关条目)
       ，并从中找到该行**真正的全局最大值** (`final_max_m`)。
    2. 利用这个 `final_max_m` 和所有的**部分最大值**及**部分指数和** (`block_Sum` 中的相关条目)，计算出该行**真正的全局指数和
       ** (`final_sum_exp_m = sum_j(exp(D_mj - final_max_m))`)。
        * 计算公式:
          `final_sum_exp_m = sum_over_threadblocks_i ( partial_sum_exp_m_i * exp(partial_max_m_i - final_max_m) )`

* **输入**:
    * `params_.reduction.ptr_Max` (即 `block_Norm`): Kernel 1 输出的部分最大值。
    * `params_.reduction.ptr_Sum` (即 `block_Sum`): Kernel 1 输出的部分指数和。
    * `params_.reduction.problem_size`: 问题维度信息，用于确定如何遍历和规约。

* **计算过程**:
    1. 每个 CUDA 线程块（或线程）负责处理矩阵 D 的一行或若干行。
    2. 对于其负责的每一行 `m`：
        * 加载所有由 Kernel 1 生成的与该行 `m` 相关的部分最大值。
        * 计算出这些部分最大值中的最大值，得到 `final_max_m`。
        * 加载所有与该行 `m` 相关的部分指数和以及对应的部分最大值。
        * 使用上述的 `final_sum_exp_m` 计算公式，结合 `final_max_m` 和每个部分统计量，计算出最终的指数和。
        * 将计算得到的 `final_max_m` 写回到 `block_Norm` 的对应位置（覆盖掉原来的部分最大值）。
        * 将计算得到的 `final_sum_exp_m` 写回到 `block_Sum` 的对应位置（覆盖掉原来的部分指数和）。

* **输出 (结果)**:
    * **`block_Norm` (全局内存, 原位更新)**: 现在存储的是矩阵 D **每一行的最终全局最大值** (`final_max_m`)。它的有效维度通常是
      `(batch_count, problem_m)`。
    * **`block_Sum` (全局内存, 原位更新)**: 现在存储的是矩阵 D **每一行的最终全局指数和** (
      `final_sum_exp_m = sum_j(exp(D_mj - final_max_m))`)。它的有效维度通常是 `(batch_count, problem_m)`。

**为什么要这么做？**
这是 Softmax 计算的规约步骤。第一步的 GEMM Kernel
是分块并行计算的，每个块只能得到局部信息。这一步则将这些局部（部分）信息汇总，得到每行真正的全局最大值和基于此最大值的正确指数和，为下一步的归一化做准备。

---

**步骤 3: 启动 `SoftmaxApplyKernel` (即 `cutlass::kernel::ApplySoftmax`)**

* **核函数类型**: `cutlass::kernel::ApplySoftmax` (在 `35_gemm_with_softmax.h` 中定义)

* **目的**:
    1. 对于矩阵 D (Kernel 1的输出) 中的每一个元素 `D_mn`：
    2. 读取其对应行的最终全局最大值 `final_max_m` (来自 `block_Norm`)。
    3. 读取其对应行的最终全局指数和 `final_sum_exp_m` (来自 `block_Sum`)。
    4. 计算 Softmax 值: `Softmax_mn = exp(D_mn - final_max_m) / final_sum_exp_m`。
    5. 将结果 `Softmax_mn` 存储到最终的输出矩阵 `block_Softmax`。

* **输入**:
    * `params_.softmax.ref_D` (即 `block_D`): Kernel 1 输出的 `alpha*A*B + beta*C` 结果。
    * `params_.softmax.ref_N` (即 `block_Norm`): Kernel 2 输出的每行最终全局最大值。
    * `params_.softmax.ref_S` (即 `block_Sum`): Kernel 2 输出的每行最终全局指数和。

* **计算过程**:
    1. 每个 CUDA 线程块负责处理最终 Softmax 输出矩阵的一个或多个瓦片。
    2. 对于其负责的每个元素 `(m,n)`：
        * 加载 `D_mn` 从 `block_D`。
        * 加载行 `m` 的 `final_max_m` 从 `block_Norm`。
        * 加载行 `m` 的 `final_sum_exp_m` 从 `block_Sum`。
        * 计算 `numerator = exp(D_mn - final_max_m)`。
        * 计算 `softmax_value = numerator / final_sum_exp_m`。 (在 `ApplySoftmax` Kernel中，实际是乘以
          `inv_sum = 1.0f / final_sum_exp_m`)
        * 将 `softmax_value` 存入 `block_Softmax` 的 `(m,n)` 位置。

* **输出 (结果)**:
    * **`block_Softmax` (全局内存)**: 存储了最终的 Softmax 计算结果。其维度与 D 矩阵相同。

**为什么要这么做？**
这是 Softmax 的最后一步，利用前两步计算得到的准确的行最大值和行指数和，对原始的 GEMM 输出（D 矩阵）进行归一化，得到每个元素的
Softmax概率值。

---

**总结流程：**

1. **Kernel 1 (GEMM + Partial Stats)**:
    * 计算 `D = alpha*A*B + beta*C`。
    * 输出 `D` 到全局内存。
    * 输出**部分**行最大值到 `block_Norm`。
    * 输出**部分**行指数和到 `block_Sum`。

2. **Kernel 2 (Final Reduction)**:
    * 读取 `block_Norm` 和 `block_Sum` 中的部分统计量。
    * 计算**最终**行最大值，并更新 `block_Norm`。
    * 计算**最终**行指数和（基于最终行最大值），并更新 `block_Sum`。

3. **Kernel 3 (Apply Softmax)**:
    * 读取 `D`、更新后的 `block_Norm` (最终行最大值)、更新后的 `block_Sum` (最终行指数和)。
    * 计算 `Softmax(D)_mn = exp(D_mn - final_max_m) / final_sum_exp_m`。
    * 输出最终 Softmax 结果到 `block_Softmax`。

这种分步计算（特别是将 Softmax 的规约部分分为两步）是并行计算中处理全局依赖（如整行最大值）的常见策略，旨在最大化并行度并有效利用
GPU 的计算资源。