这个算法的核心目标是执行一个标准的矩阵乘法 (GEMM) **并同时**对参与 GEMM 的其中一个输入矩阵 (A 或 B)沿着 K 维度进行规约 (
reduction) 操作，最后将这两个结果（GEMM 的结果矩阵 D 和规约操作的结果向量）输出。

此外，它还支持一种称为 "Split-K" 的技术。当 K 维度非常大时，可以将其切分成多个 "slices" (片)。每个 slice
上的计算可以独立进行，最后再将这些部分结果合并起来。当 `parallel_split_k` 为 `true` 且 `split_k_slices > 1` 时，会启用这种并行
Split-K 模式。

我们分两种情况讨论：

**情况 1: 非并行 Split-K 模式 ( `options.parallel_split_k` 为 `false` 或者 `options.split_k_slices` 为 `1` )**

在这种模式下，计算相对直接。

1. **`gemm_op()` 的结果：**
    * **主要计算 (GEMM):** 它会执行标准的 GEMM 运算。根据 `Gemm::Arguments` 中设置的 `alpha` 和 `beta` (此时是用户在命令行传入的
      `options.alpha` 和 `options.beta`)，计算公式为：
      `D = alpha * (A @ B) + beta * C`
      其中 `@` 代表矩阵乘法。
        * 输入矩阵 A 来自 `tensor_a.device_data()`。
        * 输入矩阵 B 来自 `tensor_b.device_ref().data()`。
        * 输入/输出矩阵 C (用于 beta 缩放和加法) 来自 `tensor_c.device_ref().data()`。
        * **GEMM 结果矩阵 D 会被存储在 `tensor_d.device_ref().data()` 指向的设备内存中。**

    * **K 维度规约 (K-Reduction):** 与此同时，`gemm_op()` 还会对指定的输入矩阵（由编译时常量 `ReduceKForA` 决定是 A 还是
      B）沿着 K 维度进行求和规约。
        * 如果 `ReduceKForA` 为 `true`，它会计算 `vector_output[m] = sum_over_k (A[m, k])`。
        * 如果 `ReduceKForA` 为 `false`，它会计算 `vector_output[n] = sum_over_k (B[k, n])`。
        * **这个规约后的向量结果会被存储在 `tensor_reduction.device_ref().data()` 指向的设备内存中。**

    * 在此模式下，`gemm_op()` 一次性完成了最终的 GEMM 和 K-Reduction 计算。

2. **`reduce_gemm_splitk_op()` 的结果：**
    * 在此模式下，这个函数**不会被调用**，因为它仅用于并行 Split-K 模式下合并 GEMM 的部分和。

3. **`reduce_vector_splitk_op()` 的结果：**
    * 在此模式下，这个函数也**不会被调用**，因为它仅用于并行 Split-K 模式下合并 K-Reduction 向量的部分和。

**情况 2: 并行 Split-K 模式 ( `options.parallel_split_k` 为 `true` 且 `options.split_k_slices` > `1` )**

在这种模式下，K 维度被分割成 `options.split_k_slices` (代码中用 `batch_count` 变量表示) 块。计算分为两步：

**步骤一：部分计算 (由 `gemm_op()` 完成)**

1. **`gemm_op()` 的结果：**
    * 此时 `gemm_op()` 的 `alpha` 参数被设置为 `1`，`beta` 参数被设置为 `0`。
    * **部分 GEMM 计算:** 对于 K 维度的每一个 slice，`gemm_op()` 会计算该 slice 内的 `A_slice @ B_slice`。它不会应用用户指定的
      `alpha` 和 `beta`，也不会加上矩阵 C。
        * **这些部分 GEMM 的结果（每个 slice 对应一个 M x N 的中间矩阵）会被存储在 `workspace.get()` 指向的设备工作空间内存区域中
          **。具体来说，是存储在 `workspace_gemm_ptr` 指向的区域。
    * **部分 K 维度规约:** 类似地，对于 K 维度的每一个 slice，`gemm_op()` 也会计算该 slice 内输入矩阵 (A 或 B) 的 K 维度规约。
        * **这些部分规约向量的结果也会被存储在 `workspace.get()` 指向的设备工作空间内存区域中**，但位于部分 GEMM
          结果之后的位置。具体来说，是存储在 `workspace_vector_ptr` 指向的区域。

    * 所以，在并行 Split-K 模式下，`gemm_op()` 的直接输出是存储在 `workspace` 中的中间（部分）结果，而不是最终的 `tensor_d` 或
      `tensor_reduction`。

**步骤二：合并部分结果 (由 `reduce_gemm_splitk_op()` 和 `reduce_vector_splitk_op()` 完成)**

2. **`reduce_gemm_splitk_op()` 的结果：**
    * **输入:** 此函数将 `gemm_op()` 在 `workspace` 中生成的部分 GEMM 结果作为输入 (`workspace_gemm_tensorref`)
      。它还会接收原始的矩阵 C (`tensor_c_tensorref`) 以及用户通过命令行指定的最终 `alpha` 和 `beta` 值 (`options.alpha`,
      `options.beta`)。
    * **操作:**
        1. 它会将所有 slice 的部分 GEMM 结果（M x N 矩阵）逐元素相加起来，得到一个总的 M x N 累加矩阵 (
           `Accum = Sum_slices (A_slice @ B_slice)` )。
        2. 然后，它应用用户指定的 `alpha` 和 `beta`，并加上矩阵 C：`D = alpha * Accum + beta * C`。
    * **输出:** **最终的 GEMM 结果矩阵 D 会被存储在 `tensor_d.device_ref().data()` (`tensor_d_tensorref`) 指向的设备内存中
      **。

3. **`reduce_vector_splitk_op()` 的结果：**
    * **输入:** 此函数将 `gemm_op()` 在 `workspace` 中生成的部分 K-Reduction 向量作为输入 (
      `workspace_vector_tensorref`)。
    * **操作:**
        1. 它会将所有 slice 的部分 K-Reduction 向量逐元素相加起来。
        2. 在此函数的参数中，`alpha` 硬编码为 `1.0f`，`beta` 硬编码为 `0.0f`，并且源张量 (`tensor_nullptr_tensorref`) 为
           `nullptr`。这意味着它执行的是一个纯粹的求和操作：
           `Final_Reduction_Vector = Sum_slices (Partial_Reduction_Vector_slice)`。
    * **输出:** **最终的 K-Reduction
      向量结果会被存储在 `tensor_reduction.device_ref().data()` (`tensor_reduction_tensorref`) 指向的设备内存中**。

**总结：**

* **`gemm_op()`**:
    * **非 Split-K**: 直接计算最终的 GEMM 结果 (存入 `tensor_d`) 和 K-Reduction 向量 (存入 `tensor_reduction`)。
    * **并行 Split-K**: 计算 GEMM 和 K-Reduction 的部分和，并将这些中间结果存入 `workspace`。
* **`reduce_gemm_splitk_op()`**:
    * **仅在并行 Split-K 模式下调用。**
    * 将 `workspace` 中的部分 GEMM 和汇总，应用用户指定的 `alpha`、`beta` 和矩阵 C，计算出最终的 GEMM 结果，并存入
      `tensor_d`。
* **`reduce_vector_splitk_op()`**:
    * **仅在并行 Split-K 模式下调用。**
    * 将 `workspace` 中的部分 K-Reduction 向量汇总，计算出最终的 K-Reduction 向量，并存入 `tensor_reduction`。

通过这种方式，算法能够有效地处理大规模 GEMM 问题，并通过 K-Reduction 融合来减少内存传输和内核启动开销，同时利用 Split-K
技术来进一步提高并行度和性能。