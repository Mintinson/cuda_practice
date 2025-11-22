#ifndef CPU_MAT_TRANS_HPP
#define CPU_MAT_TRANS_HPP

// src: m * n, dst: n * m, and reading from src is continuous, if ContiguousR is true
template <typename T, bool ContiguousR = true>
void cpu_mat_trans(const T* src, T* dst, int m, int n)
{
    if constexpr (ContiguousR) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                dst[j * m + i] = src[i * n + j];
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                dst[i * m + j] = src[j * n + i];
            }
        }
    }
}

#endif // CPU_MAT_TRANS_HPP