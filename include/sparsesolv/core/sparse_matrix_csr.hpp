/// @file sparse_matrix_csr.hpp
/// @brief Owned CSR matrix storage (IC L factor, BDDC harmonic extension, etc.)

#ifndef SPARSESOLV_CORE_SPARSE_MATRIX_CSR_HPP
#define SPARSESOLV_CORE_SPARSE_MATRIX_CSR_HPP

#include "types.hpp"
#include <vector>

namespace sparsesolv {

/// Sparse matrix in CSR format with owned storage
template<typename Scalar>
struct SparseMatrixCSR {
    std::vector<index_t> row_ptr;
    std::vector<index_t> col_idx;
    std::vector<Scalar> values;
    index_t rows = 0;
    index_t cols = 0;

    index_t nnz() const { return static_cast<index_t>(values.size()); }

    void clear() {
        row_ptr.clear();
        col_idx.clear();
        values.clear();
        rows = cols = 0;
    }

    /// y = A * x
    void multiply(const Scalar* x, Scalar* y) const {
        for (index_t i = 0; i < rows; ++i) {
            Scalar sum = Scalar(0);
            for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                sum += values[k] * x[col_idx[k]];
            }
            y[i] = sum;
        }
    }

    /// y += A * x
    void multiply_add(const Scalar* x, Scalar* y) const {
        for (index_t i = 0; i < rows; ++i) {
            for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                y[i] += values[k] * x[col_idx[k]];
            }
        }
    }
};

} // namespace sparsesolv

#endif // SPARSESOLV_CORE_SPARSE_MATRIX_CSR_HPP
