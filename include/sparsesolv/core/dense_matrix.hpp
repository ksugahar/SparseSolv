/// @file dense_matrix.hpp
/// @brief Small dense matrix with LU inverse for BDDC element-level operations

#ifndef SPARSESOLV_CORE_DENSE_MATRIX_HPP
#define SPARSESOLV_CORE_DENSE_MATRIX_HPP

#include "types.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace sparsesolv {

namespace detail {
    /// Absolute value that works for both real and complex
    template<typename T>
    inline double dense_abs(T val) { return std::abs(val); }
}

/// Small dense matrix (row-major) for element-level BDDC operations
template<typename Scalar>
class DenseMatrix {
public:
    DenseMatrix() : rows_(0), cols_(0) {}

    DenseMatrix(index_t rows, index_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, Scalar(0)) {}

    DenseMatrix(index_t rows, index_t cols, Scalar init)
        : rows_(rows), cols_(cols), data_(rows * cols, init) {}

    index_t rows() const { return rows_; }
    index_t cols() const { return cols_; }

    Scalar& operator()(index_t i, index_t j) {
        return data_[i * cols_ + j];
    }

    const Scalar& operator()(index_t i, index_t j) const {
        return data_[i * cols_ + j];
    }

    void resize(index_t rows, index_t cols) {
        rows_ = rows;
        cols_ = cols;
        data_.assign(rows * cols, Scalar(0));
    }

    void set_zero() {
        std::fill(data_.begin(), data_.end(), Scalar(0));
    }

    /// Scale row i by factor
    void scale_row(index_t i, Scalar factor) {
        for (index_t j = 0; j < cols_; ++j)
            data_[i * cols_ + j] *= factor;
    }

    /// Scale column j by factor
    void scale_col(index_t j, Scalar factor) {
        for (index_t i = 0; i < rows_; ++i)
            data_[i * cols_ + j] *= factor;
    }

    /// Negate all entries
    void negate() {
        for (auto& v : data_) v = -v;
    }

    /// Matrix multiplication: C = A * B
    static DenseMatrix multiply(const DenseMatrix& A, const DenseMatrix& B) {
        DenseMatrix C(A.rows_, B.cols_);
        for (index_t i = 0; i < A.rows_; ++i) {
            for (index_t p = 0; p < A.cols_; ++p) {
                Scalar a_ip = A(i, p);
                for (index_t j = 0; j < B.cols_; ++j) {
                    C(i, j) += a_ip * B(p, j);
                }
            }
        }
        return C;
    }

    /// Multiply-add: C += A * B
    static void multiply_add(const DenseMatrix& A, const DenseMatrix& B,
                              DenseMatrix& C) {
        for (index_t i = 0; i < A.rows_; ++i) {
            for (index_t p = 0; p < A.cols_; ++p) {
                Scalar a_ip = A(i, p);
                for (index_t j = 0; j < B.cols_; ++j) {
                    C(i, j) += a_ip * B(p, j);
                }
            }
        }
    }

    /// Matrix-vector multiply: y = A * x
    void matvec(const Scalar* x, Scalar* y) const {
        for (index_t i = 0; i < rows_; ++i) {
            Scalar sum = Scalar(0);
            for (index_t j = 0; j < cols_; ++j) {
                sum += data_[i * cols_ + j] * x[j];
            }
            y[i] = sum;
        }
    }

    /// In-place LU with partial pivoting. Returns pivot permutation.
    std::vector<index_t> lu_factorize() {
        const index_t n = rows_;
        std::vector<index_t> piv(n);
        for (index_t i = 0; i < n; ++i) piv[i] = i;

        for (index_t k = 0; k < n; ++k) {
            // Find pivot: max |A[i,k]| for i >= k
            double max_val = 0;
            index_t max_row = k;
            for (index_t i = k; i < n; ++i) {
                double val = detail::dense_abs((*this)(i, k));
                if (val > max_val) {
                    max_val = val;
                    max_row = i;
                }
            }

            if (max_val < 1e-15) {
                throw std::runtime_error(
                    "DenseMatrix::lu_factorize: singular or near-singular matrix");
            }

            // Swap rows k and max_row
            if (max_row != k) {
                std::swap(piv[k], piv[max_row]);
                for (index_t j = 0; j < n; ++j) {
                    std::swap((*this)(k, j), (*this)(max_row, j));
                }
            }

            // Eliminate below
            Scalar diag = (*this)(k, k);
            for (index_t i = k + 1; i < n; ++i) {
                Scalar factor = (*this)(i, k) / diag;
                (*this)(i, k) = factor;  // Store L factor
                for (index_t j = k + 1; j < n; ++j) {
                    (*this)(i, j) -= factor * (*this)(k, j);
                }
            }
        }
        return piv;
    }

    /// Compute inverse in-place via LU decomposition
    void invert() {
        const index_t n = rows_;
        if (n == 0) return;

        // LU factorize
        auto piv = lu_factorize();

        // Build permuted identity
        DenseMatrix inv(n, n);
        for (index_t j = 0; j < n; ++j) {
            inv(piv[j], j) = Scalar(1);
        }

        // Forward substitution: solve L * Y = P * I
        for (index_t j = 0; j < n; ++j) {
            for (index_t i = 1; i < n; ++i) {
                for (index_t k = 0; k < i; ++k) {
                    inv(i, j) -= (*this)(i, k) * inv(k, j);
                }
            }
        }

        // Backward substitution: solve U * X = Y
        for (index_t j = 0; j < n; ++j) {
            for (index_t i = n; i-- > 0;) {
                for (index_t k = i + 1; k < n; ++k) {
                    inv(i, j) -= (*this)(i, k) * inv(k, j);
                }
                inv(i, j) /= (*this)(i, i);
            }
        }

        // Copy result back
        data_ = std::move(inv.data_);
    }

private:
    index_t rows_, cols_;
    std::vector<Scalar> data_;
};

} // namespace sparsesolv

#endif // SPARSESOLV_CORE_DENSE_MATRIX_HPP
