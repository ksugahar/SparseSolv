/// @file preconditioner.hpp
/// @brief Base class for preconditioners

#ifndef SPARSESOLV_CORE_PRECONDITIONER_HPP
#define SPARSESOLV_CORE_PRECONDITIONER_HPP

#include "types.hpp"
#include "constants.hpp"
#include "parallel.hpp"
#include "sparse_matrix_view.hpp"
#include <string>
#include <memory>

namespace sparsesolv {

/// Abstract base class for preconditioners: setup(A), then apply(x, y) = M^{-1}*x
template<typename Scalar = double>
class Preconditioner {
public:
    using value_type = Scalar;

    virtual ~Preconditioner() = default;

    virtual void setup(const SparseMatrixView<Scalar>& A) = 0;
    virtual void apply(const Scalar* x, Scalar* y, index_t size) const = 0;

    void apply(const std::vector<Scalar>& x, std::vector<Scalar>& y) const {
        assert(x.size() == y.size());
        apply(x.data(), y.data(), static_cast<index_t>(x.size()));
    }

    virtual std::string name() const = 0;
    virtual bool is_ready() const { return is_setup_; }

protected:
    bool is_setup_ = false;
};

/// Identity preconditioner (no-op, copies x to y)
template<typename Scalar = double>
class IdentityPreconditioner : public Preconditioner<Scalar> {
public:
    void setup(const SparseMatrixView<Scalar>& /*A*/) override {
        this->is_setup_ = true;
    }

    void apply(const Scalar* x, Scalar* y, index_t size) const override {
        std::copy(x, x + size, y);
    }

    std::string name() const override { return "Identity"; }
};

/// Jacobi (diagonal) preconditioner: M = diag(A)
template<typename Scalar = double>
class JacobiPreconditioner : public Preconditioner<Scalar> {
public:
    void setup(const SparseMatrixView<Scalar>& A) override {
        index_t n = A.rows();
        inv_diag_.resize(n);
        for (index_t i = 0; i < n; ++i) {
            Scalar d = A.diagonal(i);
            // Avoid division by zero
            inv_diag_[i] = (std::abs(d) > constants::MIN_DIAGONAL_TOLERANCE) ? Scalar(1) / d : Scalar(1);
        }
        this->is_setup_ = true;
    }

    void apply(const Scalar* x, Scalar* y, index_t size) const override {
        parallel_for(size, [&](index_t i) {
            y[i] = inv_diag_[i] * x[i];
        });
    }

    std::string name() const override { return "Jacobi"; }

private:
    std::vector<Scalar> inv_diag_;
};

// Type aliases
using PreconditionerD = Preconditioner<double>;
using PreconditionerC = Preconditioner<complex_t>;

} // namespace sparsesolv

#endif // SPARSESOLV_CORE_PRECONDITIONER_HPP
