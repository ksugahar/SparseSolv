/// @file bddc_preconditioner.hpp
/// @brief BDDC preconditioner with two modes:
///   1. Block elimination: exact (2 iters), O(n_if^3), small problems only
///   2. Element-by-element: scalable, matches NGSolve BDDC (~15-20 iters)

#ifndef SPARSESOLV_PRECONDITIONERS_BDDC_PRECONDITIONER_HPP
#define SPARSESOLV_PRECONDITIONERS_BDDC_PRECONDITIONER_HPP

#include "../core/preconditioner.hpp"
#include "../core/dense_matrix.hpp"
#include "../core/sparse_matrix_view.hpp"
#include "../core/sparse_matrix_coo.hpp"
#include "../core/sparse_matrix_csr.hpp"
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include <complex>

namespace sparsesolv {

/// DOF classification for BDDC
enum class DOFType : uint8_t {
    Wirebasket = 0,  ///< Vertex/edge DOFs (coarse space)
    Interface  = 1   ///< Face/interior DOFs (eliminated locally)
};

/// BDDC preconditioner: block elimination (no element matrices) or element-by-element
template<typename Scalar = double>
class BDDCPreconditioner : public Preconditioner<Scalar> {
public:
    BDDCPreconditioner() = default;

    void set_element_info(std::vector<std::vector<index_t>> element_dofs,
                          std::vector<DOFType> dof_types) {
        element_dofs_ = std::move(element_dofs);
        dof_types_ = std::move(dof_types);
    }

    void set_free_dofs(std::vector<bool> free_dofs) {
        free_dofs_ = std::move(free_dofs);
    }

    /// Set true element matrices to enable element-by-element mode
    void set_element_matrices(std::vector<DenseMatrix<Scalar>> element_matrices) {
        element_matrices_ = std::move(element_matrices);
    }

    /// Skip dense coarse inverse (call set_coarse_solver() after setup instead)
    void set_use_external_coarse(bool use) { use_external_coarse_ = use; }

    /// Set external coarse solver: void(const Scalar* rhs, Scalar* sol) on compact wb vectors
    void set_coarse_solver(std::function<void(const Scalar*, Scalar*)> solver) {
        coarse_solver_ = std::move(solver);
    }

    /// Access wirebasket CSR matrix (available after setup, compact n_wb x n_wb)
    const SparseMatrixCSR<Scalar>& wirebasket_csr() const { return wb_csr_; }

    /// Access wirebasket DOF mapping (compact_idx -> global_dof)
    const std::vector<index_t>& wirebasket_dofs() const { return wb_dofs_; }

    void setup(const SparseMatrixView<Scalar>& A) override {
        n_total_ = A.rows();
        build_dof_maps();

        if (!element_matrices_.empty()) {
            setup_element_bddc();
        } else {
            setup_block_elimination(A);
        }

        this->is_setup_ = true;
    }

    void apply(const Scalar* x, Scalar* y, index_t size) const override {
        if (use_element_mode_) {
            apply_element_bddc(x, y, size);
        } else {
            apply_block_elimination(x, y, size);
        }
    }

    std::string name() const override { return "BDDC"; }

    /// Number of wirebasket DOFs
    index_t num_wirebasket_dofs() const { return n_wb_; }

    /// Number of interface DOFs
    index_t num_interface_dofs() const { return n_if_; }

    /// Whether element-by-element mode is active
    bool is_element_mode() const { return use_element_mode_; }

private:
    bool use_element_mode_ = false;
    bool use_external_coarse_ = false;
    std::function<void(const Scalar*, Scalar*)> coarse_solver_;

    // Input data
    std::vector<std::vector<index_t>> element_dofs_;
    std::vector<DOFType> dof_types_;
    std::vector<bool> free_dofs_;

    // Dimensions
    index_t n_total_ = 0;
    index_t n_wb_ = 0;       // free wirebasket DOFs
    index_t n_if_ = 0;       // free interface DOFs

    // DOF mapping (only free DOFs get compact indices)
    std::vector<index_t> wb_dofs_;   // compact_wb_idx -> global_dof
    std::vector<index_t> if_dofs_;   // compact_if_idx -> global_dof
    std::vector<index_t> wb_map_;    // global_dof -> compact_wb_idx (-1 if not wb)

    // === Block elimination mode data ===
    DenseMatrix<Scalar> aii_inv_;    // n_if x n_if (A_ii inverse)
    DenseMatrix<Scalar> awi_;        // n_wb x n_if (wb-if coupling)
    DenseMatrix<Scalar> aiw_;        // n_if x n_wb (if-wb coupling)
    DenseMatrix<Scalar> schur_inv_;  // n_wb x n_wb (Schur complement inverse)

    // === Element-by-element mode data ===
    std::vector<DenseMatrix<Scalar>> element_matrices_;
    SparseMatrixCSR<Scalar> he_csr_;       // harmonic extension (full space, if->wb)
    SparseMatrixCSR<Scalar> het_csr_;      // harmonic extension transpose (full space, wb->if)
    SparseMatrixCSR<Scalar> is_csr_;       // inner solve (full space, if->if)
    SparseMatrixCSR<Scalar> wb_csr_;       // wirebasket Schur complement (compact n_wb x n_wb)
    DenseMatrix<Scalar> wb_dense_inv_;     // dense inverse of wirebasket Schur complement
    std::vector<double> weight_;           // DOF weights

    // Work vectors for element mode apply (mutable for const apply)
    mutable std::vector<Scalar> work1_, work2_;
    mutable std::vector<Scalar> wb_work1_, wb_work2_;

    // ================================================================
    // Common: DOF map construction
    // ================================================================

    void build_dof_maps() {
        wb_dofs_.clear();
        if_dofs_.clear();
        wb_map_.assign(n_total_, -1);
        n_wb_ = 0;
        n_if_ = 0;

        for (index_t d = 0; d < n_total_; ++d) {
            // Skip non-free (Dirichlet) DOFs
            if (!free_dofs_.empty() &&
                d < static_cast<index_t>(free_dofs_.size()) &&
                !free_dofs_[d])
                continue;

            if (d < static_cast<index_t>(dof_types_.size()) &&
                dof_types_[d] == DOFType::Interface) {
                if_dofs_.push_back(d);
                n_if_++;
            } else {
                wb_map_[d] = n_wb_;
                wb_dofs_.push_back(d);
                n_wb_++;
            }
        }
    }

    // ================================================================
    // Block elimination mode
    // ================================================================

    void setup_block_elimination(const SparseMatrixView<Scalar>& A) {
        use_element_mode_ = false;

        // Extract A_ii
        DenseMatrix<Scalar> aii(n_if_, n_if_);
        for (index_t i = 0; i < n_if_; ++i)
            for (index_t j = 0; j < n_if_; ++j)
                aii(i, j) = A(if_dofs_[i], if_dofs_[j]);

        // Extract A_wi
        awi_.resize(n_wb_, n_if_);
        for (index_t i = 0; i < n_wb_; ++i)
            for (index_t j = 0; j < n_if_; ++j)
                awi_(i, j) = A(wb_dofs_[i], if_dofs_[j]);

        // Extract A_iw
        aiw_.resize(n_if_, n_wb_);
        for (index_t i = 0; i < n_if_; ++i)
            for (index_t j = 0; j < n_wb_; ++j)
                aiw_(i, j) = A(if_dofs_[i], wb_dofs_[j]);

        // Extract A_ww
        DenseMatrix<Scalar> aww(n_wb_, n_wb_);
        for (index_t i = 0; i < n_wb_; ++i)
            for (index_t j = 0; j < n_wb_; ++j)
                aww(i, j) = A(wb_dofs_[i], wb_dofs_[j]);

        aii_inv_ = aii;
        schur_inv_ = aww;

        if (n_if_ > 0) {
            // Invert A_ii
            aii_inv_.invert();

            // Schur complement: S = A_ww - A_wi * A_ii^{-1} * A_iw
            auto tmp = DenseMatrix<Scalar>::multiply(aii_inv_, aiw_);
            for (index_t i = 0; i < n_wb_; ++i) {
                for (index_t p = 0; p < n_if_; ++p) {
                    Scalar a_ip = awi_(i, p);
                    for (index_t j = 0; j < n_wb_; ++j) {
                        schur_inv_(i, j) -= a_ip * tmp(p, j);
                    }
                }
            }
        }
        schur_inv_.invert();
    }

    void apply_block_elimination(const Scalar* x, Scalar* y, index_t size) const {
        std::fill(y, y + size, Scalar(0));

        std::vector<Scalar> x_w(n_wb_), x_i(n_if_);
        for (index_t k = 0; k < n_wb_; ++k) x_w[k] = x[wb_dofs_[k]];
        for (index_t k = 0; k < n_if_; ++k) x_i[k] = x[if_dofs_[k]];

        std::vector<Scalar> u_w(n_wb_), u_i(n_if_);

        if (n_if_ > 0) {
            // Step 1: tmp_i = A_ii^{-1} * x_i
            std::vector<Scalar> tmp_i(n_if_);
            aii_inv_.matvec(x_i.data(), tmp_i.data());

            // Step 2: r_w = x_w - A_wi * tmp_i
            std::vector<Scalar> r_w(n_wb_);
            awi_.matvec(tmp_i.data(), r_w.data());
            for (index_t k = 0; k < n_wb_; ++k) r_w[k] = x_w[k] - r_w[k];

            // Step 3: u_w = S^{-1} * r_w
            schur_inv_.matvec(r_w.data(), u_w.data());

            // Step 4: u_i = A_ii^{-1} * (x_i - A_iw * u_w)
            std::vector<Scalar> tmp_i2(n_if_);
            aiw_.matvec(u_w.data(), tmp_i2.data());
            for (index_t k = 0; k < n_if_; ++k) tmp_i2[k] = x_i[k] - tmp_i2[k];
            aii_inv_.matvec(tmp_i2.data(), u_i.data());
        } else {
            // No interface DOFs: u_w = S^{-1} * x_w (S = A_ww)
            schur_inv_.matvec(x_w.data(), u_w.data());
        }

        // Scatter back to full space
        for (index_t k = 0; k < n_wb_; ++k) y[wb_dofs_[k]] = u_w[k];
        for (index_t k = 0; k < n_if_; ++k) y[if_dofs_[k]] = u_i[k];
    }

    // ================================================================
    // Element-by-element mode
    // ================================================================

    /// Element-by-element BDDC setup: process elements, assemble global operators, build coarse inverse
    void setup_element_bddc() {
        use_element_mode_ = true;

        // Initialize COO accumulators
        SparseMatrixCOO<Scalar> he_coo(n_total_, n_total_);
        SparseMatrixCOO<Scalar> het_coo(n_total_, n_total_);
        SparseMatrixCOO<Scalar> is_coo(n_total_, n_total_);
        SparseMatrixCOO<Scalar> wb_coo(n_wb_, n_wb_);

        weight_.assign(n_total_, 0.0);

        // Process each element
        for (size_t e = 0; e < element_dofs_.size(); ++e) {
            process_element(e, he_coo, het_coo, is_coo, wb_coo);
        }

        // Convert COO to CSR (sums duplicate entries)
        he_csr_ = he_coo.to_csr();
        het_csr_ = het_coo.to_csr();
        is_csr_ = is_coo.to_csr();
        wb_csr_ = wb_coo.to_csr();

        // Apply weight normalization to CSR matrices
        finalize_weights();

        // Build dense inverse for wirebasket coarse solve
        // (skip if external coarse solver will be provided)
        if (!use_external_coarse_) {
            build_coarse_dense_inverse(wb_csr_);
        }

        // Free element matrices (no longer needed)
        element_matrices_.clear();
        element_matrices_.shrink_to_fit();

        // Pre-allocate work vectors
        work1_.resize(n_total_);
        work2_.resize(n_total_);
        wb_work1_.resize(n_wb_);
        wb_work2_.resize(n_wb_);
    }

    /// Process one element: extract Schur complement, harmonic extension, inner solve → accumulate into COO
    void process_element(size_t e,
                         SparseMatrixCOO<Scalar>& he_coo,
                         SparseMatrixCOO<Scalar>& het_coo,
                         SparseMatrixCOO<Scalar>& is_coo,
                         SparseMatrixCOO<Scalar>& wb_coo) {
        const auto& el_dofs = element_dofs_[e];
        const auto& elmat = element_matrices_[e];
        index_t nel = static_cast<index_t>(el_dofs.size());

        // Classify element DOFs into wirebasket and interface
        std::vector<index_t> local_wb, local_if;
        for (index_t k = 0; k < nel; ++k) {
            index_t d = el_dofs[k];
            // Skip non-free (Dirichlet) DOFs
            if (!free_dofs_.empty() &&
                d < static_cast<index_t>(free_dofs_.size()) &&
                !free_dofs_[d])
                continue;
            if (d < static_cast<index_t>(dof_types_.size()) &&
                dof_types_[d] == DOFType::Interface)
                local_if.push_back(k);
            else
                local_wb.push_back(k);
        }

        index_t nw = static_cast<index_t>(local_wb.size());
        index_t ni = static_cast<index_t>(local_if.size());

        if (nw == 0) return;  // No wirebasket DOFs in this element

        // Extract K_ww block → becomes Schur complement after elimination
        DenseMatrix<Scalar> schur(nw, nw);
        for (index_t i = 0; i < nw; ++i)
            for (index_t j = 0; j < nw; ++j)
                schur(i, j) = elmat(local_wb[i], local_wb[j]);

        if (ni > 0) {
            DenseMatrix<Scalar> K_wi(nw, ni);
            DenseMatrix<Scalar> K_iw(ni, nw);
            DenseMatrix<Scalar> K_ii(ni, ni);

            for (index_t i = 0; i < nw; ++i)
                for (index_t j = 0; j < ni; ++j)
                    K_wi(i, j) = elmat(local_wb[i], local_if[j]);
            for (index_t i = 0; i < ni; ++i)
                for (index_t j = 0; j < nw; ++j)
                    K_iw(i, j) = elmat(local_if[i], local_wb[j]);
            for (index_t i = 0; i < ni; ++i)
                for (index_t j = 0; j < ni; ++j)
                    K_ii(i, j) = elmat(local_if[i], local_if[j]);

            // Element-level weight from K_ii diagonal
            std::vector<double> elem_weight(ni);
            for (index_t k = 0; k < ni; ++k)
                elem_weight[k] = std::abs(K_ii(k, k));

            DenseMatrix<Scalar> K_ii_inv = K_ii;
            K_ii_inv.invert();

            // harm_ext = -K_ii^{-1} * K_iw  (ni x nw)
            DenseMatrix<Scalar> harm_ext = DenseMatrix<Scalar>::multiply(K_ii_inv, K_iw);
            harm_ext.negate();

            // schur = K_ww + K_wi * harm_ext = K_ww - K_wi * K_ii^{-1} * K_iw
            DenseMatrix<Scalar>::multiply_add(K_wi, harm_ext, schur);

            // harm_ext_t = -K_wi * K_ii^{-1}  (nw x ni)
            DenseMatrix<Scalar> harm_ext_t = DenseMatrix<Scalar>::multiply(K_wi, K_ii_inv);
            harm_ext_t.negate();

            // Apply element weights (following NGSolve's AddMatrix)
            for (index_t k = 0; k < ni; ++k)
                harm_ext.scale_row(k, static_cast<Scalar>(elem_weight[k]));
            for (index_t l = 0; l < ni; ++l)
                harm_ext_t.scale_col(l, static_cast<Scalar>(elem_weight[l]));
            for (index_t k = 0; k < ni; ++k)
                K_ii_inv.scale_row(k, static_cast<Scalar>(elem_weight[k]));
            for (index_t l = 0; l < ni; ++l)
                K_ii_inv.scale_col(l, static_cast<Scalar>(elem_weight[l]));

            // Accumulate global weight for interface DOFs
            for (index_t k = 0; k < ni; ++k)
                weight_[el_dofs[local_if[k]]] += elem_weight[k];

            // Map to global DOF indices and accumulate into COO
            std::vector<index_t> g_if(ni), g_wb(nw);
            for (index_t k = 0; k < ni; ++k) g_if[k] = el_dofs[local_if[k]];
            for (index_t k = 0; k < nw; ++k) g_wb[k] = el_dofs[local_wb[k]];

            he_coo.add_submatrix(g_if.data(), ni, g_wb.data(), nw, harm_ext);
            het_coo.add_submatrix(g_wb.data(), nw, g_if.data(), ni, harm_ext_t);
            is_coo.add_submatrix(g_if.data(), ni, g_if.data(), ni, K_ii_inv);

            std::vector<index_t> c_wb(nw);
            for (index_t k = 0; k < nw; ++k) c_wb[k] = wb_map_[g_wb[k]];
            wb_coo.add_submatrix(c_wb.data(), nw, c_wb.data(), nw, schur);
        } else {
            // No interface DOFs: add K_ww directly to wirebasket
            std::vector<index_t> g_wb(nw), c_wb(nw);
            for (index_t k = 0; k < nw; ++k) {
                g_wb[k] = el_dofs[local_wb[k]];
                c_wb[k] = wb_map_[g_wb[k]];
            }
            wb_coo.add_submatrix(c_wb.data(), nw, c_wb.data(), nw, schur);
        }
    }

    /// Invert accumulated weights and scale harmonic_ext, inner_solve CSR matrices
    void finalize_weights() {
        // Invert accumulated weights
        for (index_t i = 0; i < n_total_; ++i)
            if (weight_[i] > 0.0)
                weight_[i] = 1.0 / weight_[i];

        // Scale inner_solve: is[i,j] *= w[i] * w[j]
        for (index_t i = 0; i < is_csr_.rows; ++i) {
            double wi = weight_[i];
            for (index_t k = is_csr_.row_ptr[i]; k < is_csr_.row_ptr[i + 1]; ++k) {
                index_t j = is_csr_.col_idx[k];
                is_csr_.values[k] *= static_cast<Scalar>(wi * weight_[j]);
            }
        }

        // Scale harmonic_ext: he[i,:] *= w[i]
        for (index_t i = 0; i < he_csr_.rows; ++i) {
            double wi = weight_[i];
            for (index_t k = he_csr_.row_ptr[i]; k < he_csr_.row_ptr[i + 1]; ++k)
                he_csr_.values[k] *= static_cast<Scalar>(wi);
        }

        // Scale harmonic_ext_trans: het[:,j] *= w[j]
        for (index_t i = 0; i < het_csr_.rows; ++i) {
            for (index_t k = het_csr_.row_ptr[i]; k < het_csr_.row_ptr[i + 1]; ++k) {
                index_t j = het_csr_.col_idx[k];
                het_csr_.values[k] *= static_cast<Scalar>(weight_[j]);
            }
        }
    }

    /// Convert sparse wirebasket matrix to dense and compute LU inverse
    void build_coarse_dense_inverse(const SparseMatrixCSR<Scalar>& wb_csr) {
        DenseMatrix<Scalar> wb_dense(n_wb_, n_wb_);
        for (index_t i = 0; i < n_wb_; ++i)
            for (index_t k = wb_csr.row_ptr[i]; k < wb_csr.row_ptr[i + 1]; ++k)
                wb_dense(i, wb_csr.col_idx[k]) = wb_csr.values[k];
        wb_dense_inv_ = wb_dense;
        wb_dense_inv_.invert();
    }

    /// Apply element-by-element BDDC: y = (I + he) * (S_wb^{-1} * (I + he^T) * x + is * x)
    void apply_element_bddc(const Scalar* x, Scalar* y, index_t size) const {
        // Step 1: y = x
        std::copy(x, x + size, y);

        // Step 2: y += het * x (harmonic extension transpose)
        {
            SparseMatrixView<Scalar> het_view(
                het_csr_.rows, het_csr_.cols,
                het_csr_.row_ptr.data(), het_csr_.col_idx.data(),
                het_csr_.values.data());
            het_view.multiply(x, work1_.data());
            for (index_t i = 0; i < size; ++i)
                y[i] += work1_[i];
        }

        // Step 3: Coarse solve (wirebasket inverse)
        // Gather wirebasket RHS from y
        for (index_t k = 0; k < n_wb_; ++k)
            wb_work1_[k] = y[wb_dofs_[k]];

        // Solve: wb_sol = S_wb^{-1} * wb_rhs
        if (coarse_solver_) {
            coarse_solver_(wb_work1_.data(), wb_work2_.data());
        } else {
            wb_dense_inv_.matvec(wb_work1_.data(), wb_work2_.data());
        }

        // Scatter wirebasket solution to full-space tmp
        std::fill(work1_.begin(), work1_.end(), Scalar(0));
        for (index_t k = 0; k < n_wb_; ++k)
            work1_[wb_dofs_[k]] = wb_work2_[k];

        // Step 4: tmp += inner_solve * x
        {
            SparseMatrixView<Scalar> is_view(
                is_csr_.rows, is_csr_.cols,
                is_csr_.row_ptr.data(), is_csr_.col_idx.data(),
                is_csr_.values.data());
            is_view.multiply(x, work2_.data());
            for (index_t i = 0; i < size; ++i)
                work1_[i] += work2_[i];
        }

        // Step 5: y = tmp + he * tmp (harmonic extension)
        {
            SparseMatrixView<Scalar> he_view(
                he_csr_.rows, he_csr_.cols,
                he_csr_.row_ptr.data(), he_csr_.col_idx.data(),
                he_csr_.values.data());
            he_view.multiply(work1_.data(), work2_.data());
            for (index_t i = 0; i < size; ++i)
                y[i] = work1_[i] + work2_[i];
        }
    }
};

} // namespace sparsesolv

#endif // SPARSESOLV_PRECONDITIONERS_BDDC_PRECONDITIONER_HPP
