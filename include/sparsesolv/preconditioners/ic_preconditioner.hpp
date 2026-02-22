/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * @file ic_preconditioner.hpp
 * @brief Incomplete Cholesky (IC) preconditioner
 */

#ifndef SPARSESOLV_PRECONDITIONERS_IC_PRECONDITIONER_HPP
#define SPARSESOLV_PRECONDITIONERS_IC_PRECONDITIONER_HPP

#include "../core/preconditioner.hpp"
#include "../core/solver_config.hpp"
#include "../core/sparse_matrix_csr.hpp"
#include "../core/level_schedule.hpp"
#include "../core/abmc_ordering.hpp"
#include "../core/rcm_ordering.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace sparsesolv {

/**
 * @brief Incomplete Cholesky (IC) preconditioner
 *
 * Computes an incomplete Cholesky factorization: A ≈ L * D * L^T
 * where L is a lower triangular matrix and D is diagonal.
 *
 * The shift parameter (α) modifies the diagonal during factorization
 * to improve stability: diag(L) = α * diag(A)
 *
 * Typical values for shift_parameter:
 * - 1.0: Standard IC(0) - may fail for indefinite matrices
 * - 1.05: Slightly shifted - good default
 * - 1.1-1.2: More stable for difficult problems
 *
 * Example:
 * @code
 * ICPreconditioner<double> precond(1.05);
 * precond.setup(A);
 *
 * // Apply preconditioner: y = (LDL^T)^{-1} * x
 * precond.apply(x, y, n);
 * @endcode
 *
 * @tparam Scalar The scalar type (double or complex<double>)
 */
template<typename Scalar = double>
class ICPreconditioner : public Preconditioner<Scalar> {
public:
    /**
     * @brief Construct IC preconditioner with shift parameter
     * @param shift Shift parameter for diagonal (default: 1.05)
     */
    explicit ICPreconditioner(double shift = 1.05)
        : shift_parameter_(shift)
    {}

    /**
     * @brief Setup the IC factorization from matrix A
     *
     * Computes L and D such that A ≈ L * D * L^T
     *
     * @param A Symmetric positive definite sparse matrix (CSR format)
     */
    void setup(const SparseMatrixView<Scalar>& A) override {
        const index_t n = A.rows();
        size_ = n;

        // Clear RCM state from previous setup
        rcm_ordering_.clear();
        rcm_reverse_ordering_.clear();
        rcm_csr_.clear();

        if (config_.use_abmc) {
            // ABMC path: optionally apply RCM first, then reorder + factorize
            const index_t* src_row_ptr = A.row_ptr();
            const index_t* src_col_idx = A.col_idx();

            if (config_.abmc_use_rcm) {
                // Step 0: RCM bandwidth reduction
                compute_rcm_ordering(A.row_ptr(), A.col_idx(), n,
                                     rcm_ordering_, rcm_reverse_ordering_);
                reorder_matrix_with_perm(A, rcm_ordering_, rcm_csr_);
                src_row_ptr = rcm_csr_.row_ptr.data();
                src_col_idx = rcm_csr_.col_idx.data();
            }

            // 1. Build ABMC schedule from (possibly RCM-reordered) matrix
            abmc_schedule_.build(src_row_ptr, src_col_idx, n,
                                 config_.abmc_block_size, config_.abmc_num_colors);

            // 2. Reorder the matrix according to ABMC permutation
            if (config_.abmc_use_rcm) {
                SparseMatrixView<Scalar> rcm_view(
                    rcm_csr_.rows, rcm_csr_.cols,
                    rcm_csr_.row_ptr.data(), rcm_csr_.col_idx.data(),
                    rcm_csr_.values.data());
                reorder_matrix(rcm_view);
            } else {
                reorder_matrix(A);
            }

            // 3. Extract lower triangular from the reordered matrix
            SparseMatrixView<Scalar> A_reordered(
                reordered_csr_.rows, reordered_csr_.cols,
                reordered_csr_.row_ptr.data(),
                reordered_csr_.col_idx.data(),
                reordered_csr_.values.data());
            extract_lower_triangular(A_reordered);

            // Memory management for reordered matrices
            if (!config_.abmc_reorder_spmv) {
                reordered_csr_.clear(); // SpMV uses original or RCM matrix
            }
            if (!config_.abmc_use_rcm) {
                rcm_csr_.clear();
            }
        } else {
            // Standard path: extract lower triangular directly
            extract_lower_triangular(A);
        }

        // Save original values for auto-shift restart or diagonal scaling
        if (config_.auto_shift || config_.diagonal_scaling) {
            original_values_ = L_.values;
        }

        // Compute diagonal scaling factors if enabled
        if (config_.diagonal_scaling) {
            compute_scaling_factors();
        }

        // Compute IC factorization (with auto-shift if enabled)
        compute_ic_factorization();

        // Compute transpose L^T
        compute_transpose();

        // Pre-allocate work vectors (avoid per-call heap allocation)
        work_temp_.resize(n);
        if (config_.diagonal_scaling) {
            work_temp2_.resize(n);
        }

        if (config_.use_abmc) {
            // Pre-allocate ABMC-specific work vectors
            abmc_x_perm_.resize(n);
            abmc_y_perm_.resize(n);

            // Pre-compute composite permutations to avoid multi-level indirection
            build_composite_permutations(n);
        } else {
            // Build level schedules for parallel triangular solves
            fwd_schedule_.build_from_lower(L_.row_ptr.data(), L_.col_idx.data(), n);
            bwd_schedule_.build_from_upper(Lt_.row_ptr.data(), Lt_.col_idx.data(), n);
        }

        this->is_setup_ = true;
    }

    /**
     * @brief Apply IC preconditioner: y = (LDL^T)^{-1} * x
     *
     * Solves L * D * L^T * y = x in three steps:
     * 1. Forward substitution: L * z = x
     * 2. Diagonal scaling: w = D^{-1} * z
     * 3. Backward substitution: L^T * y = w
     *
     * @param x Input vector
     * @param y Output vector
     * @param size Vector size
     */
    void apply(const Scalar* x, Scalar* y, index_t size) const override {
        if (!this->is_setup_) {
            throw std::runtime_error("ICPreconditioner::apply called before setup");
        }

        if (config_.use_abmc && abmc_schedule_.is_built()) {
            apply_abmc(x, y, size);
        } else {
            apply_level_schedule(x, y, size);
        }
    }

    std::string name() const override { return "IC"; }

    /// Get the shift parameter
    double shift_parameter() const { return shift_parameter_; }

    /// Set the shift parameter (must call setup again after changing)
    void set_shift_parameter(double shift) { shift_parameter_ = shift; }

    /// Get the actual shift used (may differ from initial if auto-shift is enabled)
    double actual_shift() const { return actual_shift_; }

    /// Set solver config (for auto-shift, diagonal scaling, etc.)
    void set_config(const SolverConfig& config) {
        config_ = config;
        shift_parameter_ = config.shift_parameter;
    }

    /// Check if ABMC reordered matrix is available for CG in reordered space
    /// Only true when abmc_reorder_spmv=true (legacy mode where CG runs in ABMC space)
    bool has_reordered_matrix() const {
        return config_.use_abmc && config_.abmc_reorder_spmv && reordered_csr_.rows > 0;
    }

    /// Get a view of the reordered full matrix (valid only if has_reordered_matrix())
    SparseMatrixView<Scalar> reordered_matrix_view() const {
        return SparseMatrixView<Scalar>(
            reordered_csr_.rows, reordered_csr_.cols,
            reordered_csr_.row_ptr.data(),
            reordered_csr_.col_idx.data(),
            reordered_csr_.values.data());
    }

    /// Get the ABMC ordering (ordering[old_row] = new_row)
    const std::vector<index_t>& abmc_ordering() const {
        return abmc_schedule_.ordering;
    }

    /// Check if RCM ordering is available
    bool has_rcm_ordering() const {
        return !rcm_ordering_.empty();
    }

    /// Get RCM permutation (ordering[old_row] = new_row)
    const std::vector<index_t>& rcm_perm() const {
        return rcm_ordering_;
    }

    /// Get RCM reverse permutation (reverse_ordering[new_row] = old_row)
    const std::vector<index_t>& rcm_reverse_perm() const {
        return rcm_reverse_ordering_;
    }

    /// Check if RCM-reordered matrix is available for SpMV
    bool has_rcm_matrix() const {
        return config_.abmc_use_rcm && !config_.abmc_reorder_spmv && rcm_csr_.rows > 0;
    }

    /// Get a view of the RCM-reordered matrix (valid only if has_rcm_matrix())
    SparseMatrixView<Scalar> rcm_matrix_view() const {
        return SparseMatrixView<Scalar>(
            rcm_csr_.rows, rcm_csr_.cols,
            rcm_csr_.row_ptr.data(),
            rcm_csr_.col_idx.data(),
            rcm_csr_.values.data());
    }

    /**
     * @brief Apply preconditioner in reordered space (no permutation)
     *
     * For use when CG operates entirely in ABMC-reordered space.
     * Input x and output y are already in reordered space.
     */
    void apply_in_reordered_space(const Scalar* x, Scalar* y, index_t size) const {
        if (!this->is_setup_) {
            throw std::runtime_error(
                "ICPreconditioner::apply_in_reordered_space called before setup");
        }

        if (config_.diagonal_scaling && !scaling_.empty()) {
            parallel_for(size, [&](index_t i) {
                abmc_x_perm_[i] = x[i] * static_cast<Scalar>(scaling_[i]);
            });
            forward_substitution_abmc(abmc_x_perm_.data(), work_temp_.data());
            backward_substitution_abmc(work_temp_.data(), abmc_x_perm_.data());
            parallel_for(size, [&](index_t i) {
                y[i] = abmc_x_perm_[i] * static_cast<Scalar>(scaling_[i]);
            });
        } else {
            forward_substitution_abmc(x, work_temp_.data());
            backward_substitution_abmc(work_temp_.data(), y);
        }
    }

    /**
     * @brief Apply with composite RCM+ABMC ordering (input in RCM space)
     *
     * Used by ICPrecondRCMABMCAdapter when CG operates in RCM-reordered
     * space (Path 2). Input x and output y are in RCM space.
     * Permutation: RCM -> ABMC -> tri-solve -> ABMC -> RCM.
     */
    void apply_rcm_abmc(const Scalar* x, Scalar* y, index_t size) const {
        const auto& perm = composite_perm_rcm_;

        if (config_.diagonal_scaling && !scaling_.empty()) {
            parallel_for(size, [&](index_t i) {
                abmc_x_perm_[perm[i]] = x[i] *
                    static_cast<Scalar>(scaling_[perm[i]]);
            });
            forward_substitution_abmc(abmc_x_perm_.data(), work_temp_.data());
            backward_substitution_abmc(work_temp_.data(), abmc_y_perm_.data());
            parallel_for(size, [&](index_t i) {
                y[i] = abmc_y_perm_[perm[i]] *
                    static_cast<Scalar>(scaling_[perm[i]]);
            });
        } else {
            parallel_for(size, [&](index_t i) {
                abmc_x_perm_[perm[i]] = x[i];
            });
            forward_substitution_abmc(abmc_x_perm_.data(), work_temp_.data());
            backward_substitution_abmc(work_temp_.data(), abmc_y_perm_.data());
            parallel_for(size, [&](index_t i) {
                y[i] = abmc_y_perm_[perm[i]];
            });
        }
    }

private:
    double shift_parameter_;
    double actual_shift_ = 0.0;      // Actual shift used (after auto-adjustment)
    index_t size_ = 0;
    SolverConfig config_;

    // IC factorization result: A ≈ L * D * L^T
    SparseMatrixCSR<Scalar> L_;      // Lower triangular factor
    SparseMatrixCSR<Scalar> Lt_;     // Upper triangular factor (L^T)
    std::vector<Scalar> inv_diag_;   // D^{-1} (inverse diagonal)

    // Auto-shift support
    std::vector<Scalar> original_values_;  // Original L values for restart

    // Diagonal scaling
    std::vector<double> scaling_;    // scaling[i] = 1/sqrt(A[i,i])

    // Level schedules for parallel triangular solves (non-ABMC path)
    LevelSchedule fwd_schedule_;     // For forward substitution (L)
    LevelSchedule bwd_schedule_;     // For backward substitution (L^T)

    // Work vectors for apply (pre-allocated in setup, avoid per-call heap alloc)
    mutable std::vector<Scalar> work_temp_;     // Intermediate result
    mutable std::vector<Scalar> work_temp2_;    // Second intermediate (diagonal scaling)

    // ABMC ordering support
    ABMCSchedule abmc_schedule_;             // Block-color schedule
    SparseMatrixCSR<Scalar> reordered_csr_;  // Temporary: reordered full matrix
    mutable std::vector<Scalar> abmc_x_perm_;   // Work vector: permuted input
    mutable std::vector<Scalar> abmc_y_perm_;   // Work vector: permuted output

    // RCM ordering support (Approach B)
    std::vector<index_t> rcm_ordering_;          // rcm_ordering_[old] = new
    std::vector<index_t> rcm_reverse_ordering_;  // rcm_reverse_ordering_[new] = old
    SparseMatrixCSR<Scalar> rcm_csr_;            // RCM-reordered full matrix

    // Composite permutation (pre-computed to avoid multi-level indirection)
    std::vector<index_t> composite_perm_;        // orig -> ABMC (or orig -> RCM -> ABMC)
    std::vector<index_t> composite_perm_rcm_;    // RCM -> ABMC (for apply_rcm_abmc)
    std::vector<double> composite_scaling_;       // scaling in original space

    // ================================================================
    // Composite permutation setup
    // ================================================================

    void build_composite_permutations(index_t n) {
        const auto& abmc_ord = abmc_schedule_.ordering;

        // composite_perm_: original space -> ABMC space
        if (config_.abmc_use_rcm && !rcm_ordering_.empty()) {
            // orig -> RCM -> ABMC
            composite_perm_.resize(n);
            for (index_t i = 0; i < n; ++i) {
                composite_perm_[i] = abmc_ord[rcm_ordering_[i]];
            }
        } else {
            composite_perm_ = abmc_ord;
        }

        // composite_perm_rcm_: RCM space -> ABMC space (for apply_rcm_abmc adapter)
        composite_perm_rcm_ = abmc_ord;

        // Pre-compute scaling in original space for the composite path
        if (config_.diagonal_scaling && !scaling_.empty()) {
            composite_scaling_.resize(n);
            for (index_t i = 0; i < n; ++i) {
                composite_scaling_[i] = scaling_[composite_perm_[i]];
            }
        }
    }

    // ================================================================
    // Apply implementations
    // ================================================================

    /**
     * @brief Apply with level scheduling (original path)
     *
     * Uses pre-allocated work_temp_ and work_temp2_ to avoid
     * per-call heap allocation.
     */
    void apply_level_schedule(const Scalar* x, Scalar* y, index_t size) const {
        const bool use_persistent = get_num_threads() > 1;

        if (config_.diagonal_scaling && !scaling_.empty()) {
            for (index_t i = 0; i < size; ++i) {
                work_temp_[i] = x[i] * static_cast<Scalar>(scaling_[i]);
            }
            if (use_persistent) {
                forward_substitution_persistent(work_temp_.data(), work_temp2_.data());
                backward_substitution_persistent(work_temp2_.data(), work_temp_.data());
            } else {
                forward_substitution(work_temp_.data(), work_temp2_.data());
                backward_substitution(work_temp2_.data(), work_temp_.data());
            }
            for (index_t i = 0; i < size; ++i) {
                y[i] = work_temp_[i] * static_cast<Scalar>(scaling_[i]);
            }
        } else {
            if (use_persistent) {
                forward_substitution_persistent(x, work_temp_.data());
                backward_substitution_persistent(work_temp_.data(), y);
            } else {
                forward_substitution(x, work_temp_.data());
                backward_substitution(work_temp_.data(), y);
            }
        }
    }

    /**
     * @brief Apply with ABMC ordering (handles both ABMC-only and RCM+ABMC)
     *
     * Uses pre-computed composite_perm_ to permute input/output in a single
     * level of indirection, regardless of whether RCM is used.
     */
    void apply_abmc(const Scalar* x, Scalar* y, index_t size) const {
        const auto& perm = composite_perm_;

        if (config_.diagonal_scaling && !composite_scaling_.empty()) {
            parallel_for(size, [&](index_t i) {
                abmc_x_perm_[perm[i]] = x[i] * static_cast<Scalar>(composite_scaling_[i]);
            });
            forward_substitution_abmc(abmc_x_perm_.data(), work_temp_.data());
            backward_substitution_abmc(work_temp_.data(), abmc_y_perm_.data());
            parallel_for(size, [&](index_t i) {
                y[i] = abmc_y_perm_[perm[i]] * static_cast<Scalar>(composite_scaling_[i]);
            });
        } else {
            parallel_for(size, [&](index_t i) {
                abmc_x_perm_[perm[i]] = x[i];
            });
            forward_substitution_abmc(abmc_x_perm_.data(), work_temp_.data());
            backward_substitution_abmc(work_temp_.data(), abmc_y_perm_.data());
            parallel_for(size, [&](index_t i) {
                y[i] = abmc_y_perm_[perm[i]];
            });
        }
    }

    // ================================================================
    // Matrix reordering
    // ================================================================

    /**
     * @brief Reorder matrix A using an arbitrary permutation
     *
     * Creates output CSR where entry A[i][j] maps to
     * A_reordered[perm[i]][perm[j]].
     */
    static void reorder_matrix_with_perm(
        const SparseMatrixView<Scalar>& A,
        const std::vector<index_t>& perm,
        SparseMatrixCSR<Scalar>& out)
    {
        const index_t n = A.rows();
        out.rows = out.cols = n;

        // Count nnz per reordered row
        std::vector<index_t> counts(n, 0);
        for (index_t i = 0; i < n; ++i) {
            auto [start, end] = A.row_range(i);
            counts[perm[i]] += (end - start);
        }

        // Build row pointers
        out.row_ptr.resize(n + 1);
        out.row_ptr[0] = 0;
        for (index_t i = 0; i < n; ++i) {
            out.row_ptr[i + 1] = out.row_ptr[i] + counts[i];
        }

        // Fill values
        const index_t total_nnz = out.row_ptr[n];
        out.col_idx.resize(total_nnz);
        out.values.resize(total_nnz);

        std::vector<index_t> pos(n);
        for (index_t i = 0; i < n; ++i) pos[i] = out.row_ptr[i];

        for (index_t i = 0; i < n; ++i) {
            index_t new_i = perm[i];
            auto [start, end] = A.row_range(i);
            for (index_t k = start; k < end; ++k) {
                index_t p = pos[new_i]++;
                out.col_idx[p] = perm[A.col_idx()[k]];
                out.values[p] = A.values()[k];
            }
        }

        // Sort each row by column index
        for (index_t i = 0; i < n; ++i) {
            index_t row_start = out.row_ptr[i];
            index_t row_end = out.row_ptr[i + 1];
            index_t row_nnz = row_end - row_start;
            if (row_nnz <= 1) continue;

            std::vector<index_t> indices(row_nnz);
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](index_t a, index_t b) {
                return out.col_idx[row_start + a] < out.col_idx[row_start + b];
            });

            std::vector<index_t> sorted_cols(row_nnz);
            std::vector<Scalar> sorted_vals(row_nnz);
            for (index_t k = 0; k < row_nnz; ++k) {
                sorted_cols[k] = out.col_idx[row_start + indices[k]];
                sorted_vals[k] = out.values[row_start + indices[k]];
            }
            for (index_t k = 0; k < row_nnz; ++k) {
                out.col_idx[row_start + k] = sorted_cols[k];
                out.values[row_start + k] = sorted_vals[k];
            }
        }
    }

    /**
     * @brief Reorder full matrix A according to ABMC permutation
     *
     * Delegates to reorder_matrix_with_perm() using the ABMC ordering.
     */
    void reorder_matrix(const SparseMatrixView<Scalar>& A) {
        reorder_matrix_with_perm(A, abmc_schedule_.ordering, reordered_csr_);
    }

    // ================================================================
    // ABMC triangular solves
    // ================================================================

    /**
     * @brief Forward substitution with ABMC ordering: solve L * y = x
     *
     * Colors processed sequentially, blocks within each color in parallel,
     * rows within each block sequentially.
     */
    void forward_substitution_abmc(const Scalar* x, Scalar* y) const {
        const index_t nc = abmc_schedule_.num_colors();
        for (index_t c = 0; c < nc; ++c) {
            const index_t blk_begin = abmc_schedule_.color_offsets[c];
            const index_t blk_end = abmc_schedule_.color_offsets[c + 1];
            const index_t num_blocks = blk_end - blk_begin;
            parallel_for(num_blocks, [&](index_t bidx) {
                const index_t blk = abmc_schedule_.color_blocks[blk_begin + bidx];
                const index_t row_begin = abmc_schedule_.block_offsets[blk];
                const index_t row_end = abmc_schedule_.block_offsets[blk + 1];

                for (index_t i = row_begin; i < row_end; ++i) {
                    Scalar s = x[i];
                    const index_t l_start = L_.row_ptr[i];
                    const index_t l_end = L_.row_ptr[i + 1] - 1; // Exclude diagonal

                    for (index_t k = l_start; k < l_end; ++k) {
                        s -= L_.values[k] * y[L_.col_idx[k]];
                    }

                    y[i] = s / L_.values[l_end];
                }
            });
        }
    }

    /**
     * @brief Backward substitution with ABMC ordering: solve L^T * y = D^{-1} * x
     *
     * Colors processed in reverse order, blocks within each color in parallel,
     * rows within each block in reverse order.
     */
    void backward_substitution_abmc(const Scalar* x, Scalar* y) const {
        const index_t nc = abmc_schedule_.num_colors();
        for (index_t c = nc; c-- > 0;) {
            const index_t blk_begin = abmc_schedule_.color_offsets[c];
            const index_t blk_end = abmc_schedule_.color_offsets[c + 1];
            const index_t num_blocks = blk_end - blk_begin;
            parallel_for(num_blocks, [&](index_t bidx) {
                const index_t blk = abmc_schedule_.color_blocks[blk_begin + bidx];
                const index_t row_begin = abmc_schedule_.block_offsets[blk];
                const index_t row_end = abmc_schedule_.block_offsets[blk + 1];

                for (index_t i = row_end; i-- > row_begin;) {
                    Scalar s = Scalar(0);
                    const index_t lt_start = Lt_.row_ptr[i] + 1; // Skip diagonal
                    const index_t lt_end = Lt_.row_ptr[i + 1];

                    for (index_t k = lt_start; k < lt_end; ++k) {
                        s -= Lt_.values[k] * y[Lt_.col_idx[k]];
                    }

                    y[i] = s * inv_diag_[i] + x[i];
                }
            });
        }
    }

    // ================================================================
    // Standard methods
    // ================================================================

    /**
     * @brief Extract lower triangular part from matrix A
     */
    void extract_lower_triangular(const SparseMatrixView<Scalar>& A) {
        const index_t n = A.rows();
        L_.rows = L_.cols = n;
        L_.row_ptr.resize(n + 1);

        // First pass: count non-zeros per row in lower triangle
        std::vector<index_t> counts(n);
        parallel_for(n, [&](index_t i) {
            auto [start, end] = A.row_range(i);
            index_t count = 0;
            for (index_t k = start; k < end; ++k) {
                if (A.col_idx()[k] <= i) {
                    ++count;
                }
            }
            counts[i] = count;
        });

        // Prefix sum for row pointers (sequential)
        L_.row_ptr[0] = 0;
        for (index_t i = 0; i < n; ++i) {
            L_.row_ptr[i + 1] = L_.row_ptr[i] + counts[i];
        }

        // Second pass: copy values
        L_.col_idx.resize(L_.row_ptr[n]);
        L_.values.resize(L_.row_ptr[n]);

        parallel_for(n, [&](index_t i) {
            auto [start, end] = A.row_range(i);
            index_t pos = L_.row_ptr[i];
            for (index_t k = start; k < end; ++k) {
                index_t j = A.col_idx()[k];
                if (j <= i) {
                    L_.col_idx[pos] = j;
                    L_.values[pos] = A.values()[k];
                    ++pos;
                }
            }
        });
    }

    /**
     * @brief Compute diagonal scaling factors: scaling[i] = 1/sqrt(A[i,i])
     */
    void compute_scaling_factors() {
        const index_t n = size_;
        scaling_.resize(n);

        for (index_t i = 0; i < n; ++i) {
            // Find diagonal element in L_
            const index_t diag_pos = L_.row_ptr[i + 1] - 1;
            if (L_.col_idx[diag_pos] == i) {
                double diag_val = std::abs(L_.values[diag_pos]);
                scaling_[i] = (diag_val > 0.0) ? 1.0 / std::sqrt(diag_val) : 1.0;
            } else {
                scaling_[i] = 1.0;
            }
        }
    }

    /**
     * @brief Apply diagonal scaling to L_ values: L[i,j] *= scaling[i] * scaling[j]
     */
    void apply_scaling_to_L() {
        const index_t n = size_;
        for (index_t i = 0; i < n; ++i) {
            for (index_t k = L_.row_ptr[i]; k < L_.row_ptr[i + 1]; ++k) {
                index_t j = L_.col_idx[k];
                L_.values[k] = original_values_[k]
                    * static_cast<Scalar>(scaling_[i])
                    * static_cast<Scalar>(scaling_[j]);
            }
        }
    }

    /**
     * @brief Compute IC factorization in-place on L_
     *
     * Modifies L_ to contain the IC factor and fills inv_diag_ with D^{-1}
     *
     * When auto_shift is enabled, automatically increases the shift parameter
     * when the factorized diagonal becomes too small (below min_diagonal_threshold).
     * This restarts the factorization with the new shift until successful or
     * max_shift_trials is exceeded.
     */
    void compute_ic_factorization() {
        const index_t n = size_;
        inv_diag_.resize(n);

        double shift = shift_parameter_;
        actual_shift_ = shift;

        bool restart = true;
        int shift_trials = 0;

        while (restart) {
            restart = false;

            if (shift_trials > 0) {
                // Restart: restore original values
                L_.values = original_values_;
                if (config_.diagonal_scaling) {
                    apply_scaling_to_L();
                }
            } else if (config_.diagonal_scaling) {
                // First pass with scaling: apply scaling to original values
                apply_scaling_to_L();
            }

            // For each row i
            for (index_t i = 0; i < n; ++i) {
                const index_t row_start = L_.row_ptr[i];
                const index_t row_end = L_.row_ptr[i + 1];

                // Process off-diagonal elements L(i, j) where j < i
                for (index_t kk = row_start; kk < row_end - 1; ++kk) {
                    const index_t j = L_.col_idx[kk];
                    if (j >= i) break;

                    Scalar s = L_.values[kk];

                    // s -= sum over k < j of: L(i,k) * L(j,k) * D^{-1}(k)
                    const index_t j_start = L_.row_ptr[j];
                    const index_t j_end = L_.row_ptr[j + 1];

                    for (index_t ii = row_start; ii < kk; ++ii) {
                        const index_t k = L_.col_idx[ii];
                        if (k >= j) break;

                        // Find L(j, k) in row j
                        for (index_t jj = j_start; jj < j_end; ++jj) {
                            if (L_.col_idx[jj] == k) {
                                s -= L_.values[ii] * L_.values[jj] * inv_diag_[k];
                                break;
                            } else if (L_.col_idx[jj] > k) {
                                break;
                            }
                        }
                    }

                    L_.values[kk] = s;
                }

                // Process diagonal element L(i, i)
                const index_t diag_pos = row_end - 1;
                if (L_.col_idx[diag_pos] != i) {
                    throw std::runtime_error(
                        "IC decomposition failed: missing diagonal element at row "
                        + std::to_string(i));
                }

                // Get the original (possibly scaled) diagonal value
                Scalar orig_diag = L_.values[diag_pos];

                // Apply shift to diagonal
                Scalar s = orig_diag * static_cast<Scalar>(shift);

                // s -= sum over k < i of: L(i,k)^2 * D^{-1}(k)
                for (index_t kk = row_start; kk < diag_pos; ++kk) {
                    const index_t k = L_.col_idx[kk];
                    if (k >= i) break;
                    s -= L_.values[kk] * L_.values[kk] * inv_diag_[k];
                }

                L_.values[diag_pos] = s;

                // Auto-shift check: if diagonal is too small and original was positive
                double abs_s = std::abs(s);
                if (config_.auto_shift) {
                    double abs_orig = std::abs(orig_diag);

                    if (abs_s < config_.min_diagonal_threshold && abs_orig > 0.0) {
                        if (shift < config_.max_shift_value &&
                            shift_trials < config_.max_shift_trials) {
                            shift += config_.shift_increment;
                            shift_trials++;
                            restart = true;
                            break; // Restart factorization with new shift
                        }
                    }
                }

                // Store D^{-1}(i) = 1/s
                if (abs_s > config_.zero_diagonal_replacement) {
                    inv_diag_[i] = Scalar(1) / s;
                } else {
                    // Zero or negative diagonal: clamp to safe value
                    Scalar safe_val = static_cast<Scalar>(config_.zero_diagonal_replacement);
                    L_.values[diag_pos] = safe_val;
                    inv_diag_[i] = Scalar(1) / safe_val;
                }
            }
        }

        actual_shift_ = shift;
    }

    /// Compute L^T (transpose of L)
    void compute_transpose() {
        Lt_ = L_.transpose();
    }

    /**
     * @brief Forward substitution: solve L * y = x
     *
     * Uses level scheduling for parallelism: rows at the same dependency
     * level are independent and processed via parallel_for.
     */
    void forward_substitution(const Scalar* x, Scalar* y) const {
        for (const auto& level : fwd_schedule_.levels) {
            const index_t level_size = static_cast<index_t>(level.size());
            parallel_for(level_size, [&](index_t idx) {
                const index_t i = level[idx];
                Scalar s = x[i];
                const index_t row_start = L_.row_ptr[i];
                const index_t row_end = L_.row_ptr[i + 1] - 1; // Exclude diagonal

                for (index_t k = row_start; k < row_end; ++k) {
                    s -= L_.values[k] * y[L_.col_idx[k]];
                }

                // Divide by diagonal element
                y[i] = s / L_.values[row_end];
            });
        }
    }

    /**
     * @brief Backward substitution: solve L^T * y = D^{-1} * x
     *
     * Uses level scheduling for parallelism.
     */
    void backward_substitution(const Scalar* x, Scalar* y) const {
        for (const auto& level : bwd_schedule_.levels) {
            const index_t level_size = static_cast<index_t>(level.size());
            parallel_for(level_size, [&](index_t idx) {
                const index_t i = level[idx];
                Scalar s = Scalar(0);
                const index_t row_start = Lt_.row_ptr[i] + 1; // Skip diagonal
                const index_t row_end = Lt_.row_ptr[i + 1];

                for (index_t k = row_start; k < row_end; ++k) {
                    s -= Lt_.values[k] * y[Lt_.col_idx[k]];
                }

                // Apply D^{-1} and add contribution from x
                y[i] = s * inv_diag_[i] + x[i];
            });
        }
    }

    /**
     * @brief Forward substitution with persistent parallel region
     *
     * Uses a single parallel_for(nthreads, ...) dispatch for all levels,
     * with spin-barrier synchronization between levels. This eliminates
     * the overhead of ~700 separate parallel_for dispatches per call.
     */
    void forward_substitution_persistent(const Scalar* x, Scalar* y) const {
        const int nthreads = get_num_threads();
        const int num_levels = fwd_schedule_.num_levels();
        SpinBarrier barrier(nthreads);

        parallel_for(static_cast<index_t>(nthreads), [&](index_t thread_id) {
            for (int lev = 0; lev < num_levels; ++lev) {
                const auto& level = fwd_schedule_.levels[lev];
                const index_t level_size = static_cast<index_t>(level.size());
                const index_t my_start = level_size * thread_id / nthreads;
                const index_t my_end = level_size * (thread_id + 1) / nthreads;

                for (index_t idx = my_start; idx < my_end; ++idx) {
                    const index_t i = level[idx];
                    Scalar s = x[i];
                    const index_t row_start = L_.row_ptr[i];
                    const index_t row_end = L_.row_ptr[i + 1] - 1;

                    for (index_t k = row_start; k < row_end; ++k) {
                        s -= L_.values[k] * y[L_.col_idx[k]];
                    }

                    y[i] = s / L_.values[row_end];
                }
                barrier.wait();
            }
        });
    }

    /**
     * @brief Backward substitution with persistent parallel region
     *
     * Uses a single parallel_for(nthreads, ...) dispatch for all levels,
     * with spin-barrier synchronization between levels.
     */
    void backward_substitution_persistent(const Scalar* x, Scalar* y) const {
        const int nthreads = get_num_threads();
        const int num_levels = bwd_schedule_.num_levels();
        SpinBarrier barrier(nthreads);

        parallel_for(static_cast<index_t>(nthreads), [&](index_t thread_id) {
            for (int lev = 0; lev < num_levels; ++lev) {
                const auto& level = bwd_schedule_.levels[lev];
                const index_t level_size = static_cast<index_t>(level.size());
                const index_t my_start = level_size * thread_id / nthreads;
                const index_t my_end = level_size * (thread_id + 1) / nthreads;

                for (index_t idx = my_start; idx < my_end; ++idx) {
                    const index_t i = level[idx];
                    Scalar s = Scalar(0);
                    const index_t row_start = Lt_.row_ptr[i] + 1;
                    const index_t row_end = Lt_.row_ptr[i + 1];

                    for (index_t k = row_start; k < row_end; ++k) {
                        s -= Lt_.values[k] * y[Lt_.col_idx[k]];
                    }

                    y[i] = s * inv_diag_[i] + x[i];
                }
                barrier.wait();
            }
        });
    }
};

/**
 * @brief Adapter that wraps ICPreconditioner for use in reordered space
 *
 * When CG operates in ABMC-reordered space, this adapter is passed as
 * the preconditioner. Its apply() delegates to
 * ICPreconditioner::apply_in_reordered_space() which skips permutations.
 */
template<typename Scalar>
class ICPrecondReorderedAdapter : public Preconditioner<Scalar> {
public:
    explicit ICPrecondReorderedAdapter(const ICPreconditioner<Scalar>& precond)
        : precond_(precond)
    {
        this->is_setup_ = true;
    }

    void setup(const SparseMatrixView<Scalar>& /*A*/) override {
        this->is_setup_ = true;
    }

    void apply(const Scalar* x, Scalar* y, index_t size) const override {
        precond_.apply_in_reordered_space(x, y, size);
    }

    std::string name() const override { return "IC-Reordered"; }

private:
    const ICPreconditioner<Scalar>& precond_;
};

/**
 * @brief Adapter for RCM+ABMC split mode
 *
 * When CG operates in RCM-reordered space, this adapter delegates to
 * ICPreconditioner::apply_rcm_abmc() which handles the RCM->ABMC
 * permutation internally for triangular solves.
 */
template<typename Scalar>
class ICPrecondRCMABMCAdapter : public Preconditioner<Scalar> {
public:
    explicit ICPrecondRCMABMCAdapter(const ICPreconditioner<Scalar>& precond)
        : precond_(precond)
    {
        this->is_setup_ = true;
    }

    void setup(const SparseMatrixView<Scalar>& /*A*/) override {
        this->is_setup_ = true;
    }

    void apply(const Scalar* x, Scalar* y, index_t size) const override {
        precond_.apply_rcm_abmc(x, y, size);
    }

    std::string name() const override { return "IC-RCM-ABMC"; }

private:
    const ICPreconditioner<Scalar>& precond_;
};

} // namespace sparsesolv

#endif // SPARSESOLV_PRECONDITIONERS_IC_PRECONDITIONER_HPP
