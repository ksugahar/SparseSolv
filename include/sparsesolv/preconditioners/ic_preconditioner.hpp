/**
 * @file ic_preconditioner.hpp
 * @brief Incomplete Cholesky (IC) preconditioner
 */

#ifndef SPARSESOLV_PRECONDITIONERS_IC_PRECONDITIONER_HPP
#define SPARSESOLV_PRECONDITIONERS_IC_PRECONDITIONER_HPP

#include "../core/preconditioner.hpp"
#include "../core/solver_config.hpp"
#include "../core/level_schedule.hpp"
#include "../core/abmc_ordering.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

namespace sparsesolv {

/**
 * @brief Sparse matrix storage for L factor (CSR format, owned)
 *
 * Internal class for storing the incomplete Cholesky factor L.
 */
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
};

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

        if (config_.use_abmc) {
            // ABMC path: reorder matrix, then extract and factorize

            // 1. Build ABMC schedule from full matrix pattern
            abmc_schedule_.build(A.row_ptr(), A.col_idx(), n,
                                 config_.abmc_block_size, config_.abmc_num_colors);

            // 2. Reorder the full matrix according to ABMC permutation
            reorder_matrix(A);

            // 3. Extract lower triangular from the reordered matrix
            SparseMatrixView<Scalar> A_reordered(
                reordered_csr_.rows, reordered_csr_.cols,
                reordered_csr_.row_ptr.data(),
                reordered_csr_.col_idx.data(),
                reordered_csr_.values.data());
            extract_lower_triangular(A_reordered);

            // Keep reordered full matrix for CG in reordered space
            // (used by solve_iccg() to run CG entirely in reordered space)
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

        if (config_.use_abmc) {
            // Pre-allocate work vectors for ABMC apply
            abmc_x_perm_.resize(n);
            abmc_y_perm_.resize(n);
            abmc_temp_.resize(n);
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
    bool has_reordered_matrix() const {
        return config_.use_abmc && reordered_csr_.rows > 0;
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

    /// Get the ABMC reverse ordering (reverse_ordering[new_row] = old_row)
    const std::vector<index_t>& abmc_reverse_ordering() const {
        return abmc_schedule_.reverse_ordering;
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
            // scaling_ was computed from L_ which is in reordered space
            for (index_t i = 0; i < size; ++i) {
                abmc_x_perm_[i] = x[i] * static_cast<Scalar>(scaling_[i]);
            }
            forward_substitution_abmc(abmc_x_perm_.data(), abmc_temp_.data());
            backward_substitution_abmc(abmc_temp_.data(), abmc_x_perm_.data());
            for (index_t i = 0; i < size; ++i) {
                y[i] = abmc_x_perm_[i] * static_cast<Scalar>(scaling_[i]);
            }
        } else {
            forward_substitution_abmc(x, abmc_temp_.data());
            backward_substitution_abmc(abmc_temp_.data(), y);
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

    // ABMC ordering support
    ABMCSchedule abmc_schedule_;             // Block-color schedule
    SparseMatrixCSR<Scalar> reordered_csr_;  // Temporary: reordered full matrix
    mutable std::vector<Scalar> abmc_x_perm_;   // Work vector: permuted input
    mutable std::vector<Scalar> abmc_y_perm_;   // Work vector: permuted output
    mutable std::vector<Scalar> abmc_temp_;     // Work vector: intermediate

    // ================================================================
    // Apply implementations
    // ================================================================

    /**
     * @brief Apply with level scheduling (original path)
     */
    void apply_level_schedule(const Scalar* x, Scalar* y, index_t size) const {
        std::vector<Scalar> temp(size);

        if (config_.diagonal_scaling && !scaling_.empty()) {
            for (index_t i = 0; i < size; ++i) {
                temp[i] = x[i] * static_cast<Scalar>(scaling_[i]);
            }
            std::vector<Scalar> temp2(size);
            forward_substitution(temp.data(), temp2.data());
            backward_substitution(temp2.data(), temp.data());
            for (index_t i = 0; i < size; ++i) {
                y[i] = temp[i] * static_cast<Scalar>(scaling_[i]);
            }
        } else {
            forward_substitution(x, temp.data());
            backward_substitution(temp.data(), y);
        }
    }

    /**
     * @brief Apply with ABMC ordering
     *
     * Permutes input to reordered space, performs ABMC-parallel
     * forward/backward substitution, then permutes output back.
     */
    void apply_abmc(const Scalar* x, Scalar* y, index_t size) const {
        const auto& ord = abmc_schedule_.ordering;
        const auto& rev = abmc_schedule_.reverse_ordering;

        if (config_.diagonal_scaling && !scaling_.empty()) {
            // Scale and permute input: x_perm[ordering[i]] = scaling[i] * x[i]
            for (index_t i = 0; i < size; ++i) {
                abmc_x_perm_[ord[i]] = x[i] * static_cast<Scalar>(scaling_[rev[ord[i]]]);
            }

            // Forward substitution in reordered space
            forward_substitution_abmc(abmc_x_perm_.data(), abmc_temp_.data());

            // Backward substitution in reordered space
            backward_substitution_abmc(abmc_temp_.data(), abmc_y_perm_.data());

            // Reverse permute and scale output: y[i] = scaling[i] * y_perm[ordering[i]]
            for (index_t i = 0; i < size; ++i) {
                y[i] = abmc_y_perm_[ord[i]] * static_cast<Scalar>(scaling_[rev[ord[i]]]);
            }
        } else {
            // Permute input: x_perm[ordering[i]] = x[i]
            for (index_t i = 0; i < size; ++i) {
                abmc_x_perm_[ord[i]] = x[i];
            }

            // Forward substitution in reordered space
            forward_substitution_abmc(abmc_x_perm_.data(), abmc_temp_.data());

            // Backward substitution in reordered space
            backward_substitution_abmc(abmc_temp_.data(), abmc_y_perm_.data());

            // Reverse permute output: y[i] = y_perm[ordering[i]]
            for (index_t i = 0; i < size; ++i) {
                y[i] = abmc_y_perm_[ord[i]];
            }
        }
    }

    // ================================================================
    // Matrix reordering
    // ================================================================

    /**
     * @brief Reorder full matrix A according to ABMC permutation
     *
     * Creates reordered_csr_ where entry A[i][j] maps to
     * A_reordered[ordering[i]][ordering[j]].
     */
    void reorder_matrix(const SparseMatrixView<Scalar>& A) {
        const index_t n = A.rows();
        const auto& ord = abmc_schedule_.ordering;

        reordered_csr_.rows = reordered_csr_.cols = n;

        // First pass: count nnz per reordered row
        std::vector<index_t> counts(n, 0);
        for (index_t i = 0; i < n; ++i) {
            auto [start, end] = A.row_range(i);
            counts[ord[i]] += (end - start);
        }

        // Build row pointers
        reordered_csr_.row_ptr.resize(n + 1);
        reordered_csr_.row_ptr[0] = 0;
        for (index_t i = 0; i < n; ++i) {
            reordered_csr_.row_ptr[i + 1] = reordered_csr_.row_ptr[i] + counts[i];
        }

        // Second pass: fill column indices and values (unsorted within each row)
        const index_t total_nnz = reordered_csr_.row_ptr[n];
        reordered_csr_.col_idx.resize(total_nnz);
        reordered_csr_.values.resize(total_nnz);

        std::vector<index_t> pos(n);
        for (index_t i = 0; i < n; ++i) pos[i] = reordered_csr_.row_ptr[i];

        for (index_t i = 0; i < n; ++i) {
            index_t new_i = ord[i];
            auto [start, end] = A.row_range(i);
            for (index_t k = start; k < end; ++k) {
                index_t p = pos[new_i]++;
                reordered_csr_.col_idx[p] = ord[A.col_idx()[k]];
                reordered_csr_.values[p] = A.values()[k];
            }
        }

        // Sort each row by column index (required for CSR invariant)
        for (index_t i = 0; i < n; ++i) {
            index_t row_start = reordered_csr_.row_ptr[i];
            index_t row_end = reordered_csr_.row_ptr[i + 1];
            index_t row_nnz = row_end - row_start;

            if (row_nnz <= 1) continue;

            // Build index array and sort
            std::vector<index_t> indices(row_nnz);
            for (index_t k = 0; k < row_nnz; ++k) indices[k] = k;
            std::sort(indices.begin(), indices.end(), [&](index_t a, index_t b) {
                return reordered_csr_.col_idx[row_start + a] <
                       reordered_csr_.col_idx[row_start + b];
            });

            // Apply permutation
            std::vector<index_t> sorted_cols(row_nnz);
            std::vector<Scalar> sorted_vals(row_nnz);
            for (index_t k = 0; k < row_nnz; ++k) {
                sorted_cols[k] = reordered_csr_.col_idx[row_start + indices[k]];
                sorted_vals[k] = reordered_csr_.values[row_start + indices[k]];
            }
            for (index_t k = 0; k < row_nnz; ++k) {
                reordered_csr_.col_idx[row_start + k] = sorted_cols[k];
                reordered_csr_.values[row_start + k] = sorted_vals[k];
            }
        }
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
        for (const auto& blocks_in_color : abmc_schedule_.color_list) {
            const index_t num_blocks = static_cast<index_t>(blocks_in_color.size());
            parallel_for(num_blocks, [&](index_t bidx) {
                const index_t blk = blocks_in_color[bidx];
                const auto& rows = abmc_schedule_.block_list[blk];

                for (index_t ridx = 0; ridx < static_cast<index_t>(rows.size()); ++ridx) {
                    const index_t i = rows[ridx];
                    Scalar s = x[i];
                    const index_t row_start = L_.row_ptr[i];
                    const index_t row_end = L_.row_ptr[i + 1] - 1; // Exclude diagonal

                    for (index_t k = row_start; k < row_end; ++k) {
                        s -= L_.values[k] * y[L_.col_idx[k]];
                    }

                    // Divide by diagonal element (last entry in row)
                    y[i] = s / L_.values[row_end];
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
        const index_t num_colors = static_cast<index_t>(abmc_schedule_.color_list.size());
        for (index_t c = num_colors; c-- > 0;) {
            const auto& blocks_in_color = abmc_schedule_.color_list[c];
            const index_t num_blocks = static_cast<index_t>(blocks_in_color.size());
            parallel_for(num_blocks, [&](index_t bidx) {
                const index_t blk = blocks_in_color[bidx];
                const auto& rows = abmc_schedule_.block_list[blk];

                for (index_t ridx = static_cast<index_t>(rows.size()); ridx-- > 0;) {
                    const index_t i = rows[ridx];
                    Scalar s = Scalar(0);
                    const index_t row_start = Lt_.row_ptr[i] + 1; // Skip diagonal
                    const index_t row_end = Lt_.row_ptr[i + 1];

                    for (index_t k = row_start; k < row_end; ++k) {
                        s -= Lt_.values[k] * y[Lt_.col_idx[k]];
                    }

                    // Apply D^{-1} and add contribution from x
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

    /**
     * @brief Compute L^T (transpose of L)
     */
    void compute_transpose() {
        const index_t n = size_;
        const index_t nnz = L_.nnz();
        Lt_.rows = Lt_.cols = n;
        Lt_.row_ptr.assign(n + 1, 0);
        Lt_.col_idx.resize(nnz);
        Lt_.values.resize(nnz);

        // Count entries per column of L (= per row of L^T)
        for (index_t k = 0; k < nnz; ++k) {
            Lt_.row_ptr[L_.col_idx[k] + 1]++;
        }

        // Cumulative sum to get row pointers (sequential - short loop)
        for (index_t i = 0; i < n; ++i) {
            Lt_.row_ptr[i + 1] += Lt_.row_ptr[i];
        }

        // Fill in values using atomic counter increments
        std::vector<index_t> counter(n, 0);
        for (index_t i = 0; i < n; ++i) {
            for (index_t k = L_.row_ptr[i]; k < L_.row_ptr[i + 1]; ++k) {
                index_t j = L_.col_idx[k];
                index_t pos = Lt_.row_ptr[j] + counter[j];
                Lt_.col_idx[pos] = i;
                Lt_.values[pos] = L_.values[k];
                counter[j]++;
            }
        }
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
};

// Type aliases
using ICPreconditionerD = ICPreconditioner<double>;
using ICPreconditionerC = ICPreconditioner<complex_t>;

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

} // namespace sparsesolv

#endif // SPARSESOLV_PRECONDITIONERS_IC_PRECONDITIONER_HPP
