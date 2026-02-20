/**
 * @file solver_config.hpp
 * @brief Unified configuration for all SparseSolv solvers
 */

#ifndef SPARSESOLV_CORE_SOLVER_CONFIG_HPP
#define SPARSESOLV_CORE_SOLVER_CONFIG_HPP

#include "types.hpp"

namespace sparsesolv {

/**
 * @brief Configuration structure for iterative solvers
 *
 * This struct centralizes all solver parameters that were previously
 * scattered across multiple setter methods. Use this to configure
 * tolerance, iteration limits, preconditioning parameters, and
 * convergence behavior.
 *
 * Example:
 * @code
 * SolverConfig config;
 * config.tolerance = 1e-10;
 * config.max_iterations = 1000;
 * config.shift_parameter = 1.05;  // for IC decomposition
 * @endcode
 */
struct SolverConfig {
    //--------------------------------------------------
    // Convergence criteria
    //--------------------------------------------------

    /// Relative tolerance for convergence (default: 1e-10)
    double tolerance = 1e-10;

    /// Maximum number of iterations (default: 1000)
    int max_iterations = 1000;

    /// Absolute tolerance threshold (solver stops if ||r|| < abs_tolerance)
    double abs_tolerance = 1e-12;

    //--------------------------------------------------
    // Normalization for convergence check
    //--------------------------------------------------

    /// How to normalize the residual for convergence check
    NormType norm_type = NormType::RHS;

    /// Custom normalization value (used when norm_type == Custom)
    double custom_norm = 1.0;

    //--------------------------------------------------
    // Preconditioning parameters
    //--------------------------------------------------

    /// Shift parameter for incomplete Cholesky decomposition
    /// Values > 1.0 improve stability, typical: 1.0-1.2
    /// (Previously called "accera" or acceleration factor)
    double shift_parameter = 1.05;

    /// Enable diagonal scaling (1/sqrt(A[i,i])) before IC factorization
    bool diagonal_scaling = false;

    //--------------------------------------------------
    // Auto-shift parameters for IC decomposition
    //--------------------------------------------------

    /// Enable automatic shift adjustment when IC factorization encounters
    /// small or negative diagonal entries
    bool auto_shift = false;

    /// Increment for automatic shift adjustment (default: 0.01)
    double shift_increment = 0.01;

    /// Maximum allowed shift value during auto-adjustment (default: 5.0)
    double max_shift_value = 5.0;

    /// Threshold below which diagonal is considered too small (triggers shift increase)
    double min_diagonal_threshold = 1e-6;

    /// Replacement value for zero or negative diagonals (default: 1e-10)
    double zero_diagonal_replacement = 1e-10;

    /// Maximum number of shift adjustment trials (default: 100)
    int max_shift_trials = 100;

    //--------------------------------------------------
    // Divergence detection
    //--------------------------------------------------

    /// Strategy for detecting divergence
    DivergenceCheck divergence_check = DivergenceCheck::None;

    /// Multiplier for divergence detection (residual > best * this value triggers count)
    double divergence_threshold = 1000.0;

    /// Number of consecutive iterations above threshold before declaring divergence
    int divergence_count = 100;

    //--------------------------------------------------
    // Result saving options
    //--------------------------------------------------

    /// Save the best result encountered during iteration (useful for non-converging solves)
    bool save_best_result = true;

    /// Save residual history for analysis/debugging
    bool save_residual_history = false;

    //--------------------------------------------------
    // Complex inner product
    //--------------------------------------------------

    /// Use conjugated inner product (a^H * b) for Hermitian systems.
    /// Default false uses unconjugated (a^T * b) for complex-symmetric systems.
    /// Has no effect for real-valued problems.
    bool conjugate = false;

    //--------------------------------------------------
    // Parallel options
    //--------------------------------------------------

    /// Number of threads for parallel operations (0 = auto)
    int num_threads = 0;

    //--------------------------------------------------
    // ABMC ordering parameters (for parallel triangular solves)
    //--------------------------------------------------

    /// Enable ABMC (Algebraic Block Multi-Color) ordering.
    /// When enabled, the preconditioner reorders the matrix to enable
    /// parallel triangular solves using a two-level hierarchy:
    /// colors (sequential) -> blocks (parallel) -> rows (sequential).
    bool use_abmc = false;

    /// Number of rows per block for ABMC ordering (block size).
    /// Larger blocks reduce parallelism overhead but may decrease
    /// the degree of parallelism. Typical values: 2-16.
    int abmc_block_size = 4;

    /// Number of colors for ABMC graph coloring.
    /// More colors allow finer-grained parallelism but increase
    /// the number of sequential synchronization points.
    /// The actual number may be increased if the graph requires it.
    int abmc_num_colors = 4;

    /// When true, CG runs entirely in ABMC-reordered space (SpMV uses
    /// reordered matrix). When false (default), CG uses the original
    /// matrix for SpMV and only the preconditioner operates in
    /// reordered space. False is usually faster because it preserves
    /// the FEM mesh ordering cache locality for SpMV.
    bool abmc_reorder_spmv = false;

    /// Enable RCM (Reverse Cuthill-McKee) preprocessing before ABMC.
    /// RCM reduces bandwidth, improving cache locality for both SpMV
    /// and triangular solves. ABMC then operates on the bandwidth-reduced
    /// matrix.
    bool abmc_use_rcm = false;

    //--------------------------------------------------
    // Builder pattern for convenient configuration
    //--------------------------------------------------

    SolverConfig& with_tolerance(double tol) {
        tolerance = tol;
        return *this;
    }

    SolverConfig& with_max_iterations(int max_iter) {
        max_iterations = max_iter;
        return *this;
    }

    SolverConfig& with_shift(double shift) {
        shift_parameter = shift;
        return *this;
    }

    SolverConfig& with_diagonal_scaling(bool enable = true) {
        diagonal_scaling = enable;
        return *this;
    }

    SolverConfig& with_residual_history(bool enable = true) {
        save_residual_history = enable;
        return *this;
    }

    SolverConfig& with_divergence_check(DivergenceCheck check, double threshold = 1000.0, int count = 100) {
        divergence_check = check;
        divergence_threshold = threshold;
        divergence_count = count;
        return *this;
    }

    SolverConfig& with_auto_shift(bool enable = true) {
        auto_shift = enable;
        return *this;
    }

    SolverConfig& with_conjugate(bool enable = true) {
        conjugate = enable;
        return *this;
    }

    SolverConfig& with_abmc(bool enable = true, int block_size = 4, int num_colors = 4) {
        use_abmc = enable;
        abmc_block_size = block_size;
        abmc_num_colors = num_colors;
        return *this;
    }

    SolverConfig& with_abmc_reorder_spmv(bool enable = true) {
        abmc_reorder_spmv = enable;
        return *this;
    }

    SolverConfig& with_abmc_rcm(bool enable = true) {
        abmc_use_rcm = enable;
        return *this;
    }
};

} // namespace sparsesolv

#endif // SPARSESOLV_CORE_SOLVER_CONFIG_HPP
