/**
 * @file sparsesolv.hpp
 * @brief Main header file for SparseSolv library
 *
 * SparseSolv provides iterative solvers for sparse linear systems,
 * designed for integration with finite element codes.
 *
 * Features:
 * - Preconditioned Conjugate Gradient (ICCG)
 * - SGS-MRTR (Symmetric Gauss-Seidel MRTR with split formula)
 * - Support for real and complex matrices
 *
 * Example usage:
 * @code
 * #include <sparsesolv/sparsesolv.hpp>
 *
 * // Create sparse matrix view (zero-copy wrapper around your CSR data)
 * sparsesolv::SparseMatrixView<double> A(n, n, row_ptr, col_idx, values);
 *
 * // Configure solver
 * sparsesolv::SolverConfig config;
 * config.tolerance = 1e-10;
 * config.max_iterations = 1000;
 *
 * // Create preconditioner
 * sparsesolv::ICPreconditioner<double> precond(1.05);
 * precond.setup(A);
 *
 * // Create solver and solve
 * sparsesolv::CGSolver<double> solver;
 * solver.set_config(config);
 *
 * std::vector<double> x(n, 0.0);
 * auto result = solver.solve(A, b, x, &precond);
 *
 * if (result.converged) {
 *     std::cout << "Converged in " << result.iterations << " iterations\n";
 * }
 * @endcode
 *
 * @author SparseSolv Contributors
 * @version 2.0.0
 */

#ifndef SPARSESOLV_HPP
#define SPARSESOLV_HPP

// Core components
#include "core/types.hpp"
#include "core/constants.hpp"
#include "core/solver_config.hpp"
#include "core/sparse_matrix_view.hpp"
#include "core/preconditioner.hpp"
#include "core/abmc_ordering.hpp"

// Preconditioners
#include "preconditioners/ic_preconditioner.hpp"
#include "preconditioners/sgs_preconditioner.hpp"

// Solvers
#include "solvers/iterative_solver.hpp"
#include "solvers/cg_solver.hpp"
#include "solvers/sgs_mrtr_solver.hpp"

namespace sparsesolv {

/**
 * @brief Library version information
 */
struct Version {
    static constexpr int major = 2;
    static constexpr int minor = 0;
    static constexpr int patch = 0;

    static std::string string() {
        return std::to_string(major) + "." +
               std::to_string(minor) + "." +
               std::to_string(patch);
    }
};

/**
 * @brief Convenience function: Solve Ax=b using ICCG
 *
 * @param A System matrix (CSR format view)
 * @param b Right-hand side vector
 * @param x Solution vector (initial guess, modified in place)
 * @param size System size
 * @param config Solver configuration
 * @return SolverResult with convergence information
 */
template<typename Scalar = double>
inline SolverResult solve_iccg(
    const SparseMatrixView<Scalar>& A,
    const Scalar* b,
    Scalar* x,
    index_t size,
    const SolverConfig& config = SolverConfig()
) {
    // Create IC preconditioner with full config (for auto-shift, diagonal scaling, etc.)
    ICPreconditioner<Scalar> precond(config.shift_parameter);
    precond.set_config(config);
    precond.setup(A);

    // Create CG solver
    CGSolver<Scalar> solver;
    solver.set_config(config);

    if (precond.has_reordered_matrix()) {
        // ABMC path: run CG entirely in reordered space
        // Permutation is done once before and after CG (not per iteration)
        const auto& ord = precond.abmc_ordering();

        // Permute b: b_reord[ord[i]] = b[i]
        std::vector<Scalar> b_reord(size);
        for (index_t i = 0; i < size; ++i) {
            b_reord[ord[i]] = b[i];
        }

        // Permute x (initial guess): x_reord[ord[i]] = x[i]
        std::vector<Scalar> x_reord(size);
        for (index_t i = 0; i < size; ++i) {
            x_reord[ord[i]] = x[i];
        }

        // CG with reordered matrix and no-permutation preconditioner adapter
        auto A_reord = precond.reordered_matrix_view();
        ICPrecondReorderedAdapter<Scalar> adapter(precond);
        auto result = solver.solve(A_reord, b_reord.data(), x_reord.data(),
                                   size, &adapter);

        // Reverse permute solution: x[i] = x_reord[ord[i]]
        for (index_t i = 0; i < size; ++i) {
            x[i] = x_reord[ord[i]];
        }

        return result;
    }

    // Standard path (level scheduling or no preconditioning)
    return solver.solve(A, b, x, size, &precond);
}

/**
 * @brief Convenience function: Solve Ax=b using ICCG with std::vector
 */
template<typename Scalar = double>
inline SolverResult solve_iccg(
    const SparseMatrixView<Scalar>& A,
    const std::vector<Scalar>& b,
    std::vector<Scalar>& x,
    const SolverConfig& config = SolverConfig()
) {
    if (x.size() != b.size()) {
        x.resize(b.size());
    }
    return solve_iccg(A, b.data(), x.data(), static_cast<index_t>(b.size()), config);
}

/**
 * @brief Convenience function: Solve Ax=b using SGS-MRTR
 *
 * Uses the specialized SGS-MRTR solver with split preconditioner formula.
 */
template<typename Scalar = double>
inline SolverResult solve_sgsmrtr(
    const SparseMatrixView<Scalar>& A,
    const Scalar* b,
    Scalar* x,
    index_t size,
    const SolverConfig& config = SolverConfig()
) {
    // Use specialized SGS-MRTR solver with split formula
    SGSMRTRSolver<Scalar> solver;
    solver.set_config(config);

    // Solve
    return solver.solve(A, b, x, size);
}

/**
 * @brief Convenience function: Solve Ax=b using SGS-MRTR with std::vector
 */
template<typename Scalar = double>
inline SolverResult solve_sgsmrtr(
    const SparseMatrixView<Scalar>& A,
    const std::vector<Scalar>& b,
    std::vector<Scalar>& x,
    const SolverConfig& config = SolverConfig()
) {
    if (x.size() != b.size()) {
        x.resize(b.size());
    }
    // Use specialized SGS-MRTR solver
    SGSMRTRSolver<Scalar> solver;
    solver.set_config(config);
    return solver.solve(A, b, x);
}

} // namespace sparsesolv

#endif // SPARSESOLV_HPP
