/// @file sparsesolv_precond.hpp
/// @brief SparseSolv preconditioners (IC, SGS, BDDC) as NGSolve BaseMatrix wrappers

#ifndef NGSOLVE_SPARSESOLV_PRECOND_HPP
#define NGSOLVE_SPARSESOLV_PRECOND_HPP

#include <basematrix.hpp>
#include <sparsematrix.hpp>

// SparseSolv headers
#include "../sparsesolv.hpp"

namespace ngla {

template<typename SCAL>
const SCAL* GetVectorData(const BaseVector& vec) {
    return vec.FV<SCAL>().Data();
}

template<typename SCAL>
SCAL* GetVectorData(BaseVector& vec) {
    return vec.FV<SCAL>().Data();
}

// ============================================================================
// Preconditioner Base Class
// ============================================================================

/// Base class: NGSolve SparseMatrix → SparseSolv view, FreeDofs handling
template<typename SCAL>
class SparseSolvPrecondBase : public BaseMatrix {
protected:
    shared_ptr<SparseMatrix<SCAL>> mat_;
    shared_ptr<BitArray> freedofs_;
    sparsesolv::index_t height_;
    sparsesolv::index_t width_;

    // CSR conversion buffers
    std::vector<sparsesolv::index_t> row_ptr_;
    std::vector<sparsesolv::index_t> col_idx_;
    std::vector<SCAL> modified_values_;

    SparseSolvPrecondBase(shared_ptr<SparseMatrix<SCAL>> mat,
                          shared_ptr<BitArray> freedofs)
        : mat_(mat)
        , freedofs_(freedofs ? make_shared<BitArray>(*freedofs) : nullptr)
        , height_(static_cast<sparsesolv::index_t>(mat->Height()))
        , width_(static_cast<sparsesolv::index_t>(mat->Width()))
    {}

    /// Convert NGSolve SparseMatrix to SparseSolv view (identity rows for constrained DOFs)
    sparsesolv::SparseMatrixView<SCAL> prepare_matrix_view() {
        const auto& firsti = mat_->GetFirstArray();
        const auto& colnr = mat_->GetColIndices();
        const auto& values = mat_->GetValues();

        const size_t nrows = firsti.Size();
        const size_t nnz = colnr.Size();

        // Convert row pointers
        row_ptr_.resize(nrows);
        sparsesolv::parallel_for(static_cast<sparsesolv::index_t>(nrows), [&](sparsesolv::index_t i) {
            row_ptr_[i] = static_cast<sparsesolv::index_t>(firsti[i]);
        });

        // Convert column indices
        col_idx_.resize(nnz);
        sparsesolv::parallel_for(static_cast<sparsesolv::index_t>(nnz), [&](sparsesolv::index_t i) {
            col_idx_[i] = static_cast<sparsesolv::index_t>(colnr[i]);
        });

        if (freedofs_) {
            modified_values_.resize(values.Size());

            sparsesolv::parallel_for(static_cast<sparsesolv::index_t>(values.Size()), [&](sparsesolv::index_t i) {
                modified_values_[i] = values[i];
            });

            sparsesolv::parallel_for(height_, [&](sparsesolv::index_t i) {
                for (sparsesolv::index_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
                    sparsesolv::index_t j = col_idx_[k];
                    if (!freedofs_->Test(i)) {
                        modified_values_[k] = (j == i) ? SCAL(1) : SCAL(0);
                    } else if (!freedofs_->Test(j)) {
                        modified_values_[k] = SCAL(0);
                    }
                }
            });

            return sparsesolv::SparseMatrixView<SCAL>(
                height_, width_,
                row_ptr_.data(), col_idx_.data(), modified_values_.data()
            );
        } else {
            return sparsesolv::SparseMatrixView<SCAL>(
                height_, width_,
                row_ptr_.data(), col_idx_.data(), values.Data()
            );
        }
    }

    /// Apply the underlying SparseSolv preconditioner (implemented by derived classes)
    virtual void apply_precond(const SCAL* x, SCAL* y) const = 0;

public:
    void Mult(const BaseVector& x, BaseVector& y) const override {
        const SCAL* x_data = GetVectorData<SCAL>(x);
        SCAL* y_data = GetVectorData<SCAL>(y);

        apply_precond(x_data, y_data);

        if (freedofs_) {
            sparsesolv::parallel_for(height_, [&](sparsesolv::index_t i) {
                if (!freedofs_->Test(i)) {
                    y_data[i] = SCAL(0);
                }
            });
        }
    }

    void MultAdd(double s, const BaseVector& x, BaseVector& y) const override {
        const SCAL* x_data = GetVectorData<SCAL>(x);
        SCAL* y_data = GetVectorData<SCAL>(y);

        std::vector<SCAL> temp(height_);
        apply_precond(x_data, temp.data());

        if (freedofs_) {
            sparsesolv::parallel_for(height_, [&](sparsesolv::index_t i) {
                if (freedofs_->Test(i)) {
                    y_data[i] += s * temp[i];
                }
            });
        } else {
            sparsesolv::parallel_for(height_, [&](sparsesolv::index_t i) {
                y_data[i] += s * temp[i];
            });
        }
    }

    int VHeight() const override { return static_cast<int>(height_); }
    int VWidth() const override { return static_cast<int>(width_); }

    AutoVector CreateRowVector() const override {
        return mat_->CreateRowVector();
    }

    AutoVector CreateColVector() const override {
        return mat_->CreateColVector();
    }
};

// ============================================================================
// IC Preconditioner
// ============================================================================
template<typename SCAL = double>
class SparseSolvICPreconditioner : public SparseSolvPrecondBase<SCAL> {
public:
    SparseSolvICPreconditioner(shared_ptr<SparseMatrix<SCAL>> mat,
                                shared_ptr<BitArray> freedofs = nullptr,
                                double shift = 1.05)
        : SparseSolvPrecondBase<SCAL>(mat, freedofs)
        , shift_(shift)
        , precond_(std::make_shared<sparsesolv::ICPreconditioner<SCAL>>(shift))
    {}

    void Update() {
        auto view = this->prepare_matrix_view();
        precond_->setup(view);
    }

    double GetShift() const { return shift_; }

    void SetShift(double shift) {
        shift_ = shift;
        precond_->set_shift_parameter(shift);
    }

protected:
    void apply_precond(const SCAL* x, SCAL* y) const override {
        precond_->apply(x, y, this->height_);
    }

private:
    double shift_;
    shared_ptr<sparsesolv::ICPreconditioner<SCAL>> precond_;
};

// ============================================================================
// SGS Preconditioner
// ============================================================================
template<typename SCAL = double>
class SparseSolvSGSPreconditioner : public SparseSolvPrecondBase<SCAL> {
public:
    SparseSolvSGSPreconditioner(shared_ptr<SparseMatrix<SCAL>> mat,
                                 shared_ptr<BitArray> freedofs = nullptr)
        : SparseSolvPrecondBase<SCAL>(mat, freedofs)
        , precond_(std::make_shared<sparsesolv::SGSPreconditioner<SCAL>>())
    {}

    void Update() {
        auto view = this->prepare_matrix_view();
        precond_->setup(view);
    }

protected:
    void apply_precond(const SCAL* x, SCAL* y) const override {
        precond_->apply(x, y, this->height_);
    }

private:
    shared_ptr<sparsesolv::SGSPreconditioner<SCAL>> precond_;
};

// ============================================================================
// BDDC Preconditioner
// ============================================================================
template<typename SCAL = double>
class SparseSolvBDDCPreconditioner : public SparseSolvPrecondBase<SCAL> {
public:
    SparseSolvBDDCPreconditioner(
        shared_ptr<SparseMatrix<SCAL>> mat,
        shared_ptr<BitArray> freedofs,
        std::vector<std::vector<sparsesolv::index_t>> element_dofs,
        std::vector<sparsesolv::DOFType> dof_types,
        std::vector<sparsesolv::DenseMatrix<SCAL>> element_matrices,
        std::string coarse_inverse = "sparsecholesky")
        : SparseSolvPrecondBase<SCAL>(mat, freedofs)
        , element_dofs_(std::move(element_dofs))
        , dof_types_(std::move(dof_types))
        , element_matrices_(std::move(element_matrices))
        , coarse_inverse_type_(std::move(coarse_inverse))
        , precond_(std::make_shared<sparsesolv::BDDCPreconditioner<SCAL>>())
    {}

    void Update() {
        auto view = this->prepare_matrix_view();
        precond_->set_element_info(element_dofs_, dof_types_);

        // Pass element matrices for element-by-element mode
        if (!element_matrices_.empty())
            precond_->set_element_matrices(element_matrices_);

        // Tell BDDC to skip dense coarse inverse if we'll use NGSolve's solver
        bool use_ngsolve_coarse = !element_matrices_.empty() &&
                                   coarse_inverse_type_ != "dense";
        if (use_ngsolve_coarse)
            precond_->set_use_external_coarse(true);

        // Pass free DOF information to BDDC
        if (this->freedofs_) {
            std::vector<bool> free_dofs(this->height_);
            for (sparsesolv::index_t i = 0; i < this->height_; ++i)
                free_dofs[i] = this->freedofs_->Test(i);
            precond_->set_free_dofs(std::move(free_dofs));
        }

        precond_->setup(view);

        // Build NGSolve sparse direct inverse for wirebasket coarse solve
        if (use_ngsolve_coarse)
            build_ngsolve_coarse_inverse();
    }

    sparsesolv::index_t NumWirebasketDofs() const {
        return precond_->num_wirebasket_dofs();
    }
    sparsesolv::index_t NumInterfaceDofs() const {
        return precond_->num_interface_dofs();
    }
    bool IsElementMode() const {
        return precond_->is_element_mode();
    }

    const std::string& GetCoarseInverseType() const { return coarse_inverse_type_; }

protected:
    void apply_precond(const SCAL* x, SCAL* y) const override {
        precond_->apply(x, y, this->height_);
    }

private:
    /// Build NGSolve sparse direct inverse (PARDISO/SparseCholesky) for wirebasket coarse solve
    void build_ngsolve_coarse_inverse() {
        const auto& wb_csr = precond_->wirebasket_csr();
        sparsesolv::index_t n_wb = precond_->num_wirebasket_dofs();

        // Build n_wb x n_wb SparseMatrix from CSR data

        // Create NGSolve SparseMatrix from wirebasket CSR
        Array<int> elsperrow(n_wb);
        for (sparsesolv::index_t i = 0; i < n_wb; ++i)
            elsperrow[i] = wb_csr.row_ptr[i + 1] - wb_csr.row_ptr[i];

        auto sp_mat = make_shared<SparseMatrix<SCAL>>(elsperrow, n_wb);
        for (sparsesolv::index_t i = 0; i < n_wb; ++i) {
            auto cols = sp_mat->GetRowIndices(i);
            auto vals = sp_mat->GetRowValues(i);
            sparsesolv::index_t off = wb_csr.row_ptr[i];
            for (int k = 0; k < elsperrow[i]; ++k) {
                cols[k] = wb_csr.col_idx[off + k];
                vals[k] = wb_csr.values[off + k];
            }
        }

        // Keep SparseMatrix alive (inverse holds a reference to it)
        coarse_mat_ = sp_mat;

        // Create sparse direct inverse
        sp_mat->SetInverseType(coarse_inverse_type_);
        coarse_inv_ = sp_mat->InverseMatrix();

        // Pre-allocate work vectors
        coarse_rhs_ = make_shared<VVector<SCAL>>(n_wb);
        coarse_sol_ = make_shared<VVector<SCAL>>(n_wb);

        // Set coarse solver callback on BDDC preconditioner
        precond_->set_coarse_solver(
            [this](const SCAL* rhs, SCAL* sol) {
                auto n = precond_->num_wirebasket_dofs();
                auto rhs_fv = coarse_rhs_->FV<SCAL>();
                auto sol_fv = coarse_sol_->FV<SCAL>();
                for (sparsesolv::index_t i = 0; i < n; ++i)
                    rhs_fv[i] = rhs[i];
                coarse_inv_->Mult(*coarse_rhs_, *coarse_sol_);
                for (sparsesolv::index_t i = 0; i < n; ++i)
                    sol[i] = sol_fv[i];
            });
    }

    std::vector<std::vector<sparsesolv::index_t>> element_dofs_;
    std::vector<sparsesolv::DOFType> dof_types_;
    std::vector<sparsesolv::DenseMatrix<SCAL>> element_matrices_;
    std::string coarse_inverse_type_;
    shared_ptr<sparsesolv::BDDCPreconditioner<SCAL>> precond_;

    // NGSolve coarse solver (when coarse_inverse_type_ != "dense")
    shared_ptr<BaseMatrix> coarse_mat_;  // wirebasket SparseMatrix (must outlive coarse_inv_)
    shared_ptr<BaseMatrix> coarse_inv_;
    mutable shared_ptr<BaseVector> coarse_rhs_;
    mutable shared_ptr<BaseVector> coarse_sol_;
};

// Type aliases for convenience
using ICPreconditioner = SparseSolvICPreconditioner<double>;
using SGSPreconditioner = SparseSolvSGSPreconditioner<double>;

// ============================================================================
// Solver Result
// ============================================================================
struct SparseSolvResult {
    bool converged = false;
    int iterations = 0;
    double final_residual = 0.0;
    std::vector<double> residual_history;
};

// ============================================================================
// SparseSolv Iterative Solver
// ============================================================================

/// Unified iterative solver: ICCG, SGSMRTR, CG. Use as BaseMatrix or .Solve().
template<typename SCAL = double>
class SparseSolvSolver : public BaseMatrix {
public:
    SparseSolvSolver(shared_ptr<SparseMatrix<SCAL>> mat,
                      const string& method = "ICCG",
                      shared_ptr<BitArray> freedofs = nullptr,
                      double tol = 1e-10,
                      int maxiter = 1000,
                      double shift = 1.05,
                      bool save_best_result = true,
                      bool save_residual_history = false,
                      bool printrates = false)
        : mat_(mat)
        , method_(method)
        , freedofs_(freedofs ? make_shared<BitArray>(*freedofs) : nullptr)
        , height_(static_cast<sparsesolv::index_t>(mat->Height()))
        , width_(static_cast<sparsesolv::index_t>(mat->Width()))
        , printrates_(printrates)
    {
        config_.tolerance = tol;
        config_.max_iterations = maxiter;
        config_.shift_parameter = shift;
        config_.save_best_result = save_best_result;
        config_.save_residual_history = save_residual_history;
    }

    /// Solve Ax = b, x initialized to zero
    void Mult(const BaseVector& x, BaseVector& y) const override {
        y = 0.0;
        Solve(x, y);
    }

    void MultAdd(double s, const BaseVector& x, BaseVector& y) const override {
        auto temp = y.CreateVector();
        temp = 0.0;
        Solve(x, *temp);
        y += s * *temp;
    }

    /// Solve Ax = b with initial guess, returns SparseSolvResult
    SparseSolvResult Solve(const BaseVector& rhs, BaseVector& sol) const {
        // Prepare matrix view (with FreeDofs handling)
        auto view = prepare_matrix();

        // Get vector data
        const SCAL* b = GetVectorData<SCAL>(rhs);
        SCAL* x = GetVectorData<SCAL>(sol);

        // For FreeDofs: work on copies with constrained DOFs zeroed
        std::vector<SCAL> b_mod;
        std::vector<SCAL> x_mod;
        const SCAL* b_ptr = b;
        SCAL* x_ptr = x;

        if (freedofs_) {
            b_mod.resize(height_);
            x_mod.resize(height_);
            sparsesolv::parallel_for(height_, [&](sparsesolv::index_t i) {
                if (freedofs_->Test(i)) {
                    b_mod[i] = b[i];
                    x_mod[i] = x[i];
                } else {
                    b_mod[i] = SCAL(0);
                    x_mod[i] = SCAL(0);
                }
            });
            b_ptr = b_mod.data();
            x_ptr = x_mod.data();
        }

        // Configure solver
        sparsesolv::SolverConfig config = config_;
        if (printrates_) {
            config.save_residual_history = true;
        }

        // Dispatch to appropriate solver
        sparsesolv::SolverResult result;

        if (method_ == "ICCG" || method_ == "iccg") {
            result = sparsesolv::solve_iccg(view, b_ptr, x_ptr, height_, config);
        } else if (method_ == "SGSMRTR" || method_ == "sgsmrtr") {
            result = sparsesolv::solve_sgsmrtr(view, b_ptr, x_ptr, height_, config);
        } else if (method_ == "CG" || method_ == "cg") {
            sparsesolv::CGSolver<SCAL> solver;
            solver.set_config(config);
            result = solver.solve(view, b_ptr, x_ptr, height_, nullptr);
        } else {
            throw std::runtime_error(
                "SparseSolvSolver: Unknown method '" + method_ +
                "'. Available: ICCG, SGSMRTR, CG");
        }

        // Copy solution back (only free DOFs)
        if (freedofs_) {
            sparsesolv::parallel_for(height_, [&](sparsesolv::index_t i) {
                if (freedofs_->Test(i)) {
                    x[i] = x_ptr[i];
                }
            });
        }

        // Print convergence info if requested
        if (printrates_ && !result.residual_history.empty()) {
            for (size_t i = 0; i < result.residual_history.size(); ++i) {
                std::cout << method_ << " iteration " << i
                          << ", residual = " << result.residual_history[i]
                          << std::endl;
            }
            if (result.converged) {
                std::cout << method_ << " converged in "
                          << result.iterations << " iterations, residual = "
                          << result.final_residual << std::endl;
            } else {
                std::cout << method_ << " NOT converged after "
                          << result.iterations << " iterations, residual = "
                          << result.final_residual << std::endl;
            }
        }

        // Build and store result
        last_result_.converged = result.converged;
        last_result_.iterations = result.iterations;
        last_result_.final_residual = result.final_residual;
        last_result_.residual_history = std::move(result.residual_history);

        return last_result_;
    }

    int VHeight() const override { return static_cast<int>(height_); }
    int VWidth() const override { return static_cast<int>(width_); }
    AutoVector CreateRowVector() const override { return mat_->CreateRowVector(); }
    AutoVector CreateColVector() const override { return mat_->CreateColVector(); }

    // Property accessors
    const string& GetMethod() const { return method_; }
    void SetMethod(const string& method) { method_ = method; }
    double GetTolerance() const { return config_.tolerance; }
    void SetTolerance(double tol) { config_.tolerance = tol; }
    int GetMaxIterations() const { return config_.max_iterations; }
    void SetMaxIterations(int maxiter) { config_.max_iterations = maxiter; }
    double GetShift() const { return config_.shift_parameter; }
    void SetShift(double shift) { config_.shift_parameter = shift; }
    bool GetSaveBestResult() const { return config_.save_best_result; }
    void SetSaveBestResult(bool save) { config_.save_best_result = save; }
    bool GetSaveResidualHistory() const { return config_.save_residual_history; }
    void SetSaveResidualHistory(bool save) { config_.save_residual_history = save; }
    bool GetPrintRates() const { return printrates_; }
    void SetPrintRates(bool print) { printrates_ = print; }
    bool GetAutoShift() const { return config_.auto_shift; }
    void SetAutoShift(bool enable) { config_.auto_shift = enable; }
    bool GetDiagonalScaling() const { return config_.diagonal_scaling; }
    void SetDiagonalScaling(bool enable) { config_.diagonal_scaling = enable; }

    // Complex inner product
    bool GetConjugate() const { return config_.conjugate; }
    void SetConjugate(bool enable) { config_.conjugate = enable; }

    // ABMC ordering
    bool GetUseABMC() const { return config_.use_abmc; }
    void SetUseABMC(bool enable) { config_.use_abmc = enable; }
    int GetABMCBlockSize() const { return config_.abmc_block_size; }
    void SetABMCBlockSize(int bs) { config_.abmc_block_size = bs; }
    int GetABMCNumColors() const { return config_.abmc_num_colors; }
    void SetABMCNumColors(int nc) { config_.abmc_num_colors = nc; }
    bool GetABMCReorderSpMV() const { return config_.abmc_reorder_spmv; }
    void SetABMCReorderSpMV(bool enable) { config_.abmc_reorder_spmv = enable; }
    bool GetABMCUseRCM() const { return config_.abmc_use_rcm; }
    void SetABMCUseRCM(bool enable) { config_.abmc_use_rcm = enable; }

    // Divergence detection
    bool GetDivergenceCheck() const {
        return config_.divergence_check == sparsesolv::DivergenceCheck::StagnationCount;
    }
    void SetDivergenceCheck(bool enable) {
        config_.divergence_check = enable
            ? sparsesolv::DivergenceCheck::StagnationCount
            : sparsesolv::DivergenceCheck::None;
    }
    double GetDivergenceThreshold() const { return config_.divergence_threshold; }
    void SetDivergenceThreshold(double threshold) { config_.divergence_threshold = threshold; }
    int GetDivergenceCount() const { return config_.divergence_count; }
    void SetDivergenceCount(int count) { config_.divergence_count = count; }

    const SparseSolvResult& GetLastResult() const { return last_result_; }

private:
    sparsesolv::SparseMatrixView<SCAL> prepare_matrix() const {
        const auto& firsti = mat_->GetFirstArray();
        const auto& colnr = mat_->GetColIndices();
        const auto& values = mat_->GetValues();

        const size_t nrows = firsti.Size();
        const size_t nnz = colnr.Size();

        // Convert row pointers
        row_ptr_.resize(nrows);
        sparsesolv::parallel_for(static_cast<sparsesolv::index_t>(nrows), [&](sparsesolv::index_t i) {
            row_ptr_[i] = static_cast<sparsesolv::index_t>(firsti[i]);
        });

        // Convert column indices
        col_idx_.resize(nnz);
        sparsesolv::parallel_for(static_cast<sparsesolv::index_t>(nnz), [&](sparsesolv::index_t i) {
            col_idx_[i] = static_cast<sparsesolv::index_t>(colnr[i]);
        });

        if (freedofs_) {
            modified_values_.resize(values.Size());

            sparsesolv::parallel_for(static_cast<sparsesolv::index_t>(values.Size()), [&](sparsesolv::index_t i) {
                modified_values_[i] = values[i];
            });

            sparsesolv::parallel_for(height_, [&](sparsesolv::index_t i) {
                for (sparsesolv::index_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
                    sparsesolv::index_t j = col_idx_[k];
                    if (!freedofs_->Test(i)) {
                        modified_values_[k] = (j == i) ? SCAL(1) : SCAL(0);
                    } else if (!freedofs_->Test(j)) {
                        modified_values_[k] = SCAL(0);
                    }
                }
            });

            return sparsesolv::SparseMatrixView<SCAL>(
                height_, width_,
                row_ptr_.data(), col_idx_.data(), modified_values_.data()
            );
        } else {
            return sparsesolv::SparseMatrixView<SCAL>(
                height_, width_,
                row_ptr_.data(), col_idx_.data(), values.Data()
            );
        }
    }

    shared_ptr<SparseMatrix<SCAL>> mat_;
    string method_;
    shared_ptr<BitArray> freedofs_;
    sparsesolv::SolverConfig config_;
    sparsesolv::index_t height_;
    sparsesolv::index_t width_;
    bool printrates_;

    // Mutable for use in const Mult/Solve methods
    mutable std::vector<sparsesolv::index_t> row_ptr_;
    mutable std::vector<sparsesolv::index_t> col_idx_;
    mutable std::vector<SCAL> modified_values_;
    mutable SparseSolvResult last_result_;
};

} // namespace ngla

#endif // NGSOLVE_SPARSESOLV_PRECOND_HPP
