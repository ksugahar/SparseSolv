/// @file sparsesolv_python_export.hpp
/// @brief Pybind11 bindings: typed classes (D/C) + factory functions with auto-dispatch

#ifndef NGSOLVE_SPARSESOLV_PYTHON_EXPORT_HPP
#define NGSOLVE_SPARSESOLV_PYTHON_EXPORT_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sparsesolv_precond.hpp"
#include <comp.hpp>
#include <type_traits>

namespace py = pybind11;

namespace ngla {

// ============================================================================
// Internal: SparseSolvResult (non-templated, called once)
// ============================================================================

inline void ExportSparseSolvResult_impl(py::module& m) {
  py::class_<SparseSolvResult>(m, "SparseSolvResult",
    "Result of a SparseSolv iterative solve (converged, iterations, final_residual, residual_history).")
    .def_readonly("converged", &SparseSolvResult::converged,
        "Whether the solver converged within tolerance")
    .def_readonly("iterations", &SparseSolvResult::iterations,
        "Number of iterations performed")
    .def_readonly("final_residual", &SparseSolvResult::final_residual,
        "Final relative residual (or best residual if save_best_result enabled)")
    .def_readonly("residual_history", &SparseSolvResult::residual_history,
        "Residual at each iteration (if save_residual_history enabled)")
    .def("__repr__", [](const SparseSolvResult& r) {
      return string("SparseSolvResult(converged=") +
             (r.converged ? "True" : "False") +
             ", iterations=" + std::to_string(r.iterations) +
             ", residual=" + std::to_string(r.final_residual) + ")";
    });
}

// ============================================================================
// Internal: Typed class registration (D/C suffix)
// ============================================================================

template<typename SCAL>
void ExportSparseSolvTyped(py::module& m, const std::string& suffix) {

  // IC Preconditioner (ICPreconditionerD / ICPreconditionerC)
  {
    std::string cls_name = "ICPreconditioner" + suffix;
    py::class_<SparseSolvICPreconditioner<SCAL>,
               shared_ptr<SparseSolvICPreconditioner<SCAL>>,
               BaseMatrix>(m, cls_name.c_str(),
               "Incomplete Cholesky preconditioner (typed). Use ICPreconditioner() factory instead.")
      .def(py::init([](shared_ptr<SparseMatrix<SCAL>> mat,
                       py::object freedofs, double shift) {
        shared_ptr<BitArray> sp_freedofs = nullptr;
        if (!freedofs.is_none())
          sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
        return make_shared<SparseSolvICPreconditioner<SCAL>>(mat, sp_freedofs, shift);
      }), py::arg("mat"), py::arg("freedofs") = py::none(), py::arg("shift") = 1.05)
      .def("Update", &SparseSolvICPreconditioner<SCAL>::Update,
          "Update preconditioner (recompute factorization after matrix change)")
      .def_property("shift",
          &SparseSolvICPreconditioner<SCAL>::GetShift,
          &SparseSolvICPreconditioner<SCAL>::SetShift,
          "Shift parameter for IC decomposition");
  }

  // SGS Preconditioner (SGSPreconditionerD / SGSPreconditionerC)
  {
    std::string cls_name = "SGSPreconditioner" + suffix;
    py::class_<SparseSolvSGSPreconditioner<SCAL>,
               shared_ptr<SparseSolvSGSPreconditioner<SCAL>>,
               BaseMatrix>(m, cls_name.c_str(),
               "Symmetric Gauss-Seidel preconditioner (typed). Use SGSPreconditioner() factory instead.")
      .def(py::init([](shared_ptr<SparseMatrix<SCAL>> mat,
                       py::object freedofs) {
        shared_ptr<BitArray> sp_freedofs = nullptr;
        if (!freedofs.is_none())
          sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
        return make_shared<SparseSolvSGSPreconditioner<SCAL>>(mat, sp_freedofs);
      }), py::arg("mat"), py::arg("freedofs") = py::none())
      .def("Update", &SparseSolvSGSPreconditioner<SCAL>::Update,
          "Update preconditioner (recompute after matrix change)");
  }

  // BDDC Preconditioner (BDDCPreconditionerD / BDDCPreconditionerC)
  {
    std::string cls_name = "BDDCPreconditioner" + suffix;
    py::class_<SparseSolvBDDCPreconditioner<SCAL>,
               shared_ptr<SparseSolvBDDCPreconditioner<SCAL>>,
               BaseMatrix>(m, cls_name.c_str(),
               "BDDC preconditioner (typed). Use BDDCPreconditioner() factory instead.")
      .def(py::init([](shared_ptr<SparseMatrix<SCAL>> mat,
                       py::object freedofs,
                       std::vector<std::vector<sparsesolv::index_t>> element_dofs,
                       std::vector<int> dof_types_int,
                       py::list element_matrices_py,
                       std::string coarse_inverse) {
        shared_ptr<BitArray> sp_freedofs = nullptr;
        if (!freedofs.is_none())
          sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
        std::vector<sparsesolv::DOFType> dof_types(dof_types_int.size());
        for (size_t i = 0; i < dof_types_int.size(); ++i)
          dof_types[i] = static_cast<sparsesolv::DOFType>(dof_types_int[i]);
        auto element_matrices = ConvertElementMatrices<SCAL>(element_matrices_py);
        return make_shared<SparseSolvBDDCPreconditioner<SCAL>>(
            mat, sp_freedofs, std::move(element_dofs),
            std::move(dof_types), std::move(element_matrices),
            std::move(coarse_inverse));
      }),
          py::arg("mat"),
          py::arg("freedofs") = py::none(),
          py::arg("element_dofs") = std::vector<std::vector<sparsesolv::index_t>>(),
          py::arg("dof_types") = std::vector<int>(),
          py::arg("element_matrices") = py::list(),
          py::arg("coarse_inverse") = "sparsecholesky")
      .def("Update", &SparseSolvBDDCPreconditioner<SCAL>::Update,
          "Update preconditioner (recompute factorization after matrix change)")
      .def_property_readonly("num_wirebasket_dofs",
          &SparseSolvBDDCPreconditioner<SCAL>::NumWirebasketDofs,
          "Number of wirebasket (coarse) DOFs")
      .def_property_readonly("num_interface_dofs",
          &SparseSolvBDDCPreconditioner<SCAL>::NumInterfaceDofs,
          "Number of interface (local) DOFs")
      .def_property_readonly("is_element_mode",
          &SparseSolvBDDCPreconditioner<SCAL>::IsElementMode,
          "Whether element-by-element BDDC mode is active");
  }

  // SparseSolv Solver (SparseSolvSolverD / SparseSolvSolverC)
  {
    std::string cls_name = "SparseSolvSolver" + suffix;
    py::class_<SparseSolvSolver<SCAL>,
               shared_ptr<SparseSolvSolver<SCAL>>,
               BaseMatrix>(m, cls_name.c_str(),
               "SparseSolv iterative solver (typed). Use SparseSolvSolver() factory instead.")
      .def(py::init([](shared_ptr<SparseMatrix<SCAL>> mat,
                       const string& method, py::object freedofs,
                       double tol, int maxiter, double shift,
                       bool save_best_result, bool save_residual_history,
                       bool printrates) {
        shared_ptr<BitArray> sp_freedofs = nullptr;
        if (!freedofs.is_none())
          sp_freedofs = py::cast<shared_ptr<BitArray>>(freedofs);
        return make_shared<SparseSolvSolver<SCAL>>(
            mat, method, sp_freedofs, tol, maxiter, shift,
            save_best_result, save_residual_history, printrates);
      }),
          py::arg("mat"),
          py::arg("method") = "ICCG",
          py::arg("freedofs") = py::none(),
          py::arg("tol") = 1e-10,
          py::arg("maxiter") = 1000,
          py::arg("shift") = 1.05,
          py::arg("save_best_result") = true,
          py::arg("save_residual_history") = false,
          py::arg("printrates") = false)
      .def("Solve", [](SparseSolvSolver<SCAL>& self,
                       const BaseVector& rhs, BaseVector& sol) {
        return self.Solve(rhs, sol);
      }, py::arg("rhs"), py::arg("sol"),
         "Solve Ax = b. Returns SparseSolvResult with convergence info.")
      .def_property("method",
          &SparseSolvSolver<SCAL>::GetMethod,
          &SparseSolvSolver<SCAL>::SetMethod,
          "Solver method: ICCG, SGSMRTR, CG")
      .def_property("tol",
          &SparseSolvSolver<SCAL>::GetTolerance,
          &SparseSolvSolver<SCAL>::SetTolerance,
          "Relative convergence tolerance")
      .def_property("maxiter",
          &SparseSolvSolver<SCAL>::GetMaxIterations,
          &SparseSolvSolver<SCAL>::SetMaxIterations,
          "Maximum number of iterations")
      .def_property("shift",
          &SparseSolvSolver<SCAL>::GetShift,
          &SparseSolvSolver<SCAL>::SetShift,
          "Shift parameter for IC preconditioner")
      .def_property("save_best_result",
          &SparseSolvSolver<SCAL>::GetSaveBestResult,
          &SparseSolvSolver<SCAL>::SetSaveBestResult,
          "Track best solution during iteration")
      .def_property("save_residual_history",
          &SparseSolvSolver<SCAL>::GetSaveResidualHistory,
          &SparseSolvSolver<SCAL>::SetSaveResidualHistory,
          "Record residual at each iteration")
      .def_property("printrates",
          &SparseSolvSolver<SCAL>::GetPrintRates,
          &SparseSolvSolver<SCAL>::SetPrintRates,
          "Print convergence information after solve")
      .def_property("auto_shift",
          &SparseSolvSolver<SCAL>::GetAutoShift,
          &SparseSolvSolver<SCAL>::SetAutoShift,
          "Enable automatic shift adjustment for IC decomposition")
      .def_property("diagonal_scaling",
          &SparseSolvSolver<SCAL>::GetDiagonalScaling,
          &SparseSolvSolver<SCAL>::SetDiagonalScaling,
          "Enable diagonal scaling for IC preconditioner")
      .def_property("divergence_check",
          &SparseSolvSolver<SCAL>::GetDivergenceCheck,
          &SparseSolvSolver<SCAL>::SetDivergenceCheck,
          "Enable stagnation-based early termination")
      .def_property("divergence_threshold",
          &SparseSolvSolver<SCAL>::GetDivergenceThreshold,
          &SparseSolvSolver<SCAL>::SetDivergenceThreshold,
          "Multiplier for divergence detection (stop if residual > best * threshold)")
      .def_property("divergence_count",
          &SparseSolvSolver<SCAL>::GetDivergenceCount,
          &SparseSolvSolver<SCAL>::SetDivergenceCount,
          "Number of consecutive bad iterations before declaring divergence")
      .def_property("conjugate",
          &SparseSolvSolver<SCAL>::GetConjugate,
          &SparseSolvSolver<SCAL>::SetConjugate,
          "Use conjugated inner product for Hermitian systems (default: False for complex-symmetric)")
      .def_property("use_abmc",
          &SparseSolvSolver<SCAL>::GetUseABMC,
          &SparseSolvSolver<SCAL>::SetUseABMC,
          "Enable ABMC (Algebraic Block Multi-Color) ordering for parallel triangular solves")
      .def_property("abmc_block_size",
          &SparseSolvSolver<SCAL>::GetABMCBlockSize,
          &SparseSolvSolver<SCAL>::SetABMCBlockSize,
          "Number of rows per block for ABMC ordering (default: 4)")
      .def_property("abmc_num_colors",
          &SparseSolvSolver<SCAL>::GetABMCNumColors,
          &SparseSolvSolver<SCAL>::SetABMCNumColors,
          "Number of colors for ABMC graph coloring (default: 4)")
      .def_property("abmc_reorder_spmv",
          &SparseSolvSolver<SCAL>::GetABMCReorderSpMV,
          &SparseSolvSolver<SCAL>::SetABMCReorderSpMV,
          "Run SpMV on ABMC-reordered matrix (default: False = use original)")
      .def_property("abmc_use_rcm",
          &SparseSolvSolver<SCAL>::GetABMCUseRCM,
          &SparseSolvSolver<SCAL>::SetABMCUseRCM,
          "Apply RCM bandwidth reduction before ABMC (default: False)")
      .def_property_readonly("last_result",
          &SparseSolvSolver<SCAL>::GetLastResult,
          "Result from the last Solve() or Mult() call");
  }
}

// ============================================================================
// Internal helpers
// ============================================================================

inline shared_ptr<BitArray> ExtractFreeDofs(py::object freedofs) {
  if (freedofs.is_none()) return nullptr;
  return py::cast<shared_ptr<BitArray>>(freedofs);
}

/// Convert Python list of nested lists to vector<DenseMatrix<SCAL>>
template<typename SCAL>
std::vector<sparsesolv::DenseMatrix<SCAL>> ConvertElementMatrices(py::list py_list) {
  std::vector<sparsesolv::DenseMatrix<SCAL>> result;
  for (size_t e = 0; e < py::len(py_list); ++e) {
    auto nested = py_list[e].cast<std::vector<std::vector<SCAL>>>();
    sparsesolv::index_t rows = static_cast<sparsesolv::index_t>(nested.size());
    sparsesolv::index_t cols = rows > 0 ?
        static_cast<sparsesolv::index_t>(nested[0].size()) : 0;
    sparsesolv::DenseMatrix<SCAL> dm(rows, cols);
    for (sparsesolv::index_t i = 0; i < rows; ++i)
      for (sparsesolv::index_t j = 0; j < cols; ++j)
        dm(i, j) = nested[i][j];
    result.push_back(std::move(dm));
  }
  return result;
}

// ============================================================================
// Internal: BDDC factory from BilinearForm (C++ element matrix extraction)
// ============================================================================

template<typename SCAL>
shared_ptr<BaseMatrix> CreateBDDCFromBilinearForm(
    shared_ptr<ngcomp::BilinearForm> bfa,
    shared_ptr<ngcomp::FESpace> fes,
    const std::string& coarse_inverse)
{
    auto mat = dynamic_pointer_cast<SparseMatrix<SCAL>>(bfa->GetMatrixPtr());
    if (!mat) throw py::type_error("BDDCPreconditioner: matrix type mismatch");
    auto freedofs = fes->GetFreeDofs(true);  // coupling=true for BDDC
    auto mesh = fes->GetMeshAccess();
    size_t ndof = fes->GetNDof();
    size_t ne = mesh->GetNE(ngfem::VOL);

    // Extract DOF classification (parallel)
    std::vector<sparsesolv::DOFType> dof_types(ndof);
    ParallelFor(ndof, [&](size_t d) {
        auto ct = fes->GetDofCouplingType(d);
        dof_types[d] = (ct == ngcomp::WIREBASKET_DOF)
            ? sparsesolv::DOFType::Wirebasket
            : sparsesolv::DOFType::Interface;
    });

    // Extract element DOFs and element matrices (parallel via IterateElements)
    std::vector<std::vector<sparsesolv::index_t>> element_dofs(ne);
    std::vector<sparsesolv::DenseMatrix<SCAL>> element_matrices(ne);
    const auto& integrators = bfa->Integrators();

    LocalHeap lh(10000000, "bddc_setup", true);  // mult_by_threads=true

    ngcomp::IterateElements(*fes, ngfem::VOL, lh,
        [&](ngcomp::FESpace::Element el, LocalHeap& lh_thread) {
            size_t elnr = el.Nr();

            // Get DOFs for this element
            auto dnums = el.GetDofs();

            // Filter to valid DOFs (>= 0)
            std::vector<int> valid_local;
            std::vector<sparsesolv::index_t> valid_global;
            for (int i = 0; i < dnums.Size(); ++i) {
                if (ngcomp::IsRegularDof(dnums[i])) {
                    valid_local.push_back(i);
                    valid_global.push_back(static_cast<sparsesolv::index_t>(dnums[i]));
                }
            }
            element_dofs[elnr] = std::move(valid_global);

            // Get FE and transformation
            auto& fe = el.GetFE();
            auto& trafo = el.GetTrafo();

            int ndof_el = dnums.Size();
            FlatMatrix<SCAL> elmat(ndof_el, ndof_el, lh_thread);
            elmat = SCAL(0);

            // Sum contributions from all integrators
            for (auto& integrator : integrators) {
                FlatMatrix<SCAL> contrib(ndof_el, ndof_el, lh_thread);
                contrib = SCAL(0);
                integrator->CalcElementMatrix(fe, trafo, contrib, lh_thread);
                elmat += contrib;
            }

            // Extract valid-DOF submatrix
            int nvalid = static_cast<int>(valid_local.size());
            sparsesolv::DenseMatrix<SCAL> dm(nvalid, nvalid);
            for (int i = 0; i < nvalid; ++i)
                for (int j = 0; j < nvalid; ++j)
                    dm(i, j) = elmat(valid_local[i], valid_local[j]);
            element_matrices[elnr] = std::move(dm);
        });

    auto p = make_shared<SparseSolvBDDCPreconditioner<SCAL>>(
        mat, freedofs, std::move(element_dofs),
        std::move(dof_types), std::move(element_matrices),
        coarse_inverse);
    p->Update();
    return p;
}

// ============================================================================
// Internal: Factory functions with auto-dispatch via mat->IsComplex()
// ============================================================================

inline void ExportSparseSolvFactories(py::module& m) {

  // ---- ICPreconditioner factory ----
  m.def("ICPreconditioner", [](shared_ptr<BaseMatrix> mat,
                                py::object freedofs, double shift) {
    auto sp_freedofs = ExtractFreeDofs(freedofs);
    shared_ptr<BaseMatrix> result;
    if (mat->IsComplex()) {
      auto sp = dynamic_pointer_cast<SparseMatrix<Complex>>(mat);
      if (!sp) throw py::type_error("ICPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvICPreconditioner<Complex>>(sp, sp_freedofs, shift);
      p->Update();
      result = p;
    } else {
      auto sp = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp) throw py::type_error("ICPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvICPreconditioner<double>>(sp, sp_freedofs, shift);
      p->Update();
      result = p;
    }
    return result;
  },
  py::arg("mat"), py::arg("freedofs") = py::none(), py::arg("shift") = 1.05,
  R"raw_string(
Incomplete Cholesky (IC) Preconditioner.

Parameters:

mat : SparseMatrix
  SPD matrix (real or complex, auto-detected).
freedofs : BitArray, optional
  Free DOFs. Constrained DOFs treated as identity.
shift : float
  Shift parameter (default: 1.05).
)raw_string");

  // ---- SGSPreconditioner factory ----
  m.def("SGSPreconditioner", [](shared_ptr<BaseMatrix> mat,
                                  py::object freedofs) {
    auto sp_freedofs = ExtractFreeDofs(freedofs);
    shared_ptr<BaseMatrix> result;
    if (mat->IsComplex()) {
      auto sp = dynamic_pointer_cast<SparseMatrix<Complex>>(mat);
      if (!sp) throw py::type_error("SGSPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvSGSPreconditioner<Complex>>(sp, sp_freedofs);
      p->Update();
      result = p;
    } else {
      auto sp = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp) throw py::type_error("SGSPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvSGSPreconditioner<double>>(sp, sp_freedofs);
      p->Update();
      result = p;
    }
    return result;
  },
  py::arg("mat"), py::arg("freedofs") = py::none(),
  R"raw_string(
Symmetric Gauss-Seidel (SGS) Preconditioner.

Parameters:

mat : SparseMatrix
  SPD matrix (real or complex, auto-detected).
freedofs : BitArray, optional
  Free DOFs. Constrained DOFs treated as identity.
)raw_string");

  // ---- BDDCPreconditioner factory ----
  m.def("BDDCPreconditioner", [](py::object first_arg,
                                  py::object freedofs,
                                  std::vector<std::vector<sparsesolv::index_t>> element_dofs,
                                  std::vector<int> dof_types_int,
                                  py::list element_matrices_py,
                                  std::string coarse_inverse) {
    // BilinearForm API: BDDCPreconditioner(a, fes, coarse_inverse=...)
    try {
      auto bfa = py::cast<shared_ptr<ngcomp::BilinearForm>>(first_arg);
      auto fes = py::cast<shared_ptr<ngcomp::FESpace>>(freedofs);
      if (bfa->GetMatrixPtr()->IsComplex())
        return CreateBDDCFromBilinearForm<Complex>(bfa, fes, coarse_inverse);
      else
        return CreateBDDCFromBilinearForm<double>(bfa, fes, coarse_inverse);
    } catch (py::cast_error&) {}

    // Matrix API: BDDCPreconditioner(mat, freedofs=..., element_dofs=..., ...)
    auto mat = py::cast<shared_ptr<BaseMatrix>>(first_arg);
    auto sp_freedofs = ExtractFreeDofs(freedofs);
    std::vector<sparsesolv::DOFType> dof_types(dof_types_int.size());
    for (size_t i = 0; i < dof_types_int.size(); ++i)
      dof_types[i] = static_cast<sparsesolv::DOFType>(dof_types_int[i]);
    shared_ptr<BaseMatrix> result;
    if (mat->IsComplex()) {
      auto sp = dynamic_pointer_cast<SparseMatrix<Complex>>(mat);
      if (!sp) throw py::type_error("BDDCPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvBDDCPreconditioner<Complex>>(
          sp, sp_freedofs, std::move(element_dofs),
          std::move(dof_types), ConvertElementMatrices<Complex>(element_matrices_py),
          std::move(coarse_inverse));
      p->Update();
      result = p;
    } else {
      auto sp = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp) throw py::type_error("BDDCPreconditioner: expected SparseMatrix");
      auto p = make_shared<SparseSolvBDDCPreconditioner<double>>(
          sp, sp_freedofs, std::move(element_dofs),
          std::move(dof_types), ConvertElementMatrices<double>(element_matrices_py),
          std::move(coarse_inverse));
      p->Update();
      result = p;
    }
    return result;
  },
  py::arg("mat"),
  py::arg("freedofs") = py::none(),
  py::arg("element_dofs") = std::vector<std::vector<sparsesolv::index_t>>(),
  py::arg("dof_types") = std::vector<int>(),
  py::arg("element_matrices") = py::list(),
  py::arg("coarse_inverse") = "sparsecholesky",
  R"raw_string(
BDDC (Balancing Domain Decomposition by Constraints) Preconditioner.

Two APIs:

1. ``BDDCPreconditioner(a, fes)`` — extracts element matrices from BilinearForm (recommended)
2. ``BDDCPreconditioner(mat, freedofs=..., element_dofs=..., dof_types=..., element_matrices=...)``

Two modes:

- Block elimination (no element_matrices): exact (~2 iters), O(n_if^3), small problems
- Element-by-element (with element_matrices): scalable, ~15-20 iters

Parameters:

mat : BilinearForm or SparseMatrix
  BilinearForm for automatic extraction, or assembled SparseMatrix.
freedofs : FESpace or BitArray
  FESpace (with BilinearForm API) or BitArray (fes.FreeDofs(True)).
element_dofs : list[list[int]]
  Element-to-DOF mapping (matrix API only).
dof_types : list[int]
  0=wirebasket, 1=interface (matrix API only).
element_matrices : list[list[list[float]]]
  True element stiffness matrices (matrix API only).
coarse_inverse : str
  Coarse solver: "sparsecholesky" (default), "pardiso", "dense".
)raw_string");

  // ---- SparseSolvSolver factory ----
  m.def("SparseSolvSolver", [](shared_ptr<BaseMatrix> mat,
                                 const string& method, py::object freedofs,
                                 double tol, int maxiter, double shift,
                                 bool save_best_result, bool save_residual_history,
                                 bool printrates, bool conjugate,
                                 bool use_abmc, int abmc_block_size, int abmc_num_colors,
                                 bool abmc_reorder_spmv, bool abmc_use_rcm) {
    auto sp_freedofs = ExtractFreeDofs(freedofs);
    shared_ptr<BaseMatrix> result;
    if (mat->IsComplex()) {
      auto sp = dynamic_pointer_cast<SparseMatrix<Complex>>(mat);
      if (!sp) throw py::type_error("SparseSolvSolver: expected SparseMatrix");
      auto solver = make_shared<SparseSolvSolver<Complex>>(
          sp, method, sp_freedofs, tol, maxiter, shift,
          save_best_result, save_residual_history, printrates);
      solver->SetConjugate(conjugate);
      solver->SetUseABMC(use_abmc);
      solver->SetABMCBlockSize(abmc_block_size);
      solver->SetABMCNumColors(abmc_num_colors);
      solver->SetABMCReorderSpMV(abmc_reorder_spmv);
      solver->SetABMCUseRCM(abmc_use_rcm);
      result = solver;
    } else {
      auto sp = dynamic_pointer_cast<SparseMatrix<double>>(mat);
      if (!sp) throw py::type_error("SparseSolvSolver: expected SparseMatrix");
      auto solver = make_shared<SparseSolvSolver<double>>(
          sp, method, sp_freedofs, tol, maxiter, shift,
          save_best_result, save_residual_history, printrates);
      solver->SetUseABMC(use_abmc);
      solver->SetABMCBlockSize(abmc_block_size);
      solver->SetABMCNumColors(abmc_num_colors);
      solver->SetABMCReorderSpMV(abmc_reorder_spmv);
      solver->SetABMCUseRCM(abmc_use_rcm);
      result = solver;
    }
    return result;
  },
  py::arg("mat"),
  py::arg("method") = "ICCG",
  py::arg("freedofs") = py::none(),
  py::arg("tol") = 1e-10,
  py::arg("maxiter") = 1000,
  py::arg("shift") = 1.05,
  py::arg("save_best_result") = true,
  py::arg("save_residual_history") = false,
  py::arg("printrates") = false,
  py::arg("conjugate") = false,
  py::arg("use_abmc") = false,
  py::arg("abmc_block_size") = 4,
  py::arg("abmc_num_colors") = 4,
  py::arg("abmc_reorder_spmv") = false,
  py::arg("abmc_use_rcm") = false,
  R"raw_string(
Iterative solver (ICCG / SGSMRTR / CG). Auto-detects real/complex.

Can be used as BaseMatrix (inverse operator) or via Solve() for detailed results.

Parameters:

mat : SparseMatrix
  SPD matrix (real or complex).
method : str
  "ICCG", "SGSMRTR", or "CG".
freedofs : BitArray, optional
  Free DOFs.
tol : float
  Convergence tolerance (default: 1e-10).
maxiter : int
  Max iterations (default: 1000).
shift : float
  IC shift parameter (default: 1.05).
save_best_result : bool
  Track best solution (default: True).
save_residual_history : bool
  Record residual history (default: False).
printrates : bool
  Print convergence info (default: False).
conjugate : bool
  Conjugated inner product for Hermitian systems (default: False).

Properties (set after construction):
  auto_shift, diagonal_scaling, divergence_check, divergence_threshold,
  divergence_count, use_abmc, abmc_block_size, abmc_num_colors,
  abmc_reorder_spmv, abmc_use_rcm.
)raw_string");
}

// ============================================================================
// Public API: Single entry point for NGSolve integration
// ============================================================================

/// Register all SparseSolv Python bindings (typed classes + factory functions)
inline void ExportSparseSolvBindings(py::module& m) {
  ExportSparseSolvResult_impl(m);
  ExportSparseSolvTyped<double>(m, "D");
  ExportSparseSolvTyped<Complex>(m, "C");
  ExportSparseSolvFactories(m);
}

} // namespace ngla

#endif // NGSOLVE_SPARSESOLV_PYTHON_EXPORT_HPP
