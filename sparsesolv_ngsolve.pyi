"""Type stubs for sparsesolv_ngsolve — SparseSolv iterative solvers for NGSolve.

Provides IC/SGS/BDDC preconditioners and ICCG/SGSMRTR iterative solvers
for use with NGSolve's sparse linear algebra.
"""

from ngsolve import BaseMatrix, BaseVector, BilinearForm, BitArray, FESpace

__all__ = [
    "SparseSolvResult",
    "ICPreconditionerD",
    "ICPreconditionerC",
    "SGSPreconditionerD",
    "SGSPreconditionerC",
    "BDDCPreconditionerD",
    "BDDCPreconditionerC",
    "SparseSolvSolverD",
    "SparseSolvSolverC",
    "ICPreconditioner",
    "SGSPreconditioner",
    "BDDCPreconditioner",
    "SparseSolvSolver",
]

# =============================================================================
# Result type
# =============================================================================

class SparseSolvResult:
    """Result of a SparseSolv iterative solve."""

    converged: bool
    """Whether the solver converged within tolerance."""
    iterations: int
    """Number of iterations performed."""
    final_residual: float
    """Final relative residual (or best residual if save_best_result enabled)."""
    residual_history: list[float]
    """Residual at each iteration (empty unless save_residual_history enabled)."""

# =============================================================================
# IC Preconditioner types
# =============================================================================

class ICPreconditionerD(BaseMatrix):
    """Incomplete Cholesky preconditioner for real (double) matrices."""

    shift: float
    """IC shift parameter."""

    def Update(self) -> None:
        """Recompute the IC factorization with current shift."""
        ...

class ICPreconditionerC(BaseMatrix):
    """Incomplete Cholesky preconditioner for complex matrices."""

    shift: float
    """IC shift parameter."""

    def Update(self) -> None:
        """Recompute the IC factorization with current shift."""
        ...

# =============================================================================
# SGS Preconditioner types
# =============================================================================

class SGSPreconditionerD(BaseMatrix):
    """Symmetric Gauss-Seidel preconditioner for real (double) matrices."""

    def Update(self) -> None:
        """Recompute the SGS factorization."""
        ...

class SGSPreconditionerC(BaseMatrix):
    """Symmetric Gauss-Seidel preconditioner for complex matrices."""

    def Update(self) -> None:
        """Recompute the SGS factorization."""
        ...

# =============================================================================
# BDDC Preconditioner types
# =============================================================================

class BDDCPreconditionerD(BaseMatrix):
    """BDDC preconditioner for real (double) matrices."""

    @property
    def num_wirebasket_dofs(self) -> int:
        """Number of wirebasket DOFs in domain decomposition."""
        ...

    @property
    def num_interface_dofs(self) -> int:
        """Number of interface DOFs in domain decomposition."""
        ...

class BDDCPreconditionerC(BaseMatrix):
    """BDDC preconditioner for complex matrices."""

    @property
    def num_wirebasket_dofs(self) -> int:
        """Number of wirebasket DOFs in domain decomposition."""
        ...

    @property
    def num_interface_dofs(self) -> int:
        """Number of interface DOFs in domain decomposition."""
        ...

# =============================================================================
# Solver types
# =============================================================================

class SparseSolvSolverD(BaseMatrix):
    """Iterative solver for real (double) matrices."""

    method: str
    """Solver method: ``"ICCG"``, ``"SGSMRTR"``, or ``"CG"``."""
    tol: float
    """Convergence tolerance (relative residual)."""
    maxiter: int
    """Maximum iterations."""
    shift: float
    """IC shift parameter."""
    auto_shift: bool
    """Auto-increase shift on IC breakdown (for semi-definite systems)."""
    diagonal_scaling: bool
    """Diagonal scaling for improved conditioning."""
    save_best_result: bool
    """Track and return the best iterate found."""
    save_residual_history: bool
    """Record residual at each iteration."""
    printrates: bool
    """Print convergence info to stdout."""
    conjugate: bool
    """Use conjugated inner product for Hermitian systems."""
    divergence_check: bool
    """Enable stagnation-based divergence detection."""
    divergence_threshold: float
    """Threshold for divergence detection."""
    divergence_count: int
    """Iterations before declaring divergence."""
    use_abmc: bool
    """Enable ABMC ordering for parallel triangular solves."""
    abmc_block_size: int
    """Rows per block in ABMC aggregation."""
    abmc_num_colors: int
    """Target number of colors for ABMC coloring."""
    abmc_reorder_spmv: bool
    """Reorder SpMV in ABMC space (experimental)."""
    abmc_use_rcm: bool
    """Use RCM ordering (experimental)."""

    @property
    def last_result(self) -> SparseSolvResult:
        """Result from the last Solve() call."""
        ...

    def Solve(self, rhs: BaseVector, sol: BaseVector) -> SparseSolvResult:
        """Solve Ax = b.

        Args:
            rhs: Right-hand side vector.
            sol: Solution vector (input: initial guess, output: solution).

        Returns:
            SparseSolvResult with convergence info.
        """
        ...

class SparseSolvSolverC(BaseMatrix):
    """Iterative solver for complex matrices."""

    method: str
    tol: float
    maxiter: int
    shift: float
    auto_shift: bool
    diagonal_scaling: bool
    save_best_result: bool
    save_residual_history: bool
    printrates: bool
    conjugate: bool
    divergence_check: bool
    divergence_threshold: float
    divergence_count: int
    use_abmc: bool
    abmc_block_size: int
    abmc_num_colors: int
    abmc_reorder_spmv: bool
    abmc_use_rcm: bool

    @property
    def last_result(self) -> SparseSolvResult: ...

    def Solve(self, rhs: BaseVector, sol: BaseVector) -> SparseSolvResult:
        """Solve Ax = b.

        Args:
            rhs: Right-hand side vector.
            sol: Solution vector (input: initial guess, output: solution).

        Returns:
            SparseSolvResult with convergence info.
        """
        ...

# =============================================================================
# Factory functions (auto-dispatch real/complex based on mat.IsComplex())
# =============================================================================

def ICPreconditioner(
    mat: BaseMatrix,
    freedofs: BitArray | None = None,
    shift: float = 1.05,
) -> ICPreconditionerD | ICPreconditionerC:
    """Incomplete Cholesky (IC) Preconditioner.

    Auto-dispatches to real/complex based on ``mat.IsComplex()``.
    Calls ``Update()`` automatically on construction.

    Args:
        mat: SPD sparse matrix (real or complex).
        freedofs: Free DOFs. Constrained DOFs treated as identity.
        shift: Shift parameter for stability (default: 1.05).
    """
    ...

def SGSPreconditioner(
    mat: BaseMatrix,
    freedofs: BitArray | None = None,
) -> SGSPreconditionerD | SGSPreconditionerC:
    """Symmetric Gauss-Seidel (SGS) Preconditioner.

    Auto-dispatches to real/complex based on ``mat.IsComplex()``.
    Calls ``Update()`` automatically on construction.

    Args:
        mat: SPD sparse matrix (real or complex).
        freedofs: Free DOFs. Constrained DOFs treated as identity.
    """
    ...

def BDDCPreconditioner(
    a: BilinearForm,
    fes: FESpace,
    coarse_inverse: str = "sparsecholesky",
) -> BDDCPreconditionerD | BDDCPreconditionerC:
    """BDDC (Balancing Domain Decomposition by Constraints) Preconditioner.

    Extracts element matrices from BilinearForm and builds element-by-element
    BDDC with wirebasket coarse space. Produces mesh-independent iteration counts.

    Auto-dispatches to real/complex based on ``a.mat.IsComplex()``.

    Args:
        a: Assembled BilinearForm.
        fes: Finite element space (for DOF classification).
        coarse_inverse: Coarse solver type —
            ``"sparsecholesky"`` (default), ``"pardiso"``, or ``"dense"``.
    """
    ...

def SparseSolvSolver(
    mat: BaseMatrix,
    method: str = "ICCG",
    freedofs: BitArray | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
    shift: float = 1.05,
    save_best_result: bool = True,
    save_residual_history: bool = False,
    printrates: bool = False,
    conjugate: bool = False,
    use_abmc: bool = False,
    abmc_block_size: int = 4,
    abmc_num_colors: int = 4,
    abmc_reorder_spmv: bool = False,
    abmc_use_rcm: bool = False,
) -> SparseSolvSolverD | SparseSolvSolverC:
    """Iterative solver (ICCG / SGSMRTR / CG).

    Can be used as a BaseMatrix inverse operator (``solver * rhs``) or
    via ``Solve()`` for detailed convergence results.

    Auto-dispatches to real/complex based on ``mat.IsComplex()``.

    Args:
        mat: SPD sparse matrix (real or complex).
        method: ``"ICCG"``, ``"SGSMRTR"``, or ``"CG"``.
        freedofs: Free DOFs.
        tol: Convergence tolerance (default: 1e-10).
        maxiter: Maximum iterations (default: 1000).
        shift: IC shift parameter (default: 1.05).
        save_best_result: Track and return best iterate (default: True).
        save_residual_history: Record residual per iteration (default: False).
        printrates: Print convergence info to stdout (default: False).
        conjugate: Conjugated inner product for Hermitian systems (default: False).
        use_abmc: Enable ABMC parallel ordering (default: False).
        abmc_block_size: Rows per ABMC block (default: 4).
        abmc_num_colors: Target ABMC colors (default: 4).
        abmc_reorder_spmv: Reorder SpMV in ABMC space (default: False).
        abmc_use_rcm: Use RCM ordering (default: False).
    """
    ...
