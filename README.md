# SparseSolv

Header-only C++17 iterative solver library for sparse linear systems, with a standalone Python extension module for [NGSolve](https://ngsolve.org/).

Fork of [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv), specialized for NGSolve finite element applications.

## Policy

**SparseSolv is provided as an independent module, separate from NGSolve.**
It is NOT embedded into NGSolve's source tree. Instead, it is built as a standalone
pybind11 extension module (`sparsesolv_ngsolve.pyd`) that links against an installed NGSolve.

## Features

### Preconditioners
- **IC** (Incomplete Cholesky) - shifted IC(0) with auto-shift for semi-definite systems
- **SGS** (Symmetric Gauss-Seidel) - no factorization required

### Iterative Solvers
- **CG** (Conjugate Gradient) - for SPD systems
- **SGS-MRTR** - MRTR with built-in SGS using split formula (no separate preconditioner)

### Combined Methods
- **ICCG** - CG + IC preconditioner
- **SGSMRTR** - SGS-MRTR (self-contained)

### Advanced Features
- Auto-shift IC decomposition for semi-definite matrices (curl-curl problems)
- Diagonal scaling for improved conditioning
- Numerical breakdown detection with convergence-aware recovery
- Best-result tracking (returns best iterate if solver doesn't converge)
- Residual history recording
- ABMC (Algebraic Block Multi-Color) ordering for parallel triangular solves
- Template support for `double` and `std::complex<double>`

### Parallelism

All parallel operations are abstracted via `core/parallel.hpp` with compile-time dispatch:

| Build Configuration | Backend | Use Case |
|---|---|---|
| `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` | NGSolve TaskManager (`ngcore::ParallelFor/ParallelReduce`) | NGSolve integration |
| `_OPENMP` | OpenMP `#pragma omp parallel for` | Standalone with OpenMP |
| (neither) | Serial loops | Standalone without threading |

## Installation

### Prerequisites

- NGSolve installed (from source or pip) with CMake config files
- CMake 3.16+
- C++17 compiler (MSVC 2022, GCC 10+, Clang 10+)
- pybind11 (fetched automatically by CMake if not found)

### Step 1: Build

```bash
git clone https://github.com/ksugahar/ngsolve-sparsesolv.git
cd ngsolve-sparsesolv
mkdir build && cd build

# Point to your NGSolve install directory containing cmake/ config files
cmake .. -DSPARSESOLV_BUILD_NGSOLVE=ON \
         -DNGSOLVE_INSTALL_DIR=/path/to/ngsolve/install/cmake

cmake --build . --config Release
```

This produces `sparsesolv_ngsolve.pyd` (Windows) or `sparsesolv_ngsolve.so` (Linux/macOS).

### Step 2: Install

Copy the built module to your Python path. For example:

```bash
# Find your NGSolve site-packages
SITE_PACKAGES=$(python -c "import ngsolve, pathlib; print(pathlib.Path(ngsolve.__file__).parent.parent)")

# Create package directory and copy
mkdir -p "$SITE_PACKAGES/sparsesolv_ngsolve"
cp build/Release/sparsesolv_ngsolve*.pyd "$SITE_PACKAGES/sparsesolv_ngsolve/"  # Windows
# cp build/sparsesolv_ngsolve*.so "$SITE_PACKAGES/sparsesolv_ngsolve/"         # Linux/macOS
echo "from .sparsesolv_ngsolve import *" > "$SITE_PACKAGES/sparsesolv_ngsolve/__init__.py"
```

### Step 3: Verify

```bash
python -c "import ngsolve; from sparsesolv_ngsolve import SparseSolvSolver; print('OK')"
```

Note: `import ngsolve` must be called before `import sparsesolv_ngsolve` to ensure
NGSolve's shared libraries are loaded.

## NGSolve Usage

### Quick Start

```python
from ngsolve import *
from sparsesolv_ngsolve import SparseSolvSolver

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
fes = H1(mesh, order=2, dirichlet="left|right|top|bottom")
u, v = fes.TnT()

a = BilinearForm(fes)
a += grad(u)*grad(v)*dx
a.Assemble()

f = LinearForm(fes)
f += 1*v*dx
f.Assemble()

gfu = GridFunction(fes)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10)
gfu.vec.data = solver * f.vec
```

### Preconditioners with NGSolve's CGSolver

```python
from sparsesolv_ngsolve import ICPreconditioner, SGSPreconditioner
from ngsolve.krylovspace import CGSolver

# IC preconditioner + NGSolve CG
pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=2000)
gfu.vec.data = inv * f.vec
```

### 3D Curl-Curl (Semi-Definite System)

```python
from netgen.occ import Box, Pnt

box = Box(Pnt(0,0,0), Pnt(1,1,1))
for face in box.faces:
    face.name = "outer"
mesh = Mesh(box.GenerateMesh(maxh=0.3))

fes = HCurl(mesh, order=1, dirichlet="outer", nograds=True)
u, v = fes.TnT()

a = BilinearForm(fes)
a += curl(u)*curl(v)*dx
a.Assemble()

f = LinearForm(fes)
f += CF((0,0,1))*v*dx
f.Assemble()

# Auto-shift IC handles semi-definite curl-curl matrix
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(),
                          tol=1e-8, maxiter=2000, shift=1.0)
solver.auto_shift = True
solver.diagonal_scaling = True

gfu = GridFunction(fes)
gfu.vec.data = solver * f.vec
```

### Complex-Symmetric vs Hermitian Systems

SparseSolv supports both complex-symmetric and Hermitian systems via the
`conjugate` parameter:

```python
# Complex-symmetric (A^T = A): e.g., eddy current (curl-curl + i*sigma*mass)
# Default: conjugate=False (unconjugated inner product a^T * b)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-8)

# Hermitian (A^H = A): e.g., real-coefficient problem in complex FE space
# Set conjugate=True (conjugated inner product a^H * b)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-8,
                          conjugate=True)
```

This corresponds to NGSolve's `CGSolver(conjugate=True/False)`.

### ABMC Ordering (Parallel Triangular Solves)

ABMC (Algebraic Block Multi-Color) ordering enables parallel execution of
triangular solves in the IC preconditioner. Without ABMC, triangular solves use
level scheduling, which can have limited parallelism for FEM matrices with deep
dependency chains.

```python
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10,
                          use_abmc=True,
                          abmc_block_size=4,
                          abmc_num_colors=4)
```

The algorithm groups nearby rows into blocks (BFS aggregation), then colors the
block adjacency graph so that blocks with the same color have no lower-triangular
dependencies. During the triangular solve:
- Colors are processed sequentially (inter-color dependencies)
- Blocks within the same color are processed in parallel
- Rows within a block are processed sequentially

The reordering is applied internally within the preconditioner; the CG solver and
SpMV operate on the original matrix ordering.

### SparseSolvSolver Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | str | `"ICCG"` | Solver method: `ICCG`, `SGSMRTR`, `CG` |
| `tol` | float | `1e-10` | Convergence tolerance |
| `maxiter` | int | `1000` | Maximum iterations |
| `shift` | float | `1.05` | IC shift parameter (stability) |
| `auto_shift` | bool | `False` | Auto-increase shift on IC breakdown |
| `diagonal_scaling` | bool | `False` | Scale matrix by diagonal |
| `save_best_result` | bool | `True` | Track and restore best iterate |
| `save_residual_history` | bool | `False` | Record residual per iteration |
| `conjugate` | bool | `False` | Conjugated inner product for Hermitian systems |
| `divergence_check` | bool | `False` | Early termination on stagnation |
| `printrates` | bool | `False` | Print convergence info to stdout |
| `use_abmc` | bool | `False` | Enable ABMC ordering for parallel triangular solves |
| `abmc_block_size` | int | `4` | Rows per block in ABMC aggregation |
| `abmc_num_colors` | int | `4` | Target number of colors for ABMC coloring |

### Available Python Classes

```python
from sparsesolv_ngsolve import (
    # Factory functions (auto-dispatch real/complex based on mat.IsComplex())
    ICPreconditioner,              # Use with NGSolve's CGSolver
    SGSPreconditioner,             # Use with NGSolve's CGSolver
    SparseSolvSolver,              # Standalone solver (ICCG, SGSMRTR)

    SparseSolvResult,              # Solve result object
)
```

Factory functions automatically dispatch to the correct type (`double` or `complex<double>`)
based on the matrix type.

## Standalone Usage (C++)

```cpp
#include <sparsesolv/sparsesolv.hpp>

// Create CSR matrix view (zero-copy)
sparsesolv::SparseMatrixView<double> A(rows, cols, row_ptr, col_idx, values);

// Solve with ICCG
sparsesolv::SolverConfig config;
config.tolerance = 1e-10;
config.max_iterations = 1000;

auto result = sparsesolv::solve_iccg(A, b, x, size, config);
if (result.converged) {
    // solution in x
}
```

## Directory Structure

```
SparseSolv/
├── include/sparsesolv/         # Header-only library
│   ├── sparsesolv.hpp          # Main header (convenience includes)
│   ├── core/                   # Types, config, matrix view, preconditioner base
│   │   ├── parallel.hpp        # Parallelism abstraction (TaskManager/OpenMP/serial)
│   │   ├── level_schedule.hpp  # Level scheduling for triangular solves
│   │   └── abmc_ordering.hpp   # ABMC ordering for parallel triangular solves
│   ├── preconditioners/        # IC, SGS implementations
│   ├── solvers/                # CG, SGS-MRTR implementations
│   └── ngsolve/                # NGSolve BaseMatrix wrappers + pybind11 export
├── ngsolve/
│   ├── python_module.cpp       # Standalone pybind11 module entry point
│   └── README.md               # Integration reference
├── tests/                      # Test suite
│   └── test_sparsesolv.py      # NGSolve pytest
├── CMakeLists.txt              # Header-only library + NGSolve module build
└── LICENSE
```

## References

1. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel Multi-Threaded
   Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE 26th International Parallel and Distributed Processing Symposium (IPDPS)*, 2012.
   — ABMC ordering algorithm.

2. Y. Tsuburaya, T. Mifune, T. Iwashita, E. Takahashi,
   "Minimum Residual-Like Methods for Solving Ax = b with Shift-and-Invert Enhanced
   Multi-Step MRTR",
   *IEEE Transactions on Magnetics*, Vol. 49, No. 5, pp. 1569–1572, 2013.
   — SGS-MRTR solver.

3. Y. Tsuburaya,
   "大規模電磁界問題の有限要素解析のための反復法の開発",
   *博士論文*, 京都大学, 2016.
   — Comprehensive reference on iterative solvers for large-scale electromagnetic FEM.

4. S. Hiruma, JP-MARs/SparseSolv,
   https://github.com/JP-MARs/SparseSolv
   — Original implementation (MPL 2.0 license).

## License

See [LICENSE](LICENSE) for details.
