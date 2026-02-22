# SparseSolv

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

Header-only C++17 iterative solver library for [NGSolve](https://ngsolve.org/) finite element applications.
Two complementary approaches for large-scale sparse linear systems:

- **BDDC preconditioner** — Element-by-element domain decomposition with wirebasket coarse space. Produces mesh-independent iteration counts. Implementation matches NGSolve's built-in BDDC and serves as a reference for developers building their own BDDC.
- **ICCG with ABMC multicolor ordering** — Incomplete Cholesky CG with parallel triangular solves via Algebraic Block Multi-Color ordering. Lower setup cost than BDDC; effective for well-conditioned problems.

Both support `double` and `std::complex<double>`, and are provided as a standalone pybind11 extension module (`sparsesolv_ngsolve`), separate from NGSolve's source tree.

Fork of [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv).

## Why SparseSolv?

NGSolve ships with direct solvers and built-in BDDC. SparseSolv adds three things that NGSolve does not provide out of the box:

1. **Transparent, standalone BDDC implementation** — NGSolve's built-in BDDC is production code embedded deep in the framework. SparseSolv reimplements the same algorithm as a readable, self-contained C++ header-only library. Iteration counts [match NGSolve's BDDC exactly](docs/03_sparsesolv_vs_ngsolve_bddc.ipynb) on the same problems, making it a reference for developers who want to study, modify, or extend BDDC.

2. **Robustness for electromagnetic (HCurl) problems** — Curl-curl FEM matrices are semi-definite, which causes standard IC factorization to break down. SparseSolv's auto-shift IC detects breakdown and adjusts automatically. More importantly, BDDC converges regardless of whether the source term is discretely divergence-free — a condition that [ICCG strictly requires but is not always easy to guarantee](docs/02_performance_comparison.ipynb).

3. **Parallel ICCG via ABMC multicolor ordering** — The triangular solve in IC preconditioning is inherently sequential. ABMC (Algebraic Block Multi-Color) ordering breaks this bottleneck, enabling parallel forward/backward substitution. This makes ICCG competitive for moderate problems where BDDC setup cost is not justified.

## When to Use What

| Problem | FE Space | Recommended | Why |
|---------|----------|-------------|-----|
| Poisson (high order) | H1 (order >= 3) | **BDDC+CG** | 2 iterations, mesh-independent |
| Poisson (low order) | H1 (order 1-2) | **ICCG** | BDDC setup cost not justified |
| Elasticity | VectorH1 | **BDDC+CG** | ICCG iteration count grows with refinement |
| Curl-curl (real) | HCurl (`nograds=True`) | **BDDC+CG** or **Shifted-ICCG** | BDDC robust to source formulation |
| Eddy current (complex) | HCurl (complex) | **BDDC+CG** (`conjugate=False`) | Complex-symmetric support |
| Small problems (< 1K DOFs) | any | Direct solver | Iterative solver overhead not justified |

See [tutorials](docs/tutorials.md) for complete examples with timing comparisons.

## Performance

Benchmarks on 3D HCurl curl-curl problems (8 threads, `nograds=True`).
Full details in [02_performance_comparison.ipynb](docs/02_performance_comparison.ipynb).

**Toroidal coil** (148K DOFs, order 2 — well-posed div-free source):

| Solver | Iterations | Wall Time | vs ICCG |
|--------|-----------|-----------|---------|
| ICCG | 513 | 13.1 s | 1.0x |
| ICCG + ABMC (8 colors) | 444 | 7.4 s | **1.8x** |
| BDDC | 47 | 4.2 s | **3.1x** |

ABMC parallelizes the triangular solve bottleneck, cutting ICCG wall time nearly in half.
BDDC goes further with mesh-independent convergence.

**Helical coil** (565K DOFs, order 2 — source formulation robustness):

| Source Formulation | ICCG | BDDC |
|---|---|---|
| Potential-based `J*v*dx` (not div-free) | 1000 iters, **diverged** | 33 iters, converged |
| Curl-based `T*curl(v)*dx` (div-free) | 161 iters, converged | 53 iters, converged |

ICCG requires the source to be discretely divergence-free — a constraint that is not always easy
to satisfy in practice. BDDC converges regardless of source formulation.

## Documentation

See [docs/](docs/) for detailed documentation (in Japanese):
- [Architecture](docs/architecture.md) — Source code structure and design
- [Algorithms](docs/algorithms.md) — Algorithm descriptions (BDDC, IC, SGS-MRTR, CG, ABMC)
- [BDDC Implementation Guide](docs/bddc_implementation_details.md) — Theory, pseudo-code, API, benchmarks
- [ABMC Implementation Guide](docs/abmc_implementation_details.md) — Parallel triangular solves, performance analysis
- [API Reference](docs/api_reference.md) — Python API reference
- [Tutorials](docs/tutorials.md) — Practical examples with all solver types
- [Development](docs/development.md) — Build, test, and development notes

### Benchmark Notebooks

| Notebook | Content |
|----------|---------|
| [01_shift_parameter.ipynb](docs/01_shift_parameter.ipynb) | IC shift parameter for semi-definite HCurl systems |
| [02_performance_comparison.ipynb](docs/02_performance_comparison.ipynb) | BDDC vs ICCG vs ICCG+ABMC performance |
| [03_sparsesolv_vs_ngsolve_bddc.ipynb](docs/03_sparsesolv_vs_ngsolve_bddc.ipynb) | SparseSolv BDDC = NGSolve BDDC equivalence |

## Features

### Preconditioners
- **BDDC** (Balancing Domain Decomposition by Constraints) — element-by-element construction, wirebasket coarse space, mesh-independent iterations
- **IC** (Incomplete Cholesky) — shifted IC(0) with auto-shift for semi-definite systems
- **SGS** (Symmetric Gauss-Seidel) — no factorization required

### Iterative Solvers
- **CG** (Conjugate Gradient) — for SPD systems, supports complex-symmetric (`conjugate=False`)
- **SGS-MRTR** — MRTR with built-in SGS using split formula

### Combined Methods
- **ICCG** — CG + IC preconditioner (with optional ABMC parallel triangular solves)
- **SGSMRTR** — SGS-MRTR (self-contained)

### Advanced Features
- Auto-shift IC decomposition for semi-definite matrices (curl-curl problems)
- Diagonal scaling for improved conditioning
- ABMC (Algebraic Block Multi-Color) ordering for parallel triangular solves
- Numerical breakdown detection with convergence-aware recovery
- Best-result tracking (returns best iterate if solver doesn't converge)
- Residual history recording

### Parallelism

All parallel operations are abstracted via `core/parallel.hpp` with compile-time dispatch:

| Build Configuration | Backend | Use Case |
|---|---|---|
| `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` | NGSolve TaskManager (`ngcore::ParallelFor/ParallelReduce`) | NGSolve integration |
| `_OPENMP` | OpenMP `#pragma omp parallel for` | Standalone with OpenMP |
| (neither) | Serial loops | Standalone without threading |

## Installation

### From Pre-built Wheel (Recommended)

Download the `.whl` file for your platform from the
[Releases](https://github.com/ksugahar/ngsolve-sparsesolv/releases) page, then install:

```bash
pip install sparsesolv_ngsolve-2.1.0-cp312-cp312-win_amd64.whl
```

Replace the filename with the one matching your Python version and OS.

### From Source

Requires [Git for Windows](https://gitforwindows.org/) (or equivalent), CMake 3.16+,
a C++17 compiler, and NGSolve (`pip install ngsolve` or built from source).

```bash
pip install git+https://github.com/ksugahar/ngsolve-sparsesolv.git
```

Or clone and build locally:

```bash
git clone https://github.com/ksugahar/ngsolve-sparsesolv.git
cd ngsolve-sparsesolv
pip install .
```

For development builds (editable install, manual CMake), see [docs/development.md](docs/development.md).

### Verify

```bash
python -c "from sparsesolv_ngsolve import SparseSolvSolver; print('OK')"
```

## NGSolve Usage

### Quick Start (ICCG)

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

### BDDC Preconditioner

```python
from sparsesolv_ngsolve import BDDCPreconditioner
from ngsolve.krylovspace import CGSolver

# BDDC takes BilinearForm + FESpace (not just the matrix)
pre = BDDCPreconditioner(a, fes, coarse_inverse="sparsecholesky")
inv = CGSolver(a.mat, pre, tol=1e-10)
gfu.vec.data = inv * f.vec
```

### ICCG with ABMC Parallel Ordering

```python
# ABMC enables parallel triangular solves in IC preconditioner
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10,
                          use_abmc=True,
                          abmc_block_size=4,
                          abmc_num_colors=4)
gfu.vec.data = solver * f.vec
```

The ABMC algorithm groups nearby rows into blocks (BFS aggregation), then colors the
block adjacency graph so that blocks with the same color have no lower-triangular
dependencies. During the triangular solve, colors are processed sequentially while
blocks within each color run in parallel.
On a 148K DOF HCurl problem with 8 threads, ABMC reduces ICCG wall time from 13.1s
to 7.4s (1.8x speedup) by parallelizing the triangular solve bottleneck.

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
    BDDCPreconditioner,            # BDDC (BilinearForm + FESpace API)
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
ngsolve-sparsesolv/
├── docs/                        # Documentation (Japanese)
├── include/sparsesolv/         # Header-only library
│   ├── sparsesolv.hpp          # Main header (convenience includes)
│   ├── core/                   # Types, config, matrix view, preconditioner base
│   │   ├── dense_matrix.hpp    # Small dense matrix with LU inverse (for BDDC)
│   │   ├── sparse_matrix_coo.hpp # COO sparse matrix (assembly)
│   │   ├── sparse_matrix_csr.hpp # CSR sparse matrix (storage)
│   │   ├── parallel.hpp        # Parallelism abstraction (TaskManager/OpenMP/serial)
│   │   ├── level_schedule.hpp  # Level scheduling for triangular solves
│   │   └── abmc_ordering.hpp   # ABMC ordering for parallel triangular solves
│   ├── preconditioners/        # IC, SGS, BDDC implementations
│   ├── solvers/                # CG, SGS-MRTR implementations
│   └── ngsolve/                # NGSolve BaseMatrix wrappers + pybind11 export
├── ngsolve/
│   └── python_module.cpp       # Standalone pybind11 module entry point
├── tests/
│   ├── test_sparsesolv.py      # Core solver/preconditioner tests
│   └── test_bddc.py            # BDDC preconditioner tests
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
   "MRTR法に基づく前処理付き反復法の数値実験",
   *電気学会研究会資料*, SA-12-64, 2012.
   — SGS-MRTR method: original domestic workshop paper.

3. Y. Tsuburaya, T. Mifune, T. Iwashita, E. Takahashi,
   "Minimum Residual-Like Methods for Solving Ax = b with Shift-and-Invert Enhanced
   Multi-Step MRTR",
   *IEEE Transactions on Magnetics*, Vol. 49, No. 5, pp. 1569–1572, 2013.
   — SGS-MRTR solver.

4. Y. Tsuburaya,
   "大規模電磁界問題の有限要素解析のための反復法の開発",
   *博士論文*, 京都大学, 2016.
   — Comprehensive reference on iterative solvers for large-scale electromagnetic FEM.

5. S. Hiruma, JP-MARs/SparseSolv,
   https://github.com/JP-MARs/SparseSolv
   — Original implementation (MPL 2.0 license).

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).

SparseSolv is provided as an independent module, separate from NGSolve.
It is built as a standalone pybind11 extension module that links against an installed NGSolve.
