"""
Benchmark: Level-Scheduling vs ABMC (old: per-iteration permute) vs ABMC-Reordered (new: CG in reordered space).
"""
import time
import sys
from ngsolve import *
from sparsesolv_ngsolve import SparseSolvSolver


def run_solve(h, mode, abmc_block_size=8, num_runs=3):
    """
    mode: "ls" = level scheduling, "abmc_old" = ABMC per-iter permute (not available anymore),
          "abmc" = ABMC with CG in reordered space (default when use_abmc=True)
    """
    mesh = Mesh(unit_cube.GenerateMesh(maxh=h))
    fes = H1(mesh, order=2, dirichlet="left|right|top|bottom|front|back")
    u, v = fes.TnT()

    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
    a.Assemble()

    f = LinearForm(fes)
    f += 1 * v * dx
    f.Assemble()

    ndof = fes.ndof
    nfree = sum(1 for i in range(ndof) if fes.FreeDofs()[i])

    use_abmc = mode != "ls"
    solver = SparseSolvSolver(a.mat, method="ICCG",
                               freedofs=fes.FreeDofs(),
                               tol=1e-10, maxiter=10000,
                               use_abmc=use_abmc,
                               abmc_block_size=abmc_block_size,
                               abmc_num_colors=4)

    gfu = GridFunction(fes)

    # Warmup (includes setup + first solve)
    gfu.vec[:] = 0
    gfu.vec.data = solver * f.vec

    times = []
    for _ in range(num_runs):
        gfu.vec[:] = 0
        t0 = time.perf_counter()
        gfu.vec.data = solver * f.vec
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    iters = solver.last_result.iterations
    return ndof, nfree, avg_time, iters


def run_benchmark(label, mesh_sizes):
    print(f"\n=== {label} ===\n")
    print(f"{'ndof':>8} {'nfree':>8} | {'Level-Sched':>14} {'it':>4} | "
          f"{'ABMC-Reord':>14} {'it':>4} | {'ratio':>7}")
    print("-" * 78)

    for h in mesh_sizes:
        ndof, nfree, t_ls, it_ls = run_solve(h, mode="ls")
        _, _, t_abmc, it_abmc = run_solve(h, mode="abmc")

        ratio = t_ls / t_abmc if t_abmc > 0 else 0
        print(f"{ndof:>8d} {nfree:>8d} | "
              f"{t_ls:>11.4f} s {it_ls:>4d} | "
              f"{t_abmc:>11.4f} s {it_abmc:>4d} | "
              f"{ratio:>6.2f}x")


if __name__ == "__main__":
    mesh_sizes = [0.2, 0.12, 0.08, 0.06, 0.04]

    # Serial
    run_benchmark("Serial (1 thread)", mesh_sizes)

    # TaskManager
    nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"\nStarting TaskManager with {nthreads} threads...")
    with TaskManager(pajetrace=False):
        run_benchmark(f"TaskManager ({nthreads} threads)", mesh_sizes)
