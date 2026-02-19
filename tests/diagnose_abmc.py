"""
Diagnose ABMC vs Level-Scheduling performance.

Measures:
1. Level scheduling stats: num levels, rows per level distribution
2. ABMC stats: num colors, blocks per color, rows per block
3. Component timing: setup, solve
"""
import time
import sys
import numpy as np
from collections import deque
from scipy.sparse import coo_matrix, csr_matrix
from ngsolve import *
from sparsesolv_ngsolve import SparseSolvSolver


def ngsolve_to_scipy_csr(mat):
    """Convert NGSolve SparseMatrix to scipy CSR via COO."""
    rows_arr, cols_arr, vals_arr = mat.COO()
    n = mat.height
    rows = np.array([rows_arr[i] for i in range(len(rows_arr))], dtype=np.int32)
    cols = np.array([cols_arr[i] for i in range(len(cols_arr))], dtype=np.int32)
    vals = np.array([vals_arr[i] for i in range(len(vals_arr))], dtype=np.float64)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def extract_freedof_submatrix(A_csr, freedofs, n):
    """Extract submatrix corresponding to free DOFs."""
    free_idx = np.array([i for i in range(n) if freedofs[i]], dtype=np.int32)
    # Use scipy fancy indexing
    A_sub = A_csr[free_idx][:, free_idx]
    return A_sub, free_idx


def analyze_level_scheduling(row_ptr, col_idx, n):
    """Compute level scheduling statistics for lower triangular solve."""
    row_level = np.zeros(n, dtype=np.int32)

    for i in range(n):
        max_dep = -1
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[k]
            if j < i:
                if row_level[j] > max_dep:
                    max_dep = int(row_level[j])
        row_level[i] = max_dep + 1

    max_level = int(row_level.max())
    num_levels = max_level + 1

    level_sizes = np.zeros(num_levels, dtype=np.int32)
    for i in range(n):
        level_sizes[row_level[i]] += 1

    return {
        'num_levels': num_levels,
        'level_sizes': level_sizes,
        'min_rows_per_level': int(level_sizes.min()),
        'max_rows_per_level': int(level_sizes.max()),
        'mean_rows_per_level': float(level_sizes.mean()),
        'median_rows_per_level': float(np.median(level_sizes)),
    }


def analyze_abmc(row_ptr, col_idx, n, block_size=4, target_colors=4):
    """Simulate ABMC BFS blocking + greedy coloring (Python reimplementation)."""
    # Stage 1: BFS aggregation
    block_assign = np.full(n, -1, dtype=np.int32)
    blocks = []
    next_unassigned = 0

    while next_unassigned < n:
        while next_unassigned < n and block_assign[next_unassigned] >= 0:
            next_unassigned += 1
        if next_unassigned >= n:
            break

        blk_id = len(blocks)
        current_block = [next_unassigned]
        block_assign[next_unassigned] = blk_id

        queue = deque()
        for k in range(row_ptr[next_unassigned], row_ptr[next_unassigned + 1]):
            j = int(col_idx[k])
            if j != next_unassigned and block_assign[j] < 0:
                queue.append(j)

        while queue and len(current_block) < block_size:
            row = queue.popleft()
            if block_assign[row] >= 0:
                continue
            block_assign[row] = blk_id
            current_block.append(row)
            if len(current_block) >= block_size:
                break
            for k in range(row_ptr[row], row_ptr[row + 1]):
                j = int(col_idx[k])
                if block_assign[j] < 0:
                    queue.append(j)

        blocks.append(current_block)

    # Stage 2: Block adjacency
    num_blocks = len(blocks)
    block_neighbors = [set() for _ in range(num_blocks)]
    for i in range(n):
        bi = int(block_assign[i])
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = int(col_idx[k])
            bj = int(block_assign[j])
            if bi != bj:
                block_neighbors[bi].add(bj)

    # Stage 3: Greedy coloring
    block_color = np.full(num_blocks, -1, dtype=np.int32)
    num_colors = target_colors

    for b in range(num_blocks):
        lower_count = sum(1 for nb in block_neighbors[b] if nb < b)
        if lower_count >= num_colors:
            num_colors = lower_count + 1

    color_lists = [[] for _ in range(num_colors)]
    candidate = 0

    for b in range(num_blocks):
        assigned = False
        for trial in range(num_colors):
            test_color = (candidate + trial) % num_colors
            conflict = any(nb < b and block_color[nb] == test_color
                           for nb in block_neighbors[b])
            if not conflict:
                block_color[b] = test_color
                color_lists[test_color].append(b)
                candidate = (test_color + 1) % num_colors
                assigned = True
                break
        if not assigned:
            block_color[b] = num_colors
            color_lists.append([b])
            num_colors += 1

    color_lists = [cl for cl in color_lists if len(cl) > 0]
    actual_colors = len(color_lists)
    blocks_per_color = [len(cl) for cl in color_lists]
    block_sizes = [len(b) for b in blocks]

    # Compute average number of neighbors per block
    avg_neighbors = np.mean([len(bn) for bn in block_neighbors])

    return {
        'num_blocks': num_blocks,
        'actual_colors': actual_colors,
        'blocks_per_color': blocks_per_color,
        'min_blocks_per_color': min(blocks_per_color),
        'max_blocks_per_color': max(blocks_per_color),
        'mean_blocks_per_color': np.mean(blocks_per_color),
        'block_sizes': block_sizes,
        'min_block_size': min(block_sizes),
        'max_block_size': max(block_sizes),
        'mean_block_size': np.mean(block_sizes),
        'avg_neighbors': avg_neighbors,
    }


def run_diagnosis(h):
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

    print(f"\n{'='*70}")
    print(f"Mesh h={h}, ndof={ndof}, nfree={nfree}")
    print(f"{'='*70}")

    # Extract CSR via scipy
    print("\nExtracting CSR pattern...")
    t0 = time.perf_counter()
    A_full = ngsolve_to_scipy_csr(a.mat)
    A_sub, free_idx = extract_freedof_submatrix(A_full, fes.FreeDofs(), ndof)
    n_active = A_sub.shape[0]
    t1 = time.perf_counter()
    print(f"  CSR extraction: {t1-t0:.3f} s, n={n_active}, nnz={A_sub.nnz}")

    # Get lower triangular for level scheduling analysis
    from scipy.sparse import tril
    A_lower = tril(A_sub, format='csr')

    row_ptr = A_lower.indptr.astype(np.int32)
    col_idx_arr = A_lower.indices.astype(np.int32)

    # Level scheduling analysis
    print("\n--- Level Scheduling Analysis ---")
    ls_stats = analyze_level_scheduling(row_ptr, col_idx_arr, n_active)
    print(f"  Num levels: {ls_stats['num_levels']}")
    print(f"  Rows per level: min={ls_stats['min_rows_per_level']}, "
          f"max={ls_stats['max_rows_per_level']}, "
          f"mean={ls_stats['mean_rows_per_level']:.1f}, "
          f"median={ls_stats['median_rows_per_level']:.1f}")

    sizes = ls_stats['level_sizes']
    print(f"  Level size distribution:")
    bins = [1, 2, 5, 10, 50, 100, 500, 1000, 10000]
    for i in range(len(bins) - 1):
        count = np.sum((sizes >= bins[i]) & (sizes < bins[i + 1]))
        if count > 0:
            pct = count / len(sizes) * 100
            print(f"    [{bins[i]:>5}, {bins[i+1]:>5}): {count:>4} levels ({pct:.1f}%)")
    count = np.sum(sizes >= bins[-1])
    if count > 0:
        pct = count / len(sizes) * 100
        print(f"    [{bins[-1]:>5},   inf): {count:>4} levels ({pct:.1f}%)")

    avg_par = n_active / ls_stats['num_levels']
    print(f"  Average parallelism: {avg_par:.1f} rows/level")
    print(f"  Theoretical speedup (Amdahl): {n_active / ls_stats['num_levels']:.1f}x")

    # ABMC analysis - full symmetric pattern
    full_row_ptr = A_sub.indptr.astype(np.int32)
    full_col_idx = A_sub.indices.astype(np.int32)

    for block_size in [4, 8, 16]:
        for target_colors in [4]:
            print(f"\n--- ABMC Analysis (block_size={block_size}, colors={target_colors}) ---")
            abmc_stats = analyze_abmc(full_row_ptr, full_col_idx, n_active,
                                       block_size, target_colors)
            print(f"  Num blocks: {abmc_stats['num_blocks']}")
            print(f"  Actual colors: {abmc_stats['actual_colors']}")
            bpc = abmc_stats['blocks_per_color']
            print(f"  Blocks per color: {bpc}")
            print(f"    min={abmc_stats['min_blocks_per_color']}, "
                  f"max={abmc_stats['max_blocks_per_color']}, "
                  f"mean={abmc_stats['mean_blocks_per_color']:.1f}")
            print(f"  Block size: min={abmc_stats['min_block_size']}, "
                  f"max={abmc_stats['max_block_size']}, "
                  f"mean={abmc_stats['mean_block_size']:.1f}")
            print(f"  Avg neighbors per block: {abmc_stats['avg_neighbors']:.1f}")

            # Effective parallelism analysis
            seq_steps = abmc_stats['actual_colors']
            rows_per_step = n_active / seq_steps
            print(f"  Sequential steps (colors): {seq_steps}")
            print(f"  Avg rows per step: {rows_per_step:.0f}")

            # Compare with level scheduling
            ratio_steps = seq_steps / ls_stats['num_levels']
            print(f"  ABMC/LS step ratio: {ratio_steps:.2f}x "
                  f"({seq_steps} ABMC steps vs {ls_stats['num_levels']} LS levels)")

    # Bandwidth analysis (cache locality)
    print(f"\n--- Cache Locality (Bandwidth) Analysis ---")
    # Original matrix bandwidth
    orig_bandwidths = []
    for i in range(n_active):
        for k in range(A_sub.indptr[i], A_sub.indptr[i + 1]):
            j = A_sub.indices[k]
            orig_bandwidths.append(abs(i - j))
    orig_bw = np.array(orig_bandwidths)
    print(f"  Original ordering:")
    print(f"    Max bandwidth: {orig_bw.max()}")
    print(f"    Mean |i-j|: {orig_bw.mean():.1f}")
    print(f"    Median |i-j|: {np.median(orig_bw):.1f}")

    # Solver timing
    print(f"\n--- Solver Timing ---")

    gfu = GridFunction(fes)

    # Warmup + first solve
    for label, use_abmc, bs, nc in [
        ("Level-Sched", False, 0, 0),
        ("ABMC(bs=4)", True, 4, 4),
        ("ABMC(bs=8)", True, 8, 4),
        ("ABMC(bs=16)", True, 16, 4),
    ]:
        kwargs = dict(method="ICCG", freedofs=fes.FreeDofs(),
                      tol=1e-10, maxiter=10000, use_abmc=use_abmc)
        if use_abmc:
            kwargs['abmc_block_size'] = bs
            kwargs['abmc_num_colors'] = nc

        solver = SparseSolvSolver(a.mat, **kwargs)

        # Warmup
        gfu.vec[:] = 0
        gfu.vec.data = solver * f.vec

        # Timed runs
        num_runs = 5
        times = []
        for _ in range(num_runs):
            gfu.vec[:] = 0
            t0 = time.perf_counter()
            gfu.vec.data = solver * f.vec
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_t = np.mean(times)
        iters = solver.last_result.iterations
        ms_per_iter = avg_t / iters * 1000
        print(f"  {label:>14}: {avg_t:.4f} s, {iters:>3} iters, "
              f"{ms_per_iter:.3f} ms/iter")


if __name__ == "__main__":
    mesh_sizes = [0.12, 0.06]

    print("\n" + "=" * 70)
    print("SERIAL ANALYSIS")
    print("=" * 70)
    for h in mesh_sizes:
        run_diagnosis(h)

    nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"\n\n{'='*70}")
    print(f"TASKMANAGER ({nthreads} threads) ANALYSIS")
    print(f"{'='*70}")
    with TaskManager(pajetrace=False):
        for h in mesh_sizes:
            run_diagnosis(h)
