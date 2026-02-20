"""
Tests for SparseSolv BDDC preconditioner.

Tests BDDC against NGSolve's built-in BDDC on 3D problems.
"""

import pytest
import numpy as np

import ngsolve
from ngsolve import *
from ngsolve.krylovspace import CGSolver
from netgen.occ import Box, Pnt
from sparsesolv_ngsolve import BDDCPreconditioner


def get_bddc_info(fes):
    """Extract element DOF mapping and DOF classification for BDDC.

    Returns:
        element_dofs: List[List[int]] - element-to-DOF mapping
        dof_types: List[int] - 0=wirebasket, 1=interface
    """
    element_dofs = []
    for el in fes.Elements(VOL):
        element_dofs.append([d for d in el.dofs if d >= 0])
    dof_types = []
    for d in range(fes.ndof):
        ct = fes.CouplingType(d)
        if ct == COUPLING_TYPE.WIREBASKET_DOF:
            dof_types.append(0)
        else:
            dof_types.append(1)
    return element_dofs, dof_types


def create_3d_poisson(maxh=0.4, order=3):
    """Create 3D Poisson problem on unit cube."""
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    for face in box.faces:
        face.name = "outer"
    mesh = box.GenerateMesh(maxh=maxh)

    fes = H1(mesh, order=order, dirichlet="outer")
    u, v = fes.TnT()

    a = BilinearForm(fes)
    a += InnerProduct(grad(u), grad(v)) * dx
    a.Assemble()

    f = LinearForm(fes)
    f += 1 * v * dx
    f.Assemble()

    return mesh, fes, a, f


def test_bddc_3d_poisson():
    """Test SparseSolv BDDC on 3D Poisson with H1 order=3."""
    mesh, fes, a, f = create_3d_poisson(maxh=0.4, order=3)
    ndof = fes.ndof
    element_dofs, dof_types = get_bddc_info(fes)

    n_wb = sum(1 for t in dof_types if t == 0)
    n_if = sum(1 for t in dof_types if t == 1)
    print(f"\nDOF={ndof}, WB={n_wb}, IF={n_if}")

    # SparseSolv BDDC
    pre = BDDCPreconditioner(
        a.mat, freedofs=fes.FreeDofs(True),
        element_dofs=element_dofs,
        dof_types=dof_types)

    gfu = GridFunction(fes)
    inv = CGSolver(a.mat, pre, printrates=False, tol=1e-10, maxiter=500)
    gfu.vec.data = inv * f.vec
    iters_ss = inv.iterations

    print(f"SparseSolv BDDC: {iters_ss} iterations")
    print(f"  WB DOFs: {pre.num_wirebasket_dofs}")
    print(f"  IF DOFs: {pre.num_interface_dofs}")

    assert inv.iterations < 500, "SparseSolv BDDC did not converge"

    # Reference: direct solver
    gfu_ref = GridFunction(fes)
    gfu_ref.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

    # Compare solutions
    err = sqrt(Integrate((gfu - gfu_ref)**2, mesh))
    ref_norm = sqrt(Integrate(gfu_ref**2, mesh))
    rel_err = err / ref_norm if ref_norm > 0 else err
    print(f"  Relative error vs direct: {rel_err:.2e}")
    assert rel_err < 1e-6, f"Solution error too large: {rel_err}"


def test_bddc_vs_ngsolve_bddc():
    """Compare SparseSolv BDDC with NGSolve's built-in BDDC."""
    mesh, fes, a, f = create_3d_poisson(maxh=0.35, order=3)
    element_dofs, dof_types = get_bddc_info(fes)

    print(f"\nDOF={fes.ndof}")

    # SparseSolv BDDC
    pre_ss = BDDCPreconditioner(
        a.mat, freedofs=fes.FreeDofs(True),
        element_dofs=element_dofs,
        dof_types=dof_types)

    gfu_ss = GridFunction(fes)
    inv_ss = CGSolver(a.mat, pre_ss, printrates=False, tol=1e-10, maxiter=500)
    gfu_ss.vec.data = inv_ss * f.vec
    iters_ss = inv_ss.iterations

    # NGSolve BDDC
    a2 = BilinearForm(fes)
    a2 += InnerProduct(grad(fes.TnT()[0]), grad(fes.TnT()[1])) * dx
    c_ng = Preconditioner(a2, type="bddc")
    a2.Assemble()

    gfu_ng = GridFunction(fes)
    inv_ng = CGSolver(a2.mat, c_ng.mat, printrates=False, tol=1e-10, maxiter=500)
    gfu_ng.vec.data = inv_ng * f.vec
    iters_ng = inv_ng.iterations

    print(f"SparseSolv BDDC: {iters_ss} iterations")
    print(f"NGSolve BDDC:    {iters_ng} iterations")

    # Both should produce similar solutions
    err = sqrt(Integrate((gfu_ss - gfu_ng)**2, mesh))
    ref_norm = sqrt(Integrate(gfu_ng**2, mesh))
    rel_err = err / ref_norm if ref_norm > 0 else err
    print(f"Relative difference: {rel_err:.2e}")
    assert rel_err < 1e-6, f"Solutions differ too much: {rel_err}"

    # SparseSolv is inexact BDDC so may need more iterations
    assert iters_ss < 200, f"SparseSolv BDDC too many iterations: {iters_ss}"


def test_bddc_mesh_independence():
    """Verify that BDDC iteration count is roughly mesh-independent."""
    iters_list = []
    for maxh in [0.4, 0.3, 0.25]:
        mesh, fes, a, f = create_3d_poisson(maxh=maxh, order=3)
        element_dofs, dof_types = get_bddc_info(fes)

        pre = BDDCPreconditioner(
            a.mat, freedofs=fes.FreeDofs(True),
            element_dofs=element_dofs,
            dof_types=dof_types)

        gfu = GridFunction(fes)
        inv = CGSolver(a.mat, pre, printrates=False, tol=1e-10, maxiter=500)
        gfu.vec.data = inv * f.vec

        iters_list.append(inv.iterations)
        print(f"maxh={maxh}, DOF={fes.ndof}, iters={inv.iterations}")

    # Iterations should not grow significantly with refinement
    # (allow 3x growth max, since inexact BDDC may be less stable)
    assert max(iters_list) < 3 * min(iters_list), \
        f"Iterations not mesh-independent: {iters_list}"


def test_bddc_shifted_curl_curl():
    """Shifted BDDC: build BDDC from regularized matrix, solve original curl-curl."""
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    for face in box.faces:
        face.name = "outer"
    mesh = box.GenerateMesh(maxh=0.3)

    fes = HCurl(mesh, order=1, dirichlet="outer", nograds=True)
    u, v = fes.TnT()

    # Original curl-curl (sigma=0)
    a = BilinearForm(fes)
    a += InnerProduct(curl(u), curl(v)) * dx
    a.Assemble()

    # Regularized matrix for BDDC
    sigma = 1e-6
    a_reg = BilinearForm(fes)
    a_reg += InnerProduct(curl(u), curl(v)) * dx
    a_reg += sigma * InnerProduct(u, v) * dx
    a_reg.Assemble()

    # Curl-based source (div J = 0 guaranteed)
    T = CF((y * (1 - y) * z * (1 - z),
            z * (1 - z) * x * (1 - x),
            x * (1 - x) * y * (1 - y)))
    f_form = LinearForm(fes)
    f_form += InnerProduct(T, curl(v)) * dx
    f_form.Assemble()

    element_dofs, dof_types = get_bddc_info(fes)
    print(f"\nHCurl DOF={fes.ndof}")

    # SparseSolv BDDC on regularized matrix, CG solves original
    pre = BDDCPreconditioner(
        a_reg.mat, freedofs=fes.FreeDofs(True),
        element_dofs=element_dofs,
        dof_types=dof_types)

    gfu = GridFunction(fes)
    inv = CGSolver(a.mat, pre, printrates=False, tol=1e-10, maxiter=500)
    gfu.vec.data = inv * f_form.vec
    print(f"Shifted SparseSolv BDDC: {inv.iterations} iterations")

    # Reference: regularized direct solver
    gfu_ref = GridFunction(fes)
    a_ref = BilinearForm(fes)
    a_ref += InnerProduct(curl(u), curl(v)) * dx
    a_ref += 1e-8 * InnerProduct(u, v) * dx
    a_ref.Assemble()
    gfu_ref.vec.data = a_ref.mat.Inverse(fes.FreeDofs(),
                                          inverse="sparsecholesky") * f_form.vec

    # Compare B = curl(A)
    B = curl(gfu)
    B_ref = curl(gfu_ref)
    B_err = sqrt(Integrate(InnerProduct(B - B_ref, B - B_ref), mesh))
    B_norm = sqrt(Integrate(InnerProduct(B_ref, B_ref), mesh))
    rel_err = B_err / B_norm if B_norm > 0 else B_err
    print(f"B = curl(A) relative error: {rel_err:.2e}")

    assert inv.iterations < 500, "Shifted BDDC did not converge"
    assert rel_err < 1e-4, f"B error too large: {rel_err}"


def get_element_matrices(a, fes):
    """Extract true element matrices using CalcElementMatrix.

    Returns:
        element_matrices: List of numpy arrays (true element stiffness matrices)
    """
    integ = a.integrators[0]
    element_matrices = []
    for el in fes.Elements(VOL):
        all_dofs = list(el.dofs)
        valid_idx = [i for i, d in enumerate(all_dofs) if d >= 0]
        fe = el.GetFE()
        trafo = el.GetTrafo()
        full_elmat = np.array(integ.CalcElementMatrix(fe, trafo, heapsize=100000))
        elmat = full_elmat[np.ix_(valid_idx, valid_idx)]
        element_matrices.append(elmat.tolist())
    return element_matrices


def test_element_bddc_3d_poisson():
    """Test element-by-element BDDC on 3D Poisson."""
    mesh, fes, a, f = create_3d_poisson(maxh=0.4, order=3)
    element_dofs, dof_types = get_bddc_info(fes)
    element_matrices = get_element_matrices(a, fes)

    print(f"\nElement BDDC: DOF={fes.ndof}, elements={len(element_dofs)}")

    # SparseSolv element-by-element BDDC
    pre = BDDCPreconditioner(
        a.mat, freedofs=fes.FreeDofs(True),
        element_dofs=element_dofs,
        dof_types=dof_types,
        element_matrices=element_matrices)

    assert pre.is_element_mode, "Should be in element mode"

    gfu = GridFunction(fes)
    inv = CGSolver(a.mat, pre, printrates=False, tol=1e-10, maxiter=500)
    gfu.vec.data = inv * f.vec
    iters_elem = inv.iterations

    print(f"  Element BDDC: {iters_elem} iterations")
    print(f"  WB DOFs: {pre.num_wirebasket_dofs}")
    print(f"  IF DOFs: {pre.num_interface_dofs}")

    assert inv.iterations < 500, "Element BDDC did not converge"

    # Reference: direct solver
    gfu_ref = GridFunction(fes)
    gfu_ref.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

    err = sqrt(Integrate((gfu - gfu_ref)**2, mesh))
    ref_norm = sqrt(Integrate(gfu_ref**2, mesh))
    rel_err = err / ref_norm if ref_norm > 0 else err
    print(f"  Relative error vs direct: {rel_err:.2e}")
    assert rel_err < 1e-6, f"Solution error too large: {rel_err}"


def test_element_bddc_vs_ngsolve():
    """Compare element-by-element BDDC with NGSolve's BDDC."""
    mesh, fes, a, f = create_3d_poisson(maxh=0.35, order=3)
    element_dofs, dof_types = get_bddc_info(fes)
    element_matrices = get_element_matrices(a, fes)

    print(f"\nElement BDDC vs NGSolve: DOF={fes.ndof}")

    # SparseSolv element-by-element BDDC
    pre_ss = BDDCPreconditioner(
        a.mat, freedofs=fes.FreeDofs(True),
        element_dofs=element_dofs,
        dof_types=dof_types,
        element_matrices=element_matrices)

    gfu_ss = GridFunction(fes)
    inv_ss = CGSolver(a.mat, pre_ss, printrates=False, tol=1e-10, maxiter=500)
    gfu_ss.vec.data = inv_ss * f.vec
    iters_ss = inv_ss.iterations

    # NGSolve BDDC
    a2 = BilinearForm(fes)
    a2 += InnerProduct(grad(fes.TnT()[0]), grad(fes.TnT()[1])) * dx
    c_ng = Preconditioner(a2, type="bddc")
    a2.Assemble()

    gfu_ng = GridFunction(fes)
    inv_ng = CGSolver(a2.mat, c_ng.mat, printrates=False, tol=1e-10, maxiter=500)
    gfu_ng.vec.data = inv_ng * f.vec
    iters_ng = inv_ng.iterations

    print(f"  Element BDDC:  {iters_ss} iterations")
    print(f"  NGSolve BDDC:  {iters_ng} iterations")

    # Solutions should match
    err = sqrt(Integrate((gfu_ss - gfu_ng)**2, mesh))
    ref_norm = sqrt(Integrate(gfu_ng**2, mesh))
    rel_err = err / ref_norm if ref_norm > 0 else err
    print(f"  Relative difference: {rel_err:.2e}")
    assert rel_err < 1e-6, f"Solutions differ: {rel_err}"

    # Iteration count should be comparable (within 2x of NGSolve)
    assert iters_ss <= 2 * iters_ng + 5, \
        f"Element BDDC needs too many iterations: {iters_ss} vs NGSolve {iters_ng}"


def test_dense_inverse():
    """Test DenseMatrix LU inverse accuracy (via BDDC assembly path)."""
    # Simple test: create a small SPD system and verify BDDC works
    from netgen.occ import Box, Pnt
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    for face in box.faces:
        face.name = "outer"
    mesh = box.GenerateMesh(maxh=0.5)

    fes = H1(mesh, order=3, dirichlet="outer")
    u, v = fes.TnT()

    a = BilinearForm(fes)
    a += InnerProduct(grad(u), grad(v)) * dx
    a.Assemble()

    f = LinearForm(fes)
    f += 1 * v * dx
    f.Assemble()

    element_dofs, dof_types = get_bddc_info(fes)

    # Just verify setup doesn't crash and produces valid result
    pre = BDDCPreconditioner(
        a.mat, freedofs=fes.FreeDofs(True),
        element_dofs=element_dofs,
        dof_types=dof_types)

    gfu = GridFunction(fes)
    inv = CGSolver(a.mat, pre, printrates=False, tol=1e-8, maxiter=300)
    gfu.vec.data = inv * f.vec

    print(f"\nDense inverse test: {inv.iterations} iterations, DOF={fes.ndof}")
    assert inv.iterations < 300, "BDDC failed to converge"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
