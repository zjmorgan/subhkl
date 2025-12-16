import os

import h5py
import numpy as np

from subhkl.optimization import FindUB


def test_sucrose(test_data_dir):
    filename = os.path.join(test_data_dir, "sucrose_mandi.h5")

    opt = FindUB(filename)

    with h5py.File(os.path.abspath(filename), "r") as f:
        U = f["sample/U"][()]
        B = f["sample/B"][()]
        R = f["goniometer/R"][()]
        h = f["peaks/h"][()]
        k = f["peaks/k"][()]
        l = f["peaks/l"][()]  # noqa: E741
        lamda = f["peaks/lambda"][()]

    assert np.isclose(np.linalg.det(U), 1.0)

    assert np.isclose(np.linalg.det(R), 1.0)

    assert np.all(np.logical_and(lamda >= 2, lamda <= 4))

    np.allclose(opt.reciprocal_lattice_B(), B)

    UB = opt.UB_matrix(U, B)

    assert np.allclose(UB, np.dot(U, B))

    kf_ki_dir, d_min, d_max = opt.uncertainty_line_segements()

    d_star = np.linalg.norm(kf_ki_dir / lamda, axis=0)

    assert np.all(np.logical_and(d_star >= 1 / d_max, d_star <= 1 / d_min))

    hkl = [h, k, l]

    d_star = kf_ki_dir / lamda

    assert np.allclose(d_star, np.einsum("ij,jk->ik", R @ UB, hkl), atol=1e-3)

    # Whether any run succeeded
    success = False

    # Number of attempted runs
    tries = 0

    while tries < 5:
        tries += 1

        num, hkl, lamda = opt.minimize(64)

        B = opt.reciprocal_lattice_B()
        U = opt.orientation_U(*opt.x)

        UB = opt.UB_matrix(U, B)

        d_star = np.linalg.norm(kf_ki_dir / lamda, axis=0)

        s = np.linalg.norm(np.einsum("ij,kj->ik", UB, hkl), axis=0)

        # Check all conditions to see if this run passed
        if num / len(lamda) > 0.95 and np.allclose(d_star, s, atol=1e-1):
            success = True
            break

    assert success


def test_lysozyme(test_data_dir):
    # FIXME Commit the real lycozyme file so we can do this test in CI
    pass
    filename = os.path.join(test_data_dir, "5vnq_mandi.h5")

    opt = FindUB(filename)

    with h5py.File(os.path.abspath(filename), "r") as f:
        U = f["sample/U"][()]
        B = f["sample/B"][()]
        R = f["goniometer/R"][()]
        h = f["peaks/h"][()]
        k = f["peaks/k"][()]
        l = f["peaks/l"][()]  # noqa: E741
        lamda = f["peaks/lambda"][()]

        assert np.isclose(np.linalg.det(U), 1.0)

        assert np.isclose(np.linalg.det(R), 1.0)

        assert np.all(np.logical_and(lamda >= 2, lamda <= 4))

        np.allclose(opt.reciprocal_lattice_B(), B)

        UB = opt.UB_matrix(U, B)

        assert np.allclose(UB, np.dot(U, B))

        kf_ki_dir, d_min, d_max = opt.uncertainty_line_segements()

        d_star = np.linalg.norm(kf_ki_dir / lamda, axis=0)

        assert np.all(np.logical_and(d_star >= 1 / d_max, d_star <= 1 / d_min))

        hkl = [h, k, l]

        d_star = kf_ki_dir / lamda

        assert np.allclose(d_star, np.einsum("ij,jk->ik", R @ UB, hkl), atol=1e-3)

        # Whether any run succeeded
        success = False

        # Number of attempted runs
        tries = 0

        while tries < 5:
            tries += 1

            num, hkl, lamda = opt.minimize(64)

            B = opt.reciprocal_lattice_B()
            U = opt.orientation_U(*opt.x)

            UB = opt.UB_matrix(U, B)

            d_star = np.linalg.norm(kf_ki_dir / lamda, axis=0)

            s = np.linalg.norm(np.einsum("ij,kj->ik", UB, hkl), axis=0)

            # Check all test conditions to see if this run passed
            if num / len(lamda) > 0.95 and np.allclose(d_star, s, atol=1e-1):
                success = True
                break

        assert success
