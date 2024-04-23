import os
directory = os.path.dirname(os.path.abspath(__file__))

import h5py

import numpy as np

from subhkl.optimization import FindUB

def test_sucrose():

    filename = os.path.join(directory, 'sucrose_mandi.h5')

    opt = FindUB(filename)

    with h5py.File(os.path.abspath(filename), 'r') as f:

        U = f['sample/U'][()]
        B = f['sample/B'][()]
        R = f['goniometer/R'][()]
        h = f['peaks/h'][()]
        k = f['peaks/k'][()]
        l = f['peaks/l'][()]
        lamda = f['peaks/lambda'][()]

    assert np.isclose(np.linalg.det(U), 1.0)

    assert np.isclose(np.linalg.det(R), 1.0)

    assert np.all(np.logical_and(lamda >= 2, lamda <= 4))

    np.allclose(opt.reciprocal_lattice_B(), B)

    UB = opt.UB_matrix(U, B)

    assert np.allclose(UB, np.dot(U, B))

    kf_ki_dir, d_min, d_max = opt.uncertainty_line_segements()

    d_star = np.linalg.norm(kf_ki_dir/lamda, axis=0)

    assert np.all(np.logical_and(d_star >= 1/d_max, d_star <= 1/d_min))

    hkl = [h,k,l]

    d_star = kf_ki_dir/lamda

    assert np.allclose(d_star, np.einsum('ij,jk->ik', R @ UB, hkl), atol=1e-3)

    num, hkl, lamda = opt.minimize(1)
    assert num/len(lamda) > 0.95

def test_lysozye():

    filename = os.path.join(directory, '5vnq_mandi.h5')

    opt = FindUB(filename)

    with h5py.File(os.path.abspath(filename), 'r') as f:

        U = f['sample/U'][()]
        B = f['sample/B'][()]
        R = f['goniometer/R'][()]
        h = f['peaks/h'][()]
        k = f['peaks/k'][()]
        l = f['peaks/l'][()]
        lamda = f['peaks/lambda'][()]

    assert np.isclose(np.linalg.det(U), 1.0)

    assert np.isclose(np.linalg.det(R), 1.0)

    assert np.all(np.logical_and(lamda >= 2, lamda <= 4))

    np.allclose(opt.reciprocal_lattice_B(), B)

    UB = opt.UB_matrix(U, B)

    assert np.allclose(UB, np.dot(U, B))

    kf_ki_dir, d_min, d_max = opt.uncertainty_line_segements()

    d_star = np.linalg.norm(kf_ki_dir/lamda, axis=0)

    assert np.all(np.logical_and(d_star >= 1/d_max, d_star <= 1/d_min))

    hkl = [h,k,l]

    d_star = kf_ki_dir/lamda

    assert np.allclose(d_star, np.einsum('ij,jk->ik', R @ UB, hkl), atol=1e-3)

    num, hkl, lamda = opt.minimize(1)
    assert num/len(lamda) > 0.95