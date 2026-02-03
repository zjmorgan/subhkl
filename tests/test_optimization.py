import numpy as np
import pytest

import scipy.linalg

import subhkl.optimization as optimization


def test_backend_flags_and_require_jax():
    # Module exposes HAS_JAX and OPTIMIZATION_BACKEND
    assert isinstance(optimization.HAS_JAX, bool)
    assert optimization.OPTIMIZATION_BACKEND in ("jax", "numpy")

    # require_jax should raise only when JAX is not available
    if optimization.HAS_JAX:
        optimization.require_jax()
    else:
        with pytest.raises(ImportError):
            optimization.require_jax()


def test_param_mapping_roundtrip():
    from subhkl.optimization import _inverse_map_param, _forward_map_param

    bounds = [0.001, 0.1, 1.0]
    test_vals = [-2.0, -0.5, 0.0, 0.3, 0.9]
    for b in bounds:
        for v in test_vals:
            norm = _inverse_map_param(v, b)
            out = _forward_map_param(norm, b)
            # Forward output must lie within [-bound, bound]
            assert out >= -b - 1e-12 and out <= b + 1e-12


def test_get_lattice_system_simple_cubic():
    from subhkl.optimization import get_lattice_system

    final, num = get_lattice_system(10.0, 10.0, 10.0, 90.0, 90.0, 90.0, "P 4 3 2")
    assert final == "Cubic"
    assert num == 1


def test_findub_load_from_dict_and_reciprocal_B():
    from subhkl.optimization import FindUB

    data = {}
    data["sample/a"] = 10.0
    data["sample/b"] = 10.0
    data["sample/c"] = 10.0
    data["sample/alpha"] = 90.0
    data["sample/beta"] = 90.0
    data["sample/gamma"] = 90.0
    data["instrument/wavelength"] = np.array([1.0, 2.0])
    data["goniometer/R"] = np.eye(3)
    data["peaks/two_theta"] = np.array([30.0])
    data["peaks/azimuthal"] = np.array([10.0])
    data["peaks/intensity"] = np.array([1.0])
    data["peaks/sigma"] = np.array([0.1])
    data["peaks/radius"] = np.array([0.0])
    data["sample/space_group"] = "P 1"

    fu = FindUB(data=data)
    B = fu.reciprocal_lattice_B()
    assert B.shape == (3, 3)

    # Recompute expected B via metric tensor
    a = data["sample/a"]
    b = data["sample/b"]
    c = data["sample/c"]
    alpha = np.deg2rad(data["sample/alpha"])
    beta = np.deg2rad(data["sample/beta"])
    gamma = np.deg2rad(data["sample/gamma"])

    g11 = a**2
    g22 = b**2
    g33 = c**2
    g12 = a * b * np.cos(gamma)
    g13 = c * a * np.cos(beta)
    g23 = b * c * np.cos(alpha)
    G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])
    B_expected = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    assert np.allclose(B, B_expected)
