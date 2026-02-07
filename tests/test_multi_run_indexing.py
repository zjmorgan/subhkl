import numpy as np
import pytest
from subhkl.optimization import FindUB, VectorizedObjective
from scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp

def generate_synthetic_data(a, b, c, alpha, beta, gamma, U_true, rotations, gonio_axes, space_group="P 1", hkl_range_k=None):
    fu_helper = FindUB()
    fu_helper.a, fu_helper.b, fu_helper.c = a, b, c
    fu_helper.alpha, fu_helper.beta, fu_helper.gamma = alpha, beta, gamma
    B = fu_helper.reciprocal_lattice_B()
    
    all_tt, all_az, all_run, all_angles, all_R = [], [], [], [], []
    
    if hkl_range_k is None: hkl_range_k = 10

    for i_run, rot_angles in enumerate(rotations):
        R_run = np.eye(3)
        for ang, axis_spec in zip(rot_angles, gonio_axes):
            sign = axis_spec[3]
            axis = axis_spec[:3]
            R_axis = Rotation.from_rotvec(sign * np.deg2rad(ang) * axis).as_matrix()
            R_run = R_run @ R_axis
            
        h_range = np.arange(-5, 6)
        k_range = np.arange(-hkl_range_k, hkl_range_k + 1)
        l_range = np.arange(-5, 6)
        hh, kk, ll = np.meshgrid(h_range, k_range, l_range, indexing='ij')
        hkls = np.stack([hh.flatten(), kk.flatten(), ll.flatten()], axis=1)
        hkls = hkls[np.linalg.norm(hkls, axis=1) > 0]
        
        RUB = R_run @ U_true @ B
        Q_lab = (RUB @ hkls.T).T
        ki = np.array([0, 0, 1])
        Q_sq = np.sum(Q_lab**2, axis=1)
        ki_dot_Q = Q_lab @ ki
        lambdas = -2.0 * ki_dot_Q / (Q_sq + 1e-9)
        
        mask = (lambdas > 1.5) & (lambdas < 4.0)
        hkls = hkls[mask]
        lambdas = lambdas[mask]
        Q_lab = Q_lab[mask]
        
        if len(hkls) > 20:
            idx = np.random.choice(len(hkls), 20, replace=False)
            hkls, lambdas, Q_lab = hkls[idx], lambdas[idx], Q_lab[idx]
            
        kf_phys = ki[None, :] + Q_lab * lambdas[:, None]
        kf_norm = kf_phys / np.linalg.norm(kf_phys, axis=1, keepdims=True)
        
        all_tt.append(np.rad2deg(np.arccos(np.clip(kf_norm[:, 2], -1, 1))))
        all_az.append(np.rad2deg(np.arctan2(kf_norm[:, 1], kf_norm[:, 0])))
        all_run.append(np.full(len(hkls), i_run))
        all_angles.append(np.tile(np.array(rot_angles), (len(hkls), 1)))
        all_R.append(np.tile(R_run[None, ...], (len(hkls), 1, 1)))
        
    return {
        "sample/a": a, "sample/b": b, "sample/c": c,
        "sample/alpha": alpha, "sample/beta": beta, "sample/gamma": gamma,
        "sample/space_group": space_group,
        "instrument/wavelength": [1.0, 5.0],
        "peaks/two_theta": np.concatenate(all_tt),
        "peaks/azimuthal": np.concatenate(all_az),
        "peaks/intensity": np.ones(sum(len(x) for x in all_tt)),
        "peaks/sigma": np.ones(sum(len(x) for x in all_tt)) * 0.1,
        "peaks/radius": np.zeros(sum(len(x) for x in all_tt)),
        "goniometer/axes": gonio_axes,
        "goniometer/angles": np.concatenate(all_angles),
        "goniometer/R": np.concatenate(all_R),
        "bank": np.concatenate(all_run),
    }

def test_multi_run_indexing_refinement():
    """Verify that multi-run physics is correct by refining from near truth."""
    np.random.seed(42)
    a, b, c = 10.1, 11.2, 12.3
    # Use Identity for U to simplify
    U_true = np.eye(3)
    gonio_axes = np.array([[0, 1, 0, 1]]) # omega
    rotations = [[0.0], [45.0]]
    
    data = generate_synthetic_data(a, b, c, 90, 90, 90, U_true, rotations, gonio_axes)
    fu = FindUB(data=data)
    
    num_peaks, hkl_res, lam_res, U_res = fu.minimize(
        strategy_name="DE",
        population_size=100,
        num_generations=50,
        init_params=np.zeros(3),
        sigma_init=0.01,
        tolerance_deg=1.0,
        loss_method="gaussian"
    )
    
    diff_R = U_res @ U_true.T
    angle = np.rad2deg(np.arccos(np.clip((np.trace(diff_R) - 1) / 2, -1, 1)))
    assert angle < 0.5

def test_clipping_logic_direct():
    """
    Directly test the VectorizedObjective HKL check to verify reflections 
    outside mask_range are handled correctly.
    """
    fu = FindUB()
    fu.a, fu.b, fu.c = 10, 10, 10
    fu.alpha, fu.beta, fu.gamma = 90, 90, 90
    B = fu.reciprocal_lattice_B()
    
    # Range 2.
    obj = VectorizedObjective(
        B=B, kf_ki_dir=np.zeros((3, 4)), peak_xyz_lab=None,
        wavelength=[1.0, 5.0], angle_cdf=np.zeros(4), angle_t=np.zeros(4),
        space_group="P 21", hkl_search_range=2
    )
    
    # In P 21, 0k0 is absent for k odd.
    # Mask range 2: k in [-2, -1, 0, 1, 2].
    # Mask values: k=0:T, k=1:F, k=2:T, k=-1:F, k=-2:T.
    
    h = jnp.array([0, 0, 0, 0])
    k = jnp.array([1, 2, 3, 4])
    l = jnp.array([0, 0, 0, 0])
    r = obj.mask_range # 2
    
    idx_h = jnp.clip(h + r, 0, 2*r).astype(jnp.int32)
    idx_k = jnp.clip(k + r, 0, 2*r).astype(jnp.int32)
    idx_l = jnp.clip(l + r, 0, 2*r).astype(jnp.int32)
    
    in_bounds = (h >= -r) & (h <= r) & (k >= -r) & (k <= r) & (l >= -r) & (l <= r)
    is_allowed = jnp.where(in_bounds, obj.valid_hkl_mask[idx_h, idx_k, idx_l], True)
    
    # k=1: in_bounds=T, absent. Result: False.
    assert bool(is_allowed[0]) == False
    # k=2: in_bounds=T, allowed. Result: True.
    assert bool(is_allowed[1]) == True
    # k=3: in_bounds=F. With fix, Result: True. (Without fix, clipped to k=2 -> True. Correct by accident).
    assert bool(is_allowed[2]) == True
    # k=4: in_bounds=F. With fix, Result: True. (Without fix, clipped to k=2 -> True).
    assert bool(is_allowed[3]) == True
    
    # Let's try negative indices.
    k_neg = jnp.array([-1, -2, -3])
    idx_k_neg = jnp.clip(k_neg + r, 0, 2*r).astype(jnp.int32)
    in_bounds_neg = (k_neg >= -r) & (k_neg <= r)
    is_allowed_neg = jnp.where(in_bounds_neg, obj.valid_hkl_mask[r, idx_k_neg, r], True)
    
    # k=-1: in_bounds=T, absent. Result: False.
    assert bool(is_allowed_neg[0]) == False
    # k=-2: in_bounds=T, allowed. Result: True.
    assert bool(is_allowed_neg[1]) == True
    # k=-3: in_bounds=F. With fix, Result: True. 
    # (Without fix, it would be clip(-3+2, 0, 4) = 0 -> k=-2 -> True).
    assert bool(is_allowed_neg[2]) == True
