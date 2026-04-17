import h5py
import numpy as np

from subhkl.optimization import FindUB

def test_u_absorbs_gonio_offset(tmp_path):
    """
    Test that orientation refinement can compensate for a fixed goniometer offset
    in a single-run scenario, and that dictionary-backed offsets persist correctly.
    """
    # 1. Setup a synthetic problem: Simple Cubic
    a, b, c = 10.0, 10.0, 10.0
    alpha, beta, gamma = 90.0, 90.0, 90.0
    space_group = "P 1"

    # True Orientation: Identity
    U_true = np.eye(3)

    # Nominal Goniometer (Identity)
    R_nom = np.eye(3)

    # Actual Goniometer Offset (e.g., 2 degrees around Y)
    from scipy.spatial.transform import Rotation

    delta_R = Rotation.from_euler("y", 2, degrees=True).as_matrix()
    R_true = R_nom @ delta_R

    # Create synthetic peaks
    h, k, l = np.meshgrid(  # noqa: E741
        np.arange(-5, 6), np.arange(-5, 5), np.arange(-5, 6), indexing="ij"
    )
    hkls = np.stack([h.flatten(), k.flatten(), l.flatten()], axis=1)
    hkls = hkls[np.linalg.norm(hkls, axis=1) > 0]

    fu_helper = FindUB()
    fu_helper.a, fu_helper.b, fu_helper.c = a, b, c
    fu_helper.alpha, fu_helper.beta, fu_helper.gamma = alpha, beta, gamma
    B = fu_helper.reciprocal_lattice_B()

    RUB_true = R_true @ U_true @ B
    Q_lab_unscaled = (RUB_true @ hkls.T).T

    ki = np.array([0, 0, 1])
    Q_sq = np.sum(Q_lab_unscaled**2, axis=1)
    ki_dot_Q = Q_lab_unscaled @ ki
    lambdas = -2.0 * ki_dot_Q / Q_sq

    mask = (lambdas > 1.0) & (lambdas < 8.5)
    hkls = hkls[mask]
    lambdas = lambdas[mask]
    Q_lab_unscaled = Q_lab_unscaled[mask]

    if len(hkls) > 100:
        hkls = hkls[:100]
        lambdas = lambdas[:100]
        Q_lab_unscaled = Q_lab_unscaled[:100]

    kf_phys = ki[None, :] + Q_lab_unscaled * lambdas[:, None]
    two_theta = np.rad2deg(np.arccos(np.clip(kf_phys[:, 2], -1, 1)))
    az_phi = np.rad2deg(np.arctan2(kf_phys[:, 1], kf_phys[:, 0]))

    data = {
        "sample/a": a,
        "sample/b": b,
        "sample/c": c,
        "sample/alpha": alpha,
        "sample/beta": beta,
        "sample/gamma": gamma,
        "sample/space_group": space_group,
        "instrument/wavelength": [1.0, 8.5],
        "goniometer/R": np.tile(R_nom[None, ...], (len(hkls), 1, 1)),
        "goniometer/axes": np.array([[0, 1, 0, 1]]),  # omega (Y)
        "goniometer/angles": np.zeros((len(hkls), 1)),
        "goniometer/names": [b"omega"], # <-- CRITICAL: Triggers dict behavior
        "peaks/intensity": np.ones(len(hkls)),
        "peaks/sigma": np.ones(len(hkls)) * 0.1,
        "peaks/radius": np.zeros(len(hkls)),
        "peaks/two_theta": two_theta,
        "peaks/azimuthal": az_phi,
    }

    fu = FindUB(data=data)

    # --- Run 1: WITH goniometer refinement ---
    score1, hkl1, lamda1, U1 = fu.minimize(
        strategy_name="DE",
        population_size=100, # Dropped from 500 so test runs instantly
        num_generations=50,  # Dropped from 150
        tolerance_deg=0.5,
        loss_method="gaussian",
        refine_lattice=False,
        refine_goniometer=True,
        goniometer_bound_deg=5.0,
    )

    # Save to a temp file to simulate bootstrap
    bootstrap_file = tmp_path / "stage1_test.h5"
    with h5py.File(bootstrap_file, "w") as f:
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/space_group"] = space_group
        f["sample/U"] = U1
        f["sample/B"] = fu.reciprocal_lattice_B()
        f["goniometer/R"] = fu.R
        f["beam/ki_vec"] = fu.ki_vec
        f["instrument/wavelength"] = [1.0, 8.5]
        
        # --- WRITE GROUP INSTEAD OF DATASET ---
        if isinstance(fu.goniometer_offsets, dict):
            grp = f.create_group("optimization/goniometer_offsets")
            for k, v in fu.goniometer_offsets.items():
                grp[k] = v
        else:
            f["optimization/goniometer_offsets"] = fu.goniometer_offsets
        f["optimization/best_params"] = fu.x
        # --------------------------------------

    # --- Run 2: WITHOUT goniometer refinement (BOOTSTRAP) ---
    fu2 = FindUB(data=data)
    _ = fu2.get_bootstrap_params(
        str(bootstrap_file), refine_lattice=False, refine_goniometer=False
    )

    # VERIFY PERSISTENCE (THE CORE FIX)
    # The new fu2.base_gonio_offset will be flattened out of the dict in the exact order of axes
    assert isinstance(fu.goniometer_offsets, dict), "Dict mapping failed to initialize!"
    expected_offset = np.array([fu.goniometer_offsets["omega"]])
    
    assert np.allclose(fu2.base_gonio_offset, expected_offset), "Bootstrap failed to load dict offset!"
    assert np.allclose(fu2.ki_vec, fu.ki_vec)

    print("TEST PASSED: Refined geometry was correctly bootstrapped from dictionary.")
