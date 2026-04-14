import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from subhkl.optimization import FindUB, VectorizedObjective


def generate_synthetic_data(
    a,
    b,
    c,
    alpha,
    beta,
    gamma,
    U_true,
    rotations,
    gonio_axes,
    sample_offset_true=None,
    lattice_system="Triclinic",
    hkl_range_k=None,
):
    fu_helper = FindUB()
    fu_helper.a, fu_helper.b, fu_helper.c = a, b, c
    fu_helper.alpha, fu_helper.beta, fu_helper.gamma = alpha, beta, gamma
    B = fu_helper.reciprocal_lattice_B()
    all_tt, all_az, all_run, all_angles, all_R, all_xyz = [], [], [], [], [], []
    if hkl_range_k is None:
        hkl_range_k = 10
    for i_run, rot_angles in enumerate(rotations):
        R_run = np.eye(3)
        for ang, axis_spec in zip(rot_angles, gonio_axes):
            sign = axis_spec[3]
            axis = axis_spec[:3]
            R_axis = Rotation.from_rotvec(sign * np.deg2rad(ang) * axis).as_matrix()
            R_run = R_run @ R_axis
        if sample_offset_true is not None:
            s_lab = R_run @ sample_offset_true
        else:
            s_lab = np.zeros(3)
        h_range = np.arange(-5, 6)
        k_range = np.arange(-hkl_range_k, hkl_range_k + 1)
        l_range = np.arange(-5, 6)
        hh, kk, ll = np.meshgrid(h_range, k_range, l_range, indexing="ij")
        hkls = np.stack([hh.flatten(), kk.flatten(), ll.flatten()], axis=1)
        hkls = hkls[np.linalg.norm(hkls, axis=1) > 0]
        Q_sample = (U_true @ B @ hkls.T).T
        Q_lab = (R_run @ Q_sample.T).T
        ki = np.array([0, 0, 1])
        Q_sq = np.sum(Q_lab**2, axis=1)
        ki_dot_Q = Q_lab @ ki
        lambdas = -2.0 * ki_dot_Q / (Q_sq + 1e-9)
        mask = (lambdas > 0.5) & (lambdas < 20.0)
        hkls = hkls[mask]
        lambdas = lambdas[mask]
        Q_lab = Q_lab[mask]
        if len(hkls) > 20:
            idx = np.random.choice(len(hkls), 20, replace=False)
            hkls, lambdas, Q_lab = hkls[idx], lambdas[idx], Q_lab[idx]
        kf_target = ki[None, :] + Q_lab * lambdas[:, None]
        kf_target = kf_target / np.linalg.norm(kf_target, axis=1, keepdims=True)
        dist_det = 0.4
        xyz_det = s_lab[None, :] + dist_det * kf_target
        all_xyz.append(xyz_det)
        kf_norm = kf_target
        all_tt.append(np.rad2deg(np.arccos(np.clip(kf_norm[:, 2], -1, 1))))
        all_az.append(np.rad2deg(np.arctan2(kf_norm[:, 1], kf_norm[:, 0])))
        all_run.append(np.full(len(hkls), i_run))
        all_angles.append(np.tile(np.array(rot_angles), (len(hkls), 1)))
        all_R.append(np.tile(R_run[None, ...], (len(hkls), 1, 1)))
    return {
        "sample/a": a,
        "sample/b": b,
        "sample/c": c,
        "sample/alpha": alpha,
        "sample/beta": beta,
        "sample/gamma": gamma,
        "sample/space_group": "P1",
        "instrument/wavelength": [0.5, 20.0],
        "peaks/two_theta": np.concatenate(all_tt),
        "peaks/azimuthal": np.concatenate(all_az),
        "peaks/intensity": np.ones(sum(len(x) for x in all_tt)),
        "peaks/sigma": np.ones(sum(len(x) for x in all_tt)) * 0.1,
        "peaks/radius": np.zeros(sum(len(x) for x in all_tt)),
        "goniometer/axes": gonio_axes,
        "goniometer/angles": np.concatenate(all_angles),
        "goniometer/R": np.concatenate(all_R),
        "bank": np.concatenate(all_run),
        "peaks/xyz": np.concatenate(all_xyz),
    }


def test_multi_run_indexing_refinement():
    np.random.seed(42)
    a, b, c = 10.1, 11.2, 12.3
    U_true = Rotation.from_euler("xyz", [15, 25, 35], degrees=True).as_matrix()
    gonio_axes = np.array([[0, 1, 0, 1]])
    rotations = [[0.0], [45.0]]
    data = generate_synthetic_data(a, b, c, 90, 90, 90, U_true, rotations, gonio_axes)
    fu = FindUB(data=data)
    num_peaks, hkl_res, lam_res, U_res = fu.minimize(
        strategy_name="DE",
        population_size=200,
        num_generations=100,
        n_runs=1,
        init_params=Rotation.from_matrix(U_true).as_rotvec(),
        sigma_init=0.001,
        tolerance_deg=0.1,
        loss_method="gaussian",
    )
    diff_R = U_res @ U_true.T
    angle = np.rad2deg(np.arccos(np.clip((np.trace(diff_R) - 1) / 2, -1, 1)))
    assert angle < 0.2


def test_sample_offset_refinement_multirun():
    np.random.seed(42)
    a, b, c = 10.0, 10.0, 10.0
    U_true = np.eye(3)
    gonio_axes = np.array([[0, 0, 1, 1]])
    rotations = [[0.0], [90.0]]
    s_true = np.array([0.002, 0.001, -0.001])
    data = generate_synthetic_data(
        a,
        b,
        c,
        90,
        90,
        90,
        U_true,
        rotations,
        gonio_axes,
        sample_offset_true=s_true,
    )
    fu = FindUB(data=data)
    num_peaks, hkl_res, lam_res, U_res = fu.minimize(
        strategy_name="DE",
        population_size=100,
        num_generations=500,
        n_runs=1,
        init_params=np.zeros(3),
        sigma_init=None,
        refine_sample=True,
        sample_bound_meters=0.005,
        tolerance_deg=0.1,
        loss_method="gaussian",
    )
    recovered_s = fu.sample_offset
    error = np.linalg.norm(recovered_s - s_true)
    assert error < 1e-2


def test_predictor_multirun_sample_rotation():
    from subhkl.instrument.detector import Detector
    from subhkl.instrument.physics import predict_reflections_on_panel

    det_config = {
        "m": 256,
        "n": 256,
        "width": 2.0,
        "height": 2.0,
        "center": [-1.0, -1.0, 0.4],
        "vhat": [0, 1, 0],
        "uhat": [1, 0, 0],
        "panel": "flat",
    }
    det = Detector(det_config)
    s_sample = np.array([0.01, 0.0, 0.0])
    RUB = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.05]])
    h, k, l = np.array([1]), np.array([0]), np.array([1])
    R1 = np.eye(3)
    R2 = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    res1 = predict_reflections_on_panel(
        det, h, k, l, R1 @ RUB, 0.1, 100.0, sample_offset=s_sample, R_all=R1
    )
    res2 = predict_reflections_on_panel(
        det, h, k, l, R2 @ RUB, 0.1, 100.0, sample_offset=s_sample, R_all=R2
    )
    assert len(res1[0]) > 0, "Peak not on panel in Run 1"
    assert len(res2[0]) > 0, "Peak not on panel in Run 2"
    shift_col = np.abs(res1[1][0] - res2[1][0])
    assert shift_col > 1.0, "Predictor ignored goniometer rotation for sample offset!"


def test_stage1_blindness_vulnerability():

    B = np.eye(3)
    xyz_lab = np.array([[0.0, 0.0, 0.4]])
    s_nom = np.array([0.01, 0.0, 0.0])
    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=np.array([[0, 0, 1]]).T,
        peak_xyz_lab=xyz_lab,
        wavelength=[1, 2],
        refine_sample=False,
        sample_nominal=s_nom,
    )
    kf_internal = obj.kf_lab_fixed
    assert np.abs(kf_internal[0, 0] - 0.0) > 1e-3, (
        "VULNERABILITY: Stage 1 is blind to nominal sample offset!"
    )


def test_multirun_peaks_per_image_vulnerability():
    """Verify if Indexer crashes or miscalculates when multiple peaks belong to the same run."""
    import numpy as np

    num_runs = 2
    peaks_per_run = 5
    total_peaks = num_runs * peaks_per_run

    B = np.eye(3)
    kf_ki_dir = np.random.randn(3, total_peaks)
    R_stack = np.stack([np.eye(3), np.eye(3)], axis=0)

    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=None,
        wavelength=[1.0, 2.0],
        static_R=R_stack,
    )
    x = np.zeros((1, 3))

    # THIS SHOULD CRASH if mapping is missing
    try:
        obj(x)
    except Exception as e:
        pytest.fail(f"Indexer CRASHED on multi-peak-per-run mapping: {e}")

    assert hasattr(obj, "peak_run_indices"), (
        "VULNERABILITY: Indexer lacks peak-to-run mapping metadata!"
    )
