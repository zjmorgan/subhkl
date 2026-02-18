import numpy as np
import h5py
from subhkl.io.parser import indexer
from subhkl.metrics import compute_metrics


def test_multi_run_geometry_compression_reproduction(tmp_path):
    """
    Reproduces the bug where multi-run indexing fails because the indexer
    compresses the rotation stack by taking the first rotation per run_index.
    """
    peaks_h5 = tmp_path / "synthetic_peaks.h5"
    output_h5 = tmp_path / "indexed.h5"

    # 1. Create synthetic data
    a, b, c = 10.0, 10.0, 10.0
    alpha, beta, gamma = 90.0, 90.0, 90.0

    # Define 4 unique rotations
    thetas = np.deg2rad([0, 1, 10, 11])
    R_stack = []
    for t in thetas:
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        R_stack.append(R)
    R_stack = np.array(R_stack)

    # Generate 5 peaks per bank to have a stronger signal
    hkls = np.array(
        [
            [1, 0, -1],
            [0, 1, -1],
            [0, 0, -1],
            [-1, -1, -1],
            [1, 1, -1],
        ]
    )
    num_hkls = len(hkls)
    np.tile(hkls, (4, 1))

    np.repeat(R_stack, num_hkls, axis=0)

    B = np.eye(3) * (1.0 / 10.0)

    xyz = []
    run_indices = []
    thetas_expanded = []
    ki = np.array([0, 0, 1])

    for i, t in enumerate(thetas):
        R = R_stack[i]
        for hkl in hkls:
            q_lab = R @ B @ hkl
            q_sq = np.sum(q_lab**2)
            # lambda = -2 (q . ki) / q^2
            lamb_val = -2 * np.dot(q_lab, ki) / q_sq

            # Use lamb_val to make it elastic: kf = lamb * q + ki
            kf = lamb_val * q_lab + ki
            # kf is now a unit vector!
            xyz.append(kf)
            run_indices.append(0 if i < 2 else 1)  # Original run_index (bugged)
            thetas_expanded.append(t)

    xyz = np.array(xyz)
    run_indices = np.array(run_indices)

    two_theta = np.rad2deg(np.arccos(xyz[:, 2]))
    azimuthal = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    num_total = len(xyz)

    # R_expanded should be per-peak for the indexer to use it
    R_per_peak = []
    for t in thetas_expanded:
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        R_per_peak.append(R)
    R_per_peak = np.array(R_per_peak)

    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = (
            alpha,
            beta,
            gamma,
        )
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.1, 10.0]

        f["peaks/two_theta"] = two_theta
        f["peaks/azimuthal"] = azimuthal
        f["peaks/intensity"] = np.ones(num_total)
        f["peaks/sigma"] = np.ones(num_total) * 0.1
        f["peaks/radius"] = np.zeros(num_total)
        f["peaks/run_index"] = run_indices.astype(np.int32)
        f["peaks/xyz"] = xyz
        f["goniometer/R"] = R_per_peak

    # 2. Run Indexer
    indexer(
        peaks_h5_filename=str(peaks_h5),
        output_peaks_filename=str(output_h5),
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        space_group="P 1",
        strategy_name="DE",
        population_size=1000,
        gens=1000,
        n_runs=1,
        tolerance_deg=0.5,
        loss_method="gaussian",
        sigma_init=3.14,
    )

    with h5py.File(output_h5, "r") as f:
        print(f"DEBUG: Output run_index: {f['peaks/run_index'][()]}")
        print(f"DEBUG: Output R stack shape: {f['goniometer/R'].shape}")

    metrics = compute_metrics(str(output_h5))
    ang_err = metrics["median_ang_err"]
    print(f"Median angular error: {ang_err:.4f} deg")

    assert ang_err < 0.5, f"Fix failed! Error still high: {ang_err}"
