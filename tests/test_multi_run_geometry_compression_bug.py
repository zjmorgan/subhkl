import numpy as np
import h5py
from unittest.mock import patch, MagicMock

# Gracefully handle the import depending on where you are in the refactor
try:
    from subhkl.io.command_line_parser import indexer
except ImportError:
    from subhkl.io.parser import indexer

from subhkl.instrument.metrics import compute_metrics


def test_multi_run_geometry_compression_reproduction(tmp_path):
    """
    Reproduces the bug where multi-run indexing fails because the indexer
    compresses the rotation stack by taking the first rotation per run_index.
    """
    peaks_h5 = tmp_path / "synthetic_peaks.h5"
    output_h5 = tmp_path / "indexed.h5"
    dummy_nexus = tmp_path / "dummy.nxs"
    dummy_nexus.touch()

    # 1. Create synthetic data
    # Use a small, distinctive non-cubic cell to ensure unambiguous indexing
    a, b, c = 4.5, 5.0, 5.5
    alpha, beta, gamma = 90.0, 90.0, 90.0

    # Define 4 well-separated rotations (Y-axis)
    thetas = np.deg2rad([0, 10, 30, 40])
    R_stack = []
    for t in thetas:
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        R_stack.append(R)
    R_stack = np.array(R_stack)

    # Use 10 peaks per rotation for a strong signal
    # Use l=-2 or l=-3 to ensure wavelengths are in the 2-6 A range for this cell
    hkls = np.array(
        [
            [1, 0, -2],
            [0, 1, -2],
            [1, 1, -2],
            [2, 0, -2],
            [0, 2, -2],
            [1, 0, -3],
            [0, 1, -3],
            [1, 1, -3],
            [2, 1, -3],
            [1, 2, -3],
        ]
    )

    from subhkl.optimization import FindUB

    fu_helper = FindUB()
    fu_helper.a, fu_helper.b, fu_helper.c = a, b, c
    fu_helper.alpha, fu_helper.beta, fu_helper.gamma = alpha, beta, gamma
    B = fu_helper.reciprocal_lattice_B()

    xyz = []
    run_indices = []
    thetas_expanded = []
    ki_hat = np.array([0, 0, 1])

    for i, t in enumerate(thetas):
        R = R_stack[i]
        for hkl in hkls:
            # Reciprocal lattice vector in lab frame (units 1/A, no 2pi)
            q_lab = R @ B @ hkl

            # Use explicit 2pi logic for Q
            Q = 2.0 * np.pi * q_lab
            Q_sq = np.sum(Q**2)

            # Generalized Laue condition: lambda = -4pi * (ki_hat . Q) / Q^2
            # (Matches |ki + Q| = |ki| with |ki| = 2pi/lambda)
            lamb_val = -4.0 * np.pi * np.dot(ki_hat, Q) / (Q_sq + 1e-9)

            # Scattered beam unit vector: kf_hat = (Q + ki) / |ki|
            # kf_hat = (lambda / 2pi) * Q + ki_hat
            kf_hat = (lamb_val / (2.0 * np.pi)) * Q + ki_hat

            xyz.append(kf_hat)
            run_indices.append(0 if i < 2 else 1)
            thetas_expanded.append(
                lamb_val
            )  # Misleading name, but we just need a list of same len

    xyz = np.array(xyz)
    run_indices = np.array(run_indices)
    num_total = len(xyz)

    # R_expanded should be per-peak for the indexer to use it
    # Re-calculate R stack for all peaks based on the original 4 rotations
    R_per_peak = []
    for i, t in enumerate(thetas):
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        for _ in range(len(hkls)):
            R_per_peak.append(R)
    R_per_peak = np.array(R_per_peak)

    # Compute expected synthetic angles to pass into the mock
    tt_synthetic = np.rad2deg(np.arccos(np.clip(xyz[:, 2], -1.0, 1.0)))
    az_synthetic = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))

    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = (
            alpha,
            beta,
            gamma,
        )
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.1, 10.0]

        f["peaks/two_theta"] = tt_synthetic
        f["peaks/azimuthal"] = az_synthetic
        f["peaks/intensity"] = np.ones(num_total)
        f["peaks/sigma"] = np.ones(num_total) * 0.1
        f["peaks/radius"] = np.zeros(num_total)
        f["peaks/run_index"] = run_indices.astype(np.int32)
        f["peaks/xyz"] = xyz
        f["goniometer/R"] = R_per_peak

        # --- NEW: Dummy pixel and bank mapping for physical reconstruction ---
        f["peaks/pixel_r"] = np.zeros(num_total)
        f["peaks/pixel_c"] = np.zeros(num_total)
        f["peaks/image_index"] = run_indices.astype(np.int32)
        f["bank"] = np.ones(num_total, dtype=np.int32)
        f["bank_ids"] = np.array([1], dtype=np.int32)

    # 2. Mock the physical geometry conversion and Run Indexer
    with (
        patch("subhkl.instrument.detector.Detector") as mock_detector,
        patch("subhkl.commands.Peaks") as mock_peaks,  # noqa: F841
        patch.dict("subhkl.config.config.beamlines", {"DUMMY": {"1": {}}}),
    ):
        # Configure the mock to return the synthetic math instead of attempting real conversions
        mock_det_instance = MagicMock()
        mock_det_instance.pixel_to_lab.return_value = xyz
        mock_det_instance.pixel_to_angles.return_value = (tt_synthetic, az_synthetic)
        mock_detector.return_value = mock_det_instance

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
            seed=42,
            sigma_init=3.14,
            instrument_name="DUMMY",
            original_nexus_filename=str(dummy_nexus),
        )

    with h5py.File(output_h5, "r") as f:
        print(f"DEBUG: Output run_index: {f['peaks/run_index'][()]}")
        print(f"DEBUG: Output R stack shape: {f['goniometer/R'].shape}")

    metrics = compute_metrics(str(output_h5))
    ang_err = metrics["median_ang_err"]
    print(f"Median angular error: {ang_err:.4f} deg")

    assert ang_err < 0.5, f"Fix failed! Error still high: {ang_err}"
