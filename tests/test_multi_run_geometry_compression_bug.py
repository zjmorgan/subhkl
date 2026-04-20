import numpy as np
import h5py
from unittest.mock import patch, MagicMock

try:
    from subhkl.io.command_line_parser import indexer
except ImportError:
    from subhkl.io.parser import indexer


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
    a, b, c = 4.5, 5.0, 5.5
    alpha, beta, gamma = 90.0, 90.0, 90.0

    # Define 4 well-separated rotations (Y-axis)
    thetas = np.deg2rad([0, 10, 30, 40])
    R_stack = []
    for t in thetas:
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        R_stack.append(R)
    R_stack = np.array(R_stack)

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
    ki_hat = np.array([0, 0, 1])

    for i, t in enumerate(thetas):
        R = R_stack[i]
        for hkl in hkls:
            q_lab = R @ B @ hkl
            Q = 2.0 * np.pi * q_lab
            Q_sq = np.sum(Q**2)
            lamb_val = -4.0 * np.pi * np.dot(ki_hat, Q) / (Q_sq + 1e-9)
            kf_hat = (lamb_val / (2.0 * np.pi)) * Q + ki_hat

            xyz.append(kf_hat)
            run_indices.append(i)  # Use distinct run indices for all 4 runs

    xyz = np.array(xyz)
    run_indices = np.array(run_indices)
    num_total = len(xyz)

    R_per_peak = []
    for i, t in enumerate(thetas):
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        for _ in range(len(hkls)):
            R_per_peak.append(R)
    R_per_peak = np.array(R_per_peak)

    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = alpha, beta, gamma
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.1, 10.0]

        # Use dummy values for angle calculations since we bypass them with the mock
        f["peaks/two_theta"] = np.zeros(num_total)
        f["peaks/azimuthal"] = np.zeros(num_total)
        f["peaks/intensity"] = np.ones(num_total)
        f["peaks/sigma"] = np.ones(num_total) * 0.1
        f["peaks/radius"] = np.zeros(num_total)
        f["peaks/run_index"] = run_indices.astype(np.int32)
        f["peaks/xyz"] = xyz
        f["goniometer/R"] = R_per_peak

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
        mock_det_instance = MagicMock()
        mock_det_instance.pixel_to_lab.return_value = xyz
        mock_det_instance.pixel_to_angles.return_value = (
            np.zeros(num_total),
            np.zeros(num_total),
        )
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

    # 3. Assert the fix!
    # If the bug exists, R will have shape (4, 3, 3) (one per run)
    # If the bug is fixed, R will have shape (40, 3, 3) (one per peak)
    with h5py.File(output_h5, "r") as f:
        output_r_stack = f["goniometer/R"][()]

        # The number of rotation matrices should exactly match the number of peaks
        assert output_r_stack.shape[0] == num_total, (
            f"Fix failed! R stack shape {output_r_stack.shape} does not match peak count {num_total}."
        )

        # It should also correctly identify 4 unique rotation states
        unique_runs = len(np.unique(f["peaks/run_index"][()]))
        assert unique_runs == 4, f"Expected 4 runs, got {unique_runs}"
