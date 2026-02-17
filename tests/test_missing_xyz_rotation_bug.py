import numpy as np
import h5py
from subhkl.io.parser import indexer
from subhkl.metrics import compute_metrics


def test_missing_xyz_rotation_regression(tmp_path):
    """
    Reproduces the bug where indexing fails when 'peaks/xyz' is missing
    because the code incorrectly assumes Lab-frame vectors are already rotated.
    """
    peaks_h5 = tmp_path / "no_xyz.h5"
    output_h5 = tmp_path / "indexed.h5"

    # Define a 90-degree rotation around Y
    thetas = np.deg2rad([90])
    R_stack = []
    for t in thetas:
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        R_stack.append(R)
    R_stack = np.array(R_stack)

    a, b, c = 10.0, 10.0, 10.0
    # Single peak at [1, 0, 0] in sample frame
    hkl = np.array([[1, 0, 0]])
    B = np.eye(3) * 0.1
    q_sample = (B @ hkl.T).T
    q_lab = (R_stack[0] @ q_sample.T).T

    ki = np.array([0, 0, 1])
    kf = q_lab + ki
    kf_dir = kf / np.linalg.norm(kf, axis=1, keepdims=True)

    # Calculate Lab angles
    two_theta = np.rad2deg(np.arccos(kf_dir[:, 2]))
    azimuthal = np.rad2deg(np.arctan2(kf_dir[:, 1], kf_dir[:, 0]))

    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = (
            90.0,
            90.0,
            90.0,
        )
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.9, 1.1]
        f["peaks/two_theta"] = two_theta
        f["peaks/azimuthal"] = azimuthal
        f["peaks/intensity"] = np.ones(1)
        f["peaks/sigma"] = np.ones(1) * 0.1
        f["peaks/radius"] = np.zeros(1)
        f["peaks/run_index"] = np.array([0], dtype=np.int32)
        # peaks/xyz is intentionally missing
        f["goniometer/R"] = R_stack

    indexer(
        peaks_h5_filename=str(peaks_h5),
        output_peaks_filename=str(output_h5),
        a=a,
        b=b,
        c=c,
        alpha=90,
        beta=90,
        gamma=90,
        space_group="P 1",
        strategy_name="DE",
        population_size=100,
        gens=50,
        n_runs=1,
        tolerance_deg=0.1,
        loss_method="gaussian",
    )

    metrics = compute_metrics(str(output_h5))
    ang_err = metrics["median_ang_err"]
    print(f"Median angular error: {ang_err:.4f} deg")

    # If the bug exists, error will be ~90 degrees or optimization will fail to find the peak
    assert ang_err < 0.1, f"Missing XYZ bug! Error: {ang_err}"
