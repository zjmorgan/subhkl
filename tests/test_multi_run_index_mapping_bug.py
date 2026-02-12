import numpy as np
import h5py
import pytest
from pathlib import Path
from subhkl.io.parser import indexer
from subhkl.metrics import compute_metrics

def generate_valid_laue_peak(q_sample, R, ki, lambda_val):
    """Generates a unit kf that satisfies the Laue condition: lambda Q = kf - ki."""
    q_lab = R @ q_sample
    kf = lambda_val * q_lab + ki
    return kf / np.linalg.norm(kf)

def test_indexer_run_mapping_consistency(tmp_path):
    """
    Exposes the bug where run_index logic in FindUB.minimize compresses geometry.
    If run_index = [0, 0, 1, 1], FindUB.minimize picks R[0] and R[2] only.
    """
    peaks_h5 = tmp_path / "mapping_peaks.h5"
    output_h5 = tmp_path / "indexed.h5"
    
    # 1. Setup physics
    a, b, c = 10.0, 10.0, 10.0
    # Q in sample frame (units 1/A). Use a vector that has negative Z in lab to get lambda > 0.
    q_sample = np.array([0.1, 0.0, -0.1])
    ki = np.array([0.0, 0.0, 1.0])
    
    # 4 unique rotations around Y
    thetas = np.deg2rad([0, 2, 10, 12])
    R_stack = []
    for t in thetas:
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        R_stack.append(R)
    R_stack = np.array(R_stack)
    
    # Generate 4 valid peaks (one per R)
    two_theta = []
    azimuthal = []
    xyz = []
    for i in range(4):
        # lambda = -2 (Q_lab . ki) / |Q_lab|^2
        q_lab = R_stack[i] @ q_sample
        lambda_val = -2.0 * np.dot(q_lab, ki) / np.dot(q_lab, q_lab)
        # Ensure lambda is positive and reasonable
        assert lambda_val > 0, f"lambda {lambda_val} must be > 0 at index {i}"
        
        kf_dir = generate_valid_laue_peak(q_sample, R_stack[i], ki, lambda_val)
        tt = np.rad2deg(np.arccos(kf_dir[2]))
        az = np.rad2deg(np.arctan2(kf_dir[1], kf_dir[0]))
        
        two_theta.append(tt)
        azimuthal.append(az)
        xyz.append(kf_dir)

    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90., 90., 90.
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.5, 15.0]
        f["peaks/two_theta"] = np.array(two_theta)
        f["peaks/azimuthal"] = np.array(azimuthal)
        f["peaks/intensity"] = np.ones(4)
        f["peaks/sigma"] = np.ones(4) * 0.1
        f["peaks/radius"] = np.zeros(4)
        f["peaks/xyz"] = np.array(xyz)
        f["peaks/run_index"] = np.array([0, 0, 1, 1], dtype=np.int32)
        f["goniometer/R"] = R_stack

    # 2. Run Indexer
    indexer(
        peaks_h5_filename=str(peaks_h5),
        output_peaks_filename=str(output_h5),
        a=a, b=b, c=c, alpha=90, beta=90, gamma=90,
        space_group="P 1",
        strategy_name="DE", population_size=200, gens=100, n_runs=1,
        tolerance_deg=0.1, loss_method="gaussian"
    )
    
    metrics = compute_metrics(str(output_h5))
    ang_err = metrics["median_ang_err"]
    print(f"\nRESULT: Median angular error: {ang_err:.4f} deg")
    
    # If the bug is fixed, error will be low (< 0.1 deg).
    assert ang_err < 0.1, f"Fix failed! Error: {ang_err}"
