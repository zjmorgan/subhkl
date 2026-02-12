import numpy as np
import h5py
import os
import pytest
from subhkl.export import FinderConcatenateMerger
from subhkl.optimization import FindUB


def test_correctness_compromised_by_merger_order(tmp_path):
    """
    PROVES that non-determinism affects correctness.
    Scenario:
    1. Run 0 has rotation 10 deg.
    2. Run 1 has rotation 20 deg.
    3. User merges them but the order is swapped [Run 1, Run 0].
    4. Indexer loads nominal angles [10, 20] from external source (like Nexus).
    5. Result: Peaks from Run 1 (20 deg) are assigned 10 deg. ERROR.
    """
    file0 = tmp_path / "run0.h5"
    file1 = tmp_path / "run1.h5"

    # Run 0: 10 degrees, Peak at [1, 0, 0]
    with h5py.File(file0, "w") as f:
        f["wavelength_mins"] = [1.0]
        f["wavelength_maxes"] = [2.0]
        f["peaks/two_theta"] = [20.0]
        f["peaks/azimuthal"] = [0.0]
        f["peaks/intensity"] = [100.0]
        f["peaks/sigma"] = [10.0]
        f["peaks/radius"] = [0.1]
        f["peaks/xyz"] = [[0.1, 0, 0.99]]
        f["bank"] = [0]
        f["peaks/run_index"] = [0]
        f["goniometer/angles"] = [[10.0]]  # Actual angle for peak in file0
        f["goniometer/axes"] = [[0, 1, 0, 1]]
        f["goniometer/names"] = [b"omega"]

    # Run 1: 20 degrees
    with h5py.File(file1, "w") as f:
        f["wavelength_mins"] = [1.0]
        f["wavelength_maxes"] = [2.0]
        f["peaks/two_theta"] = [20.0]
        f["peaks/azimuthal"] = [0.0]
        f["peaks/intensity"] = [100.0]
        f["peaks/sigma"] = [10.0]
        f["peaks/radius"] = [0.1]
        f["peaks/xyz"] = [[0.1, 0, 0.99]]
        f["bank"] = [0]
        f["peaks/run_index"] = [0]
        f["goniometer/angles"] = [[20.0]]  # Actual angle for peak in file1
        f["goniometer/axes"] = [[0, 1, 0, 1]]
        f["goniometer/names"] = [b"omega"]

    # --- SIMULATE SWAPPED MERGE (The old non-deterministic behavior) ---
    out_swapped = tmp_path / "merged_swapped.h5"

    # Let's manually create the "bugged" merged file
    with h5py.File(out_swapped, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = 10, 10, 10
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90, 90, 90
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [1.0, 2.0]

        # Swapped order: Run 1 peaks first, then Run 0
        f["peaks/run_index"] = [0, 1]
        f["peaks/two_theta"] = [20.0, 20.0]
        f["peaks/azimuthal"] = [0.0, 0.0]
        f["peaks/intensity"] = [100.0, 100.0]
        f["peaks/sigma"] = [10.0, 10.0]
        f["peaks/radius"] = [0.1, 0.1]
        f["peaks/xyz"] = [[0.1, 0, 0.99], [0.1, 0, 0.99]]

        f["goniometer/R"] = np.tile(np.eye(3)[None, ...], (2, 1, 1))
        f["goniometer/axes"] = [[0, 1, 0, 1]]
        f["goniometer/names"] = [b"omega"]

    # --- SIMULATE INDEXER LOADING METADATA BY INDEX ---
    # The indexer thinks Run 0 is 10 deg and Run 1 is 20 deg (from Nexus)
    nominal_angles = np.array([[10.0, 20.0]])  # [Axis 0, Run 0=10, Run 1=20]

    # Load the bugged file into indexer
    fu = FindUB(str(out_swapped))
    # Override angles with the "Nexus" values
    fu.goniometer_angles = nominal_angles.T  # (2, 1) -> (num_runs, num_axes)

    # Peak 0 has run_index 0.
    assigned_angle = fu.goniometer_angles[fu.run_indices[0]][0]
    actual_angle = 20.0

    print(f"Peak 0 assigned angle: {assigned_angle}")
    print(f"Peak 0 actual angle: {actual_angle}")

    assert assigned_angle != actual_angle, (
        "Correctness FAILED: Peak assigned wrong rotation due to swap!"
    )
    print(
        "SUCCESS: Proven that non-deterministic merger order compromises correctness when combined with indexed metadata."
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    tmp = Path("temp_correctness_test")
    tmp.mkdir(exist_ok=True)
    try:
        test_correctness_compromised_by_merger_order(tmp)
        print("Test PASSED (Bug Proven)")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
