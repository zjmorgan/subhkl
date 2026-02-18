import numpy as np
import h5py
from subhkl.io.parser import indexer


def test_indexer_angles_stack_shape_crash(tmp_path):
    """
    Reproduces the IndexError reported by the user:
    IndexError: index 3 is out of bounds for axis 1 with size 3

    This happens when angles_stack is (num_peaks, num_axes) but the code
    thinks it needs to broadcast it using old_run_indices on axis 1.
    """
    peaks_h5 = tmp_path / "crash_input.h5"
    output_h5 = tmp_path / "indexed.h5"

    num_peaks = 10
    num_axes = 3

    # Simulate the shape reported by the user: (10756, 3)
    # We use 10 for speed.
    angles_stack = np.zeros((num_peaks, num_axes))
    # Fill with some different values to trigger expansion
    angles_stack[:, 2] = np.linspace(0, 360, num_peaks)

    old_run_indices = np.zeros(num_peaks, dtype=np.int32)

    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = 10, 10, 10
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90, 90, 90
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.9, 1.1]

        f["peaks/two_theta"] = np.full(num_peaks, 20.0)
        f["peaks/azimuthal"] = np.zeros(num_peaks)
        f["peaks/intensity"] = np.ones(num_peaks)
        f["peaks/sigma"] = np.full(num_peaks, 0.1)
        f["peaks/radius"] = np.zeros(num_peaks)
        f["peaks/run_index"] = old_run_indices
        f["peaks/xyz"] = np.random.rand(num_peaks, 3)

        # User reported shape (N, 3). Standard is often (3, N) or (3, N_runs).
        f["goniometer/angles"] = angles_stack
        f["goniometer/axes"] = np.eye(3)  # Not used for uniqueness but needed
        f["goniometer/names"] = [b"omega", b"chi", b"phi"]

    # This used to crash with IndexError
    # because parser.py:584 did: angles_stack[:, old_run_indices]
    # Now it should succeed.

    # Setup with 5 runs to ensure index > 2
    old_run_indices = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32)
    with h5py.File(peaks_h5, "a") as f:
        del f["peaks/run_index"]
        f["peaks/run_index"] = old_run_indices

    indexer(
        peaks_h5_filename=str(peaks_h5),
        output_peaks_filename=str(output_h5),
        a=10,
        b=10,
        c=10,
        alpha=90,
        beta=90,
        gamma=90,
        space_group="P 1",
        gens=1,
        n_runs=1,
    )
    print("SUCCESS: Indexer completed without crash!")

    # Verify output run_index expansion
    with h5py.File(output_h5, "r") as f:
        new_runs = f["peaks/run_index"][()]
        print(f"Expanded runs: {len(np.unique(new_runs))}")
        assert len(np.unique(new_runs)) == 10, (
            "Should have expanded to 10 unique geometries!"
        )


if __name__ == "__main__":
    from pathlib import Path

    test_indexer_angles_stack_shape_crash(Path("."))
