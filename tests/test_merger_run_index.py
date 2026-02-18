import numpy as np
import h5py
from subhkl.export import FinderConcatenateMerger


def test_finder_merger_run_index_collision(tmp_path):
    """
    Reproduces the bug where FinderConcatenateMerger causes run_index collisions
    if input files contain multiple runs.
    """
    file1 = tmp_path / "peaks1.h5"
    file2 = tmp_path / "peaks2.h5"
    merged_h5 = tmp_path / "merged.h5"

    # 1. Create File 1 with 2 runs (0, 1)
    with h5py.File(file1, "w") as f:
        # Minimal data for merge_keys
        f["wavelength_mins"] = [0.9, 0.9]
        f["wavelength_maxes"] = [1.1, 1.1]
        f["peaks/two_theta"] = [20.0, 25.0]
        f["peaks/azimuthal"] = [0.0, 10.0]
        f["peaks/intensity"] = [100.0, 200.0]
        f["peaks/sigma"] = [10.0, 20.0]
        f["peaks/radius"] = [0.1, 0.1]
        f["peaks/xyz"] = np.random.rand(2, 3)
        f["bank"] = [0, 0]
        f["peaks/image_index"] = [0, 1]
        f["peaks/run_index"] = [0, 1]  # 2 unique runs
        f["goniometer/R"] = np.random.rand(2, 3, 3)
        f["goniometer/angles"] = np.random.rand(2, 3)
        f["goniometer/axes"] = [[0, 1, 0, 1]]
        f["goniometer/names"] = [b"omega"]

    # 2. Create File 2 with 2 runs (0, 1)
    with h5py.File(file2, "w") as f:
        f["wavelength_mins"] = [0.9, 0.9]
        f["wavelength_maxes"] = [1.1, 1.1]
        f["peaks/two_theta"] = [30.0, 35.0]
        f["peaks/azimuthal"] = [0.0, 10.0]
        f["peaks/intensity"] = [300.0, 400.0]
        f["peaks/sigma"] = [30.0, 40.0]
        f["peaks/radius"] = [0.1, 0.1]
        f["peaks/xyz"] = np.random.rand(2, 3)
        f["bank"] = [0, 0]
        f["peaks/image_index"] = [0, 1]
        f["peaks/run_index"] = [0, 1]  # 2 unique runs
        f["goniometer/R"] = np.random.rand(2, 3, 3)
        f["goniometer/angles"] = np.random.rand(2, 3)
        f["goniometer/axes"] = [[0, 1, 0, 1]]
        f["goniometer/names"] = [b"omega"]

    # 3. Merge
    merger = FinderConcatenateMerger([str(file1), str(file2)])
    merger.merge(str(merged_h5))

    with h5py.File(merged_h5, "r") as f:
        run_indices = f["peaks/run_index"][()]
        print(f"DEBUG: Merged run_indices: {run_indices}")

        # We expect 4 unique runs: [0, 1, 2, 3]
        # But File 1 (0, 1) + 0 -> (0, 1)
        # And File 2 (0, 1) + 1 -> (1, 2)
        # Result: [0, 1, 1, 2] -> COLLISION at index 1!

        unique_runs = np.unique(run_indices)
        assert len(unique_runs) == 4, (
            f"Run Index Collision! Expected 4 unique runs, got {len(unique_runs)}: {run_indices}"
        )
