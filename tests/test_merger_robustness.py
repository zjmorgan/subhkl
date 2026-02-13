import h5py
import numpy as np
import pytest
import os
from subhkl.export import BaseConcatenateMerger


def test_merger_run_offset_increment(tmp_path):
    """
    Test that run_index is correctly incremented across multiple files
    even if per_file_keys is empty.
    """
    file1 = tmp_path / "file1.h5"
    file2 = tmp_path / "file2.h5"

    with h5py.File(file1, "w") as f:
        f.create_dataset("peaks/intensity", data=np.array([10.0, 20.0]))
        f.create_dataset("peaks/run_index", data=np.array([0, 0], dtype=np.int32))

    with h5py.File(file2, "w") as f:
        f.create_dataset("peaks/intensity", data=np.array([30.0]))
        f.create_dataset("peaks/run_index", data=np.array([0], dtype=np.int32))

    output = tmp_path / "merged.h5"

    # Merge without per_file_keys
    merger = BaseConcatenateMerger(
        h5_files=[str(file1), str(file2)],
        copy_keys=[],
        merge_keys=["peaks/intensity", "peaks/run_index"],
        per_file_keys=[],
    )
    merger.merge(str(output))

    with h5py.File(output, "r") as f:
        intensities = f["peaks/intensity"][()]
        run_indices = f["peaks/run_index"][()]

        assert len(intensities) == 3
        assert np.array_equal(intensities, [10.0, 20.0, 30.0])

        # BEFORE FIX: run_indices would be [0, 0, 0] because run_offset was 0
        # AFTER FIX: run_indices should be [0, 0, 1]
        assert np.array_equal(run_indices, [0, 0, 1])


def test_merger_with_per_peak_rotation(tmp_path):
    """
    Test that per-peak rotation matrices are correctly concatenated.
    """
    file1 = tmp_path / "file1.h5"
    file2 = tmp_path / "file2.h5"

    R1 = np.eye(3)[None, :, :].repeat(2, axis=0)  # 2 peaks
    R2 = (np.eye(3) * 2)[None, :, :]  # 1 peak

    with h5py.File(file1, "w") as f:
        f.create_dataset("peaks/intensity", data=np.array([1.0, 2.0]))
        f.create_dataset("goniometer/R", data=R1)

    with h5py.File(file2, "w") as f:
        f.create_dataset("peaks/intensity", data=np.array([3.0]))
        f.create_dataset("goniometer/R", data=R2)

    output = tmp_path / "merged.h5"

    merger = BaseConcatenateMerger(
        h5_files=[str(file1), str(file2)],
        copy_keys=[],
        merge_keys=["peaks/intensity", "goniometer/R"],
        per_file_keys=[],
    )
    merger.merge(str(output))

    with h5py.File(output, "r") as f:
        R_merged = f["goniometer/R"][()]
        assert R_merged.shape == (3, 3, 3)
        assert np.array_equal(R_merged[0], np.eye(3))
        assert np.array_equal(R_merged[1], np.eye(3))
        assert np.array_equal(R_merged[2], np.eye(3) * 2)
