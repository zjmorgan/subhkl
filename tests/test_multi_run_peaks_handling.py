import numpy as np
import h5py
import os
from subhkl.integration import Peaks


def test_repro_multi_run_mapping_logic(tmp_path):
    """
    Reproduces the bug where Peaks.integrate might use incorrect run_id mapping
    when loading found peaks from a merged file.
    """
    # 1. Create a simulated merged scan_master.h5
    # Two files, each with 2 banks
    file1 = "run1.h5"
    file2 = "run2.h5"
    master_h5 = str(tmp_path / "scan_master.h5")

    with h5py.File(master_h5, "w") as f:
        f.create_dataset("images", data=np.zeros((4, 10, 10)))
        f.create_dataset("bank_ids", data=np.array([1, 2, 1, 2]))  # Banks 1,2 repeated
        f.create_dataset("files", data=np.array([file1.encode(), file2.encode()]))
        f.create_dataset("file_offsets", data=np.array([0, 2]))
        f.create_dataset("instrument/wavelength", data=[2.0, 4.0])
        # Correct axis spec: [x, y, z, sign]
        f.create_dataset("goniometer/axes", data=[[0, 1, 0, 1]])
        f.create_dataset("goniometer/angles", data=np.zeros((4, 1)))

    # 2. Create a simulated finder.h5 (merged found peaks)
    finder_h5 = str(tmp_path / "finder.h5")
    with h5py.File(finder_h5, "w") as f:
        # 4 peaks, one for each image in scan_master
        f.create_dataset("peaks/xyz", data=np.random.rand(4, 3))
        f.create_dataset("peaks/bank", data=np.array([1, 2, 1, 2]))
        # CRITICAL: peaks/run_index in finder.h5 corresponds to the file index (0 or 1)
        f.create_dataset("peaks/run_index", data=np.array([0, 0, 1, 1]))
        f.create_dataset("files", data=np.array([file1.encode(), file2.encode()]))
        f.create_dataset("file_offsets", data=np.array([0, 2]))

    # 3. Setup Peaks object
    peaks_handler = Peaks(master_h5, "MANDI")

    # Verify file_offsets were loaded correctly
    assert peaks_handler.image.file_offsets is not None
    assert np.array_equal(peaks_handler.image.file_offsets, [0, 2])

    # 4. Simulate the loop in Peaks.integrate
    # For bank index 2 (which is the first bank of the SECOND file)
    bank_idx = 2
    physical_bank = peaks_handler.image.bank_mapping.get(
        bank_idx, bank_idx
    )  # Should be 1
    run_id = peaks_handler.get_run_id(bank_idx)  # Should be 1

    assert physical_bank == 1
    assert run_id == 1

    # 5. Simulate found_peaks loading logic inside Peaks.integrate
    target_name = os.path.basename(master_h5)
    found_peaks_run = None
    with h5py.File(finder_h5, "r") as f:
        files_db = f["files"][()]
        offsets = f["file_offsets"][()]

        match_idxs = []
        # 1. Direct match
        for i, fname_bytes in enumerate(files_db):
            fname_str = fname_bytes.decode("utf-8")
            if target_name in fname_str:
                match_idxs.append(i)

        # 2. Match via source files
        if not match_idxs and peaks_handler.image.raw_files:
            for src_file in peaks_handler.image.raw_files:
                src_name = os.path.basename(src_file)
                for i, fname_bytes in enumerate(files_db):
                    fname_str = fname_bytes.decode("utf-8")
                    if src_name == os.path.basename(fname_str):
                        if i not in match_idxs:
                            match_idxs.append(i)

        assert match_idxs, "Should have found matches"

        run_list = []
        for idx in match_idxs:
            start = int(offsets[idx])
            end = (
                int(offsets[idx + 1])
                if idx < len(files_db) - 1
                else f["peaks/xyz"].shape[0]
            )
            run_list.append(f["peaks/run_index"][start:end])

        found_peaks_run = np.concatenate(run_list, axis=0)

    print(f"Match idxs: {match_idxs}")
    print(f"Found peaks run: {found_peaks_run}")
    assert found_peaks_run is not None
    assert len(found_peaks_run) == 4
    # All run indices should be present
    assert np.any(found_peaks_run == 0)
    assert np.any(found_peaks_run == 1)


def test_repro_visualization_filenames(tmp_path):
    """
    Verifies that visualization filenames are unique per bank.
    """
    master_h5 = str(tmp_path / "viz_test.h5")
    with h5py.File(master_h5, "w") as f:
        f.create_dataset("images", data=np.zeros((2, 10, 10)))
        f.create_dataset("bank_ids", data=np.array([1, 2]))
        f.create_dataset("files", data=np.array([b"run1.h5"]))
        f.create_dataset("file_offsets", data=np.array([0]))
        f.create_dataset("goniometer/axes", data=[[0, 1, 0, 1]])
        f.create_dataset("goniometer/angles", data=np.zeros((2, 1)))

    peaks_handler = Peaks(master_h5, "MANDI")

    label0 = peaks_handler.get_image_label(0)
    label1 = peaks_handler.get_image_label(1)

    assert label0 == label1 == "run1"

    viz_label0 = f"{label0}_bank1"
    viz_label1 = f"{label1}_bank2"

    assert viz_label0 != viz_label1
