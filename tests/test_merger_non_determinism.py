import numpy as np
import h5py
import os
import pytest
from subhkl.export import FinderConcatenateMerger

def test_merger_input_order_dependency(tmp_path):
    """
    Demonstrates that FinderConcatenateMerger output depends on input file order.
    Specifically, run_index assignment is cumulative.
    """
    file1 = tmp_path / "peaks1.h5"
    file2 = tmp_path / "peaks2.h5"
    
    # Create two files with 1 unique peak each
    for fpath, xyz_val in [(file1, [0.1, 0, 0.99]), (file2, [0.2, 0, 0.98])]:
        with h5py.File(fpath, "w") as f:
            f["wavelength_mins"] = [1.0]
            f["wavelength_maxes"] = [2.0]
            f["peaks/two_theta"] = [20.0]
            f["peaks/azimuthal"] = [0.0]
            f["peaks/intensity"] = [100.0]
            f["peaks/sigma"] = [10.0]
            f["peaks/radius"] = [0.1]
            f["peaks/xyz"] = [xyz_val]
            f["bank"] = [0]
            f["peaks/image_index"] = [0]
            f["peaks/run_index"] = [0]
            f["goniometer/R"] = np.eye(3)[None, ...]
            f["goniometer/angles"] = [[0.0]]
            f["goniometer/axes"] = [[0, 1, 0, 1]]
            f["goniometer/names"] = [b"omega"]

    # 1. Merge [File 1, File 2]
    out12 = tmp_path / "merged12.h5"
    merger12 = FinderConcatenateMerger([str(file1), str(file2)])
    merger12.merge(str(out12))
    
    # 2. Merge [File 2, File 1]
    out21 = tmp_path / "merged21.h5"
    merger21 = FinderConcatenateMerger([str(file2), str(file1)])
    merger21.merge(str(out21))
    
    with h5py.File(out12, "r") as f12, h5py.File(out21, "r") as f21:
        runs12 = f12["peaks/run_index"][()]
        runs21 = f21["peaks/run_index"][()]
        
        print(f"Order [1, 2] runs: {runs12}")
        print(f"Order [2, 1] runs: {runs21}")
        
        # They will both be [0, 1] probably, BUT the peaks are different.
        # Peak from File 1 is at index 0 in out12, but index 1 in out21.
        
        # If we compare them as sets of (xyz, run_index), they should ideally be the same
        # if the system was deterministic/order-invariant.
        
        xyz12 = f12["peaks/xyz"][()]
        xyz21 = f21["peaks/xyz"][()]
        
        # Create a "state" for each peak
        state12 = set(tuple(row) + (run,) for row, run in zip(xyz12, runs12))
        state21 = set(tuple(row) + (run,) for row, run in zip(xyz21, runs21))
        
        # If order-invariant, state12 == state21
        # But in reality, in out12: Peak1->Run0, Peak2->Run1
        # In out21: Peak2->Run0, Peak1->Run1
        # So Peak1 has different Run ID depending on file order!
        
        is_invariant = (state12 == state21)
        if is_invariant:
            print("DETERMINISM CONFIRMED: Peak run_index is invariant to input file order.")
            
    assert is_invariant, "Merger should now be deterministic!"

if __name__ == "__main__":
    import sys
    from pathlib import Path
    tmp = Path("temp_merger_test")
    tmp.mkdir(exist_ok=True)
    try:
        test_merger_input_order_dependency(tmp)
        print("Test PASSED (Bug Found - Non-determinism detected)")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
