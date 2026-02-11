import os
import h5py
import numpy as np
import pytest
from pathlib import Path
from subhkl.io.parser import reduce, merge_images, finder

@pytest.mark.integration
def test_multi_run_merged_finder_run_id_consistency(test_data_dir, tmp_path):
    """
    Test that peaks found on a merged image stack have correct run_index.
    This reproduces the bug where all peaks are assigned to the last run.
    """
    mesolite_dir = Path(test_data_dir) / "MANDI" / "mesolite"
    # Use 3 files to test multi-run mapping
    files = sorted(list(mesolite_dir.glob("MANDI_1161[3-5].nxs.h5")))
    if not files:
        pytest.skip("MANDI mesolite data not found")
        
    reduced_files = []
    for i, f in enumerate(files):
        out = tmp_path / f"run_{i}.reduce.h5"
        reduce(str(f), str(out), "MANDI", wavelength_min=None, wavelength_max=None)
        reduced_files.append(str(out))
        
    master_h5 = tmp_path / "scan_master.h5"
    # merge-images expects a glob pattern or space-separated list
    merge_images(" ".join(reduced_files), str(master_h5))
    
    finder_h5 = tmp_path / "finder.h5"
    finder(
        str(master_h5), 
        "MANDI", 
        output_filename=str(finder_h5),
        finder_algorithm="thresholding",
        region_growth_minimum_sigma=2.5,
        peak_minimum_pixels=50,
        peak_minimum_signal_to_noise=2.5,
        thresholding_noise_cutoff_quantile=0.95
    )
    
    with h5py.File(finder_h5, "r") as f:
        assert "peaks/run_index" in f, "run_index missing from finder output"
        run_indices = f["peaks/run_index"][()]
        unique_runs = np.unique(run_indices)
        
        print(f"Unique runs found in peaks: {unique_runs}")
        
        # We expect peaks from all 3 runs (0, 1, 2)
        # The bug causes all peaks to be assigned to run 2
        assert len(unique_runs) > 1, f"Expected multiple runs, but found only: {unique_runs}"
        assert 0 in unique_runs, "Missing peaks from run 0"
        assert 1 in unique_runs, "Missing peaks from run 1"
        assert 2 in unique_runs, "Missing peaks from run 2"
