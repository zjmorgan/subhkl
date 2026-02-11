import os
import subprocess
import pytest
from pathlib import Path
import h5py
import numpy as np

MESOLITE_FILES = [
    "MANDI_11613.nxs.h5",
    "MANDI_11614.nxs.h5"
]
INSTRUMENT = "MANDI"
LATTICE_PARAMS = [18.39, 56.55, 6.54, 90, 90, 90]
SPACE_GROUP = "F d d 2"
WAVELENGTH = [2.0, 4.5]

@pytest.mark.slow
def test_mandi_multi_run_indexing(test_data_dir, tmp_path):
    """
    Test multi-run indexing using real MANDI mesolite data.
    """
    mesolite_dir = Path(test_data_dir) / "MANDI" / "mesolite"
    finder_outputs = []
    
    # 1. Run finder on multiple files
    for filename in MESOLITE_FILES:
        input_file = mesolite_dir / filename
        if not input_file.exists():
            pytest.skip(f"Data file {filename} not found")
            
        output_file = tmp_path / f"{filename}.finder.h5"
        subprocess.run([
            "python", "-m", "subhkl.io.parser", "finder",
            str(input_file), INSTRUMENT,
            "--output-filename", str(output_file),
            "--finder-algorithm", "thresholding",
            "--thresholding-noise-cutoff-quantile", "0.99",
            "--region-growth-minimum-intensity", "3.0",
            "--region-growth-maximum-pixel-radius", "12.0",
            "--peak-center-box-size", "3",
            "--peak-smoothing-window-size", "5",
            "--peak-minimum-pixels", "40",
            "--peak-minimum-signal-to-noise", "0.0",
            "--peak-pixel-outlier-threshold", "4.0"
        ], check=True)
        finder_outputs.append(output_file)
        
    # 2. Merge finder files
    finder_list_file = tmp_path / "finder_files.txt"
    with open(finder_list_file, "w") as f:
        for out in finder_outputs:
            f.write(f"{out}\n")
            
    merged_h5 = tmp_path / "merged.h5"
    subprocess.run([
        "python", "-m", "subhkl.io.parser", "finder-merger",
        str(finder_list_file), str(merged_h5),
        *map(str, LATTICE_PARAMS),
        str(WAVELENGTH[0]), str(WAVELENGTH[1]),
        SPACE_GROUP
    ], check=True)
    
    # 3. Stage 1: Coarse multi-run indexing (0.5 deg)
    stage1_h5 = tmp_path / "stage1.h5"
    subprocess.run([
        "python", "-m", "subhkl.io.parser", "indexer",
        str(merged_h5), str(stage1_h5),
        *map(str, LATTICE_PARAMS),
        SPACE_GROUP,
        "--n-runs", "20",
        "--popsize", "500",
        "--gens", "200",
        "--strategy", "de",
        "--hkl-search-range", "35",
        "--tolerance-deg", "0.5"
    ], check=True)

    # 4. Stage 2: Fine multi-run indexing (0.1 deg)
    stage2_h5 = tmp_path / "stage2.h5"
    subprocess.run([
        "python", "-m", "subhkl.io.parser", "indexer",
        str(merged_h5), str(stage2_h5),
        *map(str, LATTICE_PARAMS),
        SPACE_GROUP,
        "--bootstrap", str(stage1_h5),
        "--n-runs", "5",
        "--popsize", "500",
        "--gens", "200",
        "--strategy", "de",
        "--hkl-search-range", "35",
        "--tolerance-deg", "0.1",
        "--loss-method", "gaussian"
    ], check=True)

    # 5. Stage 3: Super-fine refinement (0.05 deg) with CMA-ES
    indexed_h5 = tmp_path / "indexed.h5"
    subprocess.run([
        "python", "-m", "subhkl.io.parser", "indexer",
        str(merged_h5), str(indexed_h5),
        *map(str, LATTICE_PARAMS),
        SPACE_GROUP,
        "--bootstrap", str(stage2_h5),
        "--n-runs", "1",
        "--popsize", "200",
        "--gens", "100",
        "--strategy", "cma_es",
        "--hkl-search-range", "35",
        "--tolerance-deg", "0.05",
        "--loss-method", "gaussian"
    ], check=True)
    
    # 6. Verify indexing (check if peaks have HKLs assigned)
    with h5py.File(indexed_h5, "r") as f:
        assert "peaks/h" in f
        h = f["peaks/h"][()]
        indexed_count = np.sum((h != 0))
        print(f"Indexed peaks: {indexed_count} / {len(h)}")
        
        # Check run_index
        assert "peaks/run_index" in f
        run_indices = f["peaks/run_index"][()]
        assert len(run_indices) == len(h)
        if len(h) > 0:
            assert np.max(run_indices) >= 1
        
        from subhkl.metrics import compute_metrics
        m = compute_metrics(str(indexed_h5))
        ang_err = m["median_ang_err"]
        print(f"Median angular error: {ang_err}")
        
        assert indexed_count > 10
        assert ang_err < 0.3
