import os
import subprocess
import pytest
from pathlib import Path
import h5py
import numpy as np

MESOLITE_FILES = ["MANDI_11613.nxs.h5", "MANDI_11614.nxs.h5"]
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
        subprocess.run(
            [
                "python",
                "-m",
                "subhkl.io.parser",
                "finder",
                str(input_file),
                INSTRUMENT,
                "--output-filename",
                str(output_file),
                "--finder-algorithm",
                "thresholding",
                "--thresholding-noise-cutoff-quantile",
                "0.99",
                "--region-growth-minimum-intensity",
                "3.0",
                "--region-growth-maximum-pixel-radius",
                "12.0",
                "--peak-center-box-size",
                "3",
                "--peak-smoothing-window-size",
                "5",
                "--peak-minimum-pixels",
                "40",
                "--peak-minimum-signal-to-noise",
                "0.0",
                "--peak-pixel-outlier-threshold",
                "4.0",
            ],
            check=True,
        )
        finder_outputs.append(output_file)

    # 2. Merge finder files
    finder_list_file = tmp_path / "finder_files.txt"
    with open(finder_list_file, "w") as f:
        for out in finder_outputs:
            f.write(f"{out}\n")

    merged_h5 = tmp_path / "merged.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "finder-merger",
            str(finder_list_file),
            str(merged_h5),
            *map(str, LATTICE_PARAMS),
            str(WAVELENGTH[0]),
            str(WAVELENGTH[1]),
            SPACE_GROUP,
        ],
        check=True,
    )

    # --- HARDEN: Alternative image-merge workflow ---
    reduced_files = []
    for filename in MESOLITE_FILES:
        input_file = mesolite_dir / filename
        reduced_file = tmp_path / f"{filename}.reduced.h5"
        subprocess.run(
            [
                "python",
                "-m",
                "subhkl.io.parser",
                "reduce",
                str(input_file),
                str(reduced_file),
                INSTRUMENT,
            ],
            check=True,
        )
        reduced_files.append(str(reduced_file))

    scan_master_h5 = tmp_path / "scan_master.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "merge-images",
            " ".join(reduced_files),
            str(scan_master_h5),
        ],
        check=True,
    )

    finder_merged_h5 = tmp_path / "finder_merged.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "finder",
            str(scan_master_h5),
            INSTRUMENT,
            "--output-filename",
            str(finder_merged_h5),
            "--finder-algorithm",
            "thresholding",
            "--thresholding-noise-cutoff-quantile",
            "0.99",
            "--region-growth-minimum-intensity",
            "3.0",
            "--region-growth-maximum-pixel-radius",
            "12.0",
            "--peak-center-box-size",
            "3",
            "--peak-smoothing-window-size",
            "5",
            "--peak-minimum-pixels",
            "40",
            "--peak-minimum-signal-to-noise",
            "0.0",
            "--peak-pixel-outlier-threshold",
            "4.0",
        ],
        check=True,
    )

    # Compare Workflow 1 vs Workflow 2
    with h5py.File(merged_h5, "r") as f1, h5py.File(finder_merged_h5, "r") as f2:
        n_peaks1 = len(f1["peaks/two_theta"])
        n_peaks2 = len(f2["peaks/two_theta"])
        print(f"Workflow 1 (finder-merger) peaks: {n_peaks1}")
        print(f"Workflow 2 (merge-images) peaks: {n_peaks2}")

        # They should be equivalent
        assert abs(n_peaks1 - n_peaks2) < 5

    # 3. Stage 1: Coarse multi-run indexing (0.5 deg)
    stage1_h5 = tmp_path / "stage1.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "indexer",
            str(merged_h5),
            str(stage1_h5),
            *map(str, LATTICE_PARAMS),
            SPACE_GROUP,
            "--n-runs",
            "20",
            "--popsize",
            "500",
            "--gens",
            "200",
            "--strategy",
            "de",
            "--hkl-search-range",
            "35",
            "--tolerance-deg",
            "0.5",
        ],
        check=True,
    )

    # 4. Stage 2: Fine multi-run indexing (0.1 deg)
    stage2_h5 = tmp_path / "stage2.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "indexer",
            str(merged_h5),
            str(stage2_h5),
            *map(str, LATTICE_PARAMS),
            SPACE_GROUP,
            "--bootstrap",
            str(stage1_h5),
            "--n-runs",
            "5",
            "--popsize",
            "500",
            "--gens",
            "200",
            "--strategy",
            "de",
            "--hkl-search-range",
            "35",
            "--tolerance-deg",
            "0.1",
            "--loss-method",
            "gaussian",
        ],
        check=True,
    )

    # 5. Stage 3: Super-fine refinement (0.05 deg) with CMA-ES
    indexed_h5 = tmp_path / "indexed.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "indexer",
            str(merged_h5),
            str(indexed_h5),
            *map(str, LATTICE_PARAMS),
            SPACE_GROUP,
            "--bootstrap",
            str(stage2_h5),
            "--n-runs",
            "1",
            "--popsize",
            "200",
            "--gens",
            "100",
            "--strategy",
            "cma_es",
            "--hkl-search-range",
            "35",
            "--tolerance-deg",
            "0.05",
            "--loss-method",
            "gaussian",
        ],
        check=True,
    )

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

        indexed_ratio = indexed_count / len(h)
        print(f"Indexed ratio: {indexed_ratio:.2%}")

        assert indexed_count > 10
        assert indexed_ratio >= 0.75
        assert ang_err < 0.2
