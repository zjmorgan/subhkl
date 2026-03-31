import subprocess
import pytest
from pathlib import Path
import h5py
import numpy as np
import random
import time

INSTRUMENT = "MANDI"
LATTICE_PARAMS = [18.39, 56.55, 6.54, 90, 90, 90]
SPACE_GROUP = "F d d 2"
WAVELENGTH = [2.0, 4.5]

FINDER_PARAMS = [
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
]


@pytest.fixture
def random_mesolite_pair(test_data_dir):
    """Pick two random MANDI files with significantly different goniometer settings."""
    path = Path(test_data_dir) / "MANDI" / "mesolite"
    available_files = sorted(list(path.glob("MANDI_*.nxs.h5")))

    if len(available_files) < 2:
        pytest.skip(
            f"Not enough data files found in {path} (found {len(available_files)})"
        )

    rng = random.Random(42)  # Seeded for CI stability

    def get_angles(f):
        with h5py.File(f, "r") as h:
            try:
                omega = h["entry/DASlogs/BL11B:Mot:omega/value"][0]
                chi = h["entry/DASlogs/BL11B:Mot:chi/value"][0]
                phi = h["entry/DASlogs/BL11B:Mot:phi/value"][0]
                return np.array([omega, chi, phi])
            except KeyError:
                return None

    # Try to find a pair with at least 20 degrees difference in at least one angle
    start_time = time.time()
    while time.time() - start_time < 30:
        pair = rng.sample(available_files, 2)
        a1 = get_angles(pair[0])
        a2 = get_angles(pair[1])

        if a1 is not None and a2 is not None:
            diff = np.abs(a1 - a2)
            if np.any(diff >= 20.0):
                print(f"\n[Test] Selected diverse files: {[f.name for f in pair]}")
                print(f"[Test] Angles 1: {a1}")
                print(f"[Test] Angles 2: {a2}")
                print(f"[Test] Max diff: {np.max(diff):.2f} deg")
                return pair

    # Fallback to any two if diversity condition not met within timeout
    print(
        "\n[Test] WARNING: Could not find pair with >20 deg difference. Falling back to random pair."
    )
    pair = rng.sample(available_files, 2)
    return pair


def run_indexing_pipeline(input_h5, tmp_path, label):
    """Common logic for multi-stage indexing and verification."""

    # 1. Stage 1: Coarse multi-run indexing (0.5 deg)
    stage1_h5 = tmp_path / f"stage1_{label}.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "indexer",
            str(input_h5),
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

    # 2. Stage 2: Fine multi-run indexing (0.1 deg)
    stage2_h5 = tmp_path / f"stage2_{label}.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "indexer",
            str(input_h5),
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

    # 3. Stage 3: Super-fine refinement (0.05 deg) with CMA-ES
    indexed_h5 = tmp_path / f"indexed_{label}.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "indexer",
            str(input_h5),
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
            "--refine-goniometer",
        ],
        check=True,
    )

    # 4. Verify indexing
    with h5py.File(indexed_h5, "r") as f:
        assert "peaks/h" in f
        h = f["peaks/h"][()]
        indexed_count = np.sum((h != 0))
        print(f"[{label}] Indexed peaks: {indexed_count} / {len(h)}")

        from subhkl.instrument.metrics import compute_metrics

        m = compute_metrics(str(indexed_h5))
        ang_err = m["median_ang_err"]
        print(f"[{label}] Median angular error: {ang_err}")

        indexed_ratio = indexed_count / len(h)
        print(f"[{label}] Indexed ratio: {indexed_ratio:.2%}")

        assert indexed_count > 10
        assert indexed_ratio >= 0.75
        assert ang_err < 0.2

    return indexed_h5


@pytest.mark.slow
def test_mandi_multi_run_finder_merger(random_mesolite_pair, tmp_path):
    """Test indexing using the peak-finder merger workflow."""
    finder_outputs = []
    for input_file in random_mesolite_pair:
        output_file = tmp_path / f"{input_file.name}.finder.h5"
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
                *FINDER_PARAMS,
            ],
            check=True,
        )
        finder_outputs.append(output_file)

    finder_list_file = tmp_path / "finder_files.txt"
    with open(finder_list_file, "w") as f:
        for out in finder_outputs:
            f.write(f"{out}\n")

    merged_h5 = tmp_path / "merged_finder.h5"
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

    run_indexing_pipeline(merged_h5, tmp_path, "finder-merger")


@pytest.mark.slow
def test_mandi_multi_run_image_merger(random_mesolite_pair, tmp_path):
    """Test indexing using the image-merge workflow (production script style)."""
    reduced_files = []
    for input_file in random_mesolite_pair:
        reduced_file = tmp_path / f"{input_file.name}.reduced.h5"
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

    merged_h5 = tmp_path / "merged_images.h5"
    subprocess.run(
        [
            "python",
            "-m",
            "subhkl.io.parser",
            "finder",
            str(scan_master_h5),
            INSTRUMENT,
            "--output-filename",
            str(merged_h5),
            *FINDER_PARAMS,
        ],
        check=True,
    )

    run_indexing_pipeline(merged_h5, tmp_path, "image-merger")
