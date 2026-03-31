"""
Regression test for cosine loss indexer performance.
Exposes high angular deviation bug.
"""

from pathlib import Path

import pytest

from subhkl.io.parser import finder, indexer
from subhkl.instrument.metrics import compute_metrics

MESOLITE_FILE = "MANDI_11613.nxs.h5"
INSTRUMENT = "MANDI"
LATTICE_PARAMS = {
    "a": 18.39,
    "b": 56.55,
    "c": 6.54,
    "alpha": 90.0,
    "beta": 90.0,
    "gamma": 90.0,
}
SPACE_GROUP = "F d d 2"

INTEGRATOR_PARAMS = {
    "region_growth_minimum_intensity": 25.0,
    "region_growth_maximum_pixel_radius": 6.0,
    "peak_center_box_size": 3,
    "peak_smoothing_window_size": 5,
    "peak_minimum_pixels": 40,
    "peak_minimum_signal_to_noise": 4.0,
}

# Explicitly define all options to avoid typer OptionInfo errors
INDEXER_PARAMS = {
    "strategy_name": "DE",
    "n_runs": 4,
    "population_size": 100,
    "gens": 100,
    "seed": 12345,
    "sigma_init": None,
    "tolerance_deg": 0.2,
    "refine_lattice": False,
    "lattice_bound_frac": 0.05,
    "refine_goniometer": False,
    "refine_goniometer_axes": None,
    "goniometer_bound_deg": 5.0,
    "refine_sample": False,
    "sample_bound_meters": 2.0,
    "refine_beam": False,
    "beam_bound_deg": 1.0,
    "bootstrap_filename": None,
    "loss_method": "cosine",  # Target of investigation
    "d_min": None,
    "d_max": None,
    "hkl_search_range": 20,
    "search_window_size": 256,
    "batch_size": None,
    "window_batch_size": 32,
    "chunk_size": 256,
    "num_iters": 20,
    "top_k": 32,
    "B_sharpen": None,
}


@pytest.mark.slow
def test_cosine_indexer_accuracy(test_data_dir, tmp_path):
    """
    Test that the cosine indexer achieves < 0.3 deg median angular deviation.
    Known to fail currently.
    """
    mesolite_input_file = Path(test_data_dir) / "MANDI" / "mesolite" / MESOLITE_FILE
    if not mesolite_input_file.exists():
        pytest.skip("Mesolite data not found")

    finder_output = tmp_path / "mesolite.finder.h5"
    indexer_output = tmp_path / "mesolite.indexer.h5"

    finder(
        filename=str(mesolite_input_file),
        instrument=INSTRUMENT,
        output_filename=str(finder_output),
        finder_algorithm="thresholding",
        thresholding_noise_cutoff_quantile=0.99,
        **INTEGRATOR_PARAMS,
    )

    indexer(
        peaks_h5_filename=str(finder_output),
        output_peaks_filename=str(indexer_output),
        a=LATTICE_PARAMS["a"],
        b=LATTICE_PARAMS["b"],
        c=LATTICE_PARAMS["c"],
        alpha=LATTICE_PARAMS["alpha"],
        beta=LATTICE_PARAMS["beta"],
        gamma=LATTICE_PARAMS["gamma"],
        space_group=SPACE_GROUP,
        wavelength_min=2.0,
        wavelength_max=4.5,
        **INDEXER_PARAMS,
    )

    metrics = compute_metrics(str(indexer_output))
    median_ang_err = metrics["median_ang_err"]

    print(f"Cosine Indexer Median Angular Error: {median_ang_err:.4f}")

    # This assertion is expected to fail if the bug exists
    assert median_ang_err < 0.3
