"""
Integration tests for MANDI mesolite data processing workflow.

This test suite validates the complete single-run crystallography workflow:
1. Peak finding (finder)
2. Indexing (indexer)
3. Peak prediction (peak_predictor)
4. Integration (integrator)
5. MTZ export (mtz_exporter)

Test data is automatically downloaded from Zenodo (DOI: 10.5281/zenodo.18475332)
"""

import os
import tempfile
from pathlib import Path

import h5py
import pytest

from subhkl.io.parser import (
    finder,
    indexer,
    peak_predictor,
    integrator,
    mtz_exporter,
)


# Test data configuration
MESOLITE_FILE = "MANDI_11613.nxs.h5"  # Use the first file from the Zenodo dataset
INSTRUMENT = "MANDI"

# Expected lattice parameters for mesolite (from the bash script)
LATTICE_PARAMS = {
    "a": 18.39,
    "b": 56.55,
    "c": 6.54,
    "alpha": 90.0,
    "beta": 90.0,
    "gamma": 90.0,
}
SPACE_GROUP = "F d d 2"

# Integration parameters (from bash script)
INTEGRATOR_PARAMS = {
    "region_growth_minimum_intensity": 25.0,
    "region_growth_maximum_pixel_radius": 6.0,
    "peak_center_box_size": 3,
    "peak_smoothing_window_size": 5,
    "peak_minimum_pixels": 40,
    "peak_minimum_signal_to_noise": 4.0,
}

# Indexer parameters with explicit values for all typer.Option parameters
# to avoid OptionInfo type errors when calling functions directly
INDEXER_DEFAULTS = {
    "strategy_name": "DE",
    "n_runs": 1,
    "population_size": 1000,
    "gens": 100,
    "seed": 12345,
    "sigma_init": None,
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
    "loss_method": "cosine",
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


@pytest.fixture(name="mesolite_input_file")
def fixture__mesolite_input_file(test_data_dir):
    """Provide path to mesolite test data file."""
    filepath = Path(test_data_dir) / "MANDI" / "mesolite" / MESOLITE_FILE
    
    if not filepath.exists():
        pytest.skip(f"Mesolite test file not found: {filepath}")
    
    return str(filepath)


@pytest.fixture(name="temp_output_dir")
def fixture__temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestMandiMesoliteSingleRun:
    """Test suite for single-run MANDI mesolite workflow."""

    def test_01_finder(self, mesolite_input_file, temp_output_dir):
        """Test peak finding on mesolite data."""
        output_file = os.path.join(temp_output_dir, "mesolite.finder.h5")
        
        finder(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            output_filename=output_file,
            finder_algorithm="thresholding",
            thresholding_noise_cutoff_quantile=0.99,
            **INTEGRATOR_PARAMS,
        )
        
        # Verify output file was created
        assert os.path.exists(output_file), "Finder output file was not created"
        
        # Verify HDF5 structure
        with h5py.File(output_file, "r") as f:
            # Check for expected groups
            assert "peaks" in f, "Expected 'peaks' group in finder output"
            assert "instrument" in f, "Expected 'instrument' group"
            assert "goniometer" in f, "Expected 'goniometer' group"
            
            # Check peaks group has data
            peaks = f["peaks"]
            required_datasets = ["intensity", "two_theta", "azimuthal", "xyz"]
            for dataset in required_datasets:
                assert dataset in peaks, f"Missing dataset '{dataset}' in peaks"
                assert len(peaks[dataset]) > 0, f"No peaks in dataset '{dataset}'"
            
            num_peaks = len(peaks["intensity"])
            print(f"\n  Found {num_peaks} peaks")
        
        print(f"✓ Finder completed: {output_file}")

    def test_02_indexer(self, mesolite_input_file, temp_output_dir):
        """Test indexing using differential evolution strategy."""
        # First run finder
        finder_output = os.path.join(temp_output_dir, "mesolite.finder.h5")
        finder(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            output_filename=finder_output,
            finder_algorithm="thresholding",
            thresholding_noise_cutoff_quantile=0.99,
            **INTEGRATOR_PARAMS,
        )
        
        # Run indexer
        indexer_output = os.path.join(temp_output_dir, "mesolite.indexer.h5")
        indexer(
            peaks_h5_filename=finder_output,
            output_peaks_filename=indexer_output,
            a=LATTICE_PARAMS["a"],
            b=LATTICE_PARAMS["b"],
            c=LATTICE_PARAMS["c"],
            alpha=LATTICE_PARAMS["alpha"],
            beta=LATTICE_PARAMS["beta"],
            gamma=LATTICE_PARAMS["gamma"],
            space_group=SPACE_GROUP,
            wavelength_min=2.0,
            wavelength_max=4.5,
            **INDEXER_DEFAULTS,
        )
        
        # Verify output file
        assert os.path.exists(indexer_output), "Indexer output file was not created"
        
        # Verify indexing results
        with h5py.File(indexer_output, "r") as f:
            # Check sample parameters
            assert "sample" in f, "Missing 'sample' group"
            sample = f["sample"]
            
            assert "a" in sample, "Missing lattice parameter 'a'"
            assert "b" in sample, "Missing lattice parameter 'b'"
            assert "c" in sample, "Missing lattice parameter 'c'"
            assert "U" in sample, "Missing orientation matrix 'U'"
            assert "B" in sample, "Missing reciprocal lattice matrix 'B'"
            
            # Check orientation matrix shape
            U = sample["U"][()]
            assert U.shape == (3, 3), f"U matrix has incorrect shape: {U.shape}"
            
            # Check instrument parameters
            assert "instrument" in f, "Missing 'instrument' group"
            assert "goniometer" in f, "Missing 'goniometer' group"
        
        print(f"✓ Indexer completed: {indexer_output}")

    def test_03_peak_predictor(self, mesolite_input_file, temp_output_dir):
        """Test peak prediction from indexed solution."""
        # Run finder and indexer first
        finder_output = os.path.join(temp_output_dir, "mesolite.finder.h5")
        finder(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            output_filename=finder_output,
            finder_algorithm="thresholding",
            thresholding_noise_cutoff_quantile=0.99,
            **INTEGRATOR_PARAMS,
        )
        
        indexer_output = os.path.join(temp_output_dir, "mesolite.indexer.h5")
        indexer(
            peaks_h5_filename=finder_output,
            output_peaks_filename=indexer_output,
            a=LATTICE_PARAMS["a"],
            b=LATTICE_PARAMS["b"],
            c=LATTICE_PARAMS["c"],
            alpha=LATTICE_PARAMS["alpha"],
            beta=LATTICE_PARAMS["beta"],
            gamma=LATTICE_PARAMS["gamma"],
            space_group=SPACE_GROUP,
            wavelength_min=2.0,
            wavelength_max=4.5,
            **INDEXER_DEFAULTS,
        )
        
        # Run peak predictor
        predictor_output = os.path.join(temp_output_dir, "mesolite.peak_predictor.h5")
        peak_predictor(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            indexed_hdf5_filename=indexer_output,
            integration_peaks_filename=predictor_output,
            d_min=1.35,
        )
        
        # Verify output
        assert os.path.exists(predictor_output), "Peak predictor output file was not created"
        
        with h5py.File(predictor_output, "r") as f:
            # Check predicted peaks
            assert "banks" in f, "Missing 'banks' group"
            banks = f["banks"]
            
            # Should have predicted peaks
            assert len(banks.keys()) > 0, "No predicted peaks found"
            
            # Check first bank structure
            first_bank = list(banks.keys())[0]
            bank_group = banks[first_bank]
            
            # Should have Miller indices
            assert "h" in bank_group, "Missing Miller index 'h'"
            assert "k" in bank_group, "Missing Miller index 'k'"
            assert "l" in bank_group, "Missing Miller index 'l'"
            assert "i" in bank_group, "Missing pixel coordinate 'i'"
            assert "j" in bank_group, "Missing pixel coordinate 'j'"
            
            # Verify we have predictions
            assert len(bank_group["h"]) > 0, "No predicted peaks in first bank"
        
        print(f"✓ Peak predictor completed: {predictor_output}")

    def test_04_integrator(self, mesolite_input_file, temp_output_dir):
        """Test peak integration."""
        # Run full pipeline up to peak prediction
        finder_output = os.path.join(temp_output_dir, "mesolite.finder.h5")
        finder(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            output_filename=finder_output,
            finder_algorithm="thresholding",
            thresholding_noise_cutoff_quantile=0.99,
            **INTEGRATOR_PARAMS,
        )
        
        indexer_output = os.path.join(temp_output_dir, "mesolite.indexer.h5")
        indexer(
            peaks_h5_filename=finder_output,
            output_peaks_filename=indexer_output,
            a=LATTICE_PARAMS["a"],
            b=LATTICE_PARAMS["b"],
            c=LATTICE_PARAMS["c"],
            alpha=LATTICE_PARAMS["alpha"],
            beta=LATTICE_PARAMS["beta"],
            gamma=LATTICE_PARAMS["gamma"],
            space_group=SPACE_GROUP,
            wavelength_min=2.0,
            wavelength_max=4.5,
            **INDEXER_DEFAULTS,
        )
        
        predictor_output = os.path.join(temp_output_dir, "mesolite.peak_predictor.h5")
        peak_predictor(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            indexed_hdf5_filename=indexer_output,
            integration_peaks_filename=predictor_output,
            d_min=1.35,
        )
        
        # Run integrator
        integrator_output = os.path.join(temp_output_dir, "mesolite.integrator.h5")
        integrator(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            integration_peaks_filename=predictor_output,
            output_filename=integrator_output,
            **INTEGRATOR_PARAMS,
        )
        
        # Verify output
        assert os.path.exists(integrator_output), "Integrator output file was not created"
        
        with h5py.File(integrator_output, "r") as f:
            # Check for integrated intensities in peaks group
            assert "peaks" in f, "Missing 'peaks' group"
            peaks = f["peaks"]
            
            # Should have the required datasets (may be empty for single-file test)
            required_datasets = ["h", "k", "l", "intensity", "sigma"]
            for dataset in required_datasets:
                assert dataset in peaks, f"Missing dataset 'peaks/{dataset}'"
            
            num_peaks = len(peaks["h"])
            print(f"  Integrated {num_peaks} peaks")
        
        print(f"✓ Integrator completed: {integrator_output}")

    def test_05_mtz_exporter(self, mesolite_input_file, temp_output_dir):
        """Test MTZ file export."""
        # Run full pipeline
        finder_output = os.path.join(temp_output_dir, "mesolite.finder.h5")
        finder(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            output_filename=finder_output,
            finder_algorithm="thresholding",
            thresholding_noise_cutoff_quantile=0.99,
            **INTEGRATOR_PARAMS,
        )
        
        indexer_output = os.path.join(temp_output_dir, "mesolite.indexer.h5")
        indexer(
            peaks_h5_filename=finder_output,
            output_peaks_filename=indexer_output,
            a=LATTICE_PARAMS["a"],
            b=LATTICE_PARAMS["b"],
            c=LATTICE_PARAMS["c"],
            alpha=LATTICE_PARAMS["alpha"],
            beta=LATTICE_PARAMS["beta"],
            gamma=LATTICE_PARAMS["gamma"],
            space_group=SPACE_GROUP,
            wavelength_min=2.0,
            wavelength_max=4.5,
            **INDEXER_DEFAULTS,
        )
        
        predictor_output = os.path.join(temp_output_dir, "mesolite.peak_predictor.h5")
        peak_predictor(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            indexed_hdf5_filename=indexer_output,
            integration_peaks_filename=predictor_output,
            d_min=1.35,
        )
        
        integrator_output = os.path.join(temp_output_dir, "mesolite.integrator.h5")
        integrator(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            integration_peaks_filename=predictor_output,
            output_filename=integrator_output,
            **INTEGRATOR_PARAMS,
        )
        
        # Export to MTZ
        mtz_output = os.path.join(temp_output_dir, "mesolite.mtz")
        mtz_exporter(
            indexed_h5_filename=integrator_output,
            output_mtz_filename=mtz_output,
            space_group=SPACE_GROUP,
        )
        
        # Verify MTZ file was created
        assert os.path.exists(mtz_output), "MTZ output file was not created"
        assert os.path.getsize(mtz_output) > 0, "MTZ file is empty"
        
        print(f"✓ MTZ exporter completed: {mtz_output}")

    @pytest.mark.slow
    def test_full_workflow(self, mesolite_input_file, temp_output_dir):
        """
        Integration test for complete single-run workflow.
        
        This test runs all steps in sequence and validates the final output.
        Mark as 'slow' since it runs the complete pipeline.
        """
        # Define output filenames
        finder_output = os.path.join(temp_output_dir, "mesolite.finder.h5")
        indexer_output = os.path.join(temp_output_dir, "mesolite.indexer.h5")
        predictor_output = os.path.join(temp_output_dir, "mesolite.peak_predictor.h5")
        integrator_output = os.path.join(temp_output_dir, "mesolite.integrator.h5")
        mtz_output = os.path.join(temp_output_dir, "mesolite.mtz")
        
        # Step 1: Find peaks
        print("\n[1/5] Running peak finder...")
        finder(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            output_filename=finder_output,
            finder_algorithm="thresholding",
            thresholding_noise_cutoff_quantile=0.99,
            **INTEGRATOR_PARAMS,
        )
        assert os.path.exists(finder_output)
        
        # Step 2: Index peaks
        print("[2/5] Running indexer...")
        indexer(
            peaks_h5_filename=finder_output,
            output_peaks_filename=indexer_output,
            a=LATTICE_PARAMS["a"],
            b=LATTICE_PARAMS["b"],
            c=LATTICE_PARAMS["c"],
            alpha=LATTICE_PARAMS["alpha"],
            beta=LATTICE_PARAMS["beta"],
            gamma=LATTICE_PARAMS["gamma"],
            space_group=SPACE_GROUP,
            wavelength_min=2.0,
            wavelength_max=4.5,
            **INDEXER_DEFAULTS,
        )
        assert os.path.exists(indexer_output)
        
        # Step 3: Predict peaks
        print("[3/5] Running peak predictor...")
        peak_predictor(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            indexed_hdf5_filename=indexer_output,
            integration_peaks_filename=predictor_output,
            d_min=1.35,
        )
        assert os.path.exists(predictor_output)
        
        # Step 4: Integrate peaks
        print("[4/5] Running integrator...")
        integrator(
            filename=mesolite_input_file,
            instrument=INSTRUMENT,
            integration_peaks_filename=predictor_output,
            output_filename=integrator_output,
            **INTEGRATOR_PARAMS,
        )
        assert os.path.exists(integrator_output)
        
        # Step 5: Export to MTZ
        print("[5/5] Exporting to MTZ...")
        mtz_exporter(
            indexed_h5_filename=integrator_output,
            output_mtz_filename=mtz_output,
            space_group=SPACE_GROUP,
        )
        assert os.path.exists(mtz_output)
        
        # Final validation: Check we have reflections
        with h5py.File(integrator_output, "r") as f:
            total_reflections = 0
            for bank_key in f["banks"].keys():
                bank = f["banks"][bank_key]
                if "I" in bank:
                    total_reflections += len(bank["I"])
            
            assert total_reflections > 0, "No reflections found in final output"
            print(f"\n✓ Complete workflow finished successfully")
            print(f"  Total reflections: {total_reflections}")
            print(f"  Output files in: {temp_output_dir}")


# Mark the test class for pytest markers
pytestmark = pytest.mark.integration
