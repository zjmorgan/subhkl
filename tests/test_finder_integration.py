import pytest
import numpy as np
import h5py
import scipy.special
from subhkl.io.parser import finder

@pytest.mark.parametrize("algorithm", ["sparse_rbf", "thresholding", "peak_local_max"])
def test_finder_cli_integration(tmp_path, algorithm):
    """
    Integration test for all peak finder algorithms via the finder API.
    Creates a dummy NeXus-like file with a single Gaussian peak, runs the
    finder, and verifies the peak is detected and exported correctly.
    """
    input_h5 = tmp_path / f"dummy_raw_{algorithm}.h5"
    output_h5 = tmp_path / f"dummy_peaks_{algorithm}.h5"
    
    H, W = 100, 100
    bg_level = 15.0
    np.random.seed(42)
    
    # 1. Generate a mathematically perfect Erf Gaussian Peak
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    true_r, true_c = 50.0, 50.0
    true_sig, true_amp = 2.0, 150.0
    
    sig_sq2 = true_sig * np.sqrt(2.0) + 1e-6
    erf_y = scipy.special.erf((y_coords + 0.5 - true_r) / sig_sq2) - scipy.special.erf((y_coords - 0.5 - true_r) / sig_sq2)
    erf_x = scipy.special.erf((x_coords + 0.5 - true_c) / sig_sq2) - scipy.special.erf((x_coords - 0.5 - true_c) / sig_sq2)
    peak = true_amp * (np.pi / 2.0) * (true_sig**2) * erf_y * erf_x
    
    # Apply raw unscaled Poisson statistics
    image = np.random.poisson(bg_level + peak).astype(np.float32)
    
    # 2. Build the dummy HDF5
    with h5py.File(input_h5, "w") as f:
        f.create_dataset("images", data=image[np.newaxis, ...])
        f.create_dataset("bank_ids", data=np.array([1], dtype=np.int32))
        f.create_dataset("goniometer/angles", data=np.zeros((1, 1)))
        f.create_dataset("goniometer/axes", data=[[0, 1, 0, 1]])
        f.create_dataset("instrument/wavelength", data=[2.0, 4.0])
        f.attrs["instrument"] = "MANDI"
        
    # 3. Execute the Finder Wrapper (Simulating the CLI)
    finder(
        filename=str(input_h5),
        output_filename=str(output_h5),
        instrument="MANDI",
        finder_algorithm=algorithm,
        sparse_rbf_alpha=4.0,           
        sparse_rbf_min_sigma=1.0,
        sparse_rbf_max_sigma=5.0,
        sparse_rbf_loss="poisson",
        # --- LEGACY ALGORITHM FAILSAFES ---
        region_growth_minimum_intensity=10.0, 
        peak_minimum_pixels=1,         # Ensure narrow test peaks aren't discarded as cosmic rays
        peak_minimum_intensity=20.0,   # Absolute background threshold
        show_progress=False
    )
    
    # 4. Verify the Output
    assert output_h5.exists(), f"Finder failed to create output file for {algorithm}."
    
    with h5py.File(output_h5, "r") as f:
        assert "peaks/xyz" in f, f"XYZ coordinates missing from output for {algorithm}."
        xyz = f["peaks/xyz"][()]
        
        # All algorithms should easily identify this strong, isolated peak
        assert len(xyz) >= 1, f"Expected at least 1 peak, but {algorithm} returned {len(xyz)}."
