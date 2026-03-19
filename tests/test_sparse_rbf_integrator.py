import pytest
import numpy as np
import jax.numpy as jnp

from subhkl.sparse_rbf_peak_finder import (
    build_dense_padded_matrix,
    solve_ssn,
    integrate_peaks_rbf_ssn
)

def test_single_isolated_peak():
    """
    Validates that a single isolated Gaussian peak is correctly identified, 
    that sparsity suppresses incorrect shapes, and the background plane is recovered.
    """
    H, W = 50, 50
    bg_level = 15.0
    
    # Add a little Poisson-like noise to the background
    np.random.seed(42)
    image = np.random.normal(loc=bg_level, scale=2.0, size=(H, W)).astype(np.float32)
    
    # Add a known Gaussian peak at (25, 25)
    cx, cy = 25.0, 25.0
    true_sigma = 2.0
    true_amp = 100.0
    
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r2 = (x_coords - cx)**2 + (y_coords - cy)**2
    peak_profile = true_amp * np.exp(-r2 / (2 * true_sigma**2))
    
    image += peak_profile
    
    # 1. Setup inputs
    peak_centers = np.array([[cx, cy]])
    sigmas = [1.0, 2.0, 5.0]
    gamma = 2.0
    max_peaks = 5
    alpha = 1.5 # Regularization penalty
    
    # 2. Build Padded Matrix
    A, weights, volumes, actual_peaks = build_dense_padded_matrix(
        image, peak_centers, sigmas, gamma, max_peaks
    )
    
    assert actual_peaks == 1
    assert A.shape == (H * W, max_peaks * len(sigmas) + 3)
    
    # 3. Solve SSN
    intensities = jnp.array(image.flatten(), dtype=jnp.float32)
    N_c = max_peaks * len(sigmas)
    u_prime, active_set = solve_ssn(A, intensities, N_c, alpha)
    
    # 4. Verify Peak Recovery
    c_unscaled = u_prime[:N_c] / weights
    
    # The peak was at index 0, so the first 3 coefficients belong to it
    c_1 = float(c_unscaled[0]) # sigma = 1.0
    c_2 = float(c_unscaled[1]) # sigma = 2.0 (The True Shape)
    c_5 = float(c_unscaled[2]) # sigma = 5.0
    
    # L1 penalty inherently biases amplitude downward slightly, 
    # but the correct shape should dominate dramatically.
    assert c_2 > 85.0
    assert c_1 < 5.0
    assert c_5 < 5.0
    
    # 5. Verify Background Recovery
    # The last 3 elements of u_prime are [bg_const, bg_x, bg_y]
    bg_const = float(u_prime[-3])
    assert np.isclose(bg_const, bg_level, atol=2.0)

def test_overlapping_peaks_crosstalk():
    """
    Validates that the solver can decouple the intensity of two overlapping 
    peaks without artificially inflating either of them.
    """
    H, W = 50, 50
    image = np.full((H, W), 10.0, dtype=np.float32)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Peak 1
    cx1, cy1 = 20.0, 25.0
    r2_1 = (x_coords - cx1)**2 + (y_coords - cy1)**2
    image += 80.0 * np.exp(-r2_1 / (2 * 2.0**2))
    
    # Peak 2 (Highly Overlapping - distance is only 4 pixels, while sigma is 2.0)
    cx2, cy2 = 24.0, 25.0
    r2_2 = (x_coords - cx2)**2 + (y_coords - cy2)**2
    image += 60.0 * np.exp(-r2_2 / (2 * 2.0**2))
    
    peak_centers = np.array([[cx1, cy1], [cx2, cy2]])
    sigmas = [1.0, 2.0, 5.0]
    N_c = 2 * len(sigmas)
    
    A, weights, _, _ = build_dense_padded_matrix(
        image, peak_centers, sigmas, gamma=2.0, max_peaks=2
    )
    
    intensities = jnp.array(image.flatten())
    u_prime, _ = solve_ssn(A, intensities, N_c, alpha=0.5)
    c_unscaled = u_prime[:N_c] / weights
    
    # Check that both amplitudes were recovered accurately despite the overlap
    p1_amp = float(c_unscaled[1]) # Index 1 is sigma=2.0 for Peak 1
    p2_amp = float(c_unscaled[4]) # Index 4 is sigma=2.0 for Peak 2
    
    assert 70.0 < p1_amp < 85.0
    assert 50.0 < p2_amp < 65.0

def test_integrate_peaks_rbf_ssn_orchestrator():
    """
    Tests the full loop, validating that Intensity (I), Uncertainty (SIGI),
    and physical angles (two_theta, azimuthal) are properly mapped to the result object.
    """
    H, W = 40, 40
    image = np.full((H, W), 5.0, dtype=np.float32)
    cx, cy = 20.0, 20.0
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    true_sigma = 2.0
    true_amp = 50.0
    r2 = (x_coords - cx)**2 + (y_coords - cy)**2
    image += true_amp * np.exp(-r2 / (2 * true_sigma**2))

    # 1. Mock Peaks Object (Replaces the old MockImageHandler)
    class MockImageHandler:
        def __init__(self, ims):
            self.ims = ims
            self.bank_mapping = {}  # Mock bank map

        def get_run_id(self, img_key):
            return 0  # Mock run index

    class MockPeaks:
        def __init__(self, ims):
            self.image = MockImageHandler(ims)
            # Provide a minimal mock detector configuration that subhkl.Detector expects
            self.config = {
                "0": {
                    "detector": {
                        "n": H,
                        "m": W,
                        "width": W * 1.0,
                        "height": H * 1.0,
                        "pixel_size": 1.0,
                        "center": [0.0, 0.0, 100.0],
                        "fast_axis": [1.0, 0.0, 0.0],
                        "slow_axis": [0.0, -1.0, 0.0]
                    }
                }
            }

    mock_peaks_obj = MockPeaks({0: image})

    # 2. Mock peak dictionary structure: {img_key: [i, j, h, k, l, wl]}
    peak_dict = {
        0: [
            np.array([cx]),
            np.array([cy]),
            np.array([1]),
            np.array([2]),
            np.array([3]),
            np.array([1.5])
        ]
    }

    # 3. Run Orchestrator
    res = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict,
        peaks_obj=mock_peaks_obj, # Pass the new mock
        sigmas=[1.0, 2.0, 3.0],
        alpha=0.5,
        gamma=2.0,
        max_peaks=3,
        show_progress=False
    )

    assert len(res.intensity) == 1

    # Expected mathematical integral: Amplitude * 2 * pi * sigma^2
    expected_intensity = true_amp * 2 * np.pi * (true_sigma**2)

    # 4. Verify Final Export Metrics
    assert res.intensity[0] > 0
    assert np.isclose(res.intensity[0], expected_intensity, rtol=0.15)

    # Verify metadata propagation
    assert res.h[0] == 1
    assert res.k[0] == 2
    assert res.l[0] == 3
    assert res.wavelength[0] == 1.5
    assert res.run_id[0] == 0
    assert res.bank[0] == 0

    # Verify angles evaluated properly without crashing
    assert isinstance(res.tt[0], float)
    assert isinstance(res.az[0], float)

    # Verify SVD Fisher Information successfully generated an uncertainty estimate
    assert res.sigma[0] > 0.0
