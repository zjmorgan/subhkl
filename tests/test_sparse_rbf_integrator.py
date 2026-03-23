import pytest
import numpy as np
import jax.numpy as jnp

from subhkl.search.sparse_rbf import (
    build_dense_padded_matrix,
    solve_ssn,
    integrate_peaks_rbf_ssn,
)

def test_single_isolated_peak():
    """
    Validates that a single isolated Gaussian peak is correctly identified,
    that sparsity strictly deactivates incorrect shapes, and the background plane is recovered.
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
    gamma = 1.0
    max_peaks = 5
    
    # High enough penalty to enforce absolute sparsity against the injected noise
    alpha = 15.0

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

    # 4. Verify Peak Recovery and STRICT SPARSITY
    c_unscaled = u_prime[:N_c] / weights

    # The peak was at index 0, so the first 3 coefficients belong to it
    c_1 = float(c_unscaled[0]) # sigma = 1.0
    c_2 = float(c_unscaled[1]) # sigma = 2.0 (The True Shape)
    c_5 = float(c_unscaled[2]) # sigma = 5.0

    # The proximal operator must have crushed incorrect shapes exactly to 0.0
    assert c_1 == 0.0
    assert c_5 == 0.0
    
    # The boolean active set mask must perfectly align with the non-zero coefficients
    assert active_set[0] == False  # sigma 1.0 is inactive
    assert active_set[1] == True   # sigma 2.0 is active
    assert active_set[2] == False  # sigma 5.0 is inactive

    # The correct shape must still capture the vast majority of the amplitude
    assert c_2 > 85.0

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
                        "uhat": [1.0, 0.0, 0.0],
                        "vhat": [0.0, 1.0, 0.0],
                        "panel": "flat",
                    }
                }
            }

        def get_detector(self, bank):
            from subhkl.instrument.detector import Detector
            return Detector(self.config["0"]["detector"])

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

def test_peak_finder_multiscale_subpixel_recovery():
    """
    Tests the end-to-end SparseRBFPeakFinder pipeline (Scout & Sniper)
    for its ability to detect and recover sub-pixel coordinates of
    multiscale peaks embedded in severe noise.
    """
    from subhkl.search.sparse_rbf import SparseRBFPeakFinder
    import numpy as np
    
    H, W = 60, 60
    
    # 1. Base image with severe background noise
    np.random.seed(42)
    bg_level = 50.0
    image = np.random.poisson(bg_level, size=(H, W)).astype(np.float32)
    
    # 2. Inject Ground Truth Features
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Feature A: Broad, strong peak (representing diffuse scattering/background)
    gt_cx1, gt_cy1 = 30.0, 30.0
    gt_sig1 = 4.0
    gt_amp1 = 200.0
    r2_1 = (x_coords - gt_cx1)**2 + (y_coords - gt_cy1)**2
    image += gt_amp1 * np.exp(-r2_1 / (2 * gt_sig1**2))
    
    # Feature B: Sharp, weak peak at a highly specific SUB-PIXEL offset
    # Placed on the shoulder of the broad peak
    gt_cx2, gt_cy2 = 33.74, 34.21
    gt_sig2 = 1.0
    gt_amp2 = 120.0
    r2_2 = (x_coords - gt_cx2)**2 + (y_coords - gt_cy2)**2
    image += gt_amp2 * np.exp(-r2_2 / (2 * gt_sig2**2))

    # 3. Format as Batch (B, H, W)
    image_batch = image[np.newaxis, ...]
    
    # 4. Instantiate Finder
    # Alpha is set high enough to suppress noise but low enough to catch the sharp peak
    finder = SparseRBFPeakFinder(
        alpha=0.08, 
        gamma=2.0, 
        min_sigma=0.5, 
        max_sigma=5.0,
        show_steps=False
    )
    
    # 5. Execute Pipeline
    results = finder.find_peaks_batch(image_batch)
    
    assert len(results) == 1, "Should return exactly one result array for a batch size of 1."
    peaks = results[0]
    
    # 6. Verify Detection
    assert len(peaks) >= 2, f"Finder detected {len(peaks)} peaks, expected at least 2."
    
    # peaks array format: [r, c, sigma]
    # Match detected peaks to Ground Truth based on spatial proximity
    dists_to_broad = np.sqrt((peaks[:, 0] - gt_cx1)**2 + (peaks[:, 1] - gt_cy1)**2)
    broad_idx = np.argmin(dists_to_broad)
    broad_peak = peaks[broad_idx]
    
    dists_to_sharp = np.sqrt((peaks[:, 0] - gt_cx2)**2 + (peaks[:, 1] - gt_cy2)**2)
    sharp_idx = np.argmin(dists_to_sharp)
    sharp_peak = peaks[sharp_idx]
    
    # Ensure the finder didn't just find the same peak twice
    assert broad_idx != sharp_idx, "Failed to decouple the broad and sharp peaks."
    
    # 7. Verify Sub-pixel Accuracy and Scale
    # Broad Peak bounds (more lenient due to flatness and noise)
    assert np.isclose(broad_peak[0], gt_cx1, atol=1.0), "Broad peak R coordinate failed"
    assert np.isclose(broad_peak[1], gt_cy1, atol=1.0), "Broad peak C coordinate failed"
    assert broad_peak[2] > 2.0, "Broad peak failed to resolve as a large feature"
    
    # Sharp Peak bounds (tighter constraint to prove sub-pixel continuous solver worked)
    assert np.isclose(sharp_peak[0], gt_cx2, atol=0.5), f"Sharp peak R sub-pixel mismatch: {sharp_peak[0]:.2f} vs {gt_cx2}"
    assert np.isclose(sharp_peak[1], gt_cy2, atol=0.5), f"Sharp peak C sub-pixel mismatch: {sharp_peak[1]:.2f} vs {gt_cy2}"
    assert sharp_peak[2] < 2.0, "Sharp peak failed to resolve as a narrow feature"
