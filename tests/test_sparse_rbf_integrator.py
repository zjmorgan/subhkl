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
        try:
            from subhkl.peakfinder.sparse_rbf import SparseRBFPeakFinder
        except ImportError:
            from subhkl.search.sparse_rbf import SparseRBFPeakFinder

        import numpy as np

        H, W = 60, 60

        np.random.seed(42)
        bg_level = 50.0
        image = np.random.poisson(bg_level, size=(H, W)).astype(np.float32)

        # y_coords are Rows, x_coords are Cols
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # Feature A: Broad, strong peak
        gt_c1, gt_r1 = 30.0, 30.0
        gt_sig1 = 4.0
        gt_amp1 = 200.0
        r2_1 = (x_coords - gt_c1)**2 + (y_coords - gt_r1)**2
        image += gt_amp1 * np.exp(-r2_1 / (2 * gt_sig1**2))

        # Feature B: Sharp, weak peak at SUB-PIXEL offset
        gt_c2, gt_r2 = 33.74, 34.21
        gt_sig2 = 1.0
        gt_amp2 = 120.0
        r2_2 = (x_coords - gt_c2)**2 + (y_coords - gt_r2)**2
        image += gt_amp2 * np.exp(-r2_2 / (2 * gt_sig2**2))

        image_batch = image[np.newaxis, ...]

        finder = SparseRBFPeakFinder(
            alpha=15,
            gamma=2.0,
            min_sigma=0.5,
            max_sigma=5.0,
            show_steps=False
        )

        results = finder.find_peaks_batch(image_batch)
        peaks = results[0]

        assert len(peaks) >= 2, f"Finder detected {len(peaks)} peaks, expected at least 2."

        # FIX: Shifted indices from [0, 1, 2] to [1, 2, 3] to account for [intensity, r, c, sigma]
        dists_to_broad = np.sqrt((peaks[:, 1] - gt_r1)**2 + (peaks[:, 2] - gt_c1)**2)
        broad_idx = np.argmin(dists_to_broad)
        broad_peak = peaks[broad_idx]

        dists_to_sharp = np.sqrt((peaks[:, 1] - gt_r2)**2 + (peaks[:, 2] - gt_c2)**2)
        sharp_idx = np.argmin(dists_to_sharp)
        sharp_peak = peaks[sharp_idx]

        assert broad_idx != sharp_idx, "Failed to decouple the broad and sharp peaks."

        # Broad Peak bounds
        assert np.isclose(broad_peak[1], gt_r1, atol=1.0), f"Broad peak R failed: {broad_peak[1]}"
        assert np.isclose(broad_peak[2], gt_c1, atol=1.0), f"Broad peak C failed: {broad_peak[2]}"
        assert broad_peak[3] > 2.0, "Broad peak failed to resolve as a large feature"

        # Sharp Peak bounds
        assert np.isclose(sharp_peak[1], gt_r2, atol=0.5), f"Sharp R mismatch: {sharp_peak[1]:.2f} vs {gt_r2}"
        assert np.isclose(sharp_peak[2], gt_c2, atol=0.5), f"Sharp C mismatch: {sharp_peak[2]:.2f} vs {gt_c2}"
        assert sharp_peak[3] < 2.0, "Sharp peak failed to resolve as a narrow feature"

def test_poisson_vs_gaussian_sparse_flux():
    """
    Evaluates the difference between Gaussian (L2) and Poisson loss
    in a highly sparse regime. Poisson MLE strictly conserves total flux
    (counts), whereas L2 biases the amplitude downwards due to the
    heavy squared penalty on zero-count pixels near the peak center.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import SparseRBFPeakFinder
    except ImportError:
        from subhkl.search.sparse_rbf import SparseRBFPeakFinder

    import numpy as np

    H, W = 40, 40
    np.random.seed(101) # Deterministic noise floor for consistent testing

    # Ground truth: A very faint, sparse peak
    gt_c, gt_r = 20.5, 20.5
    gt_sig = 2.0
    gt_amp = 5.0 # Max expected photons is 5. Highly sparse.
    gt_flux = gt_amp * 2 * np.pi * gt_sig**2 # ~62.8 total photons

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r2 = (x_coords - gt_c)**2 + (y_coords - gt_r)**2

    # Background = 0.1 to ensure many true 0s surround the weak peak
    true_rate = 0.1 + gt_amp * np.exp(-r2 / (2 * gt_sig**2))
    image = np.random.poisson(true_rate).astype(np.float32)

    image_batch = image[np.newaxis, ...]

    # Use low alpha to ensure both solvers admit the weak peak
    finder_l2 = SparseRBFPeakFinder(
        alpha=0.5, min_sigma=1.0, max_sigma=4.0, loss='gaussian', show_steps=False
    )
    finder_pois = SparseRBFPeakFinder(
        alpha=0.5, min_sigma=1.0, max_sigma=4.0, loss='poisson', show_steps=False
    )

    peaks_l2 = finder_l2.find_peaks_batch(image_batch)[0]
    peaks_pois = finder_pois.find_peaks_batch(image_batch)[0]

    def get_target_peak(peaks):
        if len(peaks) == 0: return None
        # peaks format: [intensity, global_r, global_c, sigma]
        dists = np.sqrt((peaks[:, 1] - gt_r)**2 + (peaks[:, 2] - gt_c)**2)
        idx = np.argmin(dists)
        if dists[idx] > 3.0: return None
        return peaks[idx]

    p_l2 = get_target_peak(peaks_l2)
    p_pois = get_target_peak(peaks_pois)

    assert p_pois is not None, "Poisson finder completely missed the sparse peak."

    if p_l2 is not None:
        # Calculate recovered total flux: Amplitude * 2 * pi * sigma^2
        flux_l2 = p_l2[0] * 2 * np.pi * p_l2[3]**2
        flux_pois = p_pois[0] * 2 * np.pi * p_pois[3]**2

        error_l2 = abs(flux_l2 - gt_flux)
        error_pois = abs(flux_pois - gt_flux)

        # The Poisson flux error should be significantly smaller because
        # L2 collapses the amplitude to minimize squared distance to 0-count pixels.
        assert error_pois < error_l2, (
            f"Poisson flux error ({error_pois:.1f}) was worse than L2 error ({error_l2:.1f}). "
            f"Poisson Flux: {flux_pois:.1f}, L2 Flux: {flux_l2:.1f}, True Flux: {gt_flux:.1f}"
        )


def test_poisson_overlapping_string():
        """
        Tests the solver's ability to resolve a closely packed string of peaks
        (like a streaked reflection or multiple adjacent Bragg peaks)
        under Poisson noise, without artificially dropping or over-merging them.
        """
        try:
            from subhkl.peakfinder.sparse_rbf import SparseRBFPeakFinder
        except ImportError:
            from subhkl.search.sparse_rbf import SparseRBFPeakFinder

        import numpy as np

        H, W = 40, 80
        np.random.seed(123)
        bg_level = 20.0

        # Start with a flat expected background rate to prevent double-sampling variance
        image = np.full((H, W), bg_level, dtype=np.float32)
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # Inject 4 sharp peaks in a line, separated by just 3 pixels (sigma=1.0)
        true_peaks = [
            (20.0, 30.0, 1.0, 150.0),
            (20.0, 33.0, 1.0, 160.0),
            (20.0, 36.0, 1.0, 140.0),
            (20.0, 39.0, 1.0, 150.0),
        ]

        for r, c, sig, amp in true_peaks:
            r2 = (x_coords - c)**2 + (y_coords - r)**2
            image += amp * np.exp(-r2 / (2 * sig**2))

        # Apply true Poisson noise exactly ONCE
        image = np.random.poisson(image).astype(np.float32)
        image_batch = image[np.newaxis, ...]

        finder = SparseRBFPeakFinder(
            alpha=5.0,
            gamma=2.0,
            min_sigma=0.5,
            max_sigma=5.0,
            loss='poisson',
            show_steps=False
        )

        results = finder.find_peaks_batch(image_batch)
        peaks = results[0]

        roi_mask = (peaks[:, 1] > 15) & (peaks[:, 1] < 25) & (peaks[:, 2] > 25) & (peaks[:, 2] < 45)
        roi_peaks = peaks[roi_mask]

        # It must decouple the streak into at least 3 distinct peaks
        assert len(roi_peaks) >= 3, f"Expected to resolve at least 3 peaks in the streak, found {len(roi_peaks)}"

        medians = np.array([bg_level])[np.newaxis, ...]
        metrics = finder.compute_metrics(image_batch, medians, [peaks], global_max=1.0)
        deviance = metrics['deviance_nu']

        assert deviance < 1.5, f"Poisson Deviance/DoF is too high ({deviance:.2f}), model is severely underfitting!"
