import pytest
import numpy as np
import scipy.special
import jax.numpy as jnp

def generate_erf_peak(y_coords, x_coords, r, c, sig, amp):
    """
    Helper function to generate physically exact subpixel peaks
    using the continuous analytic Gaussian pixel integral.
    """
    sig_sq2 = sig * np.sqrt(2.0) + 1e-6
    erf_y = scipy.special.erf((y_coords + 0.5 - r) / sig_sq2) - scipy.special.erf((y_coords - 0.5 - r) / sig_sq2)
    erf_x = scipy.special.erf((x_coords + 0.5 - c) / sig_sq2) - scipy.special.erf((x_coords - 0.5 - c) / sig_sq2)
    return amp * (np.pi / 2.0) * (sig**2) * erf_y * erf_x

def test_single_isolated_peak():
    """
    Validates that a single isolated Gaussian peak is correctly integrated by the
    new Patch-Based SSN Integrator, that the best shape is activated, and that 
    the unpenalized Tikhonov debiasing accurately recovers the mass.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import SparseLaueIntegrator
    except ImportError:
        from subhkl.search.sparse_rbf import SparseLaueIntegrator
        
    import numpy as np
    
    H, W = 50, 50
    bg_level = 15.0
    
    np.random.seed(42)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Flat background
    image = np.full((H, W), bg_level, dtype=np.float32)
    
    cx, cy = 25.0, 25.0
    true_sigma = 2.0
    true_amp = 100.0
    
    # generate_erf_peak should already be in your test file
    image += generate_erf_peak(y_coords, x_coords, cy, cx, true_sigma, true_amp)
    
    # Apply Poisson noise
    image = np.random.poisson(image).astype(np.float32)
    
    # Use the new Unified Patch Integrator
    integrator = SparseLaueIntegrator(
        alpha=4.0,          # 4-sigma detection threshold
        min_sigma=1.0, 
        max_sigma=5.0, 
        gamma=1.0, 
        loss='gaussian'
    )
    
    images_batch = image[np.newaxis, ...]
    frames = [0]
    rs = [cy]
    cs = [cx]
    
    results = integrator.integrate_reflections(images_batch, frames, rs, cs)
    
    assert len(results) == 1, "The integrator dropped the peak!"
    
    intensity, r_found, c_found, sig_found, sigI_found = results[0]
    
    # 1. Did it pick the right shape from the linspace dictionary?
    assert abs(sig_found - true_sigma) < 0.25, f"Sigma warped! Expected ~{true_sigma}, Found {sig_found}"
    
    # 2. Did the debiasing properly recover the physical mass?
    expected_intensity = true_amp * 2 * np.pi * true_sigma**2
    assert np.isclose(intensity, expected_intensity, rtol=0.15), f"Debiasing failed: {intensity} vs {expected_intensity}"


def test_overlapping_peaks_crosstalk():
    """
    Validates that the patch-based integrator can independently resolve closely 
    overlapping peaks without the backgrounds swallowing each other, thanks to the 
    local median filter and robust NCC warm start.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import SparseLaueIntegrator
    except ImportError:
        from subhkl.search.sparse_rbf import SparseLaueIntegrator
        
    import numpy as np
    
    H, W = 50, 50
    bg_level = 10.0
    np.random.seed(101)
    
    image = np.full((H, W), bg_level, dtype=np.float32)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Peak 1
    cx1, cy1 = 20.0, 25.0
    true_sig1, true_amp1 = 2.0, 80.0
    image += generate_erf_peak(y_coords, x_coords, cy1, cx1, true_sig1, true_amp1)
    
    # Peak 2 (Highly overlapping, only 2-sigma away!)
    cx2, cy2 = 24.0, 25.0
    true_sig2, true_amp2 = 2.0, 60.0
    image += generate_erf_peak(y_coords, x_coords, cy2, cx2, true_sig2, true_amp2)
    
    image = np.random.poisson(image).astype(np.float32)
    
    integrator = SparseLaueIntegrator(
        alpha=4.0, 
        min_sigma=1.0, 
        max_sigma=5.0, 
        gamma=1.0, 
        loss='gaussian'
    )
    
    images_batch = image[np.newaxis, ...]
    frames = [0, 0]
    rs = [cy1, cy2]
    cs = [cx1, cx2]
    
    results = integrator.integrate_reflections(images_batch, frames, rs, cs)
    
    assert len(results) == 2, "Integrator crashed on one of the overlapping peaks!"
    
    i1, r1, c1, sig1, sigI1 = results[0]
    i2, r2, c2, sig2, sigI2 = results[1]
    
    # Ensure both survived the sparsity constraints
    assert sig1 > 0.0, "Peak 1 was crushed"
    assert sig2 > 0.0, "Peak 2 was crushed"
    
    # Because we evaluate them as independent local patches now (instead of a giant joint matrix),
    # there is a slight geometric overlap accepted into the unpenalized volume. 
    # We use a 20% tolerance to ensure crosstalk bleeding stays mathematically bounded.
    exp_i1 = true_amp1 * 2 * np.pi * true_sig1**2
    exp_i2 = true_amp2 * 2 * np.pi * true_sig2**2
   
    assert np.isclose(i1, exp_i1, rtol=0.20), f"Peak 1 Crosstalk Bleed: {i1} vs {exp_i1}"
    assert np.isclose(i2, exp_i2, rtol=0.20), f"Peak 2 Crosstalk Bleed: {i2} vs {exp_i2}"

def test_integrate_peaks_rbf_ssn_orchestrator():
    H, W = 40, 40
    image = np.full((H, W), 5.0, dtype=np.float32)
    cx, cy = 20.0, 20.0
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    true_sigma = 2.0
    true_amp = 50.0
    image += generate_erf_peak(y_coords, x_coords, cy, cx, true_sigma, true_amp)

    class MockImageHandler:
        def __init__(self, ims):
            self.ims = ims
            self.bank_mapping = {}

        def get_run_id(self, img_key):
            return 0

    class MockPeaks:
        def __init__(self, ims):
            self.image = MockImageHandler(ims)
            self.config = {
                "0": {
                    "detector": {
                        "n": H, "m": W, "width": W * 1.0, "height": H * 1.0,
                        "pixel_size": 1.0, "center": [0.0, 0.0, 100.0],
                        "uhat": [1.0, 0.0, 0.0], "vhat": [0.0, 1.0, 0.0], "panel": "flat",
                    }
                }
            }

        def get_detector(self, bank):
            from subhkl.instrument.detector import Detector
            return Detector(self.config["0"]["detector"])

    mock_peaks_obj = MockPeaks({0: image})

    peak_dict = {
        0: [
            np.array([cx]), np.array([cy]), np.array([1]), 
            np.array([2]), np.array([3]), np.array([1.5])
        ]
    }

    try:
        from subhkl.peakfinder.sparse_rbf import integrate_peaks_rbf_ssn
    except ImportError:
        from subhkl.search.sparse_rbf import integrate_peaks_rbf_ssn

    res = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict, peaks_obj=mock_peaks_obj,
        sigmas=[1.0, 2.0, 3.0], alpha=0.5, gamma=2.0, max_peaks=3, show_progress=False
    )

    assert len(res.intensity) == 1

    expected_intensity = true_amp * 2 * np.pi * (true_sigma**2)

    assert res.intensity[0] > 0
    assert np.isclose(res.intensity[0], expected_intensity, rtol=0.15)


def test_peak_finder_multiscale_subpixel_recovery():
    try:
        from subhkl.peakfinder.sparse_rbf import SparseRBFPeakFinder
    except ImportError:
        from subhkl.search.sparse_rbf import SparseRBFPeakFinder

    import numpy as np

    H, W = 60, 60

    np.random.seed(42)
    bg_level = 50.0
    image = np.random.poisson(bg_level, size=(H, W)).astype(np.float32)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    gt_c1, gt_r1 = 30.0, 30.0
    gt_sig1 = 4.0
    gt_amp1 = 200.0
    image += generate_erf_peak(y_coords, x_coords, gt_r1, gt_c1, gt_sig1, gt_amp1)

    gt_c2, gt_r2 = 33.74, 34.21
    gt_sig2 = 1.0
    gt_amp2 = 120.0
    image += generate_erf_peak(y_coords, x_coords, gt_r2, gt_c2, gt_sig2, gt_amp2)

    image_batch = image[np.newaxis, ...]

    finder = SparseRBFPeakFinder(
        alpha=2.0, gamma=1.0, min_sigma=0.5, max_sigma=5.0, show_steps=False
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    assert len(peaks) >= 2

    dists_to_broad = np.sqrt((peaks[:, 1] - gt_r1)**2 + (peaks[:, 2] - gt_c1)**2)
    broad_idx = np.argmin(dists_to_broad)
    broad_peak = peaks[broad_idx]

    dists_to_sharp = np.sqrt((peaks[:, 1] - gt_r2)**2 + (peaks[:, 2] - gt_c2)**2)
    sharp_idx = np.argmin(dists_to_sharp)
    sharp_peak = peaks[sharp_idx]

    assert broad_idx != sharp_idx

    assert np.isclose(broad_peak[1], gt_r1, atol=1.0)
    assert np.isclose(broad_peak[2], gt_c1, atol=1.0)
    assert broad_peak[3] > 2.0

    assert np.isclose(sharp_peak[1], gt_r2, atol=0.5)
    assert np.isclose(sharp_peak[2], gt_c2, atol=0.5)
    assert sharp_peak[3] < 2.0


def test_poisson_vs_gaussian_sparse_flux():
    try:
        from subhkl.peakfinder.sparse_rbf import SparseRBFPeakFinder
    except ImportError:
        from subhkl.search.sparse_rbf import SparseRBFPeakFinder

    import numpy as np

    H, W = 40, 40
    np.random.seed(101)

    gt_c, gt_r = 20.5, 20.5
    gt_sig = 2.0
    gt_amp = 5.0
    gt_flux = gt_amp * 2 * np.pi * gt_sig**2

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    true_rate = 0.1 + generate_erf_peak(y_coords, x_coords, gt_r, gt_c, gt_sig, gt_amp)
    image = np.random.poisson(true_rate).astype(np.float32)

    image_batch = image[np.newaxis, ...]

    finder_l2 = SparseRBFPeakFinder(
        gamma=-1.0, alpha=1, min_sigma=1.0, max_sigma=4.0, loss='gaussian', show_steps=False
    )
    finder_pois = SparseRBFPeakFinder(
        gamma=-1.0, alpha=1, min_sigma=1.0, max_sigma=4.0, loss='poisson', show_steps=False
    )

    peaks_l2 = finder_l2.find_peaks_batch(image_batch)[0]
    peaks_pois = finder_pois.find_peaks_batch(image_batch)[0]

    def get_target_peak(peaks):
        if len(peaks) == 0: return None
        dists = np.sqrt((peaks[:, 1] - gt_r)**2 + (peaks[:, 2] - gt_c)**2)
        idx = np.argmin(dists)
        if dists[idx] > 3.0: return None
        return peaks[idx]

    p_l2 = get_target_peak(peaks_l2)
    p_pois = get_target_peak(peaks_pois)

    assert p_pois is not None

    if p_l2 is not None:
        flux_l2 = p_l2[0] * 2 * np.pi * p_l2[3]**2
        flux_pois = p_pois[0] * 2 * np.pi * p_pois[3]**2

        error_l2 = abs(flux_l2 - gt_flux)
        error_pois = abs(flux_pois - gt_flux)

        assert error_pois < error_l2


def test_poisson_overlapping_string():
    try:
        from subhkl.peakfinder.sparse_rbf import SparseRBFPeakFinder
    except ImportError:
        from subhkl.search.sparse_rbf import SparseRBFPeakFinder

    import numpy as np

    H, W = 40, 80
    np.random.seed(123)
    bg_level = 20.0

    image = np.full((H, W), bg_level, dtype=np.float32)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    true_peaks = [
        (20.0, 30.0, 1.0, 150.0),
        (20.0, 33.0, 1.0, 160.0),
        (20.0, 36.0, 1.0, 140.0),
        (20.0, 39.0, 1.0, 150.0),
    ]

    for r, c, sig, amp in true_peaks:
        image += generate_erf_peak(y_coords, x_coords, r, c, sig, amp)

    image = np.random.poisson(image).astype(np.float32)
    image_batch = image[np.newaxis, ...]

    finder = SparseRBFPeakFinder(
        alpha=2.0, gamma=1.0, min_sigma=0.5, max_sigma=5.0, loss='poisson', show_steps=False
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    roi_mask = (peaks[:, 1] > 15) & (peaks[:, 1] < 25) & (peaks[:, 2] > 25) & (peaks[:, 2] < 45)
    roi_peaks = peaks[roi_mask]

    assert len(roi_peaks) >= 3

    # FIX: Evaluate deviance against the model's actual estimated background
    medians_ideal = np.array([bg_level])[np.newaxis, ...]
    bg_map = getattr(finder, '_last_bg_map', medians_ideal)

    metrics = finder.compute_metrics(image_batch, bg_map, [peaks], global_max=1.0)
    deviance = metrics['deviance_nu']

    # Allow < 2.5 to account for L1 shrinkage bias on highly degenerate, overlapping strings
    assert deviance < 2.5, f"Deviance too high ({deviance:.2f})"


def test_real_neutron_structured_background():
    try:
        from subhkl.peakfinder.sparse_rbf import SparseRBFPeakFinder
    except ImportError:
        from subhkl.search.sparse_rbf import SparseRBFPeakFinder
        
    import numpy as np
    
    H, W = 100, 100
    np.random.seed(42)
    
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    halo_amp = 80.0
    halo_sig = 30.0
    r2_halo = (x_coords - 50)**2 + (y_coords - 50)**2
    bg_structured = 15.0 + halo_amp * np.exp(-r2_halo / (2 * halo_sig**2))
    image = np.copy(bg_structured)
    
    true_peaks = [
        (25.0, 25.0, 1.2, 300.0), 
        (75.0, 75.0, 1.0, 80.0),  
        (50.0, 50.0, 2.0, 400.0), 
    ]
    
    for r, c, sig, amp in true_peaks:
        image += generate_erf_peak(y_coords, x_coords, r, c, sig, amp)
        
    image = np.random.poisson(image).astype(np.float32)
    image_batch = image[np.newaxis, ...]
    
    finder = SparseRBFPeakFinder(
        alpha=0.5, gamma=1.0, min_sigma=0.5, max_sigma=5.0, loss='poisson', show_steps=False
    )
    
    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]
    
    assert len(peaks) >= 3
   
    print("\n--- GEOMETRY DIAGNOSTICS ---")
    for true_r, true_c, true_sig, true_amp in true_peaks:
        dists = np.sqrt((peaks[:, 1] - true_r)**2 + (peaks[:, 2] - true_c)**2)
        closest_idx = np.argmin(dists)

        p_c, p_r, p_col, p_sig = peaks[closest_idx]
        min_dist = dists[closest_idx]

        print(f"Target: r={true_r:5.1f}, c={true_c:5.1f}, sig={true_sig:4.2f}")
        print(f"Found : r={p_r:5.2f}, c={p_col:5.2f}, sig={p_sig:4.2f} | Dist: {min_dist:.3f}")

        # Assert subpixel spatial accuracy
        assert min_dist < 0.75, f"Peak wandered! Dist: {min_dist:.2f}"

        # Assert shape preservation (allowing for dyadic grid snapping)
        assert abs(p_sig - true_sig) < 0.5, f"Sigma collapsed/exploded! True: {true_sig}, Found: {p_sig}"
    print("----------------------------\n")

    medians = np.median(image_batch, axis=(1, 2), keepdims=True)
    bg_map = getattr(finder, '_last_bg_map', medians) 

    print(f"\n--- GHOST PEAK DIAGNOSTICS ---")
    print(f"Total peaks returned by Finder: {len(peaks)} (Expected: 3)")

    # Sort peaks by amplitude (descending)
    peaks_sorted = peaks[np.argsort(peaks[:, 0])[::-1]]

    print("\nTop 10 Peaks by Amplitude:")
    for i, p in enumerate(peaks_sorted[:10]):
        print(f"  [{i}] Amp: {p[0]:6.1f} | r: {p[1]:5.1f}, c: {p[2]:5.1f} | sig: {p[3]:4.2f}")

    # Isolate ONLY the True Peaks for the strict deviance check
    # (Filtering out the 0.5-sigma background ghost bumps)
    top_peaks = peaks_sorted[:3]

    metrics_top = finder.compute_metrics(image_batch, bg_map, [top_peaks], global_max=1.0)
    deviance_top = metrics_top['deviance_nu']
    print(f"\nDeviance (Top 3 True Peaks Only): {deviance_top:.3f}")

    metrics = finder.compute_metrics(image_batch, bg_map, [peaks], global_max=1.0)
    deviance = metrics['deviance_nu']

    assert deviance < 1.5

def test_large_sensor_basic_recovery_finder():
    """
    Diagnostic Test: Simulates a 512x512 detector with a FLAT Poisson background.
    This isolates whether the GPU batching, memory, and scaling work on large arrays
    without the confounding variable of morphological halo errors.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import SparseRBFPeakFinder
    except ImportError:
        from subhkl.search.sparse_rbf import SparseRBFPeakFinder

    import numpy as np
    import scipy.special

    H, W = 512, 512
    np.random.seed(42)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Flat background: 20 photons.
    image = np.random.poisson(20.0, size=(H, W)).astype(np.float32)

    # Inject 3 strong, isolated peaks
    true_peaks = [
        (100.0, 100.0, 2.0, 300.0),
        (400.0, 400.0, 2.0, 250.0),
        (150.0, 350.0, 1.5, 400.0)
    ]
    for r, c, sig, amp in true_peaks:
        sig_sq2 = sig * np.sqrt(2.0) + 1e-6
        erf_y = scipy.special.erf((y_coords + 0.5 - r) / sig_sq2) - scipy.special.erf((y_coords - 0.5 - r) / sig_sq2)
        erf_x = scipy.special.erf((x_coords + 0.5 - c) / sig_sq2) - scipy.special.erf((x_coords - 0.5 - c) / sig_sq2)
        image += amp * (np.pi / 2.0) * (sig**2) * erf_y * erf_x

    image_batch = image[np.newaxis, ...]

    finder = SparseRBFPeakFinder(
        alpha=4.0, gamma=1.0, min_sigma=1.0, max_sigma=5.0, loss='poisson', show_steps=False
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    # 1. Did it explode into noise?
    assert len(peaks) < 15, f"Basic Finder Failed: Hallucinated {len(peaks)} peaks on a flat background!"

    # 2. Did it find the 3 real ones?
    assert len(peaks) >= 3, f"Basic Finder Failed: Missed the real peaks, only found {len(peaks)}."

def test_large_sensor_artifact_suppression():
    """
    Simulates a full 512x512 detector panel with a massive, curved 
    diffuse scattering background (halo) to ensure the solver does NOT 
    hallucinate a grid of false peaks to fit the unmodeled background curvature.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import SparseRBFPeakFinder
    except ImportError:
        from subhkl.search.sparse_rbf import SparseRBFPeakFinder
        
    import numpy as np
    import scipy.special
    
    H, W = 512, 512
    np.random.seed(42)
    
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # 1. Create a massive, curved halo that a flat plane CANNOT fit
    r2_halo = (x_coords - 256)**2 + (y_coords - 256)**2
    bg_curved = 20.0 + 150.0 * np.exp(-r2_halo / (2 * 100**2))
    
    image = np.random.poisson(bg_curved).astype(np.float32)
    
    # 2. Inject exactly TWO real peaks
    true_peaks = [
        (100.0, 100.0, 1.5, 200.0),
        (400.0, 400.0, 2.0, 250.0)
    ]
    for r, c, sig, amp in true_peaks:
        sig_sq2 = sig * np.sqrt(2.0) + 1e-6
        erf_y = scipy.special.erf((y_coords + 0.5 - r) / sig_sq2) - scipy.special.erf((y_coords - 0.5 - r) / sig_sq2)
        erf_x = scipy.special.erf((x_coords + 0.5 - c) / sig_sq2) - scipy.special.erf((x_coords - 0.5 - c) / sig_sq2)
        phi = (np.pi / 2.0) * (sig**2) * erf_y * erf_x
        image += amp * phi
        
    image_batch = image[np.newaxis, ...]
    
    # Test Peak Finder robustness to background curvature
    finder = SparseRBFPeakFinder(
        alpha=4.0, gamma=1.0, min_sigma=1.0, max_sigma=5.0, loss='poisson', show_steps=False
    )
    
    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]
    
    # The solver MUST NOT hallucinate grids. 
    # We allow a small buffer for extreme Poisson noise spikes, but it absolutely cannot be > 10.
    assert len(peaks) >= 2, f"Failed to find the 2 real peaks, found {len(peaks)}"
    assert len(peaks) < 10, f"Grid Pathology! Hallucinated {len(peaks)} peaks to fit the background."

def test_large_sensor_basic_integration():
    """
    Diagnostic Test: Tests the dense SSN Integrator on a 512x512 array with a
    FLAT Poisson background. This proves whether the massive Ht @ u matrix
    operations and active-set thresholding are intact for large arrays.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import integrate_peaks_rbf_ssn
    except ImportError:
        from subhkl.search.sparse_rbf import integrate_peaks_rbf_ssn

    import numpy as np
    import scipy.special

    H, W = 512, 512
    np.random.seed(101)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Flat background
    image = np.random.poisson(15.0, size=(H, W)).astype(np.float32)

    true_r, true_c = 256.0, 256.0
    true_sig, true_amp = 2.0, 200.0
    sig_sq2 = true_sig * np.sqrt(2.0) + 1e-6
    erf_y = scipy.special.erf((y_coords + 0.5 - true_r) / sig_sq2) - scipy.special.erf((y_coords - 0.5 - true_r) / sig_sq2)
    erf_x = scipy.special.erf((x_coords + 0.5 - true_c) / sig_sq2) - scipy.special.erf((x_coords - 0.5 - true_c) / sig_sq2)
    image += true_amp * (np.pi / 2.0) * (true_sig**2) * erf_y * erf_x

    # Mock orchestrator
    class MockImageHandler:
        def __init__(self, ims):
            self.ims = ims
            self.bank_mapping = {0: 1}
        def get_run_id(self, img_key):
            return 0

    class MockPeaks:
        def __init__(self, ims):
            self.image = MockImageHandler(ims)
        def get_detector(self, bank):
            from subhkl.instrument.detector import Detector
            return Detector({
                "n": H, "m": W, "width": W, "height": H, "pixel_size": 1.0,
                "center": [0,0,100], "uhat": [1,0,0], "vhat": [0,1,0], "panel": "flat"
            })

    # Predictions: 1 True peak (Index 0), 2 Fake peaks far away
    pred_r = np.array([true_r, 50.0, 450.0])
    pred_c = np.array([true_c, 50.0, 450.0])

    # Use distinct, non-harmonic Miller indices so the deduplicator keeps them all
    h_arr = np.array([1, 2, 3])
    k_arr = np.array([13, 17, 19])
    l_arr = np.array([1, 1, 1])

    peak_dict = {
        0: [
            pred_r, pred_c,
            h_arr, k_arr, l_arr, np.ones(3)
        ]
    }

    res = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict, peaks_obj=MockPeaks({0: image}),
        sigmas=[1.0, 2.0, 4.0], alpha=4.0, gamma=1.0, max_peaks=5, show_progress=False
    )

    intensities = np.array(res.intensity)
    sigIs = np.array(res.sigma)  # We are now properly returning statistical uncertainty
    snrs = intensities / (sigIs + 1e-9)

    # 1. Did it differentiate real vs fake using statistical confidence (SNR)?
    assert snrs[0] > 3.0, f"True peak SNR too low! Expected > 3.0, got {snrs[0]:.2f}"
    assert snrs[1] < 2.0, f"Fake peak 1 SNR too high! Expected < 2.0, got {snrs[1]:.2f}"
    assert snrs[2] < 2.0, f"Fake peak 2 SNR too high! Expected < 2.0, got {snrs[2]:.2f}"

    # 2. Did the unpenalized Measurement Phase (NNLS) correctly measure the unbiased mass?
    expected_intensity = true_amp * 2 * np.pi * true_sig**2
    found_intensity = intensities[0]

    assert np.isclose(found_intensity, expected_intensity, rtol=0.15), f"Debiasing failed: {found_intensity} vs {expected_intensity}"


def test_integrator_large_sensor_halo_suppression():
    """
    Validates that the dense matrix GPU integrator successfully subtracts the
    complex morphological halo before evaluation, and properly executes the
    debiasing loop to prevent real peaks from being crushed by the L1 penalty.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import integrate_peaks_rbf_ssn
    except ImportError:
        from subhkl.search.sparse_rbf import integrate_peaks_rbf_ssn

    import numpy as np
    import scipy.special

    H, W = 512, 512
    np.random.seed(101)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r2_halo = (x_coords - 256)**2 + (y_coords - 256)**2
    bg_curved = 15.0 + 100.0 * np.exp(-r2_halo / (2 * 120**2))

    image = np.random.poisson(bg_curved).astype(np.float32)

    true_r, true_c = 300.0, 300.0
    true_sig, true_amp = 2.0, 150.0
    sig_sq2 = true_sig * np.sqrt(2.0) + 1e-6
    erf_y = scipy.special.erf((y_coords + 0.5 - true_r) / sig_sq2) - scipy.special.erf((y_coords - 0.5 - true_r) / sig_sq2)
    erf_x = scipy.special.erf((x_coords + 0.5 - true_c) / sig_sq2) - scipy.special.erf((x_coords - 0.5 - true_c) / sig_sq2)
    image += true_amp * (np.pi / 2.0) * (true_sig**2) * erf_y * erf_x

    # Mocking framework for the Orchestrator
    class MockImageHandler:
        def __init__(self, ims):
            self.ims = ims
            self.bank_mapping = {0: 1}
        def get_run_id(self, img_key):
            return 0

    class MockPeaks:
        def __init__(self, ims):
            self.image = MockImageHandler(ims)
        def get_detector(self, bank):
            from subhkl.instrument.detector import Detector
            return Detector({
                "n": H, "m": W, "width": W, "height": H, "pixel_size": 1.0,
                "center": [0,0,100], "uhat": [1,0,0], "vhat": [0,1,0], "panel": "flat"
            })

    # Provide a grid of HKL predictions. Only ONE matches the true peak (Index 5).
    grid_i, grid_j = np.linspace(50, 450, 10), np.linspace(50, 450, 10)
    grid_i[5], grid_j[5] = true_r, true_c 
    
    # Generate 10 unique fundamental rays
    h_arr = np.arange(1, 11)
    k_arr = np.full(10, 13)
    l_arr = np.full(10, 17)
    
    peak_dict = {
        0: [
            grid_i, grid_j, 
            h_arr, k_arr, l_arr, np.ones(10)
        ]
    }

    res = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict, peaks_obj=MockPeaks({0: image}),
        sigmas=[1.0, 2.0, 4.0], alpha=5.0, gamma=1.0, max_peaks=10, show_progress=False
    )

    intensities = np.array(res.intensity)
    sigIs = np.array(res.sigma)
    snrs = intensities / (sigIs + 1e-9)

    # 1. The integrator measures everything, so we rely on SNR to reject halos
    assert snrs[5] > 3.0, f"True peak SNR too low! Expected > 3.0, got {snrs[5]:.2f}"

    # Check that all other 9 fake halo points are rejected by high uncertainty
    fake_indices = [i for i in range(10) if i != 5]
    for i in fake_indices:
        assert snrs[i] < 2.0, f"Halo trap failed! Fake peak {i} has high SNR: {snrs[i]:.2f}"

    # 2. The debiasing loop must recover the full intensity
    expected_intensity = true_amp * 2 * np.pi * true_sig**2
    found_intensity = intensities[5]

    # Allow 15% tolerance for Poisson noise variance
    assert np.isclose(found_intensity, expected_intensity, rtol=0.15), f"Halo Debias failed: {found_intensity} vs {expected_intensity}"
