import numpy as np
import pytest
from subhkl.convex_hull.peak_integrator import PeakIntegrator
from subhkl.convex_hull.region_grower import RegionGrower

def test_gaussian_center_refinement():
    """
    Test that gaussian_fit correctly refines the peak center.
    """
    # 1. Create synthetic Gaussian peak shifted from pixel grid center
    width, height = 21, 21
    true_y, true_x = 10.3, 10.7
    sigma = 1.5
    amplitude = 100.0
    bg = 5.0
    
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    dist_sq = (y - true_y)**2 + (x - true_x)**2
    intensity = bg + amplitude * np.exp(-dist_sq / (2 * sigma**2))
    
    # 2. Setup Integrator
    peak_integrator = PeakIntegrator(
        RegionGrower(distance_threshold=2.0, min_intensity=10, max_size=8),
        box_size=5,
        smoothing_window_size=3,
        min_peak_pixels=5,
        min_peak_snr=1.0,
    )
    
    # Predicted center at integer pixel (10, 10)
    predicted_center = np.array([[10, 10]])
    
    # 3. Run integration with gaussian_fit
    results, centers = peak_integrator.integrate_peaks(
        0, intensity, predicted_center, integration_method="gaussian_fit"
    )
    
    refined_y, refined_x = centers[0]
    print(f"True center: ({true_y}, {true_x})")
    print(f"Refined center: ({refined_y}, {refined_x})")
    
    # Verify refined center is close to true center
    assert np.isclose(refined_y, true_y, atol=0.1)
    assert np.isclose(refined_x, true_x, atol=0.1)
    
    # 4. Compare with free_fit (which should NOT refine to sub-pixel accuracy beyond local max)
    # Actually, free_fit currently uses local_max which returns integers.
    results_free, centers_free = peak_integrator.integrate_peaks(
        0, intensity, predicted_center, integration_method="free_fit"
    )
    
    # free_fit should return integer coordinates if box_size=1, 
    # but here box_size=5 so it might find the nearest integer peak.
    assert isinstance(centers_free[0][0], (int, np.integer)) or centers_free[0][0].is_integer()

def test_gaussian_fit_ignore_invalid():
    """
    Test that gaussian_fit handles cases where it might fail or produce bad SNR.
    """
    intensity = np.zeros((21, 21))
    # No peak here
    
    peak_integrator = PeakIntegrator(
        RegionGrower(distance_threshold=2.0, min_intensity=10, max_size=8),
        min_peak_pixels=5,
    )
    
    predicted_center = np.array([[10, 10]])
    results, centers = peak_integrator.integrate_peaks(
        0, intensity, predicted_center, integration_method="gaussian_fit"
    )
    
    # Should return None for statistics if it failed to grow region
    assert results[0][3] is None
