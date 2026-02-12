import numpy as np
import pytest
from subhkl.convex_hull.peak_integrator import PeakIntegrator
from subhkl.convex_hull.region_grower import RegionGrower
from subhkl.integration import _integrate_single_bank

def test_repro_list_indexing_error():
    """
    Reproduces 'Integration worker failed: list indices must be integers or slices, not list'
    caused by converting adjusted_centers to a list in PeakIntegrator.integrate_peaks.
    """
    # 1. Setup a simple peak
    width, height = 20, 20
    intensity = np.zeros((width, height))
    intensity[10, 10] = 100
    intensity[10, 11] = 80
    intensity[11, 10] = 80
    
    peak_integrator = PeakIntegrator(
        RegionGrower(distance_threshold=1.5, min_intensity=5, max_size=5),
        min_peak_pixels=3,
    )
    
    # 2. Integrate - this will return adjusted_centers as a LIST due to recent changes
    predicted_centers = np.array([[10, 10]])
    results, adjusted_centers = peak_integrator.integrate_peaks(
        0, intensity, predicted_centers, integration_method="free_fit"
    )
    
    assert isinstance(adjusted_centers, list), "adjusted_centers should be a list to reproduce the bug"
    
    # 3. Simulate the failure in _integrate_single_bank
    # Mock parameters for _integrate_single_bank
    det_config = {
        "m": 256, "n": 256, "width": 0.1, "height": 0.1,
        "center": [0, 0, 0.2], "vhat": [0, 1, 0], "uhat": [1, 0, 0], "panel": "flat"
    }
    peaks = ([10], [10], [1], [0], [0], [1.0]) # i, j, h, k, l, wl
    integration_params = {"peak_center_box_size": 3, "peak_smoothing_window_size": 3, 
                          "peak_minimum_pixels": 3, "peak_minimum_signal_to_noise": 0.0,
                          "peak_pixel_outlier_threshold": 4.0, "region_growth_distance_threshold": 1.5,
                          "region_growth_minimum_intensity": 5.0, "region_growth_maximum_pixel_radius": 5.0}
    viz_info = (False, None, "test")
    metrics_info = (None, None, None, 0, np.eye(3), [0,0,0], np.eye(3), np.zeros(3), np.array([0,0,1]))
    
    # This is where it fails: it tries to do refined_centers[keep] where refined_centers is a list
    # and keep is a boolean array.
    try:
        _integrate_single_bank(0, 0, intensity, peaks, det_config, integration_params, "free_fit", viz_info, metrics_info)
    except TypeError as e:
        if "list indices must be integers or slices, not list" in str(e):
            print(f"REPRODUCED: {e}")
            return # Success in reproducing
        raise e
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")
        raise e

def test_repro_gaussian_nll_warning():
    """
    Attempts to trigger the RuntimeWarning in gaussian fit.
    """
    width, height = 20, 20
    # Create data that might make the Gaussian fit go wonky (e.g. very low intensity or all zero)
    intensity = np.ones((width, height)) * 1e-6 
    intensity[10, 10] = 1.0
    
    peak_integrator = PeakIntegrator(
        RegionGrower(distance_threshold=1.5, min_intensity=0.1, max_size=5),
        min_peak_pixels=1,
    )
    
    predicted_centers = np.array([[10, 10]])
    # This might trigger the warning if model becomes <= 0 or other numerical issues
    with pytest.warns(RuntimeWarning):
        results, centers = peak_integrator.integrate_peaks(
            0, intensity, predicted_centers, integration_method="gaussian_fit"
        )
