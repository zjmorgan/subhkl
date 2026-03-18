import numpy as np
from subhkl.convex_hull.peak_integrator import PeakIntegrator
from subhkl.convex_hull.region_grower import RegionGrower
from subhkl.integration.worker import integrate_single_bank


def test_integration_worker_handles_numpy_centers():
    """
    Regression test for: 'Integration worker failed: list indices must be integers or slices, not list'
    Verifies that PeakIntegrator returns adjusted_centers as a numpy array and
    _integrate_single_bank correctly processes them.
    """
    # 1. Setup a simple peak
    width, height = 20, 20
    intensity = np.zeros((width, height))
    intensity[9:12, 9:12] = 100  # 3x3 block

    peak_integrator = PeakIntegrator(
        RegionGrower(distance_threshold=1.5, min_intensity=5, max_size=5),
        min_peak_pixels=3,
    )

    # 2. Integrate - should return adjusted_centers as a numpy array
    predicted_centers = np.array([[10, 10]])
    results, adjusted_centers = peak_integrator.integrate_peaks(
        0, intensity, predicted_centers, integration_method="free_fit"
    )

    assert isinstance(adjusted_centers, np.ndarray), (
        "adjusted_centers must be a numpy array"
    )

    # 3. Verify _integrate_single_bank doesn't crash
    det_config = {
        "m": 256,
        "n": 256,
        "width": 0.1,
        "height": 0.1,
        "center": [0, 0, 0.2],
        "vhat": [0, 1, 0],
        "uhat": [1, 0, 0],
        "panel": "flat",
    }
    # i, j, h, k, l, wl
    peaks = (
        np.array([10]),
        np.array([10]),
        np.array([1]),
        np.array([0]),
        np.array([0]),
        np.array([1.0]),
    )
    integration_params = {
        "peak_center_box_size": 3,
        "peak_smoothing_window_size": 3,
        "peak_minimum_pixels": 3,
        "peak_minimum_signal_to_noise": 0.0,
        "peak_pixel_outlier_threshold": 4.0,
        "region_growth_distance_threshold": 1.5,
        "region_growth_minimum_intensity": 5.0,
        "region_growth_maximum_pixel_radius": 5.0,
    }
    viz_info = (False, None, "test")
    # found_peaks_xyz, found_peaks_bank, found_peaks_run, run_id, RUB, angles, R, offset, ki
    metrics_info = (
        None,
        None,
        None,
        0,
        np.eye(3),
        np.zeros(3),
        np.eye(3),
        np.zeros(3),
        np.array([0, 0, 1]),
    )

    # This should NOT raise TypeError
    res = integrate_single_bank(
        0,
        0,
        intensity,
        peaks,
        det_config,
        integration_params,
        "free_fit",
        viz_info,
        metrics_info,
    )
    assert res is not None
    assert res["bank"][0] == 0  # bank_id


def test_gaussian_fit_numerical_stability():
    """
    Regression test for numerical instability in Gaussian fit NLL calculation.
    Verifies that hardened MLE logic doesn't emit RuntimeWarnings for edge-case data.
    """
    width, height = 20, 20
    # Data that might have triggered log(0) or NaNs in old logic
    intensity = np.ones((width, height)) * 1e-6
    intensity[10, 10] = 1.0

    peak_integrator = PeakIntegrator(
        RegionGrower(distance_threshold=1.5, min_intensity=0.1, max_size=5),
        min_peak_pixels=1,
    )

    predicted_centers = np.array([[10, 10]])

    # Should not emit RuntimeWarning (e.g. log(0), invalid value in log)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        results, centers = peak_integrator.integrate_peaks(
            0, intensity, predicted_centers, integration_method="gaussian_fit"
        )

    assert centers is not None
    assert len(results) > 0
