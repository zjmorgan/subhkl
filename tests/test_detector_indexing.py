import numpy as np
import pytest
from subhkl.detector import Detector

@pytest.fixture
def flat_detector():
    config = {
        "m": 256, "n": 256, "width": 0.1, "height": 0.1,
        "center": [0, 0, 0.2], "vhat": [0, 1, 0], "uhat": [1, 0, 0],
        "panel": "flat"
    }
    return Detector(config)

def test_pixel_to_angles_single_peak(flat_detector):
    tt, az = flat_detector.pixel_to_angles([100], [100])
    assert len(tt) == 1
    assert len(az) == 1

def test_pixel_to_angles_multi_peak(flat_detector):
    rows = [100, 110, 120, 130]
    cols = [100, 110, 120, 130]
    tt, az = flat_detector.pixel_to_angles(rows, cols)
    assert len(tt) == 4
    assert len(az) == 4

def test_pixel_to_angles_with_offset(flat_detector):
    rows = [100, 110]
    cols = [100, 110]
    offset = [0.01, 0.02, 0.03]
    tt, az = flat_detector.pixel_to_angles(rows, cols, sample_offset=offset)
    assert len(tt) == 2
    assert len(az) == 2

def test_pixel_to_lab_consistency(flat_detector):
    rows = [100, 110]
    cols = [100, 110]
    xyz = flat_detector.pixel_to_lab(rows, cols)
    assert xyz.shape == (2, 3)

if __name__ == "__main__":
    config = {
        "m": 256, "n": 256, "width": 0.1, "height": 0.1,
        "center": [0, 0, 0.2], "vhat": [0, 1, 0], "uhat": [1, 0, 0],
        "panel": "flat"
    }
    det = Detector(config)
    test_pixel_to_angles_single_peak(det)
    test_pixel_to_angles_multi_peak(det)
    test_pixel_to_angles_with_offset(det)
    test_pixel_to_lab_consistency(det)
    print("All tests passed!")
