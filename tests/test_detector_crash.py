import numpy as np
import pytest

from subhkl.detector import angles_from_scattering_vector


def test_detector_single_vector_crash():
    """
    Reproduce the crash in angles_from_scattering_vector when passing a single 1D vector.
    The underlying angles_from_kf function assumes 2D input (N, 3) and hardcodes axis=1.
    """
    q_vec = np.array([1.0, 0.0, 0.0])

    try:
        from numpy.exceptions import AxisError
    except ImportError:
        from numpy import AxisError

    # This is expected to work for a robust library.
    # We now expect it to SUCCEED.
    try:
        tt, az = angles_from_scattering_vector(q_vec)
        print(f"TwoTheta: {tt}, Azimuth: {az}")
        assert len(tt) == 1
        assert len(az) == 1
    except AxisError as e:
        pytest.fail(f"Crash reproduced (Fix Failed): {e}")
    except Exception as e:
        pytest.fail(f"Crashed with unexpected error: {e}")


if __name__ == "__main__":
    test_detector_single_vector_crash()
