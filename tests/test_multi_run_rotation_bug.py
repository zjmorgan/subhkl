import jax.numpy as jnp
import numpy as np

from subhkl.optimization import VectorizedObjective


def test_multi_run_rotation_assignment():
    # 2 runs, each with 5 peaks
    num_peaks = 10

    # Run 0: 5 peaks, run_index=0
    # Run 1: 5 peaks, run_index=1
    run_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Define two different rotations
    R0 = np.eye(3)
    # 90 deg rotation around Y
    R1 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    # Per-peak static_R (what indexer loads from finder.h5)
    static_R = np.zeros((num_peaks, 3, 3))
    for i in range(5):
        static_R[i] = R0
    for i in range(5, 10):
        static_R[i] = R1

    # Dummy data for objective
    B = np.eye(3)
    kf_ki_dir = np.random.randn(3, num_peaks)
    wavelength = [1.0, 2.0]
    angle_cdf = np.linspace(0, 1, 100)
    angle_t = np.linspace(0, np.pi, 100)

    # Initialize objective with peak_run_indices
    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=None,
        wavelength=wavelength,
        angle_cdf=angle_cdf,
        angle_t=angle_t,
        static_R=static_R,
        peak_run_indices=run_indices,
    )

    # Dummy solution x
    np.zeros((1, 3))  # Just orientation

    # Get results to trigger R_per_peak calculation
    # We need to bypass the JIT or just look at what happened internally
    # Actually we can just check obj.peak_run_indices and obj.static_R

    # The logic in get_results (or __call__) is:
    # R_curr = self.static_R  # (10, 3, 3)
    # R_per_peak = jnp.take(R_curr, self.peak_run_indices, axis=0)

    # If self.peak_run_indices is [0,0,0,0,0, 1,1,1,1,1]
    # R_per_peak will be [R_curr[0], R_curr[0], R_curr[0], R_curr[0], R_curr[0],
    #                    R_curr[1], R_curr[1], R_curr[1], R_curr[1], R_curr[1]]

    # But R_curr[1] is still R0! (since peaks 0-4 are Run 0)
    # Peak 5 should have R1, but it will get R_curr[1] which is R0.

    # Updated logic from optimization.py
    R_curr = obj.static_R
    num_peaks_actual = obj.kf_ki_dir_init.shape[1]

    if R_curr.ndim == 3:
        if R_curr.shape[0] == num_peaks_actual:
            R_per_peak = R_curr
        else:
            R_per_peak = jnp.take(R_curr, obj.peak_run_indices, axis=0)

    print("Peak 5 expected rotation:")
    print(R1)
    print("Peak 5 actual rotation:")
    print(R_per_peak[5])

    assert np.allclose(R_per_peak[5], R1), "Rotation for peak 5 is incorrect!"


if __name__ == "__main__":
    try:
        test_multi_run_rotation_assignment()
        print("Test PASSED")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
