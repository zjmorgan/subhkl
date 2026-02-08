import numpy as np
import jax.numpy as jnp
from subhkl.optimization import VectorizedObjective

def test_multi_run_mapping_bug():
    # Scenario: 2 images, 2 peaks per image = 4 peaks total.
    # Image 0 has rotation A, Image 1 has rotation B.
    # finder.h5 would have:
    # peaks/run_index = [0, 0, 1, 1] (mapping peaks to image indices)
    # goniometer/angles = [A, A, B, B] (per-peak angles)
    
    num_peaks = 4
    run_indices = np.array([0, 0, 1, 1])
    
    # Define two different rotations (Euler angles)
    A = np.array([10.0, 0.0, 0.0])
    B = np.array([20.0, 0.0, 0.0])
    
    # Per-peak goniometer angles as found in finder.h5
    goniometer_angles = np.array([A, A, B, B])
    
    # Dummy data for objective
    B_mat = np.eye(3)
    kf_ki_dir = np.random.randn(3, num_peaks)
    wavelength = [1.0, 2.0]
    angle_cdf = np.linspace(0, 1, 100)
    angle_t = np.linspace(0, np.pi, 100)
    
    # Define goniometer axes (e.g. just one axis for simplicity)
    gonio_axes = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    
    # Initialize objective
    obj = VectorizedObjective(
        B=B_mat,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=None,
        wavelength=wavelength,
        angle_cdf=angle_cdf,
        angle_t=angle_t,
        goniometer_axes=gonio_axes,
        goniometer_angles=goniometer_angles,
        peak_run_indices=run_indices
    )
    
    # x is dummy solution (3 orientation params)
    x = jnp.zeros((1, 3))
    
    # Trigger parameter reconstruction
    UB, B_new, sample_total, ki_vec, offsets_total, R = obj._get_physical_params_jax(x)
    
    # R has shape (S, num_runs, 3, 3)
    # obj.gonio_angles.shape[1] is 4, so num_runs = 4.
    # R[0] is (4, 3, 3)
    R_curr = R
    
    # The logic in get_results is:
    # num_peaks = 4
    # R_curr.shape[1] == 4 == num_peaks
    # So R_per_peak = R_curr
    
    # Let's check Peak 2 (which belongs to Image 1 and should have rotation B)
    # Peak 2 uses R_per_peak[0, 2]
    
    # However, if it followed peak_run_indices, it should use R_curr[0, run_indices[2]] = R_curr[0, 1]
    # R_curr[0, 1] is the rotation for the 2nd entry in gonio_angles, which is A.
    
    # Wait, the bug is:
    # If R_curr.shape[1] == num_peaks, it sets R_per_peak = R_curr.
    # Peak 2 gets R_curr[0, 2].
    # R_curr[0, 2] is calculated from gonio_angles[:, 2], which is B.
    # So Peak 2 gets rotation B. THIS IS CORRECT.
    
    # WAIT! Where is the bug then?
    # Ah! If R_curr.shape[1] != num_peaks.
    # But in the user's case, goniometer/angles HAS 288 rows. So N_runs IS 288.
    
    # Let's check the case where only SOME peaks are loaded or filtered.
    # If the indexer filters peaks by d-spacing, num_peaks changes!
    # Suppose we filter to 2 peaks: Peak 0 and Peak 2.
    # num_peaks = 2.
    # R_curr.shape[1] is still 4 (from the full goniometer_angles).
    # 4 != 2.
    # So it uses jnp.take(R_curr, peak_run_indices, axis=1)
    # peak_run_indices for filtered peaks: [0, 1]
    # R_per_peak[0] = R_curr[0] (Rotation A). Correct.
    # R_per_peak[1] = R_curr[1] (Rotation A again!). WRONG!
    # Peak 2 belongs to Image 1, but it gets R_curr[1] which is Peak 1's rotation (Image 0).
    
    print(f"Objective num_runs: {obj.gonio_angles.shape[1]}")
    
    # Case: Filtered peaks
    # Filter to peaks 0 and 2.
    filtered_run_indices = jnp.array([0, 1]) # peak_run_indices[ [0, 2] ]
    
    # Re-init objective with filtered data
    obj_filt = VectorizedObjective(
        B=B_mat,
        kf_ki_dir=kf_ki_dir[:, [0, 2]],
        peak_xyz_lab=None,
        wavelength=wavelength,
        angle_cdf=angle_cdf,
        angle_t=angle_t,
        goniometer_axes=gonio_axes,
        goniometer_angles=goniometer_angles, # Still the full 4 runs
        peak_run_indices=filtered_run_indices
    )
    
    # num_peaks is now 2.
    # R_curr.shape[1] is 4.
    # 4 != 2.
    # R_per_peak = jnp.take(R_curr, [0, 1], axis=1)
    # Peak index 1 (original peak 2) gets R_curr[1] (Rotation A).
    
    UB_f, B_f, S_f, K_f, O_f, R_f = obj_filt._get_physical_params_jax(x)
    
    # R_f is (1, 4, 3, 3)
    # We need to simulate the get_results logic
    R_per_peak_filt = jnp.take(R_f, filtered_run_indices, axis=1)
    
    # Expected rotation for second peak (original peak 2) is Rotation B
    # R_f[0, 2] is Rotation B.
    # R_f[0, 1] is Rotation A.
    
    rot_B = R_f[0, 2]
    rot_actual = R_per_peak_filt[0, 1]
    
    print("Expected rotation (B):")
    print(rot_B)
    print("Actual rotation assigned:")
    print(rot_actual)
    
    assert jnp.allclose(rot_actual, rot_B), "Mapping bug detected: Filtered peak 2 got Rotation A instead of B!"

if __name__ == "__main__":
    try:
        test_multi_run_mapping_bug()
        print("Test PASSED")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
