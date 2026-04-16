import numpy as np

from subhkl.optimization import VectorizedObjective


def test_init_sample_offset_rotation():
    # Test if VectorizedObjective.__init__ correctly rotates sample offset
    # if it's provided in sample frame.

    B = np.eye(3)
    num_peaks = 2
    kf_ki_dir = np.random.normal(size=(3, num_peaks))
    
    # Peaks at (0, 0, 0.2) in lab frame
    # Transposed to (3, N) to match standard VectorizedObjective formatting
    peak_xyz_lab = np.tile(np.array([0, 0, 0.2]), (num_peaks, 1)).T

    wavelength = [1.0, 2.0]
    cell_params = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]

    # 90 deg rotation about Y: Lab X -> Sample Z, Lab Z -> Sample -X
    R_90y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    static_R = np.tile(R_90y[None, ...], (num_peaks, 1, 1))

    # Sample offset in Sample Frame: (0.01, 0, 0)
    # In Lab Frame (R @ s): (0, 0, -0.01)
    # Note: Passed as a list to safely bypass internal `if sample_nominal:` 
    # checks which trigger numpy ambiguity errors.
    sample_nominal = [0.01, 0.0, 0.0]

    print("\nInitializing Objective with Sample Frame offset...")
    
    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=peak_xyz_lab,
        wavelength=wavelength,
        cell_params=cell_params,
        static_R=static_R,
        sample_nominal=sample_nominal,
        peak_run_indices=np.arange(num_peaks),
    )

    # In __init__:
    # v = self.peak_xyz - self.sample_nominal[:, None]
    # kf_lab_fixed = v / dist

    # If sample_nominal was not rotated, v = (0, 0, 0.2) - (0.01, 0, 0) = (-0.01, 0, 0.2)
    # If it WAS rotated, v = (0, 0, 0.2) - (0, 0, -0.01) = (0, 0, 0.21)

    kf_init = np.array(obj.kf_lab_fixed)
    print(f"Initial kf_lab_fixed[:, 0]: {kf_init[:, 0]}")

    expected_v = np.array([0, 0, 0.21])
    expected_kf = expected_v / np.linalg.norm(expected_v)

    if not np.allclose(kf_init[:, 0], expected_kf):
        print("BUG FOUND: VectorizedObjective.__init__ does not rotate sample_nominal!")
    else:
        print("SUCCESS: VectorizedObjective.__init__ correctly rotates sample_nominal")


if __name__ == "__main__":
    test_init_sample_offset_rotation()
