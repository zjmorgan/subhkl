import numpy as np

from subhkl.optimization import FindUB


def test_rotating_sample_offset():
    # 1. Physical Parameters
    a, b, c = 10.0, 10.0, 10.0
    alpha, beta, gamma = 90.0, 90.0, 90.0
    # 50mm offset in X (Sample frame) - large offset to make it obvious
    sample_offset_sample_frame = np.array([0.05, 0.0, 0.0])

    # 2. Setup Goniometer: Two runs, phi=0 and phi=90 (rotation around Y)
    # Axes format: [x, y, z, sign]
    gonio_axes = np.array([[0, 1, 0, 1]])  # Y-axis
    gonio_angles = np.array([[0.0, 90.0]])  # 2 runs

    # Rotation matrices R_sample_to_lab
    # Run 0: Identity
    R0 = np.eye(3)
    # Run 1: 90 deg around Y. X -> -Z, Z -> X
    R1 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    R_stack = np.stack([R0, R1])

    # 3. Generate Peaks
    # Peak 0: Run 0, hkl=(5,0,0). Q_sample = (0.5, 0, 0)
    # Peak 1: Run 1, hkl=(0,0,5). Q_sample = (0, 0, 0.5)

    # Lab positions of sample
    S0 = R0 @ sample_offset_sample_frame
    S1 = R1 @ sample_offset_sample_frame

    # ki = (0,0,1)
    ki = np.array([0, 0, 1])

    # Peak 0
    Q_lab0 = R0 @ np.array([0.5, 0, 0])
    kf0 = Q_lab0 + ki
    kf0 = kf0 / np.linalg.norm(kf0)
    P0 = S0 + kf0  # Detector at 1m

    # Peak 1
    Q_lab1 = R1 @ np.array([0, 0, 0.5])
    kf1 = Q_lab1 + ki
    kf1 = kf1 / np.linalg.norm(kf1)
    P1 = S1 + kf1

    # Simulated 'finder' output (angles assuming sample at origin)
    kf_orig0 = P0 / np.linalg.norm(P0)
    tt0 = np.rad2deg(np.arccos(np.clip(kf_orig0[2], -1, 1)))
    az0 = np.rad2deg(np.arctan2(kf_orig0[1], kf_orig0[0]))

    kf_orig1 = P1 / np.linalg.norm(P1)
    tt1 = np.rad2deg(np.arccos(np.clip(kf_orig1[2], -1, 1)))
    az1 = np.rad2deg(np.arctan2(kf_orig1[1], kf_orig1[0]))

    data = {
        "sample/a": a,
        "sample/b": b,
        "sample/c": c,
        "sample/alpha": alpha,
        "sample/beta": beta,
        "sample/gamma": gamma,
        "sample/space_group": "P 1",
        "instrument/wavelength": [0.5, 2.0],
        "peaks/intensity": np.array([100.0, 100.0]),
        "peaks/sigma": np.array([0.1, 0.1]),
        "peaks/radius": np.array([0.01, 0.01]),
        "peaks/two_theta": np.array([tt0, tt1]),
        "peaks/azimuthal": np.array([az0, az1]),
        "peaks/xyz": np.array([P0, P1]),
        "peaks/run_index": np.array([0, 1]),
        "goniometer/R": R_stack,
        "goniometer/axes": gonio_axes,
        "goniometer/angles": gonio_angles.T,
    }

    fu = FindUB(data=data)
    fu.base_sample_offset = sample_offset_sample_frame

    # 4. Run Indexer (Stage 2 style: refine_sample=False)
    print("\n--- Running Indexer with Rotating Sample Offset ---")
    score, hkl, lam, U = fu.minimize(
        strategy_name="DE",
        population_size=100,
        num_generations=50,
        tolerance_deg=0.1,
        refine_sample=False,
        refine_goniometer=False,
        hkl_search_range=15,
    )

    print(f"Best Score: {score:.2f}/2.0")
    if score < 1.9:
        print("FAILURE: Indexer could not find the solution.")
    else:
        print(
            "SUCCESS: Indexer found the solution with rotating sample offset model."
        )


if __name__ == "__main__":
    test_rotating_sample_offset()
