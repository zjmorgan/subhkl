import jax.numpy as jnp
import numpy as np
import pytest
from subhkl.optimization import VectorizedObjective


def test_cosine_loss_instability_at_high_index():
    """
    PROVES that the current 'cosine' loss implementation is unstable
    for high-index reflections because it scales kappa by 1/h^2 per component.
    This creates extremely tight tolerances for components near zero,
    making the loss landscape a forest of delta functions.
    """
    # 1. Setup Objective
    # Simple Cubic cell a=10
    B = np.eye(3) * 0.1
    # 1 peak at [10, 0, 0]
    hkl_true = np.array([[10.0, 0.0, 0.0]])
    # Lab scattering vector Q_l = U B h.
    # Let U = Identity. Q_l = [1.0, 0.0, 0.0]
    q_lab = B @ hkl_true.T  # (3, 1)

    # Mock data
    wavelength = [1.0, 2.0]
    angle_cdf = np.linspace(0, 1, 100)
    angle_t = np.linspace(0, np.pi, 100)

    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=q_lab,
        peak_xyz_lab=None,
        wavelength=wavelength,
        angle_cdf=angle_cdf,
        angle_t=angle_t,
        loss_method="cosine",
        tolerance_deg=0.1,
    )

    # 2. Evaluate Loss at True Orientation (U=Identity)
    UB_true = jnp.eye(3) @ jnp.array(B)
    kf_ki_sample = jnp.array(q_lab)[None, ...]  # (1, 3, 1)

    score_true, probs_true, _, _ = obj.indexer_dynamic_cosine_aniso_jax(
        UB_true[None, ...], kf_ki_sample, tolerance_rad=jnp.deg2rad(0.1)
    )

    print(f"Score at True orientation: {score_true[0]}")
    print(f"Prob at True orientation: {probs_true[0, 0]}")

    # 3. Evaluate Loss at 0.01 deg offset
    from scipy.spatial.transform import Rotation

    R_off = Rotation.from_euler("y", 0.01, degrees=True).as_matrix()
    UB_off = R_off @ B

    score_off, probs_off, _, _ = obj.indexer_dynamic_cosine_aniso_jax(
        jnp.array(UB_off)[None, ...],
        kf_ki_sample,
        tolerance_rad=jnp.deg2rad(0.1),
    )

    print(f"Score at 0.01 deg offset: {score_off[0]}")
    print(f"Prob at 0.01 deg offset: {probs_off[0, 0]}")

    # Analysis:
    # A 0.01 deg offset is 10x smaller than the 0.1 deg tolerance.
    # The probability should be near 1.0 (e.g. > 0.9).
    # If the bug exists, prob will be near 0.0 because the '0' components
    # of the HKL are penalized with effectively infinite precision.

    # Touchdown Check
    assert probs_off[0, 0] > 0.5, (
        f"instability Detected! Prob {probs_off[0, 0]:.4f} is too low for 0.01 deg offset "
        f"(within 0.1 deg tolerance). The loss landscape is too sharp."
    )


if __name__ == "__main__":
    try:
        test_cosine_loss_instability_at_high_index()
        print("Test PASSED (No Bug Found?)")
    except AssertionError as e:
        print(f"TOUCHDOWN! Bug Proven: {e}")
    except Exception as e:
        print(f"Error: {e}")
