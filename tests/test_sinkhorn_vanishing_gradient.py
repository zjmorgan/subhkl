import numpy as np
import jax
import jax.numpy as jnp
from subhkl.optimization import VectorizedObjective


def test_sinkhorn_vanishing_gradient():
    # Large unit cell
    B = np.eye(3) * (1.0 / 60.0)  # 60A cell

    # 1 peak at (1, 0, 0) in reciprocal space
    # B maps HKL to Q. For 60A cell, Q(1,0,0) = 1/60 A^-1.
    # Wavelength 1.0A means |k| = 1.0 A^-1.
    q_ideal = B @ np.array([[1.0], [0.0], [0.0]])
    kf_ki_dir = q_ideal  # Use actual Q vector magnitude, not normalized

    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=None,
        wavelength=[0.5, 2.5],
        angle_cdf=np.linspace(0, 1, 100),
        angle_t=np.linspace(0, np.pi, 100),
        hkl_search_range=10,
        space_group="P 1",
        loss_method="sinkhorn",
    )

    # Orientation offset (use Y or Z to ensure X-axis moves)
    from scipy.spatial.transform import Rotation

    R_off = Rotation.from_euler("y", 1.5, degrees=True).as_matrix()

    # Tolerance 0.1 degrees (Stage 2 setting)
    tol_rad = jnp.deg2rad(0.1)

    # Calc score and gradient
    def get_score(orient):
        # Rodrigues vector
        from subhkl.optimization import rotation_matrix_from_rodrigues_jax

        U = rotation_matrix_from_rodrigues_jax(orient)
        UB = U @ jnp.array(B)
        score, _, _, _ = obj.indexer_sinkhorn_jax(
            UB[None], jnp.array(kf_ki_dir)[None], tolerance_rad=tol_rad
        )
        return score[0]

    # rods vector for R_off
    rods_off = Rotation.from_matrix(R_off).as_rotvec()

    score_off = get_score(rods_off)
    grad_fn = jax.grad(get_score)
    grad_off = grad_fn(rods_off)

    print(f"Score at 1.5 deg offset (tol 0.1): {score_off}")
    print(f"Gradient: {grad_off}")
    grad_norm = np.linalg.norm(grad_off)
    print(f"Gradient magnitude: {grad_norm}")

    # Check for validity
    assert not np.isnan(score_off), "Score is NaN"
    assert not np.isnan(grad_norm), "Gradient is NaN"
    assert grad_norm > 1e-6, "Gradient is vanishing! (Too small)"

    print("FIX CONFIRMED: Log-Sinkhorn provides stable gradients and scores!")


if __name__ == "__main__":
    try:
        test_sinkhorn_vanishing_gradient()
        print("Test PASSED")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
