import jax.numpy as jnp
import numpy as np

from subhkl.optimization import VectorizedObjective


def test_cosine_indexer_negative_score_repro():
    """
    Reproduce the issue where indexer_dynamic_cosine_aniso_jax returns
    a score based on log-probability (which can be negative -> score positive)
    instead of linear probability sum (which should be positive -> score negative).

    The user sees "Best: -2120.9", which implies the objective returned +2120.9.
    """
    # 1. Setup minimal objective
    # 1 peak, identity orientation
    kf_ki = np.array([[[1.0, 0.0, 0.0]]])  # (1, 3, 1) - scaled by k?
    # The code expects kf_ki_sample to be kf-ki.
    # Let's just make dummy vectors.

    # B matrix (cubic)
    B = np.eye(3)

    # 1 obs
    kf_ki_sample = jnp.array([[[1.0, 0.0, 0.0]]])

    # Weights
    weights = jnp.array([1.0])

    # Create instance
    # We only need the method, but we need 'self' parameters.
    # Passing dummy values for most things.
    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=np.array([[1.0], [0.0], [0.0]]),  # (3, 1)
        peak_xyz_lab=None,
        wavelength=[1.0, 1.0],
        angle_cdf=[0, 1],
        angle_t=[0, 1],
        weights=[1.0],
        d_min=0.1,
        d_max=10.0,
    )

    # 2. Mock Internal State for Indexing
    UB = jnp.eye(3)[None, :, :]  # Batch size 1

    # We call the method directly
    # kf_ki_sample: (Batch, 3, N_obs)
    # k_sq_override: (Batch, N_obs)
    k_sq = jnp.array([[1.0]])

    # 3. Run Indexer
    score, probs, best_hkl, best_lamb = obj.indexer_dynamic_cosine_aniso_jax(
        UB, kf_ki_sample, k_sq_override=k_sq, tolerance_rad=0.01
    )

    print(f"Score: {score}")
    print(f"Probs: {probs}")

    # 4. Analyze
    # If the bug exists, score will be -log(probs).
    # If probs is small (e.g. e^-10), log(probs) is -10. Score is +10.
    # Minimizer displays {-score} -> -10.

    # If proper behavior (like soft indexer), score should be -probs.
    # If probs is e^-10 (approx 0), score is -0.

    # We suspect the bug is that score is positive (log likelihood).
    # Let's assert that score is NOT consistent with linear counting.

    # If probs is say 0.5.
    # Linear Score: -0.5
    # Log Score: -log(0.5) = +0.69

    # Check if score is roughly -log(probs)
    expected_log_score = -jnp.sum(weights * jnp.log(probs + 1e-12))
    expected_linear_score = -jnp.sum(weights * probs)

    print(f"Expected Log Score: {expected_log_score}")
    print(f"Expected Linear Score: {expected_linear_score}")

    # The Bug: The code uses log-sum (score = -sum(log_probs))
    # instead of linear sum (score = -sum(probs)).

    # We expect the code to be FIXED now.
    assert jnp.isclose(score, expected_linear_score, atol=1e-3), (
        f"Score {score} does not match linear count {expected_linear_score}! Fix failed?"
    )


if __name__ == "__main__":
    test_cosine_indexer_negative_score_repro()
