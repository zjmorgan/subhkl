import jax.numpy as jnp

from subhkl.optimization import VectorizedObjective


def test_score_sign_consistency():
    """
    Expose the inconsistency in score signs between 'gaussian' and 'cosine' losses.
    'gaussian' loss returns negative peak counts (e.g., -10.0),
    while 'cosine' loss returns positive negative-log-likelihoods (e.g., +2000.0).
    This breaks the progress bar display which expects -fitness to be a peak count.
    """
    # Mock data
    B = jnp.eye(3)
    kf_ki_dir = jnp.array([[1.0, 0.0, 0.0]])
    peak_xyz = jnp.array([[1.0, 0.0, 0.0]])
    wavelength = jnp.array([2.0, 4.0])
    angle_cdf = jnp.array([0.0, 1.0])
    angle_t = jnp.array([0.0, 1.0])

    # 1. Test Gaussian Loss
    obj_gauss = VectorizedObjective(
        B=B,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=peak_xyz,
        wavelength=wavelength,
        angle_cdf=angle_cdf,
        angle_t=angle_t,
        loss_method="gaussian",
    )

    # 2. Test Cosine Loss
    obj_cosine = VectorizedObjective(
        B=B,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=peak_xyz,
        wavelength=wavelength,
        angle_cdf=angle_cdf,
        angle_t=angle_t,
        loss_method="cosine",
    )

    # Simple identity orientation
    x = jnp.zeros((1, 3))  # Rodrigues vector [0,0,0]

    score_gauss = obj_gauss(x)[0]
    score_cosine = obj_cosine(x)[0]

    print(f"\nGaussian Score: {score_gauss}")
    print(f"Cosine Score: {score_cosine}")

    # The vulnerability: progress bar assumes -score is positive peak count
    # For gaussian, score is roughly -1.0 * (peaks indexed)
    # For cosine, score is -log_likelihood, which is a large positive number.

    # This assertion checks if the scores have the same sign convention
    # currently this is expected to FAIL because they are opposite.
    assert (score_gauss < 0) == (score_cosine < 0), (
        "Loss methods must have consistent sign conventions for progress reporting!"
    )
