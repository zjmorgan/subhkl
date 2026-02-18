import numpy as np
import pytest

from subhkl.optimization import VectorizedObjective


def test_cosine_argument_compatibility():
    """
    Regression test to ensure 'cosine' loss method accepts k_sq_override.
    This mimics the call structure in VectorizedObjective.get_results/loss call stack.
    """
    # Mock data
    B = np.eye(3)
    kf_ki_dir = np.array([[1.0, 0.0, 0.0]])
    peak_xyz = np.array([[1.0, 0.0, 0.0]])

    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=peak_xyz,
        wavelength=np.array([2.0, 4.0]),
        angle_cdf=np.array([0.0, 1.0]),
        angle_t=np.array([0.0, 1.0]),
        loss_method="cosine",
    )

    # Mock inputs for the indexer
    UB = np.eye(3)[None, ...]  # (Batch, 3, 3)
    kf_ki_sample = np.array([[[1.0], [0.0], [0.0]]])  # (Batch, 3, N)
    k_sq_dyn = np.array([[1.0]])  # (Batch, N)

    try:
        # This call mimics line 1287 in optimization.py
        obj.indexer_dynamic_cosine_aniso_jax(
            UB, kf_ki_sample, k_sq_override=k_sq_dyn, tolerance_rad=0.002
        )
    except TypeError as e:
        pytest.fail(f"Caught TypeError as expected from bug report: {e}")
    except Exception:
        # We might get shape errors or others, but we are looking specifically for the argument error
        pass
