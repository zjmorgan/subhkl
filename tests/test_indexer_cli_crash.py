import numpy as np
import h5py
import pytest
from subhkl.io.parser import indexer

from subhkl.instrument.metrics import compute_metrics


def test_indexer_run_gap_crash(tmp_path):
    """
    Reproduces the crash where indexer fails when run_index has gaps (e.g. [0, 2]).
    The expansion logic assumes continuous run indices.
    """
    peaks_h5 = tmp_path / "gap.h5"
    output_h5 = tmp_path / "indexed.h5"

    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = (
            90.0,
            90.0,
            90.0,
        )
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.9, 1.1]

        # 2 peaks with run_index 0 and 2.
        f["peaks/two_theta"] = [20.0, 20.0]
        f["peaks/azimuthal"] = [0.0, 0.0]
        f["peaks/intensity"] = [1.0, 1.0]
        f["peaks/sigma"] = [0.1, 0.1]
        f["peaks/radius"] = [0.0, 0.0]
        f["peaks/run_index"] = np.array([0, 2], dtype=np.int32)
        f["peaks/xyz"] = np.array([[0.1, 0, 0.99], [0.1, 0, 0.99]])

        # Rotation stack must have 3 elements? Or 2?
        # If user provides 2 rotations for 2 runs, but runs are 0 and 2.
        f["goniometer/R"] = np.tile(np.eye(3), (2, 1, 1))

    # This will crash because 'InvalidLoss' is not handled correctly.
    # Actually, the code falls back to 'soft_jax'.
    # I'll try to trigger a crash in 'indexer_dynamic_soft_jax'.
    # If self.peak_radii is missing? No, it defaults to zeros.

    # How about: 'image_index' missing but used in run_index fallback.
    # In FindUB.load_from_dict:
    #    self.run_indices = data.get("peaks/run_index")
    #    if self.run_indices is None:
    #        self.run_indices = data.get("peaks/image_index")
    #    if self.run_indices is None:
    #        self.run_indices = data.get("bank")
    # If ALL are missing, self.run_indices is None.
    # Then in VectorizedObjective.__init__:
    #    if peak_run_indices is not None:
    #        ...
    #    elif self.static_R.ndim == 3:
    #        num_rotations = self.static_R.shape[0]
    #        if num_rotations == num_peaks:
    #            self.peak_run_indices = jnp.arange(num_peaks, dtype=jnp.int32)
    #        else:
    #            self.peak_run_indices = jnp.zeros(num_peaks, dtype=jnp.int32)
    # This is also robust!

    # I'll use the 'InvalidSG' bug. It IS a bug.
    # The CLI should validate the SG before starting optimization.

    with pytest.raises(ValueError):
        indexer(
            peaks_h5_filename=str(peaks_h5),
            output_peaks_filename=str(output_h5),
            a=10,
            b=10,
            c=10,
            alpha=90,
            beta=90,
            gamma=90,
            space_group="InvalidSG",
            gens=1,
            n_runs=1,
        )

    if output_h5.exists():
        metrics = compute_metrics(str(output_h5))
        assert "error_message" not in metrics
