import numpy as np
import h5py
import pytest
from subhkl.io.parser import indexer


def test_multi_run_geometry_mismatch_vulnerability(tmp_path):
    """
    Reproduces the bug where indexer expansion of run_index leads to
    IndexError or mismatch in VectorizedObjective because static_R
    is NOT expanded to match.
    """
    peaks_h5 = tmp_path / "merged_input.h5"
    output_h5 = tmp_path / "indexed.h5"

    a, b, c = 10.0, 10.0, 10.0

    # Define 2 runs with 2 peaks each
    # Run 0: peaks 0, 1. Run 1: peaks 2, 3.
    # Total 4 peaks.
    run_indices = np.array([0, 0, 1, 1], dtype=np.int32)

    # 2 unique rotations (per run)
    R_stack = np.tile(np.eye(3)[None, ...], (2, 1, 1))

    # 4 unique angles (PER-PEAK)
    # This will trigger expansion of run_index from 2 runs to 4 unique geometries.
    angles = np.array([[10.0, 10.1, 20.0, 20.1]])  # (1 axis, 4 peaks)

    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = (
            90.0,
            90.0,
            90.0,
        )
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.9, 1.1]

        f["peaks/two_theta"] = [20.0, 20.0, 30.0, 30.0]
        f["peaks/azimuthal"] = [0.0, 0.0, 0.0, 0.0]
        f["peaks/intensity"] = [1.0, 1.0, 1.0, 1.0]
        f["peaks/sigma"] = [0.1, 0.1, 0.1, 0.1]
        f["peaks/radius"] = [0.0, 0.0, 0.0, 0.0]
        f["peaks/run_index"] = run_indices
        f["peaks/xyz"] = np.random.rand(4, 3)

        f["goniometer/R"] = R_stack  # (2, 3, 3)
        f["goniometer/angles"] = angles  # (1, 4)
        f["goniometer/axes"] = [[0, 1, 0, 1]]
        f["goniometer/names"] = [b"omega"]

    # This is expected to CRASH in VectorizedObjective.get_results
    # because it will try to index R_stack (size 2) with expanded run_indices (values 0..3)
    try:
        indexer(
            peaks_h5_filename=str(peaks_h5),
            output_peaks_filename=str(output_h5),
            a=a,
            b=b,
            c=c,
            alpha=90,
            beta=90,
            gamma=90,
            space_group="P 1",
            strategy_name="DE",
            gens=1,
            n_runs=1,
            refine_goniometer=False,
        )
    except Exception as e:
        print(f"BUG REPRODUCED: Indexer crashed as expected! {e}")
        return

    # If it didn't crash, let's see what happened to the run indices
    with h5py.File(output_h5, "r") as f:
        out_runs = f["peaks/run_index"][()]
        out_R = f["goniometer/R"][()]
        print(f"Output run_indices: {out_runs}")
        print(f"Output R stack shape: {out_R.shape}")

        # If out_runs has values > 1 but out_R has size 2, it's a latent mismatch bug.
        if np.max(out_runs) >= out_R.shape[0]:
            pytest.fail(
                f"Correctness VULNERABILITY: run_index expanded to {np.max(out_runs)}, "
                f"but R stack remained at size {out_R.shape[0]}!"
            )


if __name__ == "__main__":
    from pathlib import Path

    tmp = Path("temp_idx_bug")
    tmp.mkdir(exist_ok=True)
    try:
        test_multi_run_geometry_mismatch_vulnerability(tmp)
    except Exception as e:
        print(f"Test error: {e}")
