import numpy as np
import h5py
from subhkl.commands import run_finder as finder


def test_finder_output_determinism(tmp_path):
    """
    Proven source of non-determinism: Peaks.get_detector_peaks uses
    as_completed(futures), causing peak order to depend on execution timing.
    """
    # 1. Create a dummy reduced HDF5 with multiple images
    # We need enough images to ensure parallel execution might complete out of order.
    reduced_h5 = tmp_path / "dummy_reduced.h5"
    num_banks = 50
    height, width = 100, 100

    with h5py.File(reduced_h5, "w") as f:
        # Create 50 images with variable number of peaks to cause timing jitter
        images = np.zeros((num_banks, height, width))
        for i in range(num_banks):
            # Some images have many peaks, some have few
            num_peaks = 1 if (i % 2 == 0) else 20
            for p in range(num_peaks):
                images[i, 10 + p, 10 + p] = 1000.0

        f.create_dataset("images", data=images)
        f.create_dataset("bank_ids", data=np.arange(num_banks, dtype=np.int32))
        f.create_dataset("goniometer/angles", data=np.zeros((num_banks, 1)))
        f.create_dataset("goniometer/axes", data=[[0, 1, 0, 1]])
        f.create_dataset("instrument/wavelength", data=[2.0, 4.0])
        f.attrs["instrument"] = "MANDI"

    # 2. Run finder twice
    output1 = tmp_path / "finder1.h5"
    output2 = tmp_path / "finder2.h5"

    # We use thresholding which is fast
    finder_kwargs = {
        "filename": str(reduced_h5),
        "instrument": "MANDI",
        "finder_algorithm": "thresholding",
        "region_growth_minimum_intensity": 10.0,
        "peak_minimum_pixels": 1,
        "max_workers": 8,
        "show_progress": False,
    }

    # Try up to 5 times to catch it
    for attempt in range(5):
        finder(output_filename=str(output1), **finder_kwargs)
        finder(output_filename=str(output2), **finder_kwargs)

        # 3. Compare peak order
        with h5py.File(output1, "r") as f1, h5py.File(output2, "r") as f2:
            xyz1 = f1["peaks/xyz"][()]
            xyz2 = f2["peaks/xyz"][()]

            # If non-deterministic, the order of XYZ coordinates might differ
            is_equal = np.array_equal(xyz1, xyz2)

            if not is_equal:
                print(f"NON-DETERMINISM DETECTED on attempt {attempt + 1}")
                assert False, "Finder output is non-deterministic!"

    print("DETERMINISM CONFIRMED in 5 attempts.")


if __name__ == "__main__":
    # For manual testing
    from pathlib import Path

    tmp = Path("temp_test")
    tmp.mkdir(exist_ok=True)
    try:
        test_finder_output_determinism(tmp)
        print("Test PASSED (Output is deterministic - No Bug Found?)")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"Error: {e}")
