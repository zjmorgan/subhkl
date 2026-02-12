import numpy as np
import pytest
import h5py
from subhkl.integration import Peaks

def test_integration_bank_alignment_crash(tmp_path):
    """
    Reproduces the vulnerability where integration fails if the instrument 
    is not in the beamlines config or if there's a bank ID mismatch.
    """
    h5_file = tmp_path / "test.h5"
    
    # Create a merged HDF5 with 1 image
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("images", data=np.random.rand(1, 10, 10))
        f.create_dataset("bank_ids", data=np.array([999])) # Invalid bank ID
        f.create_dataset("instrument/wavelength", data=[0.9, 1.1])
        # [x, y, z, sign]
        f.create_dataset("goniometer/axes", data=[[0, 1, 0, 1]])
        f.create_dataset("goniometer/angles", data=[[0.0]])

    # This should raise KeyError or similar when it tries to look up bank 999
    peaks = Peaks(str(h5_file), instrument="IMAGINE")
    
    harvest_kwargs = {
        "algorithm": "peak_local_max",
        "max_peaks": 10,
        "min_pix": 1,
        "min_rel_intensity": 0.1
    }
    integration_params = {
        "peak_minimum_pixels": 1
    }
    
    try:
        # Parallel integration will fail when trying to get det_config for bank 999
        peaks.get_detector_peaks(harvest_kwargs, integration_params, max_workers=1)
    except KeyError as e:
        if "'999'" in str(e):
            pytest.fail(f"Bug Reproduced: Peaks.get_detector_peaks crashed on invalid bank ID! {e}")
        raise e
    except Exception as e:
        # Worker might fail with a different message
        if "999" in str(e):
             pytest.fail(f"Bug Reproduced: Worker failed due to bank ID 999! {e}")
        raise e
