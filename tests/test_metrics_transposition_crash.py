import numpy as np
import h5py
import os
from subhkl.io.parser import metrics

def test_metrics_transposition_crash():
    # Setup dummy files
    filename = "test_metrics_trans_res.h5"
    found_peaks_file = "test_metrics_trans_found.h5"
    
    # 1. Create result file (Predictor format - triggers the buggy code path)
    with h5py.File(filename, "w") as f:
        f["sample/a"] = 10.0; f["sample/b"] = 10.0; f["sample/c"] = 10.0
        f["sample/alpha"] = 90.0; f["sample/beta"] = 90.0; f["sample/gamma"] = 90.0
        f["sample/U"] = np.eye(3)
        f["goniometer/R"] = np.tile(np.eye(3)[None], (1, 1, 1)) 
        f.attrs["instrument"] = "MANDI"
        
        # Predictor format: banks/{img_idx}/...
        banks_grp = f.create_group("banks")
        run0 = banks_grp.create_group("0")
        
        # Add bank mapping
        f["bank_ids"] = np.array([1]) # Run 0 maps to Bank 1
        
        # Scenario: 33 peaks (reproducing the "length 33" error)
        num_peaks = 33
        run0["h"] = np.zeros(num_peaks)
        run0["k"] = np.zeros(num_peaks)
        run0["l"] = np.zeros(num_peaks)
        run0["wavelength"] = np.ones(num_peaks)
        run0["i"] = np.zeros(num_peaks)
        run0["j"] = np.zeros(num_peaks)
        
    # 2. Create found peaks file
    with h5py.File(found_peaks_file, "w") as f:
        # 1 observed peak
        f["peaks/xyz"] = np.zeros((1, 3))
        f["peaks/run_index"] = np.array([0])
        f.attrs["instrument"] = "MANDI"
        
    print(f"Running metrics with {num_peaks} predicted peaks...")
    try:
        # This will trigger: xyz_pred_run = det.pixel_to_lab(i_p, j_p).T
        # which becomes (3, 33).
        # KDTree will think dimension is 33.
        # tree.query(xyz_obs_run) will fail if xyz_obs_run is (1, 3).
        metrics(filename, found_peaks_file=found_peaks_file, instrument="MANDI")
        print("Metrics command completed successfully")
    except ValueError as e:
        print(f"Metrics command CRASHED as expected: {e}")
        assert "vectors of length 33" in str(e)
        print("BUG CONFIRMED: Transposition error in metrics command!")
    except Exception as e:
        print(f"Metrics command CRASHED with unexpected error: {e}")
        raise e
    finally:
        if os.path.exists(filename): os.remove(filename)
        if os.path.exists(found_peaks_file): os.remove(found_peaks_file)

if __name__ == "__main__":
    try:
        test_metrics_transposition_crash()
        print("Test PASSED (Bug Reproduced)")
    except Exception as e:
        print(f"Test FAILED: {e}")
