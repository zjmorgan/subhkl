import numpy as np
import pytest
import h5py
from subhkl.metrics import compute_metrics

def test_mtz_exporter_sg_mismatch():
    """
    Reproduces the bug where MTZExporter fails if the space group name
    is not recognized by gemmi during initialization.
    """
    from subhkl.export import MTZExporter
    import h5py
    
    indexed_h5 = "temp_dummy.h5"
    # Create a minimal indexed file
    with h5py.File(indexed_h5, "w") as f:
        f["peaks/h"] = [1, 0, 0]
        f["peaks/k"] = [0, 1, 0]
        f["peaks/l"] = [0, 0, 1]
        f["peaks/intensity"] = [100, 200, 300]
        f["peaks/sigma"] = [10, 20, 30]
        f["peaks/lambda"] = [1.0, 1.0, 1.0]
        f["peaks/two_theta"] = [20.0, 20.0, 20.0]
        f["peaks/azimuthal"] = [0.0, 0.0, 0.0]
        f["peaks/bank"] = [0, 0, 0]
        f["peaks/run_index"] = [0, 0, 0]
        f["sample/a"], f["sample/b"], f["sample/c"] = 10, 10, 10
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90, 90, 90
    
    # This should crash on invalid SG during MTZ writing
    exporter = MTZExporter(indexed_h5, space_group="InvalidSG")
    with pytest.raises(ValueError, match="Could not find space group"):
        exporter.write_mtz("test.mtz")
    
    import os
    if os.path.exists(indexed_h5):
        os.remove(indexed_h5)
    if os.path.exists("test.mtz"):
        os.remove("test.mtz")
