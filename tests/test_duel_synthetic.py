import numpy as np
import h5py
import os
import pytest
from subhkl.io.parser import index, metrics
from scipy.spatial.transform import Rotation

def create_synthetic_finder(filename):
    # Unit cell a=10 cubic
    a, b, c = 10.0, 10.0, 10.0
    alpha, beta, gamma = 90.0, 90.0, 90.0
    
    # Orientation U: 5 deg around X
    U = Rotation.from_euler('x', 5, degrees=True).as_matrix()
    
    # Goniometer R: 1 run, identity
    R0 = np.eye(3)
    
    # B matrix (1/a)
    B = np.diag([0.1, 0.1, 0.1])
    
    # Generate HKLs
    hkls = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-3, 4):
                if h==0 and k==0 and l==0: continue
                hkls.append([h, k, l])
    hkls = np.array(hkls)
    
    xyz = []
    run_idx = []
    ki = np.array([0, 0, 1])
    
    for hkl in hkls:
        q_lab = R0 @ U @ B @ hkl
        q_sq = np.sum(q_lab**2)
        # lambda = -2 (q . ki) / q^2
        lamb_val = -2 * np.dot(q_lab, ki) / q_sq
        
        if 1.5 < lamb_val < 6.0:
            kf = lamb_val * q_lab + ki
            p_lab = 0.4 * kf # 40cm detector
            xyz.append(p_lab)
            run_idx.append(0)
                
    xyz = np.array(xyz)
    run_idx = np.array(run_idx)
    
    with h5py.File(filename, 'w') as f:
        f["peaks/xyz"] = xyz
        f["peaks/run_index"] = run_idx
        f["peaks/two_theta"] = np.rad2deg(np.arccos(xyz[:, 2] / np.linalg.norm(xyz, axis=1)))
        f["peaks/azimuthal"] = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
        f["peaks/intensity"] = np.ones(len(xyz))
        f["peaks/sigma"] = np.ones(len(xyz)) * 0.1
        f["peaks/radius"] = np.zeros(len(xyz))
        f["goniometer/R"] = R0[None, ...]
        f["goniometer/axes"] = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]]) 
        f["goniometer/angles"] = np.zeros((1, 3)) 
        f["goniometer/names"] = [b"omega", b"chi", b"phi"]
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [1.0, 8.0]
        f.attrs["instrument"] = "MANDI"

def test_synthetic_indexing(tmp_path):
    finder_h5 = os.path.join(tmp_path, "synthetic_finder.h5")
    indexer_h5 = os.path.join(tmp_path, "synthetic_indexer.h5")
    
    create_synthetic_finder(finder_h5)
    
    # Load input data
    input_data = {}
    with h5py.File(finder_h5, 'r') as f:
        keys_to_load = [
            "peaks/two_theta", "peaks/azimuthal", "peaks/intensity", "peaks/sigma", 
            "peaks/radius", "peaks/xyz",
            "goniometer/R", "goniometer/axes", "goniometer/angles", "goniometer/names",
            "peaks/run_index", "instrument/wavelength"
        ]
        for k in keys_to_load:
            if k in f: input_data[k] = f[k][()]
        input_data["sample/a"] = f["sample/a"][()]
        input_data["sample/b"] = f["sample/b"][()]
        input_data["sample/c"] = f["sample/c"][()]
        input_data["sample/alpha"] = f["sample/alpha"][()]
        input_data["sample/beta"] = f["sample/beta"][()]
        input_data["sample/gamma"] = f["sample/gamma"][()]
        input_data["sample/space_group"] = f["sample/space_group"][()]
        input_data['instrument'] = f.attrs['instrument']

    # Run indexer
    # Use small popsize for fast test
    index(
        input_data=input_data,
        output_peaks_filename=indexer_h5,
        n_runs=1,
        population_size=200,
        gens=50,
        tolerance_deg=0.5,
        loss_method="gaussian",
        instrument_name="MANDI"
    )
    
    # Run metrics
    # Capture output to check for median error
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        metrics(indexer_h5, instrument="MANDI")
    
    output = f.getvalue()
    print(output)
    
    # Parse METRICS line: METRICS: d_med d_mean d_max ang_med ang_mean ang_max
    for line in output.splitlines():
        if line.startswith("METRICS:"):
            parts = line.split()
            ang_med = float(parts[4])
            assert ang_med < 1.0 # Should be well within 1 degree
            break
