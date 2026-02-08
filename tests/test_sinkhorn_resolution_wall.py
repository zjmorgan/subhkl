import numpy as np
import jax.numpy as jnp
from subhkl.optimization import VectorizedObjective

def test_sinkhorn_resolution_wall():
    # Scenario: Search range 5, but observation is at high resolution (HKL 10).
    hkl_search_range = 5
    
    # Initialize objective with P 1
    B = np.eye(3)
    # Obs vector pointing at (10, 0, 0).
    # In P1 with B=Identity, q_theory = (10, 0, 0).
    # |q_theory| = 10. 
    # lambda = 1.0. |k| = 10. (Dummy)
    kf_ki_dir = np.array([[10.0], [0.0], [0.0]]) 
    wavelength = [0.1, 2.0]
    angle_cdf = np.linspace(0, 1, 100)
    angle_t = np.linspace(0, np.pi, 100)
    
    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=None,
        wavelength=wavelength,
        angle_cdf=angle_cdf,
        angle_t=angle_t,
        hkl_search_range=hkl_search_range,
        space_group="P 1",
        loss_method="sinkhorn"
    )
    
    # Run sinkhorn indexer
    UB = jnp.eye(3)[None]
    kf_ki_sample = jnp.array(kf_ki_dir)[None]
    
    # We need to simulate k_sq_dyn to match the scale
    k_sq_dyn = jnp.sum(kf_ki_sample**2, axis=1) # 100
    
    score, probs, best_hkl, best_lamb = obj.indexer_sinkhorn_jax(UB, kf_ki_sample, k_sq_override=k_sq_dyn, tolerance_rad=0.01)
    
    found_hkl = np.array(best_hkl[0, 0])
    print(f"Observed HKL (ideal): [10, 0, 0]")
    print(f"Sinkhorn search range parameter: {hkl_search_range}")
    print(f"Sinkhorn matched to HKL: {found_hkl}")
    
    # Check if it matched to the high-res HKL
    h_found = found_hkl[0]
    
    print(f"Matched true high-res HKL? {h_found == 10}")
    
    assert h_found == 10, f"Sinkhorn failed to match the true high-res HKL! Found {h_found}"
    
    print("FIX CONFIRMED: Sinkhorn indexer now correctly sizes pool to include observed data.")

if __name__ == "__main__":
    try:
        test_sinkhorn_resolution_wall()
        print("Test PASSED")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
