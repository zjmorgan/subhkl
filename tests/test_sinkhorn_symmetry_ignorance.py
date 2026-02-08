import numpy as np
import jax.numpy as jnp
from subhkl.optimization import VectorizedObjective

def test_sinkhorn_ignores_symmetry():
    # Space group with absences: I 2 2 2 (h+k+l must be even)
    # HKL (1, 0, 0) is forbidden.
    
    hkl_search_range = 5
    
    # Initialize objective
    B = np.eye(3)
    # Obs vector pointing at (1, 0, 0)
    kf_ki_dir = np.array([[1.0], [0.0], [0.0]]) 
    wavelength = [0.5, 2.0]
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
        space_group="I 2 2 2",
        centering="I",
        loss_method="sinkhorn"
    )
    
    # Run sinkhorn indexer simulation
    UB = jnp.eye(3)[None] # Identity orientation
    kf_ki_sample = jnp.array(kf_ki_dir)[None]
    
    # Call indexer_sinkhorn_jax
    score, probs, best_hkl, best_lamb = obj.indexer_sinkhorn_jax(UB, kf_ki_sample, tolerance_rad=0.01)
    
    found_hkl = np.array(best_hkl[0, 0])
    print(f"Observed direction: {kf_ki_dir.flatten()}")
    print(f"Sinkhorn matched to HKL: {found_hkl}")
    
    # Check if matched HKL is forbidden
    h, k, l = found_hkl
    is_forbidden = ((h + k + l) % 2 != 0)
    
    print(f"Is matched HKL forbidden in I 2 2 2? {is_forbidden}")
    
    assert not is_forbidden, "Fix failed: Sinkhorn matched to a forbidden reflection!"
    print("FIX CONFIRMED: Sinkhorn indexer now respects symmetry and rejects forbidden reflections!")

if __name__ == "__main__":
    try:
        test_sinkhorn_ignores_symmetry()
        print("Test PASSED (Bug Found)")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
