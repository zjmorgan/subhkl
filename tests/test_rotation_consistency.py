import numpy as np
import jax.numpy as jnp
from subhkl.config.goniometer import calc_goniometer_rotation_matrix
from subhkl.optimization import VectorizedObjective

def test_rotation_consistency():
    # MANDI axes from reduction_settings.json
    axes = [
        [0, 1, 0, 1], # omega
        [0, 0, 1, 1], # chi
        [0, 1, 0, 1]  # phi
    ]
    
    # Test with large angles
    angles = [45.0, 30.0, 90.0]
    
    # 1. Calc using CPU version
    R_cpu = calc_goniometer_rotation_matrix(axes, angles)
    
    # 2. Calc using JAX version via VectorizedObjective
    # We need to mock a VectorizedObjective enough to call compute_goniometer_R_jax
    class MockObjective(VectorizedObjective):
        def __init__(self, axes, angles):
            self.gonio_axes = jnp.array(axes)
            self.gonio_angles = jnp.array(angles)[:, None] # 1 run
            self.num_gonio_axes = len(axes)
            self.goniometer_bound_deg = 5.0
            self.gonio_nominal_offsets = jnp.zeros(len(axes))
            
    obj = MockObjective(axes, angles)
    
    # compute_goniometer_R_jax expects norm offsets (0.5 = no offset)
    gonio_norm = jnp.full((1, len(axes)), 0.5)
    R_jax = obj.compute_goniometer_R_jax(gonio_norm)
    
    R_jax_mat = np.array(R_jax[0, 0])
    
    print("R_cpu:")
    print(R_cpu)
    print("R_jax:")
    print(R_jax_mat)
    
    assert np.allclose(R_cpu, R_jax_mat), "CPU and JAX rotations differ!"

if __name__ == "__main__":
    try:
        test_rotation_consistency()
        print("Test PASSED")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
