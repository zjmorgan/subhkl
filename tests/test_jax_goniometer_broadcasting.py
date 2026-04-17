import pytest
import numpy as np
import jax.numpy as jnp
from subhkl.optimization import VectorizedObjective

def test_jax_1_to_n_broadcasting_bug():
    """
    Directly tests the JAX VectorizedObjective to ensure that 1:n motor mappings
    (e.g., 4 motors driving 5 physical axes) correctly broadcast the refined offsets
    without throwing a shape mismatch error during the matrix accumulation loop.
    """
    # Minimal inputs to satisfy the objective function
    B = np.eye(3)
    kf_ki_dir = np.array([[1.0], [0.0], [0.0]])
    peak_xyz = np.array([[1.0], [0.0], [0.0]])
    wavelength = [1.0, 2.0]
    
    # Define 5 physical rotation axes (e.g., a virtual kappa stage)
    axes = [
        [0, 1, 0, -1],  # Axis 0: phi
        [1, 0, 0, -1],  # Axis 1: alpha tilt
        [0, 1, 0, 1],   # Axis 2: kappa
        [1, 0, 0, 1],   # Axis 3: alpha antitilt
        [0, 1, 0, 1],   # Axis 4: omega
    ]
    
    # CRITICAL TEST FIX: The parser generates 5 angles (one for each axis in the JSON), 
    # not 4 motors. This matches the (5, 220) shape that triggered the crash in production.
    angles = np.array([
        [45.0, 45.0, 45.0],  # Axis 0 (phi)
        [24.0, 24.0, 24.0],  # Axis 1 (alpha tilt)
        [30.0, 30.0, 30.0],  # Axis 2 (kappa)
        [24.0, 24.0, 24.0],  # Axis 3 (alpha antitilt)
        [10.0, 10.0, 10.0],  # Axis 4 (omega)
    ])
    
    # Map the 5 axes to the 4 unique motors
    motor_map = [0, 1, 2, 1, 3]
    
    # Initialize the objective
    obj = VectorizedObjective(
        B=B,
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=peak_xyz,
        wavelength=wavelength,
        goniometer_axes=axes,
        goniometer_angles=angles,
        refine_goniometer=True,
        goniometer_refine_mask=np.ones(4, dtype=bool),
        goniometer_bound_deg=5.0,
        motor_map=motor_map, 
    )
    
    # Create a dummy population of 2 members
    # Parameters = 3 (orientation) + 4 (refined motors) = 7
    x = jnp.full((2, 7), 0.5)
    x = x.at[0, 4].set(0.6) 

    try:
        # Without the fix, this line crashes with:
        # TypeError: add got incompatible shapes for broadcasting: (2, 4, 1), (1, 5, 3)
        results = obj._get_physical_params_jax(x)
        
        offsets_total = results[4]
        R = results[5]
        
    except Exception as e:
        pytest.fail(f"Broadcasting bug triggered! JAX raised: {e}")
        
    # 1. The optimizer should have allocated exactly 4 offset parameters
    assert offsets_total.shape == (2, 4), "Optimizer failed to collapse to 4 refinement parameters!"
    
    # 2. The resulting Rotation matrix stack should cover both population members, and all 3 peaks
    assert R.shape == (2, 3, 3, 3), f"Expected Rotation shape (2, 3, 3, 3), got {R.shape}"
