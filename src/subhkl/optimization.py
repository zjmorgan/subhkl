import os
from functools import partial

import h5py
import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.interpolate

import gemmi

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jscipy_linalg

from evosax.algorithms import DifferentialEvolution, PSO, CMA_ES

from subhkl.detector import scattering_vector_from_angles

# Try to import tqdm for a progress bar
try:
    from tqdm import trange
except ImportError:
    trange = None


def get_lattice_system(a, b, c, alpha, beta, gamma, centering, atol_len=0.05, atol_ang=0.5):
    """
    Detect crystal system constraints based on initial unit cell parameters and centering.
    
    Returns
    -------
    system_name : str
        'Cubic', 'Hexagonal', 'Tetragonal', 'Rhombohedral', 'Orthorhombic', 'Monoclinic', 'Triclinic'
    num_params : int
        Number of free lattice parameters.
    """
    # Check angles
    is_90 = lambda x: np.isclose(x, 90.0, atol=atol_ang)
    is_120 = lambda x: np.isclose(x, 120.0, atol=atol_ang)
    
    # Check lengths
    eq = lambda x, y: np.isclose(x, y, atol=atol_len)
    
    # Explicitly handle Centering-based hints
    if centering == 'R':
        # Rhombohedral lattice
        # Case 1: Hexagonal settings (a=b, alpha=beta=90, gamma=120)
        if is_90(alpha) and is_90(beta) and is_120(gamma) and eq(a, b):
            return 'Hexagonal', 2 # Treated as hexagonal parameters a, c
        # Case 2: Primitive Rhombohedral settings (a=b=c, alpha=beta=gamma)
        elif eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma):
            return 'Rhombohedral', 2 # a, alpha
            
    if centering == 'H':
        return 'Hexagonal', 2

    # Geometric detection for P, I, F, C, A, B
    if is_90(alpha) and is_90(beta) and is_90(gamma):
        if eq(a, b) and eq(b, c):
            return 'Cubic', 1  # a
        elif eq(a, b):
            return 'Tetragonal', 2  # a, c
        elif eq(a, c) or eq(b, c):
            # Non-standard tetragonal setting or just Orthorhombic
            return 'Orthorhombic', 3 # a, b, c
        else:
            return 'Orthorhombic', 3 # a, b, c

    elif is_90(alpha) and is_90(beta) and is_120(gamma):
        if eq(a, b):
            return 'Hexagonal', 2 # a, c
        
    elif eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma):
        return 'Rhombohedral', 2 # a, alpha

    # Monoclinic Check (standard setting: b unique axis -> alpha=gamma=90)
    if is_90(alpha) and is_90(gamma) and not is_90(beta):
        return 'Monoclinic', 4 # a, b, c, beta
    
    # Fallback to Triclinic
    return 'Triclinic', 6


def rotation_matrix_from_axis_angle_jax(axis, angle_rad):
    """
    Compute rotation matrix from axis and angle using JAX.
    
    Parameters
    ----------
    axis : array (3,)
    angle_rad : array (S, M) or scalar
    
    Returns
    -------
    R : array (S, M, 3, 3)
    """
    # Normalize axis
    u = axis / jnp.linalg.norm(axis)
    ux, uy, uz = u

    # Cross product matrix K
    K = jnp.array([
        [0.0, -uz, uy],
        [uz, 0.0, -ux],
        [-uy, ux, 0.0]
    ]) # (3, 3)

    # I + sin(theta) K + (1-cos(theta)) K^2
    # Handle broadcasting for angle_rad which might be (S, M)
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    t = 1.0 - c

    # Broadcast shapes
    # K: (1, 1, 3, 3)
    # s, t: (S, M, 1, 1)
    
    eye = jnp.eye(3)
    
    R = eye + s[..., None, None] * K + t[..., None, None] * (K @ K)
    return R


class VectorizedObjectiveJAX:
    """
    JAX-compatible vectorized objective function for evosax.
    """
    def __init__(self, B, centering, kf_ki_dir, wavelength, angle_cdf, angle_t, weights=None, softness=0.15,
                 cell_params=None, refine_lattice=False, lattice_bound_frac=0.05, lattice_system='Triclinic',
                 goniometer_axes=None, goniometer_angles=None, refine_goniometer=False, goniometer_bound_deg=5.0,
                 peak_radii=None):
        """
        Parameters
        ----------
        B : array (3, 3)
            B matrix (from reciprocal_lattice_B). Used if refine_lattice=False.
        centering : str
            Bravais lattice centering
        kf_ki_dir : array (3, M)
            difference between incident and scattering directions for M
            reflections. 
            If refine_goniometer=True, this must be in the LAB frame.
            If refine_goniometer=False, this is in the CRYSTAL frame (pre-rotated by fixed R).
        wavelength : array (2,)
            wavelength lower and upper bounds [min, max]
        angle_cdf : array
            CDF values for angle interpolation (from FindUB._angle_cdf)
        angle_t : array
            Angle values for interpolation (from FindUB._angle_t)
        softness : float
            Shape parameter for rounding hkls
        cell_params : array (6,)
            Initial unit cell parameters [a, b, c, alpha, beta, gamma].
            Required if refine_lattice=True.
        refine_lattice : bool
            If True, refine unit cell parameters.
        lattice_bound_frac : float
            Fractional bound for lattice parameter refinement (e.g. 0.05 = +/- 5%).
        lattice_system : str
            Detected crystal system for parameter constraints.
        goniometer_axes : list or array
            List of goniometer axes specifications (as in config).
        goniometer_angles : array (N_axes, M)
            Base angles for each axis for each peak.
        refine_goniometer : bool
            If True, refine goniometer angle offsets.
        goniometer_bound_deg : float
            Bound for goniometer angle optimization in degrees.
        peak_radii : array (M,) or None
            Angular radius (radians) of each peak for weighting.
        """
        self.B = jnp.array(B)
        self.kf_ki_dir = jnp.array(kf_ki_dir)

        # Pre-calculate k_sq from the invariant input vector (q_lab equivalent)
        self.k_sq_invariant = jnp.sum(self.kf_ki_dir**2, axis=0) # Shape (M,)

        self.softness = softness
        self.centering = centering
        self.angle_cdf = jnp.array(angle_cdf)
        self.angle_t = jnp.array(angle_t)
        
        self.refine_lattice = refine_lattice
        self.lattice_system = lattice_system
        
        self.refine_goniometer = refine_goniometer
        self.goniometer_bound_deg = goniometer_bound_deg

        # --- Lattice Refinement Setup ---
        if self.refine_lattice:
            if cell_params is None:
                raise ValueError("cell_params must be provided if refine_lattice is True")
            self.cell_init = jnp.array(cell_params) # Full 6 params [a, b, c, alpha, beta, gamma]
            
            # Determine free parameters based on system
            if self.lattice_system == 'Cubic': # (a)
                self.free_params_init = self.cell_init[0:1]
            elif self.lattice_system == 'Hexagonal': # (a, c)
                self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[2]])
            elif self.lattice_system == 'Tetragonal': # (a, c)
                self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[2]])
            elif self.lattice_system == 'Rhombohedral': # (a, alpha)
                self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[3]])
            elif self.lattice_system == 'Orthorhombic': # (a, b, c)
                self.free_params_init = self.cell_init[0:3]
            elif self.lattice_system == 'Monoclinic': # (a, b, c, beta)
                self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[1], self.cell_init[2], self.cell_init[4]])
            else: # Triclinic (a, b, c, alpha, beta, gamma)
                self.free_params_init = self.cell_init

            # Define bounds for free params
            delta = jnp.abs(self.free_params_init) * lattice_bound_frac
            self.lat_min = self.free_params_init - delta
            self.lat_max = self.free_params_init + delta

        # --- Goniometer Refinement Setup ---
        if self.refine_goniometer:
            if goniometer_axes is None or goniometer_angles is None:
                raise ValueError("goniometer_axes and goniometer_angles must be provided for refinement")
            
            self.gonio_axes = jnp.array(goniometer_axes) # (N_axes, 4)
            self.gonio_angles = jnp.array(goniometer_angles) # (N_axes, M)
            
            # We optimize one offset per axis (assuming offset is constant for the dataset)
            # Or should it be per peak? Usually goniometer zero-point error is constant.
            self.num_gonio_axes = self.gonio_axes.shape[0]
            
            # Bounds for offsets (centered at 0)
            self.gonio_min = jnp.full(self.num_gonio_axes, -goniometer_bound_deg)
            self.gonio_max = jnp.full(self.num_gonio_axes, goniometer_bound_deg)


        # Ensure wavelength is a JAX array
        wavelength = jnp.array(wavelength)
        
        # Wavelength bounds for analytical search
        self.wl_min_val = wavelength[0]
        self.wl_max_val = wavelength[1]

        # Scan parameters for indexer_dynamic_jax
        self.num_candidates = 64

        # Handle weights: if None, default to 1.0 for everyone
        if weights is None:
            self.weights = jnp.ones(self.kf_ki_dir.shape[1])
        else:
            self.weights = jnp.array(weights)
            
        # Handle peak radii (for variable softness)
        if peak_radii is None:
            self.peak_radii = jnp.zeros(self.kf_ki_dir.shape[1])
        else:
            self.peak_radii = jnp.array(peak_radii)

        # Pre-calculate the maximum possible score (sum of all weights)
        self.max_score = jnp.sum(self.weights)

    def reconstruct_cell_params(self, params_norm):
        """
        Expand normalized free parameters back to full 6 unit cell parameters.
        Returns full parameters in real units (Angstroms/Degrees).
        """
        # Map [0, 1] to physical range for free params
        p_free = self.lat_min + params_norm * (self.lat_max - self.lat_min)
        
        # Helper for creating full array (Batch size S)
        S = params_norm.shape[0]
        
        # Default angles
        deg90 = jnp.full((S,), 90.0)
        deg120 = jnp.full((S,), 120.0)
        
        if self.lattice_system == 'Cubic':
            # p_free: [a]
            a = p_free[:, 0]
            return jnp.stack([a, a, a, deg90, deg90, deg90], axis=1)
            
        elif self.lattice_system == 'Hexagonal':
            # p_free: [a, c]
            a = p_free[:, 0]
            c = p_free[:, 1]
            return jnp.stack([a, a, c, deg90, deg90, deg120], axis=1)
            
        elif self.lattice_system == 'Tetragonal':
            # p_free: [a, c]
            a = p_free[:, 0]
            c = p_free[:, 1]
            return jnp.stack([a, a, c, deg90, deg90, deg90], axis=1)
            
        elif self.lattice_system == 'Rhombohedral':
            # p_free: [a, alpha]
            a = p_free[:, 0]
            alpha = p_free[:, 1]
            return jnp.stack([a, a, a, alpha, alpha, alpha], axis=1)
            
        elif self.lattice_system == 'Orthorhombic':
            # p_free: [a, b, c]
            a, b, c = p_free[:, 0], p_free[:, 1], p_free[:, 2]
            return jnp.stack([a, b, c, deg90, deg90, deg90], axis=1)
            
        elif self.lattice_system == 'Monoclinic':
            # p_free: [a, b, c, beta]
            a, b, c, beta = p_free[:, 0], p_free[:, 1], p_free[:, 2], p_free[:, 3]
            return jnp.stack([a, b, c, deg90, beta, deg90], axis=1)
            
        else: # Triclinic
            return p_free

    def compute_B_jax(self, cell_params_norm):
        """
        Compute B matrices from normalized free lattice parameters.
        """
        # Reconstruct full 6 params
        p = self.reconstruct_cell_params(cell_params_norm)
        
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        # Angles provided in degrees, convert to radians
        deg2rad = jnp.pi / 180.0
        alpha = p[:, 3] * deg2rad
        beta  = p[:, 4] * deg2rad
        gamma = p[:, 5] * deg2rad

        # Construct Metric Tensor G
        g11 = a**2
        g22 = b**2
        g33 = c**2
        g12 = a * b * jnp.cos(gamma)
        g13 = a * c * jnp.cos(beta)
        g23 = b * c * jnp.cos(alpha)

        # G shape: (S, 3, 3)
        row1 = jnp.stack([g11, g12, g13], axis=-1)
        row2 = jnp.stack([g12, g22, g23], axis=-1)
        row3 = jnp.stack([g13, g23, g33], axis=-1)
        G = jnp.stack([row1, row2, row3], axis=-2)

        # Reciprocal Metric Tensor G* = inv(G)
        G_star = jnp.linalg.inv(G)

        # B is Cholesky of G* (Upper triangular, B^T B = G*)
        B = jscipy_linalg.cholesky(G_star, lower=False)
        
        return B
    
    def compute_goniometer_R_jax(self, gonio_offsets_norm):
        """
        Compute goniometer rotation matrices for each peak and particle.
        
        Parameters
        ----------
        gonio_offsets_norm : array (S, N_axes)
            Normalized offsets [0, 1]
            
        Returns
        -------
        R : array (S, M, 3, 3)
            Goniometer rotation matrices.
        """
        # Map normalized parameters to physical offsets
        offsets = self.gonio_min + gonio_offsets_norm * (self.gonio_max - self.gonio_min) # (S, N_axes)
        
        # Base angles: (N_axes, M)
        # Add offsets: (S, N_axes, 1) + (N_axes, M) -> (S, N_axes, M)
        angles_deg = offsets[:, :, None] + self.gonio_angles[None, :, :]
        
        S = offsets.shape[0]
        M = self.gonio_angles.shape[1]
        
        # Initialize R as Identity (S, M, 3, 3)
        R = jnp.eye(3)[None, None, ...].repeat(S, axis=0).repeat(M, axis=1)
        
        # Compose rotations: R = R_0 @ R_1 @ ... @ R_k (or in reverse depending on stack)
        # Mantid SetGoniometer applies matrices in order pushed: R = R0 * R1 * ...
        # My goniometer.py: matrix = matrix @ axis_matrix
        
        deg2rad = jnp.pi / 180.0
        
        for i in range(self.num_gonio_axes):
            axis_spec = self.gonio_axes[i]
            direction = axis_spec[:3]
            sign = axis_spec[3]
            
            # (S, M)
            theta = sign * angles_deg[:, i, :] * deg2rad
            
            # Compute R_i for this axis
            # (S, M, 3, 3)
            Ri = rotation_matrix_from_axis_angle_jax(direction, theta)
            
            # R_new = R_old @ Ri
            R = jnp.einsum('smij,smjk->smik', R, Ri)
            
        return R

    def orientation_U_jax(self, param):
        """
        Compute orientation matrices (U) from angles using JAX.
        
        Parameters
        ----------
        param : array, (S, N)
            Optimization parameters. First 3 are rotation params.
        """
        # Unpack explicitly by column index to handle S > 3 (lattice refinement)
        u0 = param[:, 0]
        u1 = param[:, 1]
        u2 = param[:, 2]

        theta = jnp.arccos(1 - 2 * u0)
        phi = 2 * jnp.pi * u1

        # Rotation axis (w)
        w = jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        ).T # (S, 3)

        # Rotation angle (omega)
        omega = jnp.interp(u2, self.angle_cdf, self.angle_t) # (S,)

        # JAX implementation of axis-angle to rotation matrix
        wx, wy, wz = w.T # (S,) each
        c = jnp.cos(omega) # (S,)
        s = jnp.sin(omega) # (S,)
        t = 1.0 - c # (S,)

        # Identity matrices
        I = jnp.eye(3)[None, :, :].repeat(param.shape[0], axis=0) # (S, 3, 3)

        # Skew-symmetric cross-product matrix K
        K = jnp.array(
            [
                [jnp.zeros_like(wx), -wz, wy],
                [wz, jnp.zeros_like(wy), -wx],
                [-wy, wx, jnp.zeros_like(wz)]
            ]
        ) # (3, 3, S)
        K = jnp.transpose(K, (2, 0, 1)) # (S, 3, 3)

        # K^2
        K2 = jnp.einsum('sij,sjk->sik', K, K) # (S, 3, 3)

        U = I + s[:, None, None] * K + t[:, None, None] * K2 # (S, 3, 3)

        return U

    def indexer_dynamic_jax(self, UB, kf_ki_sample, softness=0.001):
        """
        Indexing with Dynamic Miller Sampling using lax.scan.
        
        Iterates over integer candidates sequentially to avoid exploding memory usage.

        UB: (S, 3, 3)
        kf_ki_sample: (S, 3, M) - Scattering vectors in Sample Frame
        """
        UB_inv = jnp.linalg.inv(UB) # (S, 3, 3)

        # 1. Project ray into HKL space
        # v = UB_inv @ k_sample. 
        # v shape: (S, 3, M)
        v = jnp.einsum("sij,sjm->sim", UB_inv, kf_ki_sample)

        # 2. Determine Dominant Axis & Integer Start
        abs_v = jnp.abs(v)
        max_v_val = jnp.max(abs_v, axis=1) # (S, M)

        # We start searching from the integer corresponding to max wavelength
        n_start = max_v_val / self.wl_max_val
        start_int = jnp.ceil(n_start) # (S, M)

        k_sq = self.k_sq_invariant[None, :]

        # --- Scan Body Function ---
        # Carry state: (min_dist_sq, best_lambda, best_hkl)
        # All shapes (S, M) or (S, 3, M)

        initial_carry = (
            jnp.full(max_v_val.shape, jnp.inf), # min_dist_sq
            jnp.zeros(max_v_val.shape),         # best_lambda
            jnp.zeros((v.shape[0], 3, v.shape[2]), dtype=jnp.int32) # best_hkl
        )

        def scan_body(carry, i):
            curr_min_dist, curr_best_lamb, curr_best_hkl = carry

            # Current Integer Candidate: n = start_int + i
            n = start_int + i

            # Avoid division by zero
            n_safe = jnp.where(n == 0, 1e-9, n)

            # Initial Lambda Guess
            lamda_cand = max_v_val / n_safe

            # Validity Mask (range check)
            valid_cand = (lamda_cand >= self.wl_min_val) & (lamda_cand <= self.wl_max_val)

            # Map to HKL (using candidate lambda)
            # hkl = v / lambda
            hkl_float = v / lamda_cand[:, None, :] # (S, 3, M)
            hkl_int = jnp.round(hkl_float).astype(jnp.int32)

            # Predict Q back in Sample Frame: Q = UB @ hkl
            q_int = jnp.einsum("sij,sjm->sim", UB, hkl_int)

            # Analytic Refinement of Lambda
            # k . Q
            k_dot_q = jnp.sum(kf_ki_sample * q_int, axis=1) # (S, M)

            safe_dot = jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
            lambda_opt = k_sq / safe_dot 

            # Clamp to range
            lambda_clamped = jnp.clip(lambda_opt, self.wl_min_val, self.wl_max_val)

            # Calculate Residuals in Q-space
            # Q_obs = k / lambda
            q_obs = kf_ki_sample / lambda_clamped[:, None, :]
            diff_vec = q_obs - q_int
            dist_sq = jnp.sum(diff_vec**2, axis=1) # (S, M)

            # Apply Validity Masks
            # 1. Lambda Range (dist -> inf if invalid)
            dist_sq = jnp.where(valid_cand, dist_sq, jnp.inf)

            # 2. Symmetry / Centering
            h = hkl_int[:, 0, :]
            k = hkl_int[:, 1, :]
            l = hkl_int[:, 2, :]

            valid_sym = jnp.full_like(h, True, dtype=bool)
            if self.centering == "A": valid_sym = (k + l) % 2 == 0
            elif self.centering == "B": valid_sym = (h + l) % 2 == 0
            elif self.centering == "C": valid_sym = (h + k) % 2 == 0
            elif self.centering == "I": valid_sym = (h + k + l) % 2 == 0
            elif self.centering == "F": valid_sym = ((h + k) % 2 == 0) & ((l + h) % 2 == 0) & ((k + l) % 2 == 0)
            elif self.centering == "R": valid_sym = (h + k + l) % 3 == 0
            elif self.centering == "R_obv": valid_sym = (-h + k + l) % 3 == 0
            elif self.centering == "R_rev": valid_sym = (h - k + l) % 3 == 0

            dist_sq = jnp.where(valid_sym, dist_sq, jnp.inf)

            # Update Best
            update_mask = dist_sq < curr_min_dist

            new_min_dist = jnp.where(update_mask, dist_sq, curr_min_dist)
            new_best_lamb = jnp.where(update_mask, lambda_clamped, curr_best_lamb)

            # Update HKL (broadcast mask to 3 components)
            new_best_hkl = jnp.where(update_mask[:, None, :], hkl_int, curr_best_hkl)

            return (new_min_dist, new_best_lamb, new_best_hkl), None

        # Execute Scan
        scan_indices = jnp.arange(self.num_candidates)
        final_carry, _ = jax.lax.scan(scan_body, initial_carry, scan_indices)

        min_dist_sq, min_lamb, int_hkl = final_carry

        # --- Scoring ---
        # Variable Sigma (Radius in Q space)
        k_norm = jnp.sqrt(k_sq)

        # Avoid division by zero if min_lamb is 0 (unindexed peaks)
        safe_min_lamb = jnp.where(min_lamb == 0, 1.0, min_lamb)
        radius_q = (k_norm / safe_min_lamb) * self.peak_radii[None, :]
        effective_sigma = softness + radius_q

        peak_probs = self.weights[None, :] * jnp.exp(-min_dist_sq / (2 * effective_sigma**2))
        total_score = jnp.sum(peak_probs, axis=1)

        # HKL comes out as (S, 3, M). Transpose to (S, M, 3) for compatibility
        return self.max_score - total_score, total_score, int_hkl.transpose((0, 2, 1)), min_lamb

    # Use partial to make 'self' a static argument for JIT
    @partial(jax.jit, static_argnames='self')
    def __call__(self, x):
        """
        JIT-compiled objective function.

        Parameters
        ----------
        x : array (S, num_dims)
            Refineable parameters. 
            [rot(3), lat(N_lat)?, gonio(N_gonio)?]

        Returns
        -------
        error : array (S,)
            Indexing error for each particle.
        """
        # Pointer to current position in x
        idx = 0
        
        # 1. Orientation (3 params)
        rot_params = x[:, idx:idx+3]
        U = self.orientation_U_jax(rot_params) # (S, 3, 3)
        idx += 3

        # 2. Lattice (optional)
        if self.refine_lattice:
            n_lat = self.free_params_init.size
            cell_params_norm = x[:, idx:idx+n_lat]
            B = self.compute_B_jax(cell_params_norm) # (S, 3, 3)
            idx += n_lat
            # UB = U @ B (per particle)
            UB = jnp.einsum("sij,sjk->sik", U, B)
        else:
            # Fixed B
            UB = jnp.einsum("sij,jk->sik", U, self.B)

        # 3. Goniometer (optional)
        if self.refine_goniometer:
            n_gon = self.num_gonio_axes
            gonio_params_norm = x[:, idx:idx+n_gon]
            idx += n_gon
            
            # Compute R (S, M, 3, 3)
            R = self.compute_goniometer_R_jax(gonio_params_norm)
            
            # Transform Lab Vectors to Crystal Frame
            # q_lab is (3, M) [self.kf_ki_dir]
            # q_cryst = R^T @ q_lab
            # einsum: s m j i (R), k m (q_lab) -> s i m
            # We want sum over j. R_ji * q_j
            # R is (S, M, 3, 3). q_lab is (3, M).
            # We want (S, 3, M).
            kf_ki_vec = jnp.einsum("smji,jm->sim", R, self.kf_ki_dir)
            
        else:
            # Fixed R already applied in self.kf_ki_dir (if passed appropriately)
            # Expand (3, M) to (S, 3, M)
            kf_ki_vec = self.kf_ki_dir[None, ...].repeat(x.shape[0], axis=0)

        # Use the dynamic indexer instead of the soft one
        error, _, _, _ = self.indexer_dynamic_jax(UB, kf_ki_vec, softness=self.softness)

        return error


class FindUB:
    """
    Optimizer of crystal orientation from peaks and known lattice parameters.
    """

    def __init__(self, filename=None):
        """
        Find :math:`UB` from peaks.

        Parameters
        ----------
        filename : str, optional
            Filename of found peaks. The default is None.

        """
        self.goniometer_axes = None
        self.goniometer_angles = None
        self.goniometer_offsets = None # To store refined offsets
        self.goniometer_names = None # To store axis names

        if filename is not None:
            self.load_peaks(filename)

        t = np.linspace(0, np.pi, 1024)
        cdf = (t - np.sin(t)) / np.pi

        # Store data for JAX
        self._angle_cdf = cdf
        self._angle_t = t

        # Keep scipy interpolator for non-JAX methods
        self._angle = scipy.interpolate.interp1d(cdf, t, kind="linear")

    def load_peaks(self, filename):
        """
        Obtain peak information from .h5 file.

        Parameters
        ----------
        filename : str
            HDF5 file of peak information.

        """

        with h5py.File(os.path.abspath(filename), "r") as f:
            self.a = f["sample/a"][()]
            self.b = f["sample/b"][()]
            self.c = f["sample/c"][()]
            self.alpha = f["sample/alpha"][()]
            self.beta = f["sample/beta"][()]
            self.gamma = f["sample/gamma"][()]
            self.wavelength = f["instrument/wavelength"][()]
            self.R = f["goniometer/R"][()]
            self.two_theta = f["peaks/two_theta"][()]
            self.az_phi = f["peaks/azimuthal"][()]
            self.intensity = f["peaks/intensity"][()]
            self.sigma_intensity = f["peaks/sigma"][()]
            self.centering = f["sample/centering"][()].decode("utf-8")
            
            # Load goniometer raw data if available
            if "goniometer/axes" in f:
                 self.goniometer_axes = f["goniometer/axes"][()]
            if "goniometer/angles" in f:
                 self.goniometer_angles = f["goniometer/angles"][()]
            if "goniometer/names" in f:
                 self.goniometer_names = [n.decode('utf-8') for n in f["goniometer/names"][()]]

    def get_consistent_U_for_symmetry(self, U_mat, B_mat):
        """
        Return the proper rotations for this spacegroup and pick
        a consistent U matrix among all symmetry-related possibilities.
        """

        uc = gemmi.UnitCell(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        )

        # extract the proper rotations from the point group
        gops = gemmi.find_lattice_symmetry(uc, self.centering, max_obliq=3.0)

        # gemmi stores rotation matrices as integers scaled by 24 to handle
        # denominators like 2, 3, 4, 6 exactly.
        transforms = [ np.array(g.rot) // 24 for g in gops.sym_ops ]

        # Filter for determinant approx +1
        transforms = [M for M in transforms if np.isclose(np.linalg.det(M), 1.0)]

        # select a rotation that maximes the trace of UB
        cost, T = -np.inf, np.eye(3)
        for M in transforms:
            UBp = U_mat @ B_mat @ np.linalg.inv(M) @ np.linalg.inv(B_mat)
            trace = np.trace(UBp)
            if trace > cost:
                cost = trace
                T = M.copy()

        # the new U matrix
        U_prime = U_mat @ B_mat @ np.linalg.inv(T) @ np.linalg.inv(B_mat)

        return U_prime, T

    def uncertainty_line_segements(self):
        """
        The scattering vector scaled with the (unknown) wavelength.

        Returns
        -------
        kf_ki_dir : list
            Difference between scattering and incident beam directions.

        """

        kf_ki_dir = scattering_vector_from_angles(self.two_theta, self.az_phi)

        # self.R.shape == (M, 3, 3)
        return np.einsum("mji,jm->im", self.R, kf_ki_dir)
        # (3, M)

    def metric_G_tensor(self):
        """
        Calculate the metric tensor :math:`G`.
        """
        alpha = np.deg2rad(self.alpha)
        beta = np.deg2rad(self.beta)
        gamma = np.deg2rad(self.gamma)

        g11 = self.a**2
        g22 = self.b**2
        g33 = self.c**2
        g12 = self.a * self.b * np.cos(gamma)
        g13 = self.c * self.a * np.cos(beta)
        g23 = self.b * self.c * np.cos(alpha)

        G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])

        return G

    def metric_G_star_tensor(self):
        """
        Calculate the reciprocal metric tensor :math:`G^*`.
        """
        return np.linalg.inv(self.metric_G_tensor())

    def reciprocal_lattice_B(self):
        """
        The reciprocal lattice :math:`B`-matrix.
        """
        Gstar = self.metric_G_star_tensor()
        return scipy.linalg.cholesky(Gstar, lower=False)

    def minimize_evosax(
        self,
        strategy_name: str, 
        population_size: int = 1000, 
        num_generations: int = 100, 
        n_runs: int = 1, 
        seed: int = 0,
        softness: float = 1e-3,
        init_params: np.ndarray = None,
        refine_lattice: bool = False,
        lattice_bound_frac: float = 0.05,
        goniometer_axes: list = None,
        goniometer_angles: np.ndarray = None,
        refine_goniometer: bool = False,
        goniometer_bound_deg: float = 5.0,
        goniometer_names: list = None
    ):
        """
        Minimize the objective function using evosax JAX-based algorithms.
        """

        # Auto-detect goniometer info from loaded file if not provided
        if goniometer_axes is None and self.goniometer_axes is not None:
             goniometer_axes = self.goniometer_axes
        
        # Need to handle angles carefully. 
        # self.goniometer_angles should be (M, N) or (N, M) depending on how saved.
        # parser/integration saves it as list of arrays/lists per peak. 
        # integration: gonio_angles_out.extend([raw] * num_peaks). 
        # So shape is (Total_Peaks, N_axes).
        # Optimization expects (N_axes, M) for easy broadcasting? Or (M, N_axes)?
        # Let's check `compute_goniometer_R_jax`.
        # It expects `gonio_angles` as (N_axes, M).
        # So if loaded from file (M, N), we transpose.
        
        if goniometer_angles is None and self.goniometer_angles is not None:
             # Transpose to (N_axes, M)
             goniometer_angles = self.goniometer_angles.T
             
        # Also auto-detect names
        if goniometer_names is None and self.goniometer_names is not None:
             goniometer_names = self.goniometer_names

        # 1. Prepare vectors
        # If refining goniometer, we need UNROTATED (Lab) vectors.
        # If NOT refining, we use the ROTATED (Crystal) vectors stored in self.R
        
        kf_ki_dir_lab = scattering_vector_from_angles(self.two_theta, self.az_phi)
        
        if refine_goniometer:
            if goniometer_axes is None or goniometer_angles is None:
                raise ValueError("If refine_goniometer=True, axes and angles must be provided.")
            # Input to objective is Lab frame
            kf_ki_input = kf_ki_dir_lab
        else:
            # Input to objective is Crystal frame (using fixed R)
            kf_ki_input = np.einsum("mji,jm->im", self.R, kf_ki_dir_lab)

        # 2. Use Signal-to-Noise (I / sigma)
        weights = self.intensity / (self.sigma_intensity + 1e-6)
        weights = weights / np.mean(weights)
        weights = np.clip(weights, 0, 10.0)

        # Prepare cell parameters
        cell_params_init = np.array([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])

        # Detect crystal system for constraints
        lattice_system, num_lattice_params = get_lattice_system(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.centering
        )
        
        if refine_lattice:
            print(f"Lattice Refinement Enabled.")
            print(f"Detected System: {lattice_system} ({num_lattice_params} free parameters).")
            
        if refine_goniometer:
            print(f"Goniometer Refinement Enabled (Bounds: +/- {goniometer_bound_deg} deg).")
        
        # 3. Instantiate the JAX-compatible objective function
        objective = VectorizedObjectiveJAX(
            self.reciprocal_lattice_B(),
            self.centering,
            kf_ki_input,
            np.array(self.wavelength), 
            self._angle_cdf,           
            self._angle_t,             
            weights=weights,
            softness=softness,
            cell_params=cell_params_init,
            refine_lattice=refine_lattice,
            lattice_bound_frac=lattice_bound_frac,
            lattice_system=lattice_system,
            goniometer_axes=goniometer_axes,
            goniometer_angles=goniometer_angles,
            refine_goniometer=refine_goniometer,
            goniometer_bound_deg=goniometer_bound_deg
        )

        # Determine dimensions
        # 3 dimensions for rotation (u0, u1, u2)
        # + num_lattice_params (if refined)
        # + num_goniometer_axes (if refined)
        num_dims = 3
        if refine_lattice:
            num_dims += num_lattice_params
        if refine_goniometer:
            num_dims += len(goniometer_axes)
            
        sample_solution = jnp.zeros(num_dims)

        # 4. Initialize strategy
        if strategy_name.lower() == "de":
            strategy = DifferentialEvolution(solution=sample_solution, population_size=population_size)
            strategy_type = 'population_based'
        elif strategy_name.lower() == "pso":
            strategy = PSO(solution=sample_solution, population_size=population_size)
            strategy_type = 'population_based'
        elif strategy_name.lower() == "cma_es":
            strategy = CMA_ES(solution=sample_solution, population_size=population_size)
            strategy_type = 'distribution_based'
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        params = strategy.default_params

        @jax.jit
        def es_step(rng, state, params):
            rng, rng_ask, rng_tell = jax.random.split(rng, 3)
            population, state = strategy.ask(rng_ask, state, params)
            population_clipped = jnp.clip(population, 0.0, 1.0)
            fitness = objective(population_clipped)
            state, metrics = strategy.tell(rng_tell, population_clipped, fitness, state, params)
            return rng, state, metrics

        best_overall_fitness = jnp.inf
        best_overall_member = None

        for i in range(n_runs):
            run_seed = seed + i
            print(f"\n--- Starting Run {i+1}/{n_runs} (Seed: {run_seed}) ---")

            rng = jax.random.PRNGKey(run_seed)
            rng, rng_pop, rng_init = jax.random.split(rng, 3)

            if init_params is None:
                if strategy_type == 'population_based':
                    population_init = jax.random.uniform(rng_pop, (population_size, num_dims))
                    fitness_init = objective(population_init)
                    state = strategy.init(rng_init, population_init, fitness_init, params)
                elif strategy_type == 'distribution_based':
                    solution_init = jax.random.uniform(rng_pop, (num_dims, ))
                    state = strategy.init(rng_init, solution_init, params)
            else:
                # Handle restart logic
                start_sol = jnp.array(init_params)

                # MODIFIED: Allow extending the solution if dimensions increased
                if start_sol.shape[0] != num_dims:
                    if start_sol.shape[0] < num_dims:
                        print(f"Bootstrapping: extending solution from {start_sol.shape[0]} to {num_dims} dims.")
                        # Calculate how many new parameters we need (e.g. goniometer offsets)
                        n_new = num_dims - start_sol.shape[0]
                        # Append zeros (assuming 0.5 is 'neutral')
                        padding = jnp.full((n_new,), 0.5)

                        start_sol = jnp.concatenate([start_sol, padding])

                        if strategy_type == 'population_based':
                            # Create a population around the padded solution
                            noise = jax.random.normal(rng_pop, (population_size, num_dims)) * 0.05
                            population_init = jnp.clip(start_sol + noise, 0.0, 1.0)
                            fitness_init = objective(population_init)
                            state = strategy.init(rng_init, population_init, fitness_init, params)
                        elif strategy_type == 'distribution_based':
                            # Initialize distribution at the padded solution
                            state = strategy.init(rng_init, start_sol, params)
                    else:
                        print(f"Warning: init_params shape {start_sol.shape} mismatch with required {num_dims}. Restarting random.")
                        if strategy_type == 'population_based':
                             population_init = jax.random.uniform(rng_pop, (population_size, num_dims))
                             fitness_init = objective(population_init)
                             state = strategy.init(rng_init, population_init, fitness_init, params)
                        else:
                             state = strategy.init(rng_init, jax.random.uniform(rng_pop, (num_dims, )), params)
                else:
                    if strategy_type == 'population_based':
                        noise = jax.random.normal(rng_pop, (population_size, num_dims)) * 0.05
                        population_init = jnp.clip(start_sol + noise, 0.0, 1.0)
                        fitness_init = objective(population_init)
                        state = strategy.init(rng_init, population_init, fitness_init, params)
                    elif strategy_type == 'distribution_based':
                        state = strategy.init(rng_init, start_sol, params)

            pbar = range(num_generations)
            if trange is not None:
                pbar = trange(num_generations, desc=f"Run {i+1}/{n_runs}")

            for gen in pbar:
                rng, state, metrics = es_step(rng, state, params)
                if trange is not None:
                    pbar.set_description(f"Run {i+1} Gen: {gen+1}/{num_generations} | Best Fitness: {metrics['best_fitness']:.4f}")

            current_run_fitness = state.best_fitness
            current_run_member = state.best_solution
            print(f"Run {i+1} finished. Best fitness: {current_run_fitness:.4f}")

            if current_run_fitness < best_overall_fitness:
                best_overall_fitness = current_run_fitness
                best_overall_member = current_run_member
                print(f"!!! New best solution found in Run {i+1} !!!")

        print(f"\n--- All {n_runs} runs complete ---")
        print(f"Best overall fitness: {best_overall_fitness:.4f}")
        
        self.x = np.array(best_overall_member)

        # 5. Process results
        idx = 0
        rot_params = self.x[idx:idx+3]
        idx += 3
        
        # Calculate final U (orientation)
        U = objective.orientation_U_jax(rot_params[None])[0] # (3, 3)

        # Calculate final B (reciprocal lattice)
        if refine_lattice:
            # Extract normalized lattice params
            cell_norm = self.x[None, idx:idx+num_lattice_params] 
            idx += num_lattice_params
            
            p_full_real = objective.reconstruct_cell_params(cell_norm)
            p = np.array(p_full_real[0])
            
            B_new = objective.compute_B_jax(cell_norm)[0] # (3, 3)
            
            print("--- Refined Lattice Parameters ---")
            print(f"a: {p[0]:.4f}, b: {p[1]:.4f}, c: {p[2]:.4f}")
            print(f"alpha: {p[3]:.4f}, beta: {p[4]:.4f}, gamma: {p[5]:.4f}")
            
            self.a, self.b, self.c = p[0], p[1], p[2]
            self.alpha, self.beta, self.gamma = p[3], p[4], p[5]
            B = B_new
        else:
            B = self.reciprocal_lattice_B()
            
        # Refined Goniometer
        kf_ki_vec = np.array(kf_ki_input)
        if refine_goniometer:
            gonio_norm = self.x[None, idx:idx+len(goniometer_axes)]
            # Recompute R (S=1, M, 3, 3)
            R_refined = objective.compute_goniometer_R_jax(gonio_norm)[0] # (M, 3, 3)
            
            # Update self.R
            self.R = np.array(R_refined)
            
            # Print and Store offsets
            offsets_val = objective.gonio_min + gonio_norm * (objective.gonio_max - objective.gonio_min)
            print("--- Refined Goniometer Offsets (deg) ---")
            if goniometer_names is not None:
                for name, val in zip(goniometer_names, offsets_val[0]):
                    print(f"{name}: {val:.4f}")
            else:
                print(offsets_val[0])
            self.goniometer_offsets = offsets_val[0]
            
            # Transform vectors to Crystal Frame for indexing report
            # R is (M, 3, 3), kf_ki_vec (Lab) is (3, M)
            # R^T @ kf_ki_vec
            kf_ki_vec = np.einsum("mji,jm->im", self.R, kf_ki_vec)
        else:
            # Vectors already in crystal frame
            kf_ki_vec = kf_ki_input

        # Enforce Symmetry consistency
        U_new, _ = self.get_consistent_U_for_symmetry(U, B)
        
        # Recalculate score with final U and B
        UB_final = U_new @ B
        _, score, hkl, lamb = objective.indexer_dynamic_jax(UB_final[None], kf_ki_vec[None], softness=softness)

        return score[0], hkl[0], lamb[0], U_new
