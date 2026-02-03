import os
import warnings
from functools import partial

import h5py
import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.interpolate


import gemmi

# Import JAX with fallback from utils (centralized)
from subhkl.utils import (
    jax,
    jnp,
    jscipy_linalg,
    HAS_JAX,
    OPTIMIZATION_BACKEND,
    DifferentialEvolution,
    PSO,
    CMA_ES,
    Mesh,
    NamedSharding,
    P,
)

from subhkl.detector import scattering_vector_from_angles
from subhkl.spacegroup import generate_hkl_mask, get_centering, get_space_group_object

try:
    from tqdm import trange
except ImportError:
    trange = None


def require_jax():
    """
    Check if JAX is available and raise an informative error if not.

    Raises
    ------
    ImportError
        If JAX and evosax are not installed.
    """
    if not HAS_JAX:
        raise ImportError(
            "JAX and evosax are required for this functionality. "
            'Install with: pip install -e ".[jax]" or pip install jax jaxlib evosax'
        )

# ==============================================================================
# 1. HELPER FUNCTIONS: Parameter Mapping
# ==============================================================================

def _inverse_map_param(value, bound):
    if bound < 1e-12: return 0.5
    norm = (value + bound) / (2.0 * bound)
    return np.clip(norm, 0.0, 1.0)

def _forward_map_param(norm, bound):
    return norm * 2.0 * bound - bound

def _inverse_map_lattice(value, nominal, frac_bound):
    delta = np.abs(nominal) * frac_bound
    min_val = nominal - delta
    max_val = nominal + delta
    if (max_val - min_val) < 1e-12: return 0.5
    norm = (value - min_val) / (max_val - min_val)
    return np.clip(norm, 0.0, 1.0)

def _forward_map_lattice(norm, nominal, frac_bound):
    delta = np.abs(nominal) * frac_bound
    min_val = nominal - delta
    max_val = nominal + delta
    return min_val + norm * (max_val - min_val)

def _get_active_lattice_indices(lattice_system):
    if lattice_system == 'Cubic': return [0]
    elif lattice_system == 'Hexagonal': return [0, 2]
    elif lattice_system == 'Tetragonal': return [0, 2]
    elif lattice_system == 'Rhombohedral': return [0, 3]
    elif lattice_system == 'Orthorhombic': return [0, 1, 2]
    elif lattice_system == 'Monoclinic': return [0, 1, 2, 4]
    else: return [0, 1, 2, 3, 4, 5]

def get_lattice_system(a, b, c, alpha, beta, gamma, space_group_name, atol_len=0.05, atol_ang=0.5):
    """
    Determine the Lattice System for refinement based on Space Group and Geometry.
    
    Logic:
    1. Determine 'Expected' system from Space Group (Bravais Lattice).
    2. Determine 'Geometric' system from parameter values (e.g. 90 deg, a=b).
    3. Constraint Check: If Geometry violates Space Group (e.g. 88 deg for Cubic), WARN.
    4. Override: If Expected symmetry is LOWER than Geometric (e.g. P1 SG but 90-90-90 params),
       force LOWER symmetry (Triclinic) to allow full refinement.
    """
    
    # --- 1. Determine Expected System from Space Group ---
    try:
        sg = get_space_group_object(space_group_name)
        # Gemmi CrystalSystem: triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic
        sys_str = str(sg.crystal_system()).split('.')[-1].lower() 
        centering = sg.centring_type() # 'P', 'F', 'I', 'R', etc.
    except Exception:
        sys_str = 'triclinic' # Fallback
        centering = 'P'

    # Map to internal types
    expected = 'Triclinic'
    if sys_str == 'cubic': expected = 'Cubic'
    elif sys_str == 'hexagonal': expected = 'Hexagonal'
    elif sys_str == 'trigonal': 
        expected = 'Rhombohedral' if centering == 'R' else 'Hexagonal'
    elif sys_str == 'tetragonal': expected = 'Tetragonal'
    elif sys_str == 'orthorhombic': expected = 'Orthorhombic'
    elif sys_str == 'monoclinic': expected = 'Monoclinic'

    # --- 2. Check Constraints & Warn ---
    is_90 = lambda x: np.isclose(x, 90.0, atol=atol_ang)
    is_120 = lambda x: np.isclose(x, 120.0, atol=atol_ang)
    eq = lambda x, y: np.isclose(x, y, atol=atol_len)

    violation_msg = []
    
    if expected == 'Cubic':
        if not (eq(a, b) and eq(b, c)): violation_msg.append("a=b=c")
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)): violation_msg.append("angles=90")
    elif expected == 'Hexagonal':
        if not eq(a, b): violation_msg.append("a=b")
        if not (is_90(alpha) and is_90(beta) and is_120(gamma)): violation_msg.append("angles=90,90,120")
    elif expected == 'Rhombohedral':
        if not (eq(a, b) and eq(b, c)): violation_msg.append("a=b=c")
        if not (eq(alpha, beta) and eq(beta, gamma)): violation_msg.append("alpha=beta=gamma")
    elif expected == 'Tetragonal':
        if not eq(a, b): violation_msg.append("a=b")
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)): violation_msg.append("angles=90")
    elif expected == 'Orthorhombic':
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)): violation_msg.append("angles=90")
    elif expected == 'Monoclinic':
        # Assuming b-unique or c-unique depending on settings, roughly check if at least two are 90
        count90 = sum([is_90(alpha), is_90(beta), is_90(gamma)])
        if count90 < 2: violation_msg.append("at least two angles=90")

    if violation_msg:
        warnings.warn(
            f"\n[Lattice System] Input parameters violate {space_group_name} ({expected}) constraints: {', '.join(violation_msg)}.\n"
            f"optimization will enforce {expected} constraints, which may cause a jump in parameters."
        )

    # --- 3. Geometric Inference (Legacy Logic) ---
    geometric = 'Triclinic'
    if is_90(alpha) and is_90(beta) and is_90(gamma):
        if eq(a, b) and eq(b, c): geometric = 'Cubic'
        elif eq(a, b): geometric = 'Tetragonal'
        else: geometric = 'Orthorhombic'
    elif is_90(alpha) and is_90(beta) and is_120(gamma):
        if eq(a, b): geometric = 'Hexagonal'
    elif centering == 'R' and eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma):
        geometric = 'Rhombohedral'
    elif sum([is_90(alpha), is_90(beta), is_90(gamma)]) >= 2:
        geometric = 'Monoclinic'

    # --- 4. Hierarchy and Override ---
    # Rank symmetries: Lower number = Lower Symmetry (More free params)
    ranks = {
        'Triclinic': 0, 'Monoclinic': 1, 'Orthorhombic': 2, 
        'Tetragonal': 3, 'Rhombohedral': 4, 'Hexagonal': 4, 'Cubic': 5
    }
    
    rank_exp = ranks.get(expected, 0)
    rank_geo = ranks.get(geometric, 0)

    # Decision Rule:
    # 1. If Expected is LOWER symmetry than Geometric (e.g. P1 SG, but 90-90-90 params),
    #    we MUST use Expected (Triclinic) to allow refinement of angles away from 90.
    # 2. If Expected is HIGHER symmetry (e.g. Cubic SG, but wonky params),
    #    we MUST use Expected (Cubic) to enforce Space Group symmetry (despite the warning).
    
    final_system = expected
    
    # Calculate free params count
    # Triclinic(6), Mono(4), Ortho(3), Tet(2), Hex(2), Rho(2), Cub(1)
    if final_system == 'Triclinic': num = 6
    elif final_system == 'Monoclinic': num = 4
    elif final_system == 'Orthorhombic': num = 3
    elif final_system == 'Tetragonal': num = 2
    elif final_system == 'Hexagonal': num = 2
    elif final_system == 'Rhombohedral': num = 2
    elif final_system == 'Cubic': num = 1
    else: num = 6

    if rank_exp < rank_geo:
        print(f"Lattice System Override: Geometry suggests {geometric}, but Space Group {space_group_name} requires {expected}. Enforcing {expected} (Lower Symmetry).")
    
    return final_system, num

def rotation_matrix_from_axis_angle_jax(axis, angle_rad):
    u = axis / jnp.linalg.norm(axis)
    ux, uy, uz = u
    K = jnp.array([[0.0, -uz, uy],[uz, 0.0, -ux],[-uy, ux, 0.0]]) 
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    eye = jnp.eye(3)
    R = eye + s[..., None, None] * K + (1.0 - c)[..., None, None] * (K @ K)
    return R

def rotation_matrix_from_rodrigues_jax(w):
    theta = jnp.linalg.norm(w) + 1e-9 
    k = w / theta
    K = jnp.array([[0.0, -k[2], k[1]],[k[2], 0.0, -k[0]],[-k[1], k[0], 0.0]])
    I = jnp.eye(3)
    R = I + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
    return R

# ==============================================================================
# 2. VECTORIZED OBJECTIVE (JAX)
# ==============================================================================

class VectorizedObjective:
    def __init__(self, B, kf_ki_dir, peak_xyz_lab, wavelength, angle_cdf, angle_t, weights=None, softness=0.01,
                 cell_params=None, refine_lattice=False, lattice_bound_frac=0.05, lattice_system='Triclinic',
                 goniometer_axes=None, goniometer_angles=None, refine_goniometer=False, goniometer_bound_deg=5.0,
                 goniometer_refine_mask=None, goniometer_nominal_offsets=None,
                 refine_sample=False, sample_bound_meters=0.002, sample_nominal=None,
                 refine_beam=False, beam_bound_deg=1.0, beam_nominal=None,
                 peak_radii=None, loss_method='gaussian', 
                 hkl_search_range=15, d_min=5.0, d_max=100.0, search_window_size=256, window_batch_size=32,
                 chunk_size=4096, num_iters=20, top_k=32,
                 space_group="P 1"):
        
        self.B = jnp.array(B)
        self.kf_ki_dir_init = jnp.array(kf_ki_dir)
        self.k_sq_init = jnp.sum(self.kf_ki_dir_init**2, axis=0)
        
        # Fixed Detector Vectors (kf) from input Q (assuming ideal beam)
        # Note: self.kf_ki_dir_init was calculated as (Pixel - IdealBeam).
        # So kf_lab = kf_ki_dir_init + IdealBeam.
        self.kf_lab_fixed = self.kf_ki_dir_init + jnp.array([0., 0., 1.])[:, None]
        self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(self.kf_lab_fixed, axis=0)
        
        if peak_xyz_lab is not None:
            self.peak_xyz = jnp.array(peak_xyz_lab.T) 
        else:
            self.peak_xyz = None

        self.refine_sample = refine_sample
        self.sample_bound = sample_bound_meters
        if sample_nominal is None: self.sample_nominal = jnp.zeros(3)
        else: self.sample_nominal = jnp.array(sample_nominal)

        self.refine_beam = refine_beam
        self.beam_bound_deg = beam_bound_deg
        if beam_nominal is None: self.beam_nominal = jnp.array([0.0, 0.0, 1.0])
        else: self.beam_nominal = jnp.array(beam_nominal)

        self.softness = softness
        self.loss_method = loss_method
        self.angle_cdf = jnp.array(angle_cdf)
        self.angle_t = jnp.array(angle_t)
        self.space_group = space_group

        self.refine_lattice = refine_lattice
        self.lattice_system = lattice_system
        self.lattice_bound_frac = lattice_bound_frac
        self.refine_goniometer = refine_goniometer
        self.goniometer_bound_deg = goniometer_bound_deg

        if self.refine_lattice:
            if cell_params is None: raise ValueError("cell_params required")
            self.cell_init = jnp.array(cell_params) 
            if self.lattice_system == 'Cubic': self.free_params_init = self.cell_init[0:1]
            elif self.lattice_system == 'Hexagonal': self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[2]])
            elif self.lattice_system == 'Tetragonal': self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[2]])
            elif self.lattice_system == 'Rhombohedral': self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[3]])
            elif self.lattice_system == 'Orthorhombic': self.free_params_init = self.cell_init[0:3]
            elif self.lattice_system == 'Monoclinic': self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[1], self.cell_init[2], self.cell_init[4]])
            else: self.free_params_init = self.cell_init

        if goniometer_axes is not None:
            self.gonio_axes = jnp.array(goniometer_axes) 
            self.gonio_angles = jnp.array(goniometer_angles) 
            self.num_gonio_axes = self.gonio_axes.shape[0]
            if goniometer_refine_mask is not None: self.gonio_mask = np.array(goniometer_refine_mask, dtype=bool)
            else: self.gonio_mask = np.ones(self.num_gonio_axes, dtype=bool)
            self.num_active_gonio = np.sum(self.gonio_mask)
            
            if goniometer_nominal_offsets is None: self.gonio_nominal_offsets = jnp.zeros(self.num_gonio_axes)
            else: self.gonio_nominal_offsets = jnp.array(goniometer_nominal_offsets)
            
            self.gonio_min = jnp.full(self.num_gonio_axes, -goniometer_bound_deg) 
            self.gonio_max = jnp.full(self.num_gonio_axes, goniometer_bound_deg)
        else:
            self.gonio_axes = None
            self.num_gonio_axes = 0

        wavelength = jnp.array(wavelength)
        self.wl_min_val = wavelength[0]
        self.wl_max_val = wavelength[1]
        self.num_candidates = 64
        
        if weights is None: self.weights = jnp.ones(self.kf_ki_dir_init.shape[1])
        else: self.weights = jnp.array(weights)
        if peak_radii is None: self.peak_radii = jnp.zeros(self.kf_ki_dir_init.shape[1])
        else: self.peak_radii = jnp.array(peak_radii)
        
        self.max_score = jnp.sum(self.weights)
        self.d_min = d_min
        self.d_max = d_max
        self.search_window_size = search_window_size
        self.window_batch_size = window_batch_size

        self.chunk_size = chunk_size
        self.num_iters = num_iters
        self.top_k = top_k

        # --- Search Window Heuristic Warning ---
        if self.loss_method == 'forward':
            # Calculate Volume (Real Space) from B matrix (Reciprocal Basis, 2pi included)
            # V_real = (2pi)^3 / det(B)
            det_B = float(np.abs(np.linalg.det(self.B)))
            if det_B > 1e-9:
                vol_real = 1.0 / det_B
                # Peak Density Heuristic: N approx Vol / d^3
                # Factor 0.0025 empirically determined for +/- 2 deg coverage on MANDI
                heuristic_win = int((vol_real / (self.d_min**3)) * 0.0025)
                # Clamp for sanity in warning logic
                heuristic_win = max(64, heuristic_win)
                
                if self.search_window_size < (heuristic_win * 0.75):
                    warnings.warn(
                        f"\n[WARNING] search_window_size ({self.search_window_size}) is likely too small "
                        f"for resolution {self.d_min:.2f}A and Volume {vol_real:.0f}A^3.\n"
                        f"Binary search indexer may miss valid peaks.\n"
                        f"RECOMMENDED SIZE: >= {heuristic_win}\n"
                    )

        # --- HKL Mask Generation ---
        r = jnp.arange(-hkl_search_range, hkl_search_range + 1)
        h, k, l = jnp.meshgrid(r, r, r, indexing='ij')
        hkl_pool = jnp.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)
        zero_mask = ~jnp.all(hkl_pool == 0, axis=0)
        hkl_pool = hkl_pool[:, zero_mask]
        q_cart = self.B @ hkl_pool 

        # NOTE: For Sinkhorn, we keep the flat pool available directly
        self.pool_hkl_flat = hkl_pool

        phis = jnp.arctan2(q_cart[1], q_cart[0])
        sort_idx = jnp.argsort(phis)
        self.pool_phi_sorted = phis[sort_idx]
        self.pool_hkl_sorted = hkl_pool[:, sort_idx] 
        self.mask_range = hkl_search_range
        print(f"Generating HKL mask for Space Group: {self.space_group} (Range: +/-{self.mask_range})")
        mask_cpu = generate_hkl_mask(self.mask_range, self.mask_range, self.mask_range, self.space_group)
        self.valid_hkl_mask = jnp.array(mask_cpu)

    def _get_physical_params_jax(self, x):
        """Reconstruct physical parameters (Base + Delta) for a batch of solutions x."""
        idx = 0
        rot_params = x[:, idx:idx+3]
        U = self.orientation_U_jax(rot_params) 
        idx += 3

        if self.refine_lattice:
            n_lat = self.free_params_init.size
            cell_params_norm = x[:, idx:idx+n_lat]
            B = self.compute_B_jax(cell_params_norm) 
            idx += n_lat
            UB = jnp.einsum("sij,sjk->sik", U, B)
        else:
            B = self.B 
            UB = jnp.einsum("sij,jk->sik", U, B)

        if self.refine_sample:
            s_norm = x[:, idx:idx+3]
            idx += 3
            sample_delta = _forward_map_param(s_norm, self.sample_bound)
            sample_total = self.sample_nominal + sample_delta
        else:
            sample_total = self.sample_nominal[None, :].repeat(x.shape[0], axis=0)

        if self.refine_beam:
            bound_rad = jnp.deg2rad(self.beam_bound_deg)
            tx = _forward_map_param(x[:, idx], bound_rad)
            ty = _forward_map_param(x[:, idx+1], bound_rad)
            idx += 2
            ki_vec = jnp.tile(self.beam_nominal[None, :], (x.shape[0], 1))
            ki_vec = ki_vec.at[:, 0].add(tx)
            ki_vec = ki_vec.at[:, 1].add(ty)
            ki_vec = ki_vec / jnp.linalg.norm(ki_vec, axis=1, keepdims=True)
        else:
            ki_vec = self.beam_nominal[None, :].repeat(x.shape[0], axis=0)

        if self.refine_goniometer:
            n_active = self.num_active_gonio
            if n_active > 0:
                active_params = x[:, idx:idx+n_active]
                idx += n_active
                batch_size = x.shape[0]
                gonio_norm = jnp.full((batch_size, self.num_gonio_axes), 0.5)
                gonio_norm = gonio_norm.at[:, self.gonio_mask].set(active_params)
            else:
                gonio_norm = jnp.full((x.shape[0], self.num_gonio_axes), 0.5)
            
            offsets_delta = _forward_map_param(gonio_norm, self.goniometer_bound_deg)
            offsets_total = self.gonio_nominal_offsets + offsets_delta
            R = self.compute_goniometer_R_jax(gonio_norm) # Helper assumes input is norm
        else:
            if self.gonio_axes is not None:
                offsets_total = self.gonio_nominal_offsets[None, :].repeat(x.shape[0], axis=0)
                # To calculate R from Nominal (fixed), we pass 0.5 to helper
                gonio_norm = jnp.full((x.shape[0], self.num_gonio_axes), 0.5)
                R = self.compute_goniometer_R_jax(gonio_norm)
            else:
                offsets_total = None
                R = None

        return UB, B, sample_total, ki_vec, offsets_total, R

    def reconstruct_cell_params(self, params_norm):
        p_free = _forward_map_lattice(params_norm, self.free_params_init, self.lattice_bound_frac)
        S = params_norm.shape[0]
        deg90 = jnp.full((S,), 90.0)
        deg120 = jnp.full((S,), 120.0)
        if self.lattice_system == 'Cubic':
            a = p_free[:, 0]
            return jnp.stack([a, a, a, deg90, deg90, deg90], axis=1)
        elif self.lattice_system == 'Hexagonal':
            a, c = p_free[:, 0], p_free[:, 1]
            return jnp.stack([a, a, c, deg90, deg90, deg120], axis=1)
        elif self.lattice_system == 'Tetragonal':
            a, c = p_free[:, 0], p_free[:, 1]
            return jnp.stack([a, a, c, deg90, deg90, deg90], axis=1)
        elif self.lattice_system == 'Rhombohedral':
            a, alpha = p_free[:, 0], p_free[:, 1]
            return jnp.stack([a, a, a, alpha, alpha, alpha], axis=1)
        elif self.lattice_system == 'Orthorhombic':
            a, b, c = p_free[:, 0], p_free[:, 1], p_free[:, 2]
            return jnp.stack([a, b, c, deg90, deg90, deg90], axis=1)
        elif self.lattice_system == 'Monoclinic':
            a, b, c, beta = p_free[:, 0], p_free[:, 1], p_free[:, 2], p_free[:, 3]
            return jnp.stack([a, b, c, deg90, beta, deg90], axis=1)
        else: return p_free

    def compute_B_jax(self, cell_params_norm):
        p = self.reconstruct_cell_params(cell_params_norm)
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        deg2rad = jnp.pi / 180.0
        alpha, beta, gamma = p[:, 3] * deg2rad, p[:, 4] * deg2rad, p[:, 5] * deg2rad
        g11, g22, g33 = a**2, b**2, c**2
        g12, g13, g23 = a * b * jnp.cos(gamma), a * c * jnp.cos(beta), b * c * jnp.cos(alpha)
        row1 = jnp.stack([g11, g12, g13], axis=-1)
        row2 = jnp.stack([g12, g22, g23], axis=-1)
        row3 = jnp.stack([g13, g23, g33], axis=-1)
        G = jnp.stack([row1, row2, row3], axis=-2)
        G_star = jnp.linalg.inv(G)
        B = jscipy_linalg.cholesky(G_star, lower=False)
        return B
    
    def compute_goniometer_R_jax(self, gonio_offsets_norm):
        # NOTE: This helper uses Norm to calc Delta, then adds Nominal from self.
        offsets_delta = _forward_map_param(gonio_offsets_norm, self.goniometer_bound_deg)
        total_offsets = self.gonio_nominal_offsets + offsets_delta
        
        angles_deg = total_offsets[:, :, None] + self.gonio_angles[None, :, :]
        S, M = total_offsets.shape[0], self.gonio_angles.shape[1]
        R = jnp.eye(3)[None, None, ...].repeat(S, axis=0).repeat(M, axis=1)
        deg2rad = jnp.pi / 180.0
        for i in range(self.num_gonio_axes):
            axis_spec = self.gonio_axes[i]
            direction = axis_spec[:3]
            sign = axis_spec[3]
            theta = sign * angles_deg[:, i, :] * deg2rad
            Ri = rotation_matrix_from_axis_angle_jax(direction, theta)
            R = jnp.einsum('smij,smjk->smik', R, Ri)
        return R

    def orientation_U_jax(self, param):
        U = jax.vmap(rotation_matrix_from_rodrigues_jax)(param)
        return U

    def indexer_dynamic_soft_jax(self, UB, kf_ki_sample, k_sq_override=None, softness=0.01):
        UB_inv = jnp.linalg.inv(UB)
        v = jnp.einsum("sij,sjm->sim", UB_inv, kf_ki_sample)
        abs_v = jnp.abs(v)
        max_v_val = jnp.max(abs_v, axis=1)
        n_start = max_v_val / self.wl_max_val
        start_int = jnp.ceil(n_start)
        k_sq = k_sq_override if k_sq_override is not None else self.k_sq_init[None, :]
        k_norm = jnp.sqrt(k_sq)
        
        initial_carry = (jnp.zeros(max_v_val.shape), jnp.zeros(max_v_val.shape), jnp.zeros((v.shape[0], 3, v.shape[2]), dtype=jnp.int32), jnp.zeros(max_v_val.shape))
        def scan_body(carry, i):
            curr_sum, curr_max, curr_best_hkl, curr_best_lamb = carry
            n = start_int + i
            n_safe = jnp.where(n == 0, 1e-9, n)
            lamda_cand = max_v_val / n_safe
            hkl_float = v / lamda_cand[:, None, :]
            hkl_int = jnp.round(hkl_float).astype(jnp.int32)
            q_int = jnp.einsum("sij,sjm->sim", UB, hkl_int)
            k_dot_q = jnp.sum(kf_ki_sample * q_int, axis=1)
            safe_dot = jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
            lambda_opt = jnp.clip(k_sq / safe_dot, self.wl_min_val, self.wl_max_val)
            q_obs = kf_ki_sample / lambda_opt[:, None, :]
            dist_sq = jnp.sum((q_obs - q_int)**2, axis=1)
            safe_lamb = jnp.where(lambda_opt == 0, 1.0, lambda_opt)
            effective_sigma = softness + (k_norm / safe_lamb) * self.peak_radii[None, :]
            prob = jnp.exp(-dist_sq / (2 * effective_sigma**2))
            
            valid_cand = (lamda_cand >= self.wl_min_val) & (lamda_cand <= self.wl_max_val)
            
            # --- FIX: ADDED RESOLUTION FILTER (d_min / d_max) ---
            # 1. Calc |Q|^2 for predicted HKL
            q_sq_pred = jnp.sum(q_int**2, axis=1)
            # 2. Convert to d = 1/|Q| (Crystallographic units)
            d_pred = 1.0 / jnp.sqrt(q_sq_pred + 1e-9)
            valid_res = (d_pred >= self.d_min) & (d_pred <= self.d_max)

            h, k, l = hkl_int[:, 0, :], hkl_int[:, 1, :], hkl_int[:, 2, :]
            r = self.mask_range
            idx_h = jnp.clip(h + r, 0, 2*r).astype(jnp.int32)
            idx_k = jnp.clip(k + r, 0, 2*r).astype(jnp.int32)
            idx_l = jnp.clip(l + r, 0, 2*r).astype(jnp.int32)
            is_allowed = self.valid_hkl_mask[idx_h, idx_k, idx_l]
            
            # Combine masks
            final_mask = valid_cand & is_allowed & valid_res
            
            prob = jnp.where(final_mask, prob, 0.0)
            
            new_sum = curr_sum + prob
            update_mask = prob > curr_max
            new_max = jnp.where(update_mask, prob, curr_max)
            new_best_hkl = jnp.where(update_mask[:, None, :], hkl_int, curr_best_hkl)
            new_best_lamb = jnp.where(update_mask, lambda_opt, curr_best_lamb)
            return (new_sum, new_max, new_best_hkl, new_best_lamb), None
        
        final_carry, _ = jax.lax.scan(scan_body, initial_carry, jnp.arange(self.num_candidates))
        accum_probs, _, best_hkl, best_lamb = final_carry
        score = -jnp.sum(self.weights * accum_probs, axis=1)
        return score, accum_probs, best_hkl.transpose((0, 2, 1)), best_lamb

    def indexer_dynamic_cosine_aniso_jax(self, UB, kf_ki_sample, k_sq_override=None, softness=0.01):
        UB_inv = jnp.linalg.inv(UB)
        v = jnp.einsum("sij,sjm->sim", UB_inv, kf_ki_sample)
        abs_v = jnp.abs(v)
        max_v_val = jnp.max(abs_v, axis=1)
        n_start = max_v_val / self.wl_max_val
        start_int = jnp.ceil(n_start)
        recip_len_sq = jnp.sum(UB**2, axis=1)
        kappa = recip_len_sq / ((softness + 1e-9)**2 * 4 * jnp.pi**2)
        kappa = kappa[:, :, None]
        initial_carry = (
            jnp.zeros(max_v_val.shape),         
            jnp.full(max_v_val.shape, -1e9),    
            jnp.zeros((v.shape[0], 3, v.shape[2]), dtype=jnp.int32), 
            jnp.zeros(max_v_val.shape)          
        )
        def scan_body(carry, i):
            curr_sum, curr_max, curr_best_hkl, curr_best_lamb = carry
            n = start_int + i
            ratio = n / max_v_val
            hkl_float = v * ratio[:, None, :]
            lamda_cand = 1.0 / ratio
            cos_terms = kappa * (jnp.cos(2 * jnp.pi * hkl_float) - 1.0)
            log_prob = jnp.sum(cos_terms, axis=1) 
            prob = jnp.exp(log_prob)
            
            # --- VALIDATION LOGIC ---
            valid_cand = (lamda_cand >= self.wl_min_val) & (lamda_cand <= self.wl_max_val)
            
            # 1. ADDED: Resolution Filter (Crystallographic convention d = 1/|Q|)
            # hkl_float is the fractional HKL estimate. We need |B * hkl|
            # Approx: Use integer HKL for speed, or float for accuracy. 
            # Using float is safer for the filter.
            q_vecs = jnp.einsum("sij,sjm->sim", UB, hkl_float)
            q_sq = jnp.sum(q_vecs**2, axis=1) # |Q|^2 = 1/d^2
            d_est = 1.0 / jnp.sqrt(q_sq + 1e-9)
            valid_res = (d_est >= self.d_min) & (d_est <= self.d_max)

            # 2. Symmetry Mask
            hkl_int = jnp.round(hkl_float).astype(jnp.int32)
            h, k, l = hkl_int[:, 0, :], hkl_int[:, 1, :], hkl_int[:, 2, :]
            r = self.mask_range
            idx_h = jnp.clip(h + r, 0, 2*r).astype(jnp.int32)
            idx_k = jnp.clip(k + r, 0, 2*r).astype(jnp.int32)
            idx_l = jnp.clip(l + r, 0, 2*r).astype(jnp.int32)
            is_allowed = self.valid_hkl_mask[idx_h, idx_k, idx_l]
            
            # Combine all masks
            final_mask = valid_cand & valid_res & is_allowed
            
            prob = jnp.where(final_mask, prob, 0.0)
            new_sum = curr_sum + prob
            score_tracked = jnp.where(final_mask, log_prob, -1e9)
            
            update_mask = score_tracked > curr_max
            new_max = jnp.where(update_mask, score_tracked, curr_max)
            new_best_hkl = jnp.where(update_mask[:, None, :], hkl_int, curr_best_hkl)
            new_best_lamb = jnp.where(update_mask, lamda_cand, curr_best_lamb)
            return (new_sum, new_max, new_best_hkl, new_best_lamb), None
        
        final_carry, _ = jax.lax.scan(scan_body, initial_carry, jnp.arange(self.num_candidates))
        accum_probs, _, best_hkl, best_lamb = final_carry
        score = -jnp.sum(self.weights * accum_probs, axis=1)
        return score, accum_probs, best_hkl.transpose((0, 2, 1)), best_lamb

    def indexer_dynamic_binary_jax(self, UB, kf_ki_sample, k_sq_override=None, softness=0.01, window_batch_size=32):
        k_sq = k_sq_override if k_sq_override is not None else self.k_sq_init[None, :]
        k_norm = jnp.sqrt(k_sq)
        UB_inv = jnp.linalg.inv(UB)
        hkl_float = jnp.einsum("sij,sjm->sim", UB_inv, kf_ki_sample)
        hkl_cart_approx = jnp.einsum("ij,sjm->sim", self.B, hkl_float)
        phi_obs = jnp.arctan2(hkl_cart_approx[:, 1, :], hkl_cart_approx[:, 0, :])
        idx_centers = jnp.searchsorted(self.pool_phi_sorted, phi_obs)
        half_win = self.search_window_size // 2
        raw_offsets = jnp.arange(-half_win, half_win + 1)
        pad_len = (window_batch_size - (raw_offsets.shape[0] % window_batch_size)) % window_batch_size
        offsets_padded = jnp.pad(raw_offsets, (0, pad_len), constant_values=raw_offsets[-1])
        offset_batches = offsets_padded.reshape(-1, window_batch_size)
        init_min_dist = jnp.full(idx_centers.shape, 1e9)
        init_best_hkl = jnp.zeros(idx_centers.shape + (3,))
        init_best_lamb = jnp.zeros(idx_centers.shape)
        init_carry = (init_min_dist, init_best_hkl, init_best_lamb)
        def scan_body(carry, batch_offsets):
            curr_min_dist, curr_best_hkl, curr_best_lamb = carry
            gather_idx = idx_centers[..., None] + batch_offsets[None, None, :]
            pool_T = self.pool_hkl_sorted.T 
            hkl_cands = jnp.take(pool_T, gather_idx, axis=0, mode='wrap') 
            q_pred = jnp.einsum("sij,smwj->smwi", UB, hkl_cands)
            k_obs = jnp.transpose(kf_ki_sample, (0, 2, 1))[:, :, None, :]
            k_dot_q = jnp.sum(k_obs * q_pred, axis=3)
            lambda_opt = k_sq[..., None] / jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
            valid_lamb = (lambda_opt >= self.wl_min_val) & (lambda_opt <= self.wl_max_val)
            q_sq = jnp.sum(q_pred**2, axis=3)
            d_spacings = 1.0 / jnp.sqrt(q_sq + 1e-9) # crystallographic convention
            valid_res = (d_spacings >= self.d_min) & (d_spacings <= self.d_max)
            h, k, l = hkl_cands[..., 0], hkl_cands[..., 1], hkl_cands[..., 2]
            r = self.mask_range
            idx_h = jnp.clip(h + r, 0, 2*r).astype(jnp.int32)
            idx_k = jnp.clip(k + r, 0, 2*r).astype(jnp.int32)
            idx_l = jnp.clip(l + r, 0, 2*r).astype(jnp.int32)
            valid_sym = self.valid_hkl_mask[idx_h, idx_k, idx_l]
            valid_mask = valid_lamb & valid_res & valid_sym
            q_obs_opt = k_obs / jnp.where(lambda_opt==0, 1.0, lambda_opt)[..., None]
            diff = q_obs_opt - q_pred
            dist_sq = jnp.sum(diff**2, axis=3)
            dist_sq_masked = jnp.where(valid_mask, dist_sq, 1e9)
            batch_min_dist = jnp.min(dist_sq_masked, axis=2)
            batch_best_local_idx = jnp.argmin(dist_sq_masked, axis=2)
            batch_best_hkl = jnp.take_along_axis(hkl_cands, batch_best_local_idx[..., None, None], axis=2).squeeze(axis=2)
            batch_best_lamb = jnp.take_along_axis(lambda_opt, batch_best_local_idx[..., None], axis=2).squeeze(axis=2)
            improve_mask = batch_min_dist < curr_min_dist
            new_min_dist = jnp.where(improve_mask, batch_min_dist, curr_min_dist)
            new_best_hkl = jnp.where(improve_mask[..., None], batch_best_hkl, curr_best_hkl)
            new_best_lamb = jnp.where(improve_mask, batch_best_lamb, curr_best_lamb)
            return (new_min_dist, new_best_hkl, new_best_lamb), None
        final_carry, _ = jax.lax.scan(scan_body, init_carry, offset_batches)
        best_dist_sq, best_hkl, best_lamb = final_carry
        effective_sigma = softness + (k_norm / jnp.where(best_lamb==0, 1.0, best_lamb)) * self.peak_radii[None, :]
        probs = jnp.exp(-best_dist_sq / (2 * effective_sigma**2 + 1e-9))
        score = -jnp.sum(self.weights * probs, axis=1)
        return score, probs, best_hkl, best_lamb

    # ==========================================================================
    # OPTIMIZED SINKHORN-EM INDEXER (Memory Efficient + Rotation Trick)
    # ==========================================================================
    def indexer_sinkhorn_jax(self, UB, kf_ki_sample, k_sq_override=None, softness=0.01, num_iters=20, epsilon=1.0, top_k=32,
                             chunk_size=256):
        """
        Robust Memory-Efficient Sinkhorn with Rotated-Observer Optimization.
        
        Args:
            UB: (Batch, 3, 3) Orientation Matrix
            kf_ki_sample: (Batch, 3, N_obs) Observed directions
            top_k: Number of nearest neighbors to keep (sparsity factor).
            softness: vMF sigma (controls kernel width).
            epsilon: Sinkhorn entropy regularizer (kept at 1.0 due to internal scaling).
        """
        # 1. Setup Data
        hkl_pool = self.pool_hkl_flat # (3, N_hkl)
        
        # --- Observer Rotation Trick ---
        # Instead of q_theory = UB @ h, we project Obs into Crystal frame:
        # Cost = r_obs . (UB h) = (r_obs @ UB) . h
        # kf_ki_sample shape is (Batch, 3, N_obs) -> Axis 1 is spatial (x,y,z)
        
        # Normalize Obs
        norm_obs = jnp.linalg.norm(kf_ki_sample, axis=1, keepdims=True)
        r_obs_unit = kf_ki_sample / (norm_obs + 1e-9)
        
        # Re-project Unit Obs: (Batch, 3, N_obs) x (Batch, 3, 3) -> (Batch, N_obs, 3)
        # We contract spatial dim 'i' (size 3)
        r_obs_proj_unit = jnp.einsum("sin,sij->snj", r_obs_unit, UB)
        
        k_sq_obs = k_sq_override if k_sq_override is not None else self.k_sq_init[None, :]
        
        batch_size, _, n_obs = kf_ki_sample.shape
        _, n_hkl = hkl_pool.shape
        
        # 2. Block-wise Top-K Search
        # Chunk size significantly reduced to prevent OOM / thrashing
        num_chunks = (n_hkl + chunk_size - 1) // chunk_size
        
        def scan_topk(carry, i):
            curr_vals, curr_idxs = carry 
            idx_start = i * chunk_size
            
            # Slice HKL Pool (Static, small)
            # Use dynamic_slice to handle last chunk boundary gracefully
            # (If last chunk < 256, JAX will pad or error depending on implementation, 
            # here we use simple slicing assuming hkl_pool is sufficient or padding isn't critical for top-k)
            hkl_chunk = jax.lax.dynamic_slice(hkl_pool, (0, idx_start), (3, chunk_size))
            
            # 1. Compute Norms |UB h| for this chunk
            # q_chunk = UB @ h_chunk. Shape (Batch, 3, Chunk).
            q_chunk = jnp.einsum("sij,jk->sik", UB, hkl_chunk)
            q_sq_chunk = jnp.sum(q_chunk**2, axis=1)
            norm_q_chunk = jnp.sqrt(q_sq_chunk + 1e-9) # (Batch, Chunk)
            
            # 2. Dot Product using Rotated Observer
            # (Batch, N_obs, 3) @ (3, Chunk) -> (Batch, N_obs, Chunk)
            dot_raw = jnp.matmul(r_obs_proj_unit, hkl_chunk)
            
            # 3. Apply Norm
            # cosine = dot_raw / |UB h|
            dots_chunk = dot_raw / (norm_q_chunk[:, None, :] + 1e-9)
            
            # 4. Top-K Selection
            global_idxs = jnp.arange(chunk_size) + idx_start
            global_idxs_broadcast = jnp.tile(global_idxs[None, None, :], (batch_size, n_obs, 1))
            
            combined_vals = jnp.concatenate([curr_vals, dots_chunk], axis=2)
            combined_idxs = jnp.concatenate([curr_idxs, global_idxs_broadcast], axis=2)
            
            vals, top_k_indices = jax.lax.top_k(combined_vals, top_k)
            idxs = jnp.take_along_axis(combined_idxs, top_k_indices, axis=2)
            return (vals, idxs), None

        init_vals = jnp.full((batch_size, n_obs, top_k), -1e9)
        init_idxs = jnp.zeros((batch_size, n_obs, top_k), dtype=jnp.int32)
        
        (top_vals, top_idxs), _ = jax.lax.scan(scan_topk, (init_vals, init_idxs), jnp.arange(num_chunks))
        
        # 3. Robust Scaling
        max_possible_L = 1.0 / (softness**2)
        scale_factor = 50.0 / max_possible_L
        log_K = (1.0 / softness**2) * top_vals * scale_factor
        
        # 4. Filter Logic (Re-calculate norms for Top-K only)
        # Gather HKL vectors: (Batch, N_obs, K, 3)
        hkl_selected = jnp.take(hkl_pool.T, top_idxs, axis=0) 
        
        # Compute q_selected = UB @ hkl_selected
        # (Batch, 3, 3) @ (Batch, N_obs, K, 3) -> (Batch, N_obs, K, 3)
        hkl_flat_sel = hkl_selected.reshape(batch_size, -1, 3)
        q_flat_sel = jnp.einsum("sij,smj->smi", UB, hkl_flat_sel)
        q_selected = q_flat_sel.reshape(batch_size, n_obs, top_k, 3)
        
        q_sq_selected = jnp.sum(q_selected**2, axis=3)
        
        # Lambda & Resolution Checks
        lambda_sparse = jnp.sqrt(k_sq_obs)[:, :, None] / (jnp.sqrt(q_sq_selected) + 1e-9)
        mask_lambda = (lambda_sparse >= self.wl_min_val) & (lambda_sparse <= self.wl_max_val)
        
        d_sparse = 1.0 / jnp.sqrt(q_sq_selected + 1e-9)
        mask_res = (d_sparse >= self.d_min) & (d_sparse <= self.d_max)
        
        valid_mask = mask_lambda & mask_res
        log_K = jnp.where(valid_mask, log_K, -100.0)

        # 5. Outlier (Dustbin) & Softmax
        log_K_dustbin = jnp.full((batch_size, n_obs, 1), -5.0)
        log_K_extended = jnp.concatenate([log_K, log_K_dustbin], axis=2)
        
        # Row Normalization (Softmax)
        log_P = log_K_extended - jax.nn.logsumexp(log_K_extended, axis=2, keepdims=True)
        P = jnp.exp(log_P)
        
        # 6. Score
        P_match = P[:, :, :-1]
        log_K_match = log_K 
        
        matched_scores = jnp.sum(P_match * log_K_match, axis=2)
        weighted_score = jnp.sum(self.weights * matched_scores, axis=1)

        # Metrics
        probs = jnp.max(P_match, axis=2)
        best_k_idx = jnp.argmax(P_match, axis=2)
        best_hkl_idx = jnp.take_along_axis(top_idxs, best_k_idx[:, :, None], axis=2).squeeze(2)
        best_hkl = jnp.take(hkl_pool.T, best_hkl_idx, axis=0)
        best_lamb = jnp.take_along_axis(lambda_sparse, best_k_idx[:, :, None], axis=2).squeeze(2)

        return -weighted_score, probs, best_hkl, best_lamb


    @partial(jax.jit, static_argnames='self')
    def __call__(self, x):
        UB, _, sample_total, ki_vec, _, R = self._get_physical_params_jax(x)
        
        if self.refine_sample:
            s = sample_total[:, :, None]
            p = self.peak_xyz[None, :, :]
            v = p - s
            dist = jnp.sqrt(jnp.sum(v**2, axis=1, keepdims=True))
            kf = v / dist
            ki = ki_vec[:, :, None]
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)
        else:
            kf = self.kf_lab_fixed[None, :, :].repeat(x.shape[0], axis=0)
            ki = ki_vec[:, :, None]
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)

        if R is not None:
            if R.ndim == 4: kf_ki_vec = jnp.einsum("smji,sjm->sim", R, q_lab)
            else: kf_ki_vec = jnp.einsum("sji,sjm->sim", R, q_lab)
        else:
            kf_ki_vec = q_lab

        if self.loss_method == 'forward':
            score, _, _, _ = self.indexer_dynamic_binary_jax(UB, kf_ki_vec, k_sq_override=k_sq_dyn, softness=self.softness, window_batch_size=self.window_batch_size)
        elif self.loss_method == 'cosine':
            score, _, _, _ = self.indexer_dynamic_cosine_aniso_jax(UB, kf_ki_vec, k_sq_override=k_sq_dyn, softness=self.softness)
        elif self.loss_method == 'sinkhorn':
            score, _, _, _ = self.indexer_sinkhorn_jax(UB, kf_ki_vec, k_sq_override=k_sq_dyn, softness=self.softness,
                                                       chunk_size=self.chunk_size, num_iters=self.num_iters, top_k=self.top_k)
        else:
            score, _, _, _ = self.indexer_dynamic_soft_jax(UB, kf_ki_vec, k_sq_override=k_sq_dyn, softness=self.softness)

        return score

# ==============================================================================
# 3. MAIN CLASS
# ==============================================================================

class FindUB:
    def __init__(self, filename=None, data=None):
        self.goniometer_axes = None
        self.goniometer_angles = None
        self.goniometer_offsets = None 
        self.goniometer_names = None 
        self.sample_offset = None
        self.peak_xyz = None
        self.ki_vec = None
        self.base_sample_offset = np.zeros(3)
        self.base_gonio_offset = None 

        if filename is not None:
            self.load_peaks(filename)
        elif data is not None:
            self.load_from_dict(data)

        t = np.linspace(0, np.pi, 1024)
        cdf = (t - np.sin(t)) / np.pi
        self._angle_cdf = cdf
        self._angle_t = t
        self._angle = scipy.interpolate.interp1d(cdf, t, kind="linear")

    def load_from_dict(self, data):
        """Load data from a dictionary instead of a file."""
        self.a = data["sample/a"]
        self.b = data["sample/b"]
        self.c = data["sample/c"]
        self.alpha = data["sample/alpha"]
        self.beta = data["sample/beta"]
        self.gamma = data["sample/gamma"]
        self.wavelength = data["instrument/wavelength"]
        self.R = data["goniometer/R"]
        self.two_theta = data["peaks/two_theta"]
        self.az_phi = data["peaks/azimuthal"]
        self.intensity = data["peaks/intensity"]
        self.sigma_intensity = data["peaks/sigma"]
        self.radii = data["peaks/radius"]
        
        # Handle bytes vs string for Space Group
        sg = data["sample/space_group"]
        if isinstance(sg, bytes):
            self.space_group = sg.decode("utf-8")
        else:
            self.space_group = str(sg)

        if "peaks/xyz" in data: self.peak_xyz = data["peaks/xyz"]
        if "goniometer/axes" in data: self.goniometer_axes = data["goniometer/axes"]
        if "goniometer/angles" in data: self.goniometer_angles = data["goniometer/angles"]
        
        if "goniometer/names" in data:
            names = data["goniometer/names"]
            # Handle list of bytes vs list of strings
            self.goniometer_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in names]
            
        if "beam/ki_vec" in data: self.ki_vec = data["beam/ki_vec"]
        else: self.ki_vec = np.array([0., 0., 1.])

    def load_peaks(self, filename):
        with h5py.File(os.path.abspath(filename), "r") as f:
            # Create a dict from the file content and reuse logic
            data = {}
            data["sample/a"] = f["sample/a"][()]
            data["sample/b"] = f["sample/b"][()]
            data["sample/c"] = f["sample/c"][()]
            data["sample/alpha"] = f["sample/alpha"][()]
            data["sample/beta"] = f["sample/beta"][()]
            data["sample/gamma"] = f["sample/gamma"][()]
            data["instrument/wavelength"] = f["instrument/wavelength"][()]
            data["goniometer/R"] = f["goniometer/R"][()]
            data["peaks/two_theta"] = f["peaks/two_theta"][()]
            data["peaks/azimuthal"] = f["peaks/azimuthal"][()]
            data["peaks/intensity"] = f["peaks/intensity"][()]
            data["peaks/sigma"] = f["peaks/sigma"][()]
            data["peaks/radius"] = f["peaks/radius"][()]
            data["sample/space_group"] = f["sample/space_group"][()]
            
            if "peaks/xyz" in f: data["peaks/xyz"] = f["peaks/xyz"][()]
            if "goniometer/axes" in f: data["goniometer/axes"] = f["goniometer/axes"][()]
            if "goniometer/angles" in f: data["goniometer/angles"] = f["goniometer/angles"][()]
            if "goniometer/names" in f: data["goniometer/names"] = f["goniometer/names"][()]
            if "beam/ki_vec" in f: data["beam/ki_vec"] = f["beam/ki_vec"][()]
            
            self.load_from_dict(data)

    def reciprocal_lattice_B(self):
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
        return scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    def get_bootstrap_params(self, bootstrap_filename, 
                             refine_lattice=False, lattice_bound_frac=0.05,
                             refine_sample=False, sample_bound_meters=0.002,
                             refine_beam=False, beam_bound_deg=1.0,
                             refine_goniometer=False, goniometer_bound_deg=5.0, 
                             refine_goniometer_axes=None):
        print(f"Bootstrapping from physical solution: {bootstrap_filename}")
        
        with h5py.File(bootstrap_filename, "r") as f:
            raw_x = None
            if "optimization/best_params" in f: raw_x = f["optimization/best_params"][()]
            b_a = f["sample/a"][()]
            b_b = f["sample/b"][()]
            b_c = f["sample/c"][()]
            b_alpha = f["sample/alpha"][()]
            b_beta = f["sample/beta"][()]
            b_gamma = f["sample/gamma"][()]
            b_offset = f["sample/offset"][()] if "sample/offset" in f else np.zeros(3)
            b_ki = f["beam/ki_vec"][()] if "beam/ki_vec" in f else np.array([0., 0., 1.])
            b_gonio_offsets = None
            if "optimization/goniometer_offsets" in f: b_gonio_offsets = f["optimization/goniometer_offsets"][()]

        new_params = []
        if raw_x is not None: new_params.append(raw_x[:3])
        else: new_params.append(np.zeros(3))

        if refine_lattice:
            self.a, self.b, self.c = b_a, b_b, b_c
            self.alpha, self.beta, self.gamma = b_alpha, b_beta, b_gamma
            print(f"  > Recentered Lattice: {self.a:.2f}, {self.b:.2f}, {self.c:.2f}...")
            lat_sys, _ = get_lattice_system(self.a, self.b, self.c, self.alpha, self.beta, self.gamma, get_centering(self.space_group))
            active_indices = _get_active_lattice_indices(lat_sys)
            new_params.append(np.full(len(active_indices), 0.5))

        if refine_sample:
            if self.peak_xyz is not None:
                print(f"  > Setting Base Sample Offset: {b_offset}")
                self.base_sample_offset = b_offset # Store only
                new_params.append(np.full(3, 0.5)) 
            else:
                new_params.append(np.full(3, 0.5))

        if refine_beam:
            print(f"  > Recentered Beam Vector: {b_ki}")
            self.ki_vec = b_ki 
            new_params.append(np.full(2, 0.5))

        if refine_goniometer:
            active_mask = []
            if refine_goniometer_axes is not None and self.goniometer_names is not None:
                for name in self.goniometer_names:
                    is_active = any(req in name for req in refine_goniometer_axes)
                    active_mask.append(is_active)
            else:
                active_mask = [True] * len(self.goniometer_axes)
            
            if b_gonio_offsets is not None:
                print(f"  > Setting Base Goniometer Offsets: {b_gonio_offsets}")
                self.base_gonio_offset = b_gonio_offsets # Store only
                n_active = sum(active_mask)
                new_params.append(np.full(n_active, 0.5))
            else:
                self.base_gonio_offset = np.zeros(len(self.goniometer_axes))
                n_active = sum(active_mask)
                new_params.append(np.full(n_active, 0.5))

        return np.concatenate([np.atleast_1d(p) for p in new_params])

    def minimize(
        self,
        strategy_name: str,
        population_size: int = 1000,
        num_generations: int = 100,
        n_runs: int = 1,
        seed: int = 0,
        softness: float = 0.01,
        loss_method: str = 'gaussian',
        init_params: np.ndarray = None,
        refine_lattice: bool = False,
        lattice_bound_frac: float = 0.05,
        goniometer_axes: list = None,
        goniometer_angles: np.ndarray = None,
        refine_goniometer: bool = False,
        goniometer_bound_deg: float = 5.0,
        goniometer_names: list = None,
        refine_goniometer_axes: list = None,
        refine_sample: bool = False,
        sample_bound_meters: float = 2.0,
        refine_beam: bool = False,
        beam_bound_deg: float = 1.0,
        d_min: float = None,
        d_max: float = None,
        hkl_search_range: int = 20,
        search_window_size: int = 256,
        window_batch_size: int = 32,
        chunk_size: int = 2048,
        num_iters: int = 20,
        top_k: int = 32,
        batch_size: int = None,
        sigma_init: float = None,
        B_sharpen: float = 50,
    ):
        if goniometer_axes is None and self.goniometer_axes is not None:
             goniometer_axes = self.goniometer_axes
        if goniometer_angles is None and self.goniometer_angles is not None:
             goniometer_angles = self.goniometer_angles.T
        if goniometer_names is None and self.goniometer_names is not None:
             goniometer_names = self.goniometer_names

        kf_ki_dir_lab = scattering_vector_from_angles(self.two_theta, self.az_phi)
        num_obs = kf_ki_dir_lab.shape[1]
        
        if goniometer_axes is not None:
            kf_ki_input = kf_ki_dir_lab
        else:
            kf_ki_input = np.einsum("mji,jm->im", self.R, kf_ki_dir_lab)

        goniometer_refine_mask = None
        if refine_goniometer and refine_goniometer_axes is not None:
            if self.goniometer_names is None:
                print("Warning: refine_goniometer_axes provided but goniometer_names not found. Refining ALL.")
            else:
                mask = []
                print(f"Refining specific goniometer axes: {refine_goniometer_axes}")
                for name in self.goniometer_names:
                    should_refine = any(req in name for req in refine_goniometer_axes)
                    mask.append(should_refine)
                goniometer_refine_mask = np.array(mask, dtype=bool)
                print(f"Goniometer Mask: {goniometer_refine_mask} (Names: {self.goniometer_names})")

        snr = self.intensity / (self.sigma_intensity + 1e-6)

        if B_sharpen is not None:
            theta_rad = np.deg2rad(self.two_theta) / 2.0
            sin_sq_theta = np.sin(theta_rad)**2
            wilson_correction = np.exp(B_sharpen * sin_sq_theta)
            weights = snr * wilson_correction
            weights = weights / np.mean(weights)
        else:
            weights = snr

        weights = np.clip(weights, 0, 10.0)

        cell_params_init = np.array([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])
        lattice_system, num_lattice_params = get_lattice_system(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.space_group,
        )
        
        if refine_lattice:
            print(f"Lattice Refinement Enabled.")
            print(f"Detected System: {lattice_system} ({num_lattice_params} free parameters).")

        if loss_method == "forward" and (d_min is None or d_max is None):
            raise ValueError(f"Need to supply --d_min and --d_max for loss_method=='forward'")

        objective = VectorizedObjective(
            self.reciprocal_lattice_B(),
            kf_ki_input,
            self.peak_xyz,
            np.array(self.wavelength),
            self._angle_cdf,
            self._angle_t,
            weights=weights,
            softness=softness,
            space_group=self.space_group,
            loss_method=loss_method,
            cell_params=cell_params_init,
            peak_radii=self.radii,
            refine_lattice=refine_lattice,
            lattice_bound_frac=lattice_bound_frac,
            lattice_system=lattice_system,
            goniometer_axes=goniometer_axes,
            goniometer_angles=goniometer_angles,
            refine_goniometer=refine_goniometer,
            goniometer_refine_mask=goniometer_refine_mask,
            goniometer_nominal_offsets=self.base_gonio_offset,
            refine_sample=refine_sample,
            sample_bound_meters=sample_bound_meters,
            sample_nominal=self.base_sample_offset,
            refine_beam=refine_beam,
            beam_bound_deg=beam_bound_deg,
            beam_nominal=self.ki_vec, 
            goniometer_bound_deg=goniometer_bound_deg,
            hkl_search_range=hkl_search_range,
            d_min=d_min,
            d_max=d_max,
            window_batch_size=window_batch_size,
            search_window_size=search_window_size,
            chunk_size=chunk_size,
            num_iters=num_iters,
            top_k=top_k,
        )
        print(f"Objective initialized with {loss_method} loss. Softness: {softness}")

        num_dims = 3
        if refine_lattice: num_dims += num_lattice_params
        if refine_sample:
            if self.peak_xyz is None: refine_sample = False
            else: num_dims += 3
        if refine_beam:
            if self.peak_xyz is None: refine_sample = False
            else: num_dims += 2
        if refine_goniometer:
            if goniometer_refine_mask is not None: num_dims += np.sum(goniometer_refine_mask)
            else: num_dims += len(goniometer_axes)

        start_sol_processed = None
        if init_params is not None:
            start_sol = jnp.array(init_params)
            if start_sol.shape[0] != num_dims:
                if start_sol.shape[0] < num_dims:
                    n_new = num_dims - start_sol.shape[0]
                    start_sol_processed = jnp.concatenate([start_sol, jnp.full((n_new,), 0.5)])
                else:
                    start_sol_processed = start_sol[:num_dims]
            else:
                start_sol_processed = start_sol

        sample_solution = jnp.zeros(num_dims)
        target_sigma = sigma_init if sigma_init else (0.01 if start_sol_processed is not None else 3.14)
        print(f"Strategy: {strategy_name.upper()} | Target Sigma: {target_sigma}")

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

        es_params = strategy.default_params
        
        def init_single_run(rng, start_sol):
            rng, rng_pop, rng_init = jax.random.split(rng, 3)
            
            if start_sol is not None:
                if strategy_type == 'population_based':
                    noise = jax.random.normal(rng_pop, (population_size, num_dims)) * 0.05
                    p_orient = start_sol[:3] + noise[:, :3]
                    p_rest = jnp.clip(start_sol[3:] + noise[:, 3:], 0.0, 1.0)
                    population_init = jnp.concatenate([p_orient, p_rest], axis=1)
                    fitness_init = objective(population_init)
                    state = strategy.init(rng_init, population_init, fitness_init, es_params)
                else:
                    state = strategy.init(rng_init, start_sol, es_params)
                    state = state.replace(std=target_sigma)
            else:
                if strategy_type == 'population_based':
                    pop_orient = jax.random.normal(rng_pop, (population_size, 3)) * target_sigma
                    rng_rest, _ = jax.random.split(rng_pop)
                    pop_rest = jax.random.uniform(rng_rest, (population_size, max(0, num_dims-3)))
                    population_init = jnp.concatenate([pop_orient, pop_rest], axis=1)
                    fitness_init = objective(population_init)
                    state = strategy.init(rng_init, population_init, fitness_init, es_params)
                else:
                    mean_orient = jnp.zeros(3)
                    mean_rest = jnp.full((max(0, num_dims-3),), 0.5)
                    solution_init = jnp.concatenate([mean_orient, mean_rest])
                    state = strategy.init(rng_init, solution_init, es_params)
                    state = state.replace(std=target_sigma)
            return state

        mesh = Mesh(np.array(jax.devices()), ('i'))

        def step_single_run(rng, state):
            rng, rng_ask, rng_tell = jax.random.split(rng, 3)
            x, state_ask = strategy.ask(rng_ask, state, es_params)
            x_orient = x[:, :3]
            x_rest = jnp.clip(x[:, 3:], 0.0, 1.0)
            x_valid = jnp.concatenate([x_orient, x_rest], axis=1)
            fitness = objective(x_valid)

            # parallelize population across GPUs
            x_valid = jax.lax.with_sharding_constraint(x_valid, NamedSharding(mesh, P('i')))

            state_tell, metrics = strategy.tell(rng_tell, x_valid, fitness, state_ask, es_params)
            return rng, state_tell, metrics

        init_batch_jit = jax.jit(jax.vmap(init_single_run, in_axes=(0, None)))
        step_batch_jit = jax.jit(jax.vmap(step_single_run, in_axes=(0, 0)))

        exec_batch_size = batch_size if batch_size is not None else n_runs
        print(f"\n--- Starting {n_runs} Runs (Batch Size: {exec_batch_size}) ---")
        
        seeds = jnp.arange(seed, seed + n_runs)
        all_keys = jax.vmap(jax.random.PRNGKey)(seeds)
        
        batch_keys_list = []
        batch_states_list = []
        
        for b_i in range(int(np.ceil(n_runs / exec_batch_size))):
            start_idx = b_i * exec_batch_size
            end_idx = min((b_i + 1) * exec_batch_size, n_runs)
            b_keys = all_keys[start_idx:end_idx]
            b_state = init_batch_jit(b_keys, start_sol_processed)
            batch_keys_list.append(b_keys)
            batch_states_list.append(b_state)

        pbar = range(num_generations)
        if trange is not None:
            pbar = trange(num_generations, desc="Optimizing")

        for gen in pbar:
            current_gen_best = np.inf
            for b_i in range(len(batch_keys_list)):
                curr_keys = batch_keys_list[b_i]
                curr_state = batch_states_list[b_i]
                next_keys, next_state, _ = step_batch_jit(curr_keys, curr_state)
                batch_keys_list[b_i] = next_keys
                batch_states_list[b_i] = next_state
                b_min = jnp.min(next_state.best_fitness)
                if b_min < current_gen_best:
                    current_gen_best = b_min
            if trange is not None:
                pbar.set_description(f"Gen {gen+1} | Best: {-current_gen_best:.1f}/{num_obs}")

        all_fitness_list = []
        all_solutions_list = []
        for b_state in batch_states_list:
            all_fitness_list.append(b_state.best_fitness)
            all_solutions_list.append(b_state.best_solution)
            
        all_fitness = jnp.concatenate(all_fitness_list, axis=0)
        all_solutions = jnp.concatenate(all_solutions_list, axis=0)

        best_idx = np.argmin(all_fitness)
        best_overall_fitness = all_fitness[best_idx]
        best_overall_member = all_solutions[best_idx]
        
        print(f"\n--- Optimization Complete ---")
        print(f"Best overall peaks: {-best_overall_fitness:.2f} (from Run {best_idx+1})")
        
        self.x = np.array(best_overall_member)
        
        x_batch = jnp.array(self.x[None, :])
        UB_final_batch, B_new_batch, s_total_batch, ki_vec_batch, offsets_total_batch, R_batch = objective._get_physical_params_jax(x_batch)
        
        UB_final = np.array(UB_final_batch[0])
        self.sample_offset = np.array(s_total_batch[0])
        self.ki_vec = np.array(ki_vec_batch[0]).flatten()
        if offsets_total_batch is not None: self.goniometer_offsets = np.array(offsets_total_batch[0])
        if R_batch is not None: self.R = np.array(R_batch[0])

        idx = 0
        rot_params = self.x[idx:idx+3]
        idx += 3
        U = objective.orientation_U_jax(rot_params[None])[0]

        if refine_lattice:
            print("--- Refined Lattice Parameters ---")
            idx_lat = 3
            cell_norm = jnp.array(self.x[None, idx_lat:idx_lat+num_lattice_params])
            p_full = np.array(objective.reconstruct_cell_params(cell_norm)[0])
            print(f"a: {p_full[0]:.4f}, b: {p_full[1]:.4f}, c: {p_full[2]:.4f}")
            print(f"alpha: {p_full[3]:.4f}, beta: {p_full[4]:.4f}, gamma: {p_full[5]:.4f}")
            self.a, self.b, self.c = p_full[0], p_full[1], p_full[2]
            self.alpha, self.beta, self.gamma = p_full[3], p_full[4], p_full[5]

        if refine_sample:
             print(f"--- Refined Sample Offset (mm) ---")
             print(f"X: {1000*self.sample_offset[0]:.4f}, Y: {1000*self.sample_offset[1]:.4f}, Z: {1000*self.sample_offset[2]:.4f}")

        if refine_beam:
            print("-- Refined Beam Direction ---")
            print(f"(ki_x, ki_y, ki_z): ({self.ki_vec[0]:.3f}, {self.ki_vec[1]:.3f}, {self.ki_vec[2]:.3f})")

        if self.goniometer_offsets is not None:
            print("--- Refined Goniometer Offsets (deg) ---")
            if goniometer_names is not None:
                for name, val in zip(goniometer_names, self.goniometer_offsets):
                    print(f"{name}: {val:.4f}")
            else:
                print(self.goniometer_offsets)

        # Final Score Recalculation
        if refine_sample:
            s = self.sample_offset
            p = self.peak_xyz.T + self.base_sample_offset[:, None] # Reconstruct P_raw
            v = p - s[:, None]
            dist = np.linalg.norm(v, axis=0)
            kf = v / dist
            ki = self.ki_vec[:, None]
            q_lab = kf - ki
            k_sq_dyn = np.sum(q_lab**2, axis=0)[None, :]
        else:
            # FIX: Use kf_lab_fixed (Ideal Beam) + refined ki
            kf_fixed = objective.kf_lab_fixed 
            ki = self.ki_vec[:, None]
            q_lab = kf_fixed - ki
            k_sq_dyn = np.sum(q_lab**2, axis=0)[None, :]

        if self.R.ndim == 3: kf_ki_vec = np.einsum("mji,jm->im", self.R, q_lab)
        else: kf_ki_vec = np.einsum("ji,jm->im", self.R, q_lab)

        UB_final = np.array(UB_final)

        if loss_method == 'forward':
            score, accum_probs, hkl, lamb = objective.indexer_dynamic_binary_jax(UB_final[None], kf_ki_vec[None], k_sq_override=k_sq_dyn, softness=softness, window_batch_size=window_batch_size)
        elif loss_method == 'cosine':
            score, accum_probs, hkl, lamb = objective.indexer_dynamic_cosine_aniso_jax(UB_final[None], kf_ki_vec[None], softness=softness, k_sq_override=k_sq_dyn)
        elif loss_method == 'sinkhorn':
            score, accum_probs, hkl, lamb = objective.indexer_sinkhorn_jax(UB_final[None], kf_ki_vec[None], softness=softness, k_sq_override=k_sq_dyn, chunk_size=chunk_size, num_iters=num_iters, top_k=top_k)
        else:
            score, accum_probs, hkl, lamb = objective.indexer_dynamic_soft_jax(UB_final[None], kf_ki_vec[None], softness=softness, k_sq_override=k_sq_dyn)

        num_peaks_soft = float(np.sum(accum_probs[0]))
        print(f"Final Solution indexed {num_peaks_soft:.2f}/{num_obs} peaks (unweighted count).")

        return num_peaks_soft, hkl[0], lamb[0], U
