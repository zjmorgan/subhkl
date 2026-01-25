import os
from functools import partial

import h5py
import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.interpolate

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jscipy_linalg

from evosax.algorithms import DifferentialEvolution, PSO, CMA_ES

from subhkl.detector import scattering_vector_from_angles
from subhkl.spacegroup import generate_hkl_mask, get_centering

try:
    from tqdm import trange
except ImportError:
    trange = None


def get_lattice_system(a, b, c, alpha, beta, gamma, centering, atol_len=0.05, atol_ang=0.5):
    is_90 = lambda x: np.isclose(x, 90.0, atol=atol_ang)
    is_120 = lambda x: np.isclose(x, 120.0, atol=atol_ang)
    eq = lambda x, y: np.isclose(x, y, atol=atol_len)
    
    if centering == 'R':
        if is_90(alpha) and is_90(beta) and is_120(gamma) and eq(a, b):
            return 'Hexagonal', 2 
        elif eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma):
            return 'Rhombohedral', 2 
            
    if centering == 'H':
        return 'Hexagonal', 2

    if is_90(alpha) and is_90(beta) and is_90(gamma):
        if eq(a, b) and eq(b, c):
            return 'Cubic', 1  
        elif eq(a, b):
            return 'Tetragonal', 2 
        elif eq(a, c) or eq(b, c):
            return 'Orthorhombic', 3 
        else:
            return 'Orthorhombic', 3 

    elif is_90(alpha) and is_90(beta) and is_120(gamma):
        if eq(a, b):
            return 'Hexagonal', 2 
        
    elif eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma):
        return 'Rhombohedral', 2 

    if is_90(alpha) and is_90(gamma) and not is_90(beta):
        return 'Monoclinic', 4 
    
    return 'Triclinic', 6


def rotation_matrix_from_axis_angle_jax(axis, angle_rad):
    u = axis / jnp.linalg.norm(axis)
    ux, uy, uz = u
    K = jnp.array([
        [0.0, -uz, uy],
        [uz, 0.0, -ux],
        [-uy, ux, 0.0]
    ]) 
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    t = 1.0 - c
    eye = jnp.eye(3)
    R = eye + s[..., None, None] * K + t[..., None, None] * (K @ K)
    return R

def rotation_matrix_from_rodrigues_jax(w):
    """
    Maps a 3D parameter vector (Rodrigues/Rotation vector) to a 3x3 Rotation Matrix.
    Bijective mapping from R^3 to SO(3).
    """
    theta = jnp.linalg.norm(w) + 1e-9 # Avoid div by zero
    k = w / theta
    K = jnp.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0]
    ])
    # Rodrigues formula: I + sin(t)K + (1-cos(t))K^2
    I = jnp.eye(3)
    R = I + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
    return R

class VectorizedObjectiveJAX:
    """
    JAX-compatible vectorized objective function for evosax.
    """
    def __init__(self, B, kf_ki_dir, peak_xyz_lab, wavelength, angle_cdf, angle_t, weights=None, softness=0.01,
                 cell_params=None, refine_lattice=False, lattice_bound_frac=0.05, lattice_system='Triclinic',
                 goniometer_axes=None, goniometer_angles=None, refine_goniometer=False, goniometer_bound_deg=5.0,
                 goniometer_refine_mask=None, 
                 refine_sample=False, sample_bound_meters=0.002,
                 peak_radii=None, loss_method='gaussian', 
                 hkl_search_range=15, d_min=5.0, d_max=100.0, search_window_size=256, window_batch_size=32,
                 space_group="P 1"):
        
        self.B = jnp.array(B)
        
        # Pre-calculated vectors (used if sample refinement is OFF)
        self.kf_ki_dir_init = jnp.array(kf_ki_dir)
        self.k_sq_init = jnp.sum(self.kf_ki_dir_init**2, axis=0)
        
        # Peak Coordinates (used if sample refinement is ON)
        # xyz is (N, 3) -> transpose to (3, N)
        if peak_xyz_lab is not None:
            self.peak_xyz = jnp.array(peak_xyz_lab.T) 
        else:
            self.peak_xyz = None

        self.refine_sample = refine_sample
        self.sample_bound = sample_bound_meters

        self.softness = softness
        self.loss_method = loss_method
        self.angle_cdf = jnp.array(angle_cdf)
        self.angle_t = jnp.array(angle_t)
       
        self.space_group = space_group

        self.refine_lattice = refine_lattice
        self.lattice_system = lattice_system
        self.refine_goniometer = refine_goniometer
        self.goniometer_bound_deg = goniometer_bound_deg

        # --- Lattice Refinement Setup ---
        if self.refine_lattice:
            if cell_params is None:
                raise ValueError("cell_params must be provided if refine_lattice is True")
            self.cell_init = jnp.array(cell_params) 
            
            if self.lattice_system == 'Cubic': self.free_params_init = self.cell_init[0:1]
            elif self.lattice_system == 'Hexagonal': self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[2]])
            elif self.lattice_system == 'Tetragonal': self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[2]])
            elif self.lattice_system == 'Rhombohedral': self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[3]])
            elif self.lattice_system == 'Orthorhombic': self.free_params_init = self.cell_init[0:3]
            elif self.lattice_system == 'Monoclinic': self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[1], self.cell_init[2], self.cell_init[4]])
            else: self.free_params_init = self.cell_init

            delta = jnp.abs(self.free_params_init) * lattice_bound_frac
            self.lat_min = self.free_params_init - delta
            self.lat_max = self.free_params_init + delta

        if self.refine_goniometer:
            self.gonio_axes = jnp.array(goniometer_axes) 
            self.gonio_angles = jnp.array(goniometer_angles) 
            self.num_gonio_axes = self.gonio_axes.shape[0]
            self.gonio_min = jnp.full(self.num_gonio_axes, -goniometer_bound_deg)
            self.gonio_max = jnp.full(self.num_gonio_axes, goniometer_bound_deg)

            if goniometer_refine_mask is not None:
                self.gonio_mask = np.array(goniometer_refine_mask, dtype=bool)
            else:
                self.gonio_mask = np.ones(self.num_gonio_axes, dtype=bool)

            self.num_active_gonio = np.sum(self.gonio_mask)

        wavelength = jnp.array(wavelength)
        self.wl_min_val = wavelength[0]
        self.wl_max_val = wavelength[1]
        self.num_candidates = 64

        if weights is None: self.weights = jnp.ones(self.kf_ki_dir_init.shape[1])
        else: self.weights = jnp.array(weights)
            
        if peak_radii is None: self.peak_radii = jnp.zeros(self.kf_ki_dir_init.shape[1])
        else: self.peak_radii = jnp.array(peak_radii)

        self.max_score = jnp.sum(self.weights)

        # --- Forward Search / Binary Search Setup ---
        self.d_min = d_min
        self.d_max = d_max
        self.search_window_size = search_window_size
        self.window_batch_size = window_batch_size

        # 1. Generate Static HKL Pool
        r = jnp.arange(-hkl_search_range, hkl_search_range + 1)
        h, k, l = jnp.meshgrid(r, r, r, indexing='ij')
        hkl_pool = jnp.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)
        zero_mask = ~jnp.all(hkl_pool == 0, axis=0)
        hkl_pool = hkl_pool[:, zero_mask]

        # 2. Convert Pool to Spherical Coordinates (Crystal Frame)
        q_cart = self.B @ hkl_pool 
        phis = jnp.arctan2(q_cart[1], q_cart[0])
        sort_idx = jnp.argsort(phis)
        self.pool_phi_sorted = phis[sort_idx]
        self.pool_hkl_sorted = hkl_pool[:, sort_idx] 

        # --- Generate Space Group Mask ---
        # Mask covers range [-range, +range]
        # Size is (2*range+1)^3
        self.mask_range = hkl_search_range
        print(f"Generating HKL mask for Space Group: {self.space_group} (Range: +/-{self.mask_range})")

        # Generate on CPU via gemmi
        mask_cpu = generate_hkl_mask(self.mask_range, self.mask_range, self.mask_range, self.space_group)

        # Move to JAX/GPU
        self.valid_hkl_mask = jnp.array(mask_cpu)

    def reconstruct_cell_params(self, params_norm):
        p_free = self.lat_min + params_norm * (self.lat_max - self.lat_min)
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
        offsets = self.gonio_min + gonio_offsets_norm * (self.gonio_max - self.gonio_min) 
        angles_deg = offsets[:, :, None] + self.gonio_angles[None, :, :]
        S, M = offsets.shape[0], self.gonio_angles.shape[1]
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
            h, k, l = hkl_int[:, 0, :], hkl_int[:, 1, :], hkl_int[:, 2, :]

            r = self.mask_range
            in_bounds = (h >= -r) & (h <= r) & \
                        (k >= -r) & (k <= r) & \
                        (l >= -r) & (l <= r)

            # 2. Lookup in Mask
            # Shift indices: -r maps to 0, 0 maps to r, +r maps to 2r
            idx_h = h + r
            idx_k = k + r
            idx_l = l + r

            # Use 'clip' to be safe for lookup, but 'in_bounds' ensures we ignore invalid ones
            safe_h = jnp.clip(idx_h, 0, 2*r).astype(jnp.int32)
            safe_k = jnp.clip(idx_k, 0, 2*r).astype(jnp.int32)
            safe_l = jnp.clip(idx_l, 0, 2*r).astype(jnp.int32)

            # Lookup: (S, M)
            is_allowed = self.valid_hkl_mask[safe_h, safe_k, safe_l]
            valid_sym = in_bounds & is_allowed

            prob = jnp.where(valid_cand & valid_sym, prob, 0.0)
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
            valid_cand = (lamda_cand >= self.wl_min_val) & (lamda_cand <= self.wl_max_val)
            hkl_int = jnp.round(hkl_float).astype(jnp.int32)
            h, k, l = hkl_int[:, 0, :], hkl_int[:, 1, :], hkl_int[:, 2, :]

            r = self.mask_range
            in_bounds = (h >= -r) & (h <= r) & \
                        (k >= -r) & (k <= r) & \
                        (l >= -r) & (l <= r)

            # 2. Lookup in Mask
            # Shift indices: -r maps to 0, 0 maps to r, +r maps to 2r
            idx_h = h + r
            idx_k = k + r
            idx_l = l + r

            # Use 'clip' to be safe for lookup, but 'in_bounds' ensures we ignore invalid ones
            safe_h = jnp.clip(idx_h, 0, 2*r).astype(jnp.int32)
            safe_k = jnp.clip(idx_k, 0, 2*r).astype(jnp.int32)
            safe_l = jnp.clip(idx_l, 0, 2*r).astype(jnp.int32)

            # Lookup: (S, M)
            is_allowed = self.valid_hkl_mask[safe_h, safe_k, safe_l]

            valid_sym = in_bounds & is_allowed

            prob = jnp.where(valid_cand & valid_sym, prob, 0.0)
            new_sum = curr_sum + prob
            score_tracked = jnp.where(valid_cand & valid_sym, log_prob, -1e9)
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
        remainder = raw_offsets.shape[0] % window_batch_size
        if remainder != 0:
            pad_len = window_batch_size - remainder
            offsets_padded = jnp.pad(raw_offsets, (0, pad_len), constant_values=raw_offsets[-1])
        else:
            offsets_padded = raw_offsets
        num_batches = offsets_padded.shape[0] // window_batch_size
        offset_batches = offsets_padded.reshape(num_batches, window_batch_size)

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
            d_spacings = 2 * jnp.pi / jnp.sqrt(q_sq + 1e-9)
            valid_res = (d_spacings >= self.d_min) & (d_spacings <= self.d_max)
            valid_mask = valid_lamb & valid_res

            h, k, l = hkl_cands[..., 0], hkl_cands[..., 1], hkl_cands[..., 2]
            # 2. Lookup in Mask
            # Shift indices: -r maps to 0, 0 maps to r, +r maps to 2r
            r = self.mask_range
            idx_h = h + r
            idx_k = k + r
            idx_l = l + r

            # Use 'clip' to be safe for lookup, but 'in_bounds' ensures we ignore invalid ones
            safe_h = jnp.clip(idx_h, 0, 2*r).astype(jnp.int32)
            safe_k = jnp.clip(idx_k, 0, 2*r).astype(jnp.int32)
            safe_l = jnp.clip(idx_l, 0, 2*r).astype(jnp.int32)

            # Lookup: (S, M, W)
            valid_sym = self.valid_hkl_mask[safe_h, safe_k, safe_l]

            q_obs_opt = k_obs / jnp.where(lambda_opt==0, 1.0, lambda_opt)[..., None]
            diff = q_obs_opt - q_pred
            dist_sq = jnp.sum(diff**2, axis=3)
            dist_sq_masked = jnp.where(valid_mask & valid_sym, dist_sq, 1e9)
            
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
        return score, probs, best_hkl.transpose((0, 2, 1)), best_lamb

    @partial(jax.jit, static_argnames='self')
    def __call__(self, x):
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
            UB = jnp.einsum("sij,jk->sik", U, self.B)

        sample_offset = jnp.zeros((x.shape[0], 3))
        if self.refine_sample:
            if self.peak_xyz is not None:
                s_norm = x[:, idx:idx+3]
                idx += 3
                sample_offset = (s_norm - 0.5) * 2.0 * self.sample_bound
        
        if self.refine_sample:
            s = sample_offset[:, :, None]
            p = self.peak_xyz[None, :, :]
            v = p - s 
            dist = jnp.sqrt(jnp.sum(v**2, axis=1, keepdims=True))
            kf = v / dist
            ki = jnp.array([0.0, 0.0, 1.0])[None, :, None]
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1) 
        else:
            q_lab = self.kf_ki_dir_init[None, ...].repeat(x.shape[0], axis=0)
            k_sq_dyn = self.k_sq_init[None, :].repeat(x.shape[0], axis=0)

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

            R = self.compute_goniometer_R_jax(gonio_norm)
            if R.ndim == 4:
                 kf_ki_vec = jnp.einsum("smji,sjm->sim", R, q_lab)
            else:
                 kf_ki_vec = jnp.einsum("sji,sjm->sim", R, q_lab)
        else:
            if R.ndim == 3:
                 kf_ki_vec = jnp.einsum("mji,sjm->sim", self.R, q_lab)
            else:
                 kf_ki_vec = jnp.einsum("ji,sjm->sim", self.R, q_lab)

        if self.loss_method == 'forward':
            score, _, _, _ = self.indexer_dynamic_binary_jax(UB, kf_ki_vec, k_sq_override=k_sq_dyn, softness=self.softness,
                window_batch_size=self.window_batch_size)
        elif self.loss_method == 'cosine':
            score, _, _, _ = self.indexer_dynamic_cosine_aniso_jax(UB, kf_ki_vec, k_sq_override=k_sq_dyn, softness=self.softness)
        else:
            score, _, _, _ = self.indexer_dynamic_soft_jax(UB, kf_ki_vec, k_sq_override=k_sq_dyn, softness=self.softness)

        return score

class FindUB:
    """
    Optimizer of crystal orientation from peaks and known lattice parameters.
    """

    def __init__(self, filename=None):
        self.goniometer_axes = None
        self.goniometer_angles = None
        self.goniometer_offsets = None 
        self.goniometer_names = None 
        self.sample_offset = None
        self.peak_xyz = None

        if filename is not None:
            self.load_peaks(filename)

        t = np.linspace(0, np.pi, 1024)
        cdf = (t - np.sin(t)) / np.pi
        self._angle_cdf = cdf
        self._angle_t = t
        self._angle = scipy.interpolate.interp1d(cdf, t, kind="linear")

    def load_peaks(self, filename):
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
            self.radii = f["peaks/radius"][()]
            self.space_group = f["sample/space_group"][()].decode("utf-8")
            
            if "peaks/xyz" in f:
                self.peak_xyz = f["peaks/xyz"][()]
            
            if "goniometer/axes" in f:
                 self.goniometer_axes = f["goniometer/axes"][()]
            if "goniometer/angles" in f:
                 self.goniometer_angles = f["goniometer/angles"][()]
            if "goniometer/names" in f:
                 self.goniometer_names = [n.decode('utf-8') for n in f["goniometer/names"][()]]

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

    def minimize_evosax(
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
        d_min: float = None,
        d_max: float = None,
        hkl_search_range: int = 20,
        search_window_size: int = 256,
        window_batch_size: int = 32,
        batch_size: int = None,
        sigma_init: float = None,
    ):
        if goniometer_axes is None and self.goniometer_axes is not None:
             goniometer_axes = self.goniometer_axes
        if goniometer_angles is None and self.goniometer_angles is not None:
             goniometer_angles = self.goniometer_angles.T
        if goniometer_names is None and self.goniometer_names is not None:
             goniometer_names = self.goniometer_names

        kf_ki_dir_lab = scattering_vector_from_angles(self.two_theta, self.az_phi)
        num_obs = kf_ki_dir_lab.shape[1]
        
        if refine_goniometer:
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

        weights = self.intensity / (self.sigma_intensity + 1e-6)
        weights = weights / np.mean(weights)
        weights = np.clip(weights, 0, 10.0)

        cell_params_init = np.array([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])
        lattice_system, num_lattice_params = get_lattice_system(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma, get_centering(self.space_group),
        )
        
        if refine_lattice:
            print(f"Lattice Refinement Enabled.")
            print(f"Detected System: {lattice_system} ({num_lattice_params} free parameters).")

        if loss_method == "forward" and (d_min is None or d_max is None):
            raise ValueError(f"Need to supply --d_min and --d_max for loss_method=='forward'")

        objective = VectorizedObjectiveJAX(
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
            refine_sample=refine_sample,
            sample_bound_meters=sample_bound_meters,
            goniometer_bound_deg=goniometer_bound_deg,
            hkl_search_range=hkl_search_range,
            d_min=d_min,
            d_max=d_max,
            window_batch_size=window_batch_size
        )
        print(f"Objective initialized with {loss_method} loss. Softness: {softness}")

        num_dims = 3
        if refine_lattice:
            num_dims += num_lattice_params
        if refine_sample:
            if self.peak_xyz is None:
                print("Warning: refine_sample requested but peaks/xyz not found. Disabling sample refinement.")
                refine_sample = False
            else:
                num_dims += 3
        if refine_goniometer:
            if goniometer_refine_mask is not None:
                num_dims += np.sum(goniometer_refine_mask)
            else:
                num_dims += len(goniometer_axes)

        start_sol_processed = None
        if init_params is not None:
            start_sol = jnp.array(init_params)
            if start_sol.shape[0] != num_dims:
                if start_sol.shape[0] < num_dims:
                    print(f"Bootstrapping: extending solution from {start_sol.shape[0]} to {num_dims} dims.")
                    n_new = num_dims - start_sol.shape[0]
                    padding = jnp.full((n_new,), 0.5)
                    start_sol_processed = jnp.concatenate([start_sol, padding])
                else:
                    n_gonio = len(goniometer_axes) if refine_goniometer else 0
                    if n_gonio > 0:
                        sliced_sol = jnp.concatenate([start_sol[:3], start_sol[-n_gonio:]])
                    else:
                        sliced_sol = start_sol[:3]

                    if sliced_sol.shape[0] == num_dims:
                        print(f"Bootstrapping: reducing solution from {start_sol.shape[0]} to {num_dims} dims.")
                        start_sol_processed = sliced_sol
                    else:
                        print(f"Warning: init_params shape {start_sol.shape} mismatch. Ignoring.")
                        start_sol_processed = None
            else:
                start_sol_processed = start_sol

        sample_solution = jnp.zeros(num_dims)

        target_sigma = sigma_init
        if target_sigma is None:
            if start_sol_processed is not None:
                target_sigma = 0.05
            else:
                target_sigma = 3.14
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

        def step_single_run(rng, state):
            rng, rng_ask, rng_tell = jax.random.split(rng, 3)
            x, state_ask = strategy.ask(rng_ask, state, es_params)
            x_orient = x[:, :3]
            x_rest = x[:, 3:]
            x_rest_clipped = jnp.clip(x_rest, 0.0, 1.0)
            x_valid = jnp.concatenate([x_orient, x_rest_clipped], axis=1)
            fitness = objective(x_valid)
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
        
        total_batches = int(np.ceil(n_runs / exec_batch_size))
        
        print("Initializing populations...")
        for b_i in range(total_batches):
            start_idx = b_i * exec_batch_size
            end_idx = min((b_i + 1) * exec_batch_size, n_runs)
            b_keys = all_keys[start_idx:end_idx]
            b_state = init_batch_jit(b_keys, start_sol_processed)
            batch_keys_list.append(b_keys)
            batch_states_list.append(b_state)

        pbar = range(num_generations)
        if trange is not None:
            pbar = trange(num_generations, desc="Optimizing")

        global_best_fitness = np.inf

        for gen in pbar:
            current_gen_best = np.inf
            
            for b_i in range(total_batches):
                curr_keys = batch_keys_list[b_i]
                curr_state = batch_states_list[b_i]
                next_keys, next_state, _ = step_batch_jit(curr_keys, curr_state)
                batch_keys_list[b_i] = next_keys
                batch_states_list[b_i] = next_state
                b_min = jnp.min(next_state.best_fitness)
                if b_min < current_gen_best:
                    current_gen_best = b_min

            if current_gen_best < global_best_fitness:
                global_best_fitness = current_gen_best

            if trange is not None:
                pbar.set_description(
                    f"Gen {gen+1} | Best: {-global_best_fitness:.1f}/{num_obs}"
                )

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

        idx = 0
        rot_params = self.x[idx:idx+3]
        idx += 3
        U = objective.orientation_U_jax(rot_params[None])[0]

        if refine_lattice:
            cell_norm = self.x[None, idx:idx+num_lattice_params] 
            idx += num_lattice_params
            p_full_real = objective.reconstruct_cell_params(cell_norm)
            p = np.array(p_full_real[0])
            B_new = objective.compute_B_jax(cell_norm)[0]
            print("--- Refined Lattice Parameters ---")
            print(f"a: {p[0]:.4f}, b: {p[1]:.4f}, c: {p[2]:.4f}")
            print(f"alpha: {p[3]:.4f}, beta: {p[4]:.4f}, gamma: {p[5]:.4f}")
            self.a, self.b, self.c = p[0], p[1], p[2]
            self.alpha, self.beta, self.gamma = p[3], p[4], p[5]
            B = B_new
        else:
            B = self.reciprocal_lattice_B()

        if refine_sample:
             s_norm = self.x[idx:idx+3]
             idx += 3
             self.sample_offset = (s_norm - 0.5) * 2.0 * sample_bound_meters
             print(f"--- Refined Sample Offset (mm) ---")
             print(f"X: {1000*self.sample_offset[0]:.4f}, Y: {1000*self.sample_offset[1]:.4f}, Z: {1000*self.sample_offset[2]:.4f}")

        if refine_goniometer:
            n_active = np.sum(goniometer_refine_mask) if goniometer_refine_mask is not None else len(goniometer_axes)
            if n_active > 0:
                active_norm = self.x[None, idx:idx+n_active]
                gonio_norm = np.full((1, len(goniometer_axes)), 0.5)
                if goniometer_refine_mask is not None:
                    gonio_norm[:, goniometer_refine_mask] = active_norm
                else:
                    gonio_norm = active_norm
            else:
                gonio_norm = np.full((1, len(goniometer_axes)), 0.5)

        kf_ki_vec = np.array(kf_ki_input)
        if refine_goniometer:
            R_refined = objective.compute_goniometer_R_jax(gonio_norm)[0]
            self.R = np.array(R_refined)
            offsets_val = objective.gonio_min + gonio_norm * (objective.gonio_max - objective.gonio_min)
            print("--- Refined Goniometer Offsets (deg) ---")
            if goniometer_names is not None:
                for name, val in zip(goniometer_names, offsets_val[0]):
                    print(f"{name}: {val:.4f}")
            else:
                print(offsets_val[0])
            self.goniometer_offsets = offsets_val[0]
            if self.R.ndim == 3:
                 kf_ki_vec = np.einsum("mji,jm->im", self.R, kf_ki_vec)
            else:
                 kf_ki_vec = np.einsum("ji,jm->im", self.R, kf_ki_vec)
        else:
            kf_ki_vec = kf_ki_input

        UB_final = U @ B

        if loss_method == 'forward':
            # Need to re-calculate Q magnitude if sample offset changed
            k_sq_dyn = None
            if refine_sample:
                s = self.sample_offset # (3,)
                p = self.peak_xyz.T # (3, M)
                v = p - s[:, None]
                dist = np.linalg.norm(v, axis=0)
                kf = v / dist
                ki = np.array([0, 0, 1])[:, None]
                q_lab = kf - ki
                k_sq_dyn = np.sum(q_lab**2, axis=0)[None, :]
                
            score, accum_probs, hkl, lamb = objective.indexer_dynamic_binary_jax(
                UB_final[None], kf_ki_vec[None], k_sq_override=k_sq_dyn, softness=softness, window_batch_size=window_batch_size
            )
        elif loss_method == 'cosine':
            score, accum_probs, hkl, lamb = objective.indexer_dynamic_cosine_aniso_jax(UB_final[None], kf_ki_vec[None], softness=softness)
        else:
            score, accum_probs, hkl, lamb = objective.indexer_dynamic_soft_jax(UB_final[None], kf_ki_vec[None], softness=softness)

        num_peaks_soft = float(jnp.sum(accum_probs[0]))
        print(f"Final Solution indexed {num_peaks_soft:.2f}/{num_obs} peaks (unweighted count).")

        return num_peaks_soft, hkl[0], lamb[0], U
