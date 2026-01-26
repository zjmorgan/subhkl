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


# --- Helper Functions for Parameter Mapping ---

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

def get_lattice_system(a, b, c, alpha, beta, gamma, centering, atol_len=0.05, atol_ang=0.5):
    is_90 = lambda x: np.isclose(x, 90.0, atol=atol_ang)
    is_120 = lambda x: np.isclose(x, 120.0, atol=atol_ang)
    eq = lambda x, y: np.isclose(x, y, atol=atol_len)
    if centering == 'R':
        if is_90(alpha) and is_90(beta) and is_120(gamma) and eq(a, b): return 'Hexagonal', 2 
        elif eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma): return 'Rhombohedral', 2 
    if centering == 'H': return 'Hexagonal', 2
    if is_90(alpha) and is_90(beta) and is_90(gamma):
        if eq(a, b) and eq(b, c): return 'Cubic', 1  
        elif eq(a, b): return 'Tetragonal', 2 
        elif eq(a, c) or eq(b, c): return 'Orthorhombic', 3 
        else: return 'Orthorhombic', 3 
    elif is_90(alpha) and is_90(beta) and is_120(gamma):
        if eq(a, b): return 'Hexagonal', 2 
    elif eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma): return 'Rhombohedral', 2 
    if is_90(alpha) and is_90(gamma) and not is_90(beta): return 'Monoclinic', 4 
    return 'Triclinic', 6

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

class VectorizedObjectiveJAX:
    def __init__(self, B, kf_ki_dir, peak_xyz_lab, wavelength, angle_cdf, angle_t, weights=None, softness=0.01,
                 cell_params=None, refine_lattice=False, lattice_bound_frac=0.05, lattice_system='Triclinic',
                 goniometer_axes=None, goniometer_angles=None, refine_goniometer=False, goniometer_bound_deg=5.0,
                 goniometer_refine_mask=None, goniometer_nominal_offsets=None,
                 refine_sample=False, sample_bound_meters=0.002, sample_nominal=None,
                 refine_beam=False, beam_bound_deg=1.0, beam_nominal=None,
                 peak_radii=None, loss_method='gaussian', 
                 hkl_search_range=15, d_min=5.0, d_max=100.0, search_window_size=256, window_batch_size=32,
                 space_group="P 1"):
        
        self.B = jnp.array(B)
        self.kf_ki_dir_init = jnp.array(kf_ki_dir)
        self.k_sq_init = jnp.sum(self.kf_ki_dir_init**2, axis=0)
        self.kf_lab_fixed = self.kf_ki_dir_init + jnp.array([0., 0., 1.])[:, None]
        self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(self.kf_lab_fixed, axis=0)
        
        if peak_xyz_lab is not None: self.peak_xyz = jnp.array(peak_xyz_lab.T) 
        else: self.peak_xyz = None

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

        r = jnp.arange(-hkl_search_range, hkl_search_range + 1)
        h, k, l = jnp.meshgrid(r, r, r, indexing='ij')
        hkl_pool = jnp.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)
        zero_mask = ~jnp.all(hkl_pool == 0, axis=0)
        hkl_pool = hkl_pool[:, zero_mask]
        q_cart = self.B @ hkl_pool 
        phis = jnp.arctan2(q_cart[1], q_cart[0])
        sort_idx = jnp.argsort(phis)
        self.pool_phi_sorted = phis[sort_idx]
        self.pool_hkl_sorted = hkl_pool[:, sort_idx] 
        self.mask_range = hkl_search_range
        print(f"Generating HKL mask for Space Group: {self.space_group} (Range: +/-{self.mask_range})")
        mask_cpu = generate_hkl_mask(self.mask_range, self.mask_range, self.mask_range, self.space_group)
        self.valid_hkl_mask = jnp.array(mask_cpu)

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
            h, k, l = hkl_int[:, 0, :], hkl_int[:, 1, :], hkl_int[:, 2, :]
            r = self.mask_range
            idx_h = jnp.clip(h + r, 0, 2*r).astype(jnp.int32)
            idx_k = jnp.clip(k + r, 0, 2*r).astype(jnp.int32)
            idx_l = jnp.clip(l + r, 0, 2*r).astype(jnp.int32)
            is_allowed = self.valid_hkl_mask[idx_h, idx_k, idx_l]
            valid_sym = (h >= -r) & (h <= r) & (k >= -r) & (k <= r) & (l >= -r) & (l <= r) & is_allowed
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
            idx_h = jnp.clip(h + r, 0, 2*r).astype(jnp.int32)
            idx_k = jnp.clip(k + r, 0, 2*r).astype(jnp.int32)
            idx_l = jnp.clip(l + r, 0, 2*r).astype(jnp.int32)
            is_allowed = self.valid_hkl_mask[idx_h, idx_k, idx_l]
            prob = jnp.where(valid_cand & is_allowed, prob, 0.0)
            new_sum = curr_sum + prob
            score_tracked = jnp.where(valid_cand & is_allowed, log_prob, -1e9)
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
            d_spacings = 2 * jnp.pi / jnp.sqrt(q_sq + 1e-9)
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
            
            R = self.compute_goniometer_R_jax(gonio_norm)
        else:
            if self.gonio_axes is not None:
                offsets_total = self.gonio_nominal_offsets[None, :].repeat(x.shape[0], axis=0)
                # Compute R from nominal (norm=0.5 corresponds to zero delta)
                gonio_norm = jnp.full((x.shape[0], self.num_gonio_axes), 0.5)
                R = self.compute_goniometer_R_jax(gonio_norm)
            else:
                offsets_total = None
                R = None

        return UB, B, sample_total, ki_vec, offsets_total, R

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
            # FIX: Use kf_lab_fixed (Ideal Beam) + refined ki
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
        else:
            score, _, _, _ = self.indexer_dynamic_soft_jax(UB, kf_ki_vec, k_sq_override=k_sq_dyn, softness=self.softness)

        return score

class FindUB:
    def __init__(self, filename=None):
        self.goniometer_axes = None
        self.goniometer_angles = None
        self.goniometer_offsets = None 
        self.goniometer_names = None 
        self.sample_offset = None
        self.peak_xyz = None
        self.ki_vec = None
        self.base_sample_offset = np.zeros(3)
        self.base_gonio_offset = None 

        if filename is not None: self.load_peaks(filename)

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
            if "peaks/xyz" in f: self.peak_xyz = f["peaks/xyz"][()]
            if "goniometer/axes" in f: self.goniometer_axes = f["goniometer/axes"][()]
            if "goniometer/angles" in f: self.goniometer_angles = f["goniometer/angles"][()]
            if "goniometer/names" in f: self.goniometer_names = [n.decode('utf-8') for n in f["goniometer/names"][()]]
            if "beam/ki_vec" in f: self.ki_vec = f["beam/ki_vec"][()]
            else: self.ki_vec = np.array([0., 0., 1.])

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
                self.base_sample_offset = b_offset # FIX: Store base, DO NOT MODIFY peak_xyz
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
                self.base_gonio_offset = b_gonio_offsets # FIX: Store base, DO NOT MODIFY gonio_angles
                n_active = sum(active_mask)
                new_params.append(np.full(n_active, 0.5))
            else:
                self.base_gonio_offset = np.zeros(len(self.goniometer_axes))
                n_active = sum(active_mask)
                new_params.append(np.full(n_active, 0.5))

        return np.concatenate([np.atleast_1d(p) for p in new_params])

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
        refine_beam: bool = False,
        beam_bound_deg: float = 1.0,
        d_min: float = None,
        d_max: float = None,
        hkl_search_range: int = 20,
        search_window_size: int = 256,
        window_batch_size: int = 32,
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
        
        # FIX: Logic for kf_ki_input passed to Objective (irrelevant if gonio present, but good for fallback)
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
            goniometer_nominal_offsets=self.base_gonio_offset,
            refine_sample=refine_sample,
            sample_bound_meters=sample_bound_meters,
            sample_nominal=self.base_sample_offset,
            refine_beam=refine_beam,
            beam_bound_deg=beam_bound_deg,
            beam_nominal=self.ki_vec, 
            goniometer_bound_deg=goniometer_bound_deg,
            hkl_search_range=hkl_search_range,
            search_window_size=search_window_size,
            d_min=d_min,
            d_max=d_max,
            window_batch_size=window_batch_size
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
            # (Truncate/Pad logic omitted for brevity, assuming standard bootstrap fits)
            if start_sol.shape[0] != num_dims:
                # Simple truncation/pad
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

        # DEBUG: Validate the initial score
        if start_sol_processed is not None:
            init_score_batch = objective(start_sol_processed[None, :])
            print(f"\nDEBUG: Initial Bootstrapped Score (unweighted peaks): {-init_score_batch[0]:.2f}")

        if strategy_name.lower() == "cma_es":
            strategy = CMA_ES(solution=sample_solution, population_size=population_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        es_params = strategy.default_params
        
        def init_single_run(rng, start_sol):
            rng, rng_pop, rng_init = jax.random.split(rng, 3)
            if start_sol is not None:
                state = strategy.init(rng_init, start_sol, es_params)
                state = state.replace(std=target_sigma)
            else:
                solution_init = jnp.zeros(num_dims) # Simplified
                state = strategy.init(rng_init, solution_init, es_params)
                state = state.replace(std=target_sigma)
            return state

        def step_single_run(rng, state):
            rng, rng_ask, rng_tell = jax.random.split(rng, 3)
            x, state_ask = strategy.ask(rng_ask, state, es_params)
            x_orient = x[:, :3]
            x_rest = jnp.clip(x[:, 3:], 0.0, 1.0)
            x_valid = jnp.concatenate([x_orient, x_rest], axis=1)
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
        else:
            score, accum_probs, hkl, lamb = objective.indexer_dynamic_soft_jax(UB_final[None], kf_ki_vec[None], softness=softness, k_sq_override=k_sq_dyn)

        num_peaks_soft = float(np.sum(accum_probs[0]))
        print(f"Final Solution indexed {num_peaks_soft:.2f}/{num_obs} peaks (unweighted count).")

        return num_peaks_soft, hkl[0], lamb[0], U
