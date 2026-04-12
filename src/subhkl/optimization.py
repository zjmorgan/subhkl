import os
import warnings
from functools import partial

import h5py
import numpy as np
import scipy.linalg

from subhkl.instrument.detector import scattering_vector_from_angles
from subhkl.core.spacegroup import get_space_group_object

from subhkl.utils.shim import (
    CMA_ES,
    HAS_JAX,
    OPTIMIZATION_BACKEND,
    PSO,
    DifferentialEvolution,
    Mesh,
    NamedSharding,
    P,
    jax,
    lax,
    jnp,
    jnp_update_add,
    jnp_update_set,
    jscipy_linalg,
)

if HAS_JAX:
    jax.config.update("jax_enable_x64", True)

__all__ = ["OPTIMIZATION_BACKEND"]

try:
    from tqdm import trange
except ImportError:
    trange = None


def require_jax():
    if not HAS_JAX:
        raise ImportError(
            "JAX and evosax are required for this functionality. "
            'Install with: pip install -e ".[jax]" or pip install jax jaxlib evosax'
        )


def _inverse_map_param(value, bound):
    if bound < 1e-12:
        return 0.5
    norm = (value + bound) / (2.0 * bound)
    return np.clip(norm, 0.0, 1.0)


def _forward_map_param(norm, bound):
    return norm * 2.0 * bound - bound


def _inverse_map_lattice(value, nominal, frac_bound):
    delta = np.abs(nominal) * frac_bound
    min_val = nominal - delta
    max_val = nominal + delta
    if (max_val - min_val) < 1e-12:
        return 0.5
    norm = (value - min_val) / (max_val - min_val)
    return np.clip(norm, 0.0, 1.0)


def _forward_map_lattice(norm, nominal, frac_bound):
    delta = np.abs(nominal) * frac_bound
    min_val = nominal - delta
    max_val = nominal + delta
    return min_val + norm * (max_val - min_val)


def _get_active_lattice_indices(lattice_system):
    if lattice_system == "Cubic": return [0]
    if lattice_system in ("Hexagonal", "Tetragonal"): return [0, 2]
    if lattice_system == "Rhombohedral": return [0, 3]
    if lattice_system == "Orthorhombic": return [0, 1, 2]
    if lattice_system == "Monoclinic": return [0, 1, 2, 4]
    return [0, 1, 2, 3, 4, 5]


def get_lattice_system(a, b, c, alpha, beta, gamma, space_group_name, atol_len=0.05, atol_ang=0.5):
    try:
        sg = get_space_group_object(space_group_name)
        sys_str = str(sg.crystal_system()).split(".")[-1].lower()
        centering = sg.centring_type()
    except Exception:
        sys_str = "triclinic"
        centering = "P"

    expected = "Triclinic"
    if sys_str == "cubic": expected = "Cubic"
    elif sys_str == "hexagonal": expected = "Hexagonal"
    elif sys_str == "trigonal": expected = "Rhombohedral" if centering == "R" else "Hexagonal"
    elif sys_str == "tetragonal": expected = "Tetragonal"
    elif sys_str == "orthorhombic": expected = "Orthorhombic"
    elif sys_str == "monoclinic": expected = "Monoclinic"

    def is_90(x): return np.isclose(x, 90.0, atol=atol_ang)
    def is_120(x): return np.isclose(x, 120.0, atol=atol_ang)
    def eq(x, y): return np.isclose(x, y, atol=atol_len)

    violation_msg = []
    if expected == "Cubic":
        if not (eq(a, b) and eq(b, c)): violation_msg.append("a=b=c")
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)): violation_msg.append("angles=90")
    elif expected == "Hexagonal":
        if not eq(a, b): violation_msg.append("a=b")
        if not (is_90(alpha) and is_90(beta) and is_120(gamma)): violation_msg.append("angles=90,90,120")
    elif expected == "Rhombohedral":
        if not (eq(a, b) and eq(b, c)): violation_msg.append("a=b=c")
        if not (eq(alpha, beta) and eq(beta, gamma)): violation_msg.append("alpha=beta=gamma")
    elif expected == "Tetragonal":
        if not eq(a, b): violation_msg.append("a=b")
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)): violation_msg.append("angles=90")
    elif expected == "Orthorhombic":
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)): violation_msg.append("angles=90")
    elif expected == "Monoclinic":
        count90 = sum([is_90(alpha), is_90(beta), is_90(gamma)])
        if count90 < 2: violation_msg.append("at least two angles=90")

    if violation_msg:
        warnings.warn(
            f"\n[Lattice System] Input parameters violate {space_group_name} ({expected}) constraints: {', '.join(violation_msg)}.\n"
            f"optimization will enforce {expected} constraints, which may cause a jump in parameters.",
            stacklevel=2,
        )

    geometric = "Triclinic"
    if is_90(alpha) and is_90(beta) and is_90(gamma):
        if eq(a, b) and eq(b, c): geometric = "Cubic"
        elif eq(a, b): geometric = "Tetragonal"
        else: geometric = "Orthorhombic"
    elif is_90(alpha) and is_90(beta) and is_120(gamma):
        if eq(a, b): geometric = "Hexagonal"
    elif centering == "R" and eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma):
        geometric = "Rhombohedral"
    elif sum([is_90(alpha), is_90(beta), is_90(gamma)]) >= 2:
        geometric = "Monoclinic"

    ranks = {"Triclinic": 0, "Monoclinic": 1, "Orthorhombic": 2, "Tetragonal": 3, "Rhombohedral": 4, "Hexagonal": 4, "Cubic": 5}
    rank_exp = ranks.get(expected, 0)
    rank_geo = ranks.get(geometric, 0)

    final_system = expected
    num = {
        "Triclinic": 6, "Monoclinic": 4, "Orthorhombic": 3,
        "Tetragonal": 2, "Hexagonal": 2, "Rhombohedral": 2, "Cubic": 1
    }.get(final_system, 6)

    if rank_exp < rank_geo:
        print(f"Lattice System Override: Geometry suggests {geometric}, but Space Group {space_group_name} requires {expected}. Enforcing {expected}.")

    return final_system, num


def rotation_matrix_from_axis_angle_jax(axis, angle_rad):
    u = axis / jnp.linalg.norm(axis)
    ux, uy, uz = u
    K = jnp.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    eye = jnp.eye(3)
    return eye + s[..., None, None] * K + (1.0 - c)[..., None, None] * (K @ K)


def rotation_matrix_from_rodrigues_jax(w):
    theta = jnp.linalg.norm(w) + 1e-9
    k = w / theta
    K = jnp.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    eye = jnp.eye(3)
    return eye + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)


class VectorizedObjective:
    def __init__(
        self,
        B,
        kf_ki_dir,
        peak_xyz_lab,
        wavelength,
        tolerance_deg=0.1,
        cell_params=None,
        refine_lattice=False,
        lattice_bound_frac=0.05,
        lattice_system="Triclinic",
        goniometer_axes=None,
        goniometer_angles=None,
        refine_goniometer=False,
        goniometer_bound_deg=5.0,
        goniometer_refine_mask=None,
        goniometer_nominal_offsets=None,
        refine_sample=False,
        sample_bound_meters=0.002,
        sample_nominal=None,
        refine_beam=False,
        beam_bound_deg=1.0,
        beam_nominal=None,
        static_R=None,
        kf_lab_fixed_vectors=None,
        peak_run_indices=None,
    ):
        self.B = jnp.array(B)
        self.kf_ki_dir_init = jnp.array(kf_ki_dir)
        if self.kf_ki_dir_init.ndim == 2 and self.kf_ki_dir_init.shape[0] != 3:
            self.kf_ki_dir_init = self.kf_ki_dir_init.T

        self.k_sq_init = jnp.sum(self.kf_ki_dir_init**2, axis=0)
        num_peaks = self.kf_ki_dir_init.shape[1]
        self.tolerance_rad = jnp.deg2rad(tolerance_deg)

        self.static_R = jnp.array(static_R) if static_R is not None else jnp.eye(3)

        if peak_run_indices is not None:
            self.peak_run_indices = jnp.array(peak_run_indices, dtype=jnp.int32)
            if self.static_R.ndim == 3:
                max_run = jnp.max(self.peak_run_indices)
                if max_run >= self.static_R.shape[0] and self.static_R.shape[0] == 1:
                    self.static_R = jnp.tile(self.static_R, (max_run + 1, 1, 1))
        elif self.static_R.ndim == 3 and self.static_R.shape[0] == num_peaks:
            self.peak_run_indices = jnp.arange(num_peaks, dtype=jnp.int32)
        else:
            self.peak_run_indices = jnp.zeros(num_peaks, dtype=jnp.int32)

        if self.static_R.ndim == 3:
            self.peak_run_indices = jnp.clip(self.peak_run_indices, 0, self.static_R.shape[0] - 1)

        if peak_xyz_lab is not None:
            p_xyz = jnp.array(peak_xyz_lab)
            self.peak_xyz = p_xyz.T if p_xyz.shape[0] != 3 else p_xyz
        else:
            self.peak_xyz = None

        self.refine_sample = refine_sample
        self.sample_bound = sample_bound_meters
        self.sample_nominal = jnp.array(sample_nominal) if sample_nominal is not None else jnp.zeros(3)

        self.refine_beam = refine_beam
        self.beam_bound_deg = beam_bound_deg
        self.beam_nominal = jnp.array(beam_nominal) if beam_nominal is not None else jnp.array([0.0, 0.0, 1.0])

        self.kf_lab_fixed = None
        if self.peak_xyz is not None:
            v = self.peak_xyz - self.sample_nominal[:, None]
            dist = jnp.linalg.norm(v, axis=0)
            self.kf_lab_fixed = v / jnp.where(dist == 0, 1.0, dist[None, :])

        if kf_lab_fixed_vectors is not None and self.kf_lab_fixed is None:
            q_vecs = jnp.array(kf_lab_fixed_vectors)
            q_vecs = q_vecs.T if q_vecs.shape[0] != 3 else q_vecs
            self.kf_lab_fixed = q_vecs + self.beam_nominal[:, None]
            self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(self.kf_lab_fixed, axis=0)

        if self.kf_lab_fixed is None:
            q_vecs = self.kf_ki_dir_init
            self.kf_lab_fixed = q_vecs + self.beam_nominal[:, None]
            self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(self.kf_lab_fixed, axis=0)

        self.refine_lattice = refine_lattice
        self.lattice_system = lattice_system
        self.lattice_bound_frac = lattice_bound_frac
        self.refine_goniometer = refine_goniometer
        self.goniometer_bound_deg = goniometer_bound_deg

        if self.refine_lattice:
            self.cell_init = jnp.array(cell_params)
            if self.lattice_system == "Cubic": self.free_params_init = self.cell_init[0:1]
            elif self.lattice_system in ("Hexagonal", "Tetragonal"): self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[2]])
            elif self.lattice_system == "Rhombohedral": self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[3]])
            elif self.lattice_system == "Orthorhombic": self.free_params_init = self.cell_init[0:3]
            elif self.lattice_system == "Monoclinic": self.free_params_init = jnp.array([self.cell_init[0], self.cell_init[1], self.cell_init[2], self.cell_init[4]])
            else: self.free_params_init = self.cell_init

        if goniometer_axes is not None:
            axes = jnp.array(goniometer_axes)
            if axes.ndim == 2 and axes.shape[1] == 3:
                axes = jnp.concatenate([axes, jnp.ones((axes.shape[0], 1))], axis=1)
            self.gonio_axes = axes

            angles = jnp.array(goniometer_angles)
            if angles.ndim == 2 and angles.shape[0] != self.gonio_axes.shape[0]:
                angles = angles.T
            self.gonio_angles = angles
            self.num_gonio_axes = self.gonio_axes.shape[0]

            if self.gonio_angles.shape[1] == num_peaks:
                self.peak_run_indices = jnp.arange(num_peaks, dtype=jnp.int32)

            self.gonio_mask = np.array(goniometer_refine_mask, dtype=bool) if goniometer_refine_mask is not None else np.ones(self.num_gonio_axes, dtype=bool)
            self.num_active_gonio = np.sum(self.gonio_mask)
            self.gonio_nominal_offsets = jnp.array(goniometer_nominal_offsets) if goniometer_nominal_offsets is not None else jnp.zeros(self.num_gonio_axes)
        else:
            self.gonio_axes = None
            self.num_gonio_axes = 0

        wavelength = jnp.array(wavelength)
        self.wl_min_val = wavelength[0]
        self.wl_max_val = wavelength[1]
        self.num_candidates = 64

    def orientation_U_jax(self, param):
        """Vectorized U matrix calculation from Rodrigues vectors"""
        U = jax.vmap(rotation_matrix_from_rodrigues_jax)(param)
        return U

    def _get_physical_params_jax(self, x):
        idx = 0
        rot_params = x[:, idx : idx + 3]
        U = self.orientation_U_jax(rot_params)
        idx += 3

        if self.refine_lattice:
            n_lat = self.free_params_init.size
            cell_params_norm = x[:, idx : idx + n_lat]
            p = _forward_map_lattice(cell_params_norm, self.free_params_init, self.lattice_bound_frac)
            
            S = cell_params_norm.shape[0]
            deg90, deg120 = jnp.full((S,), 90.0), jnp.full((S,), 120.0)
            
            if self.lattice_system == "Cubic":
                a = p[:, 0]
                p = jnp.stack([a, a, a, deg90, deg90, deg90], axis=1)
            elif self.lattice_system == "Hexagonal":
                a, c = p[:, 0], p[:, 1]
                p = jnp.stack([a, a, c, deg90, deg90, deg120], axis=1)
            elif self.lattice_system == "Tetragonal":
                a, c = p[:, 0], p[:, 1]
                p = jnp.stack([a, a, c, deg90, deg90, deg90], axis=1)
            elif self.lattice_system == "Rhombohedral":
                a, alpha = p[:, 0], p[:, 1]
                p = jnp.stack([a, a, a, alpha, alpha, alpha], axis=1)
            elif self.lattice_system == "Orthorhombic":
                a, b, c = p[:, 0], p[:, 1], p[:, 2]
                p = jnp.stack([a, b, c, deg90, deg90, deg90], axis=1)
            elif self.lattice_system == "Monoclinic":
                a, b, c, beta = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
                p = jnp.stack([a, b, c, deg90, beta, deg90], axis=1)

            deg2rad = jnp.pi / 180.0
            a, b, c = p[:, 0], p[:, 1], p[:, 2]
            alpha, beta, gamma = p[:, 3] * deg2rad, p[:, 4] * deg2rad, p[:, 5] * deg2rad
            
            g11, g22, g33 = a**2, b**2, c**2
            g12, g13, g23 = a*b*jnp.cos(gamma), a*c*jnp.cos(beta), b*c*jnp.cos(alpha)
            G = jnp.stack([
                jnp.stack([g11, g12, g13], axis=-1),
                jnp.stack([g12, g22, g23], axis=-1),
                jnp.stack([g13, g23, g33], axis=-1)
            ], axis=-2)
            
            B = jscipy_linalg.cholesky(jnp.linalg.inv(G), lower=False)
            idx += n_lat
            UB = jnp.matmul(U, B)
        else:
            B = self.B
            UB = jnp.matmul(U, B[None, ...])

        if self.refine_sample:
            s_norm = x[:, idx : idx + 3]
            idx += 3
            sample_delta = _forward_map_param(s_norm, self.sample_bound)
            sample_total = self.sample_nominal + sample_delta
        else:
            sample_total = self.sample_nominal[None, :].repeat(x.shape[0], axis=0)

        if self.refine_beam:
            bound_rad = jnp.deg2rad(self.beam_bound_deg)
            tx = _forward_map_param(x[:, idx], bound_rad)
            ty = _forward_map_param(x[:, idx + 1], bound_rad)
            idx += 2
            ki_vec = jnp.tile(self.beam_nominal[None, :], (x.shape[0], 1))
            ki_vec = jnp_update_add(ki_vec, (slice(None), 0), tx)
            ki_vec = jnp_update_add(ki_vec, (slice(None), 1), ty)
            ki_vec = ki_vec / jnp.linalg.norm(ki_vec, axis=1, keepdims=True)
        else:
            ki_vec = self.beam_nominal[None, :].repeat(x.shape[0], axis=0)

        if self.refine_goniometer:
            gonio_norm = jnp.full((x.shape[0], self.num_gonio_axes), 0.5)
            if self.num_active_gonio > 0:
                gonio_norm = jnp_update_set(gonio_norm, (slice(None), self.gonio_mask), x[:, idx : idx + self.num_active_gonio])
                idx += self.num_active_gonio

            offsets_delta = _forward_map_param(gonio_norm, self.goniometer_bound_deg)
            offsets_total = self.gonio_nominal_offsets + offsets_delta
            
            angles_deg = offsets_total[:, :, None] + self.gonio_angles[None, :, :]
            S, M = offsets_total.shape[0], self.gonio_angles.shape[1]
            R = jnp.eye(3)[None, None, ...].repeat(S, axis=0).repeat(M, axis=1)
            deg2rad = jnp.pi / 180.0
            
            for i in range(self.num_gonio_axes):
                direction = self.gonio_axes[i][0:3]
                theta = self.gonio_axes[i][3] * angles_deg[:, i, :] * deg2rad
                Ri = rotation_matrix_from_axis_angle_jax(direction, theta)
                R = jnp.matmul(R, Ri)
        elif self.gonio_axes is not None:
            offsets_total = self.gonio_nominal_offsets[None, :].repeat(x.shape[0], axis=0)
            angles_deg = offsets_total[:, :, None] + self.gonio_angles[None, :, :]
            S, M = offsets_total.shape[0], self.gonio_angles.shape[1]
            R = jnp.eye(3)[None, None, ...].repeat(S, axis=0).repeat(M, axis=1)
            deg2rad = jnp.pi / 180.0
            for i in range(self.num_gonio_axes):
                direction = self.gonio_axes[i][0:3]
                theta = self.gonio_axes[i][3] * angles_deg[:, i, :] * deg2rad
                Ri = rotation_matrix_from_axis_angle_jax(direction, theta)
                R = jnp.matmul(R, Ri)
        else:
            offsets_total = None
            R = None

        return UB, B, sample_total, ki_vec, offsets_total, R

    def indexer_dynamic_soft_jax(self, ub_mat, kf_ki_sample, k_sq_override=None, tolerance_rad=0.002):
        ub_inv = jnp.linalg.inv(ub_mat)
        v = jnp.matmul(ub_inv, kf_ki_sample)
        abs_v = jnp.abs(v)
        max_v_val = jnp.max(abs_v, axis=1)
        n_start = max_v_val / self.wl_max_val
        start_int = jnp.ceil(n_start)
        k_sq = k_sq_override if k_sq_override is not None else self.k_sq_init[None, :]

        initial_carry = (
            jnp.inf * jnp.ones(max_v_val.shape),
            jnp.zeros((v.shape[0], 3, v.shape[2]), dtype=jnp.int32),
            jnp.zeros(max_v_val.shape),
        )

        def scan_body(carry, i):
            curr_min, curr_best_hkl, curr_best_lamb = carry
            n = start_int + i
            n_safe = jnp.where(n == 0, 1e-9, n)
            lamda_cand = max_v_val / n_safe
            hkl_float = v / lamda_cand[:, None, :]
            hkl_int = jnp.round(hkl_float).astype(jnp.int32)
            
            q_int = jnp.matmul(ub_mat, hkl_int.astype(jnp.float32))
            k_dot_q = jnp.sum(kf_ki_sample * q_int, axis=1)
            safe_dot = jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
            lambda_opt = k_sq / safe_dot
            
            delta_hkl = jnp.sin(jnp.pi * hkl_float) / jnp.pi
            dist = jnp.linalg.norm(delta_hkl, axis=1)

            update_mask = dist < curr_min
            new_min = jnp.where(update_mask, dist, curr_min)
            new_best_hkl = jnp.where(update_mask[:, None, :], hkl_int, curr_best_hkl)
            new_best_lamb = jnp.where(update_mask, lambda_opt, curr_best_lamb)
            return (new_min, new_best_hkl, new_best_lamb), None

        final_carry, _ = lax.scan(scan_body, initial_carry, jnp.arange(self.num_candidates))
        dist_min, best_hkl, best_lamb = final_carry
        loss = jnp.mean(dist_min, axis=1)
        return loss, dist_min, best_hkl.transpose((0, 2, 1)), best_lamb

    @partial(jax.jit, static_argnames="self")
    def get_results(self, x):
        original_S = x.shape[0]
        pad_size = max(0, 2 - original_S)
        x_pad = jnp.pad(x, ((0, pad_size), (0, 0)), mode="edge")

        UB, _, sample_total, ki_vec, _, R = self._get_physical_params_jax(x_pad)

        R_curr = R if R is not None else self.static_R

        if R_curr is not None:
            if R_curr.ndim == 4:
                R_per_peak = R_curr[:, self.peak_run_indices, :, :]
            elif R_curr.ndim == 3:
                R_per_peak = R_curr[self.peak_run_indices, :, :]
            else:
                R_per_peak = R_curr
        else:
            R_per_peak = None

        if self.peak_xyz is not None:
            if R_per_peak is not None:
                if R_per_peak.ndim == 4:
                    s_lab = jnp.matmul(R_per_peak, sample_total[:, None, :, None]).squeeze(-1)
                    s = s_lab.transpose(0, 2, 1)
                elif R_per_peak.ndim == 3:
                    s_lab = jnp.matmul(R_per_peak[None, ...], sample_total[:, None, :, None]).squeeze(-1)
                    s = s_lab.transpose(0, 2, 1)
                else:
                    s_lab = jnp.matmul(R_per_peak[None, ...], sample_total[:, :, None]).squeeze(-1)
                    s = s_lab[:, :, None]
            else:
                s = sample_total[:, :, None]

            v = self.peak_xyz[None, :, :] - s
            dist = jnp.sqrt(jnp.sum(v**2, axis=1, keepdims=True))
            kf = v / jnp.where(dist == 0, 1.0, dist)
            ki = ki_vec[:, :, None]
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)
        else:
            kf = self.kf_lab_fixed[None, :, :].repeat(x.shape[0], axis=0)
            ki = ki_vec[:, :, None]
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)

        if R_per_peak is not None:
            q_lab_vec = q_lab.transpose(0, 2, 1)[..., None]
            if R_per_peak.ndim == 4:
                RT = R_per_peak.transpose(0, 1, 3, 2)
                kf_ki_vec_T = jnp.matmul(RT, q_lab_vec).squeeze(-1)
            elif R_per_peak.ndim == 3:
                RT = R_per_peak.transpose(0, 2, 1)[None, ...]
                kf_ki_vec_T = jnp.matmul(RT, q_lab_vec).squeeze(-1)
            else:
                RT = R_per_peak.T[None, None, ...]
                kf_ki_vec_T = jnp.matmul(RT, q_lab_vec).squeeze(-1)
            kf_ki_vec = kf_ki_vec_T.transpose(0, 2, 1)
        else:
            kf_ki_vec = q_lab

        res = self.indexer_dynamic_soft_jax(
            UB,
            kf_ki_vec,
            k_sq_override=k_sq_dyn,
            tolerance_rad=self.tolerance_rad,
        )

        return jax.tree.map(
            lambda arr: arr[:original_S] if hasattr(arr, "shape") and arr.ndim > 0 else arr,
            res,
        )

    @partial(jax.jit, static_argnames="self")
    def __call__(self, x):
        score, _, _, _ = self.get_results(x)
        return score


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

    def load_from_dict(self, data):
        self.a = data["sample/a"]
        self.b = data["sample/b"]
        self.c = data["sample/c"]
        self.alpha = data["sample/alpha"]
        self.beta = data["sample/beta"]
        self.gamma = data["sample/gamma"]
        self.wavelength = data["instrument/wavelength"]
        self.R = data.get("goniometer/R")
        self.two_theta = data["peaks/two_theta"]
        self.az_phi = data["peaks/azimuthal"]
        self.intensity = data["peaks/intensity"]
        self.sigma_intensity = data["peaks/sigma"]

        r_stack = data.get("goniometer/R")
        idx_run = data.get("peaks/run_index")
        idx_img = data.get("peaks/image_index")
        idx_bank = data.get("bank")
        if idx_bank is None:
            idx_bank = data.get("bank_ids")

        if r_stack is not None and r_stack.ndim == 3:
            num_rot = r_stack.shape[0]
            if idx_run is not None and int(np.max(idx_run)) + 1 == num_rot:
                self.run_indices = idx_run
            elif idx_img is not None and int(np.max(idx_img)) + 1 == num_rot:
                self.run_indices = idx_img
            elif idx_bank is not None and int(np.max(idx_bank)) + 1 == num_rot:
                self.run_indices = idx_bank
            else:
                self.run_indices = idx_run if idx_run is not None else idx_img
        else:
            self.run_indices = idx_run if idx_run is not None else idx_img

        if self.run_indices is None:
            self.run_indices = idx_bank

        if self.run_indices is None:
            num_peaks = len(data["peaks/two_theta"])
            self.run_indices = np.zeros(num_peaks, dtype=int)

        sg = data["sample/space_group"]
        self.space_group = sg.decode("utf-8") if isinstance(sg, bytes) else str(sg)

        if "sample/offset" in data: self.base_sample_offset = data["sample/offset"]
        if "peaks/xyz" in data: self.peak_xyz = data["peaks/xyz"]
        if "goniometer/axes" in data: self.goniometer_axes = data["goniometer/axes"]
        if "goniometer/angles" in data: self.goniometer_angles = data["goniometer/angles"]
        if "goniometer/names" in data:
            self.goniometer_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in data["goniometer/names"]]
        self.ki_vec = data["beam/ki_vec"] if "beam/ki_vec" in data else np.array([0.0, 0.0, 1.0])

    def load_peaks(self, filename):
        with h5py.File(os.path.abspath(filename), "r") as f:
            data = {
                "sample/a": f["sample/a"][()], "sample/b": f["sample/b"][()], "sample/c": f["sample/c"][()],
                "sample/alpha": f["sample/alpha"][()], "sample/beta": f["sample/beta"][()], "sample/gamma": f["sample/gamma"][()],
                "instrument/wavelength": f["instrument/wavelength"][()], "goniometer/R": f["goniometer/R"][()],
                "peaks/two_theta": f["peaks/two_theta"][()], "peaks/azimuthal": f["peaks/azimuthal"][()],
                "peaks/intensity": f["peaks/intensity"][()], "peaks/sigma": f["peaks/sigma"][()], "peaks/radius": f["peaks/radius"][()],
                "sample/space_group": f["sample/space_group"][()]
            }
            if "peaks/run_index" in f: data["peaks/run_index"] = f["peaks/run_index"][()]
            if "peaks/image_index" in f: data["peaks/image_index"] = f["peaks/image_index"][()]
            if "bank" in f: data["bank"] = f["bank"][()]
            if "bank_ids" in f: data["bank_ids"] = f["bank_ids"][()]
            if "peaks/xyz" in f: data["peaks/xyz"] = f["peaks/xyz"][()]
            if "goniometer/axes" in f: data["goniometer/axes"] = f["goniometer/axes"][()]
            if "goniometer/angles" in f: data["goniometer/angles"] = f["goniometer/angles"][()]
            if "goniometer/names" in f: data["goniometer/names"] = f["goniometer/names"][()]
            if "beam/ki_vec" in f: data["beam/ki_vec"] = f["beam/ki_vec"][()]
            self.load_from_dict(data)

    def reciprocal_lattice_B(self):
        alpha, beta, gamma = np.deg2rad([self.alpha, self.beta, self.gamma])
        g11, g22, g33 = self.a**2, self.b**2, self.c**2
        g12 = self.a * self.b * np.cos(gamma)
        g13 = self.c * self.a * np.cos(beta)
        g23 = self.b * self.c * np.cos(alpha)
        G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])
        return scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    def get_bootstrap_params(
        self,
        bootstrap_filename,
        refine_lattice=False,
        lattice_bound_frac=0.05,
        refine_sample=False,
        sample_bound_meters=0.002,
        refine_beam=False,
        beam_bound_deg=1.0,
        refine_goniometer=False,
        goniometer_bound_deg=5.0,
        refine_goniometer_axes=None,
    ):
        print(f"Bootstrapping from physical solution: {bootstrap_filename}")
        with h5py.File(bootstrap_filename, "r") as f:
            raw_x = f["optimization/best_params"][()] if "optimization/best_params" in f else None
            b_a, b_b, b_c = f["sample/a"][()], f["sample/b"][()], f["sample/c"][()]
            b_alpha, b_beta, b_gamma = f["sample/alpha"][()], f["sample/beta"][()], f["sample/gamma"][()]
            b_offset = f["sample/offset"][()] if "sample/offset" in f else np.zeros(3)
            b_ki = f["beam/ki_vec"][()] if "beam/ki_vec" in f else np.array([0.0, 0.0, 1.0])
            b_gonio_offsets = f["optimization/goniometer_offsets"][()] if "optimization/goniometer_offsets" in f else None

            if "sample/U" in f:
                U_initial = f["sample/U"][()]
                from scipy.spatial.transform import Rotation as R
                rodrigues_vec = R.from_matrix(U_initial).as_rotvec()
                if raw_x is None: raw_x = rodrigues_vec
                else: raw_x[:3] = rodrigues_vec

        new_params = [raw_x[:3] if raw_x is not None else np.zeros(3)]

        if refine_lattice:
            self.a, self.b, self.c = b_a, b_b, b_c
            self.alpha, self.beta, self.gamma = b_alpha, b_beta, b_gamma
            lat_sys, _ = get_lattice_system(self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.space_group)
            new_params.append(np.full(len(_get_active_lattice_indices(lat_sys)), 0.5))

        if b_offset is not None: self.base_sample_offset = b_offset
        if refine_sample: new_params.append(np.full(3, 0.5))

        if b_ki is not None: self.ki_vec = b_ki
        if refine_beam: new_params.append(np.full(2, 0.5))

        if b_gonio_offsets is not None:
            self.base_gonio_offset = b_gonio_offsets
        else:
            self.base_gonio_offset = np.zeros(len(self.goniometer_axes)) if self.goniometer_axes is not None else None

        if refine_goniometer:
            active_mask = [True] * len(self.goniometer_axes)
            if refine_goniometer_axes is not None and self.goniometer_names is not None:
                active_mask = [any(req in name for req in refine_goniometer_axes) for name in self.goniometer_names]
            new_params.append(np.full(sum(active_mask), 0.5))

        return np.concatenate([np.atleast_1d(p) for p in new_params])

    def minimize(
        self,
        strategy_name: str,
        population_size: int = 1000,
        num_generations: int = 100,
        n_runs: int = 1,
        seed: int = 0,
        tolerance_deg: float = 0.1,
        init_params: np.ndarray | None = None,
        refine_lattice: bool = False,
        lattice_bound_frac: float = 0.05,
        goniometer_axes: list | None = None,
        goniometer_angles: np.ndarray | None = None,
        refine_goniometer: bool = False,
        goniometer_bound_deg: float = 5.0,
        goniometer_names: list | None = None,
        refine_goniometer_axes: list | None = None,
        refine_sample: bool = False,
        sample_bound_meters: float = 2.0,
        refine_beam: bool = False,
        beam_bound_deg: float = 1.0,
        batch_size: int | None = None,
        sigma_init: float | None = None,
        **kwargs
    ):
        require_jax()

        if goniometer_axes is None and self.goniometer_axes is not None:
            goniometer_axes = self.goniometer_axes
        if goniometer_angles is None and self.goniometer_angles is not None:
            goniometer_angles = self.goniometer_angles
        if goniometer_names is None and self.goniometer_names is not None:
            goniometer_names = self.goniometer_names

        kf_ki_dir_lab = scattering_vector_from_angles(self.two_theta, self.az_phi)
        num_obs = kf_ki_dir_lab.shape[1]

        static_R_input = self.R if self.R is not None else np.eye(3)
        if self.run_indices is not None:
            max_run_id = int(np.max(self.run_indices))
            num_runs_range = max_run_id + 1
            unique_runs, first_indices = np.unique(self.run_indices, return_index=True)

            def has_variation(data, indices):
                if data is None: return False
                for r in unique_runs:
                    mask = indices == r
                    if np.sum(mask) <= 1: continue
                    subset = data[mask] if data.ndim == 2 else data[mask, ...]
                    if not np.allclose(subset, subset[0:1], atol=1e-7): return True
                return False

            can_reduce_angles = (goniometer_angles is not None and goniometer_angles.shape[1] == num_obs and not has_variation(goniometer_angles.T, self.run_indices))
            can_reduce_R = (self.R is not None and self.R.ndim == 3 and self.R.shape[0] == num_obs and not has_variation(self.R, self.run_indices))

            if can_reduce_angles:
                new_angles = np.zeros((goniometer_angles.shape[0], num_runs_range))
                new_angles[:] = goniometer_angles[:, first_indices[0:1]]
                new_angles[:, unique_runs] = goniometer_angles[:, first_indices]
                goniometer_angles = new_angles

            if can_reduce_R:
                new_R = np.zeros((num_runs_range, 3, 3))
                new_R[:] = self.R[first_indices[0:1]]
                new_R[unique_runs] = self.R[first_indices]
                static_R_input = new_R
            elif self.R is not None and self.R.ndim == 3 and self.R.shape[0] == num_obs:
                static_R_input = self.R
                self.run_indices = np.arange(num_obs, dtype=np.int32)
            elif (goniometer_angles is not None and goniometer_angles.shape[1] == num_obs):
                self.run_indices = np.arange(num_obs, dtype=np.int32)

        goniometer_refine_mask = None
        if refine_goniometer and refine_goniometer_axes is not None:
            if self.goniometer_names is not None:
                mask = [any(req in name for req in refine_goniometer_axes) for name in self.goniometer_names]
                goniometer_refine_mask = np.array(mask, dtype=bool)
            else:
                goniometer_refine_mask = np.ones(len(goniometer_axes), dtype=bool)

        cell_params_init = np.array([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])
        lattice_system, num_lattice_params = get_lattice_system(self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.space_group)

        objective = VectorizedObjective(
            self.reciprocal_lattice_B(),
            kf_ki_dir_lab,
            self.peak_xyz,
            np.array(self.wavelength),
            tolerance_deg=tolerance_deg,
            cell_params=cell_params_init,
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
            static_R=static_R_input,
            kf_lab_fixed_vectors=kf_ki_dir_lab,
            peak_run_indices=self.run_indices,
        )

        num_dims = 3 + (num_lattice_params if refine_lattice else 0)
        if refine_sample and self.peak_xyz is not None: num_dims += 3
        if refine_beam and self.peak_xyz is not None: num_dims += 2
        if refine_goniometer: num_dims += np.sum(goniometer_refine_mask) if goniometer_refine_mask is not None else len(goniometer_axes)

        start_sol_processed = None
        if init_params is not None:
            start_sol = jnp.array(init_params)
            if start_sol.shape[0] < num_dims:
                start_sol_processed = jnp.concatenate([start_sol, jnp.full((num_dims - start_sol.shape[0],), 0.5)])
            else:
                start_sol_processed = start_sol[:num_dims]

        sample_solution = jnp.zeros(num_dims)
        target_sigma = sigma_init or (0.01 if start_sol_processed is not None else 3.14)

        if strategy_name.lower() == "de": strategy = DifferentialEvolution(solution=sample_solution, population_size=population_size); strategy_type = "population_based"
        elif strategy_name.lower() == "pso": strategy = PSO(solution=sample_solution, population_size=population_size); strategy_type = "population_based"
        elif strategy_name.lower() == "cma_es": strategy = CMA_ES(solution=sample_solution, population_size=population_size); strategy_type = "distribution_based"
        else: raise ValueError(f"Unknown strategy: {strategy_name}")

        es_params = strategy.default_params

        def init_single_run(rng, start_sol):
            rng, rng_pop, rng_init = jax.random.split(rng, 3)
            if start_sol is not None:
                if strategy_type == "population_based":
                    noise = jax.random.normal(rng_pop, (population_size, num_dims)) * target_sigma
                    population_init = jnp.concatenate([start_sol[:3] + noise[:, :3], jnp.clip(start_sol[3:] + noise[:, 3:], 0.0, 1.0)], axis=1)
                    state = strategy.init(rng_init, population_init, objective(population_init), es_params)
                else:
                    state = strategy.init(rng_init, start_sol, es_params).replace(std=target_sigma)
            elif strategy_type == "population_based":
                pop_orient = jax.random.normal(rng_pop, (population_size, 3)) * target_sigma
                pop_rest = jax.random.uniform(jax.random.split(rng_pop)[0], (population_size, max(0, num_dims - 3)))
                population_init = jnp.concatenate([pop_orient, pop_rest], axis=1)
                state = strategy.init(rng_init, population_init, objective(population_init), es_params)
            else:
                solution_init = jnp.concatenate([jnp.zeros(3), jnp.full((max(0, num_dims - 3),), 0.5)])
                state = strategy.init(rng_init, solution_init, es_params).replace(std=target_sigma)
            return state

        mesh = Mesh(np.array(jax.devices()), ("i")) if HAS_JAX else None

        def step_single_run(rng, state):
            rng, rng_ask, rng_tell = jax.random.split(rng, 3)
            x, state_ask = strategy.ask(rng_ask, state, es_params)
            x_valid = jnp.concatenate([x[:, :3], jnp.clip(x[:, 3:], 0.0, 1.0)], axis=1)
            
            if mesh: x_valid = jax.lax.with_sharding_constraint(x_valid, NamedSharding(mesh, P("i")))
            
            state_tell, metrics = strategy.tell(rng_tell, x_valid, objective(x_valid), state_ask, es_params)
            return rng, state_tell, metrics

        init_batch_jit = jax.jit(jax.vmap(init_single_run, in_axes=(0, None)))
        step_batch_jit = jax.jit(jax.vmap(step_single_run, in_axes=(0, 0)))

        exec_batch_size = batch_size if batch_size is not None else n_runs
        seeds = jnp.arange(seed, seed + n_runs)
        all_keys = jax.vmap(jax.random.PRNGKey)(seeds)
        batch_keys_list, batch_states_list = [], []

        for b_i in range(int(np.ceil(n_runs / exec_batch_size))):
            start_idx, end_idx = b_i * exec_batch_size, min((b_i + 1) * exec_batch_size, n_runs)
            batch_keys_list.append(all_keys[start_idx:end_idx])
            batch_states_list.append(init_batch_jit(batch_keys_list[-1], start_sol_processed))

        pbar = trange(num_generations, desc="Optimizing") if trange else range(num_generations)
        for gen in pbar:
            current_gen_best = np.inf
            for b_i in range(len(batch_keys_list)):
                batch_keys_list[b_i], batch_states_list[b_i], _ = step_batch_jit(batch_keys_list[b_i], batch_states_list[b_i])
                current_gen_best = min(current_gen_best, jnp.min(batch_states_list[b_i].best_fitness))
            if trange: pbar.set_description(f"Gen {gen + 1} | Best Loss: {current_gen_best:.5f}")

        all_loss = jnp.concatenate([b.best_fitness for b in batch_states_list], axis=0)
        all_solutions = jnp.concatenate([b.best_solution for b in batch_states_list], axis=0)

        best_idx = np.argmin(all_loss)
        best_overall_loss, best_overall_member = all_loss[best_idx], all_solutions[best_idx]

        print("Polishing solution with BFGS refinement...")
        from scipy.optimize import minimize as scipy_minimize
        res_ref = scipy_minimize(
            lambda x_flat: float(objective(x_flat[None, :])[0]),
            np.array(best_overall_member),
            jac=lambda x_flat: np.array(jax.grad(lambda x: objective(x[None, :])[0])(x_flat)),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0) if i >= 3 else (None, None) for i in range(num_dims)],
            options={"maxiter": 50},
        )

        if res_ref.success and res_ref.fun < best_overall_loss:
            best_overall_member, best_overall_loss = res_ref.x, res_ref.fun

        self.x = np.array(best_overall_member)
        x_batch = jnp.array(self.x[None, :])
        (UB_final_batch, B_new_batch, s_total_batch, ki_vec_batch, offsets_total_batch, R_batch) = objective._get_physical_params_jax(x_batch)

        self.sample_offset = np.array(s_total_batch[0])
        self.ki_vec = np.array(ki_vec_batch[0]).flatten()
        if offsets_total_batch is not None: self.goniometer_offsets = np.array(offsets_total_batch[0])
        if R_batch is not None: self.R = np.array(R_batch[0])

        rot_params = self.x[:3]
        U = objective.orientation_U_jax(rot_params[None])[0]

        if refine_lattice:
            cell_norm = jnp.array(self.x[None, 3 : 3 + num_lattice_params])
            p_full = np.array(objective.reconstruct_cell_params(cell_norm)[0])
            self.a, self.b, self.c = p_full[:3]
            self.alpha, self.beta, self.gamma = p_full[3:]

        loss_score, dist_min, hkl, lamb = objective.get_results(x_batch)
        dist_min_final = np.array(dist_min[0])
        
        # Calculate number of correctly indexed peaks based on sub-pixel HKL distance
        mask = dist_min_final < 0.15
        num_indexed = int(np.sum(mask))
        
        hkl_final = np.array(hkl[0])
        hkl_final[~mask] = 0

        print(f"Final Solution indexed {num_indexed}/{num_obs} peaks.")

        return num_indexed, hkl_final, np.array(lamb[0]), np.array(U)
