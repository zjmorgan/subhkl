import os
import warnings
from functools import partial

import h5py
import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.spatial

from subhkl.instrument.detector import scattering_vector_from_angles
from subhkl.core.spacegroup import (
    generate_hkl_mask,
    get_centering,
    get_space_group_object,
)

# Import JAX with fallback from utils (centralized)
from subhkl.shim import (
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
    if lattice_system == "Cubic":
        return [0]
    if lattice_system == "Hexagonal" or lattice_system == "Tetragonal":
        return [0, 2]
    if lattice_system == "Rhombohedral":
        return [0, 3]
    if lattice_system == "Orthorhombic":
        return [0, 1, 2]
    if lattice_system == "Monoclinic":
        return [0, 1, 2, 4]
    return [0, 1, 2, 3, 4, 5]


def get_lattice_system(
    a, b, c, alpha, beta, gamma, space_group_name, atol_len=0.05, atol_ang=0.5
):
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
        sys_str = str(sg.crystal_system()).split(".")[-1].lower()
        centering = sg.centring_type()  # 'P', 'F', 'I', 'R', etc.
    except Exception:
        sys_str = "triclinic"  # Fallback
        centering = "P"

    # Map to internal types
    expected = "Triclinic"
    if sys_str == "cubic":
        expected = "Cubic"
    elif sys_str == "hexagonal":
        expected = "Hexagonal"
    elif sys_str == "trigonal":
        expected = "Rhombohedral" if centering == "R" else "Hexagonal"
    elif sys_str == "tetragonal":
        expected = "Tetragonal"
    elif sys_str == "orthorhombic":
        expected = "Orthorhombic"
    elif sys_str == "monoclinic":
        expected = "Monoclinic"

    # --- 2. Check Constraints & Warn ---
    def is_90(x):
        return np.isclose(x, 90.0, atol=atol_ang)

    def is_120(x):
        return np.isclose(x, 120.0, atol=atol_ang)

    def eq(x, y):
        return np.isclose(x, y, atol=atol_len)

    violation_msg = []

    if expected == "Cubic":
        if not (eq(a, b) and eq(b, c)):
            violation_msg.append("a=b=c")
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)):
            violation_msg.append("angles=90")
    elif expected == "Hexagonal":
        if not eq(a, b):
            violation_msg.append("a=b")
        if not (is_90(alpha) and is_90(beta) and is_120(gamma)):
            violation_msg.append("angles=90,90,120")
    elif expected == "Rhombohedral":
        if not (eq(a, b) and eq(b, c)):
            violation_msg.append("a=b=c")
        if not (eq(alpha, beta) and eq(beta, gamma)):
            violation_msg.append("alpha=beta=gamma")
    elif expected == "Tetragonal":
        if not eq(a, b):
            violation_msg.append("a=b")
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)):
            violation_msg.append("angles=90")
    elif expected == "Orthorhombic":
        if not (is_90(alpha) and is_90(beta) and is_90(gamma)):
            violation_msg.append("angles=90")
    elif expected == "Monoclinic":
        # Assuming b-unique or c-unique depending on settings, roughly check if at least two are 90
        count90 = sum([is_90(alpha), is_90(beta), is_90(gamma)])
        if count90 < 2:
            violation_msg.append("at least two angles=90")

    if violation_msg:
        warnings.warn(
            f"\n[Lattice System] Input parameters violate {space_group_name} ({expected}) constraints: {', '.join(violation_msg)}.\n"
            f"optimization will enforce {expected} constraints, which may cause a jump in parameters.",
            stacklevel=2,
        )

    # --- 3. Geometric Inference (Legacy Logic) ---
    geometric = "Triclinic"
    if is_90(alpha) and is_90(beta) and is_90(gamma):
        if eq(a, b) and eq(b, c):
            geometric = "Cubic"
        elif eq(a, b):
            geometric = "Tetragonal"
        else:
            geometric = "Orthorhombic"
    elif is_90(alpha) and is_90(beta) and is_120(gamma):
        if eq(a, b):
            geometric = "Hexagonal"
    elif (
        centering == "R"
        and eq(a, b)
        and eq(b, c)
        and eq(alpha, beta)
        and eq(beta, gamma)
    ):
        geometric = "Rhombohedral"
    elif sum([is_90(alpha), is_90(beta), is_90(gamma)]) >= 2:
        geometric = "Monoclinic"

    # --- 4. Hierarchy and Override ---
    # Rank symmetries: Lower number = Lower Symmetry (More free params)
    ranks = {
        "Triclinic": 0,
        "Monoclinic": 1,
        "Orthorhombic": 2,
        "Tetragonal": 3,
        "Rhombohedral": 4,
        "Hexagonal": 4,
        "Cubic": 5,
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
    if final_system == "Triclinic":
        num = 6
    elif final_system == "Monoclinic":
        num = 4
    elif final_system == "Orthorhombic":
        num = 3
    elif (
        final_system == "Tetragonal"
        or final_system == "Hexagonal"
        or final_system == "Rhombohedral"
    ):
        num = 2
    elif final_system == "Cubic":
        num = 1
    else:
        num = 6

    if rank_exp < rank_geo:
        print(
            f"Lattice System Override: Geometry suggests {geometric}, but Space Group {space_group_name} requires {expected}. Enforcing {expected} (Lower Symmetry)."
        )

    return final_system, num


def rotation_matrix_from_axis_angle_jax(axis, angle_rad):
    u = axis / jnp.linalg.norm(axis)
    ux, uy, uz = u
    K = jnp.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    eye = jnp.eye(3)
    R = eye + s[..., None, None] * K + (1.0 - c)[..., None, None] * (K @ K)
    return R


def rotation_matrix_from_rodrigues_jax(w):
    theta = jnp.linalg.norm(w) + 1e-9
    k = w / theta
    K = jnp.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    eye = jnp.eye(3)
    R = eye + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
    return R


# ==============================================================================
# 2. VECTORIZED OBJECTIVE (JAX)
# ==============================================================================


class VectorizedObjective:
    def __init__(
        self,
        B,
        kf_ki_dir,
        peak_xyz_lab,
        wavelength,
        angle_cdf,
        angle_t,
        weights=None,
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
        peak_radii=None,
        loss_method="gaussian",
        hkl_search_range=15,
        d_min=None,
        d_max=100.0,
        search_window_size=256,
        window_batch_size=32,
        chunk_size=4096,
        num_iters=20,
        top_k=32,
        space_group="P 1",
        centering="P",
        static_R=None,
        kf_lab_fixed_vectors=None,
        peak_run_indices=None,
    ):
        self.B = jnp.array(B)
        self.kf_ki_dir_init = jnp.array(kf_ki_dir)
        if self.kf_ki_dir_init.ndim == 2:
            if self.kf_ki_dir_init.shape[0] != 3 and self.kf_ki_dir_init.shape[1] == 3:
                self.kf_ki_dir_init = self.kf_ki_dir_init.T

        self.k_sq_init = jnp.sum(self.kf_ki_dir_init**2, axis=0)
        num_peaks = self.kf_ki_dir_init.shape[1]

        self.centering = centering

        # Convert tolerance from degrees to radians
        self.tolerance_rad = jnp.deg2rad(tolerance_deg)

        # FIX: Handle Static Rotation (R) correctly
        if static_R is not None:
            self.static_R = jnp.array(static_R)
        else:
            self.static_R = jnp.eye(3)

        # Handle Peak-to-Run mapping metadata
        if peak_run_indices is not None:
            self.peak_run_indices = jnp.array(peak_run_indices, dtype=jnp.int32)
            # Validation: Ensure R stack is large enough for the max run_index
            if self.static_R.ndim == 3:
                max_run = jnp.max(self.peak_run_indices)
                num_rot = self.static_R.shape[0]
                if max_run >= num_rot:
                    # If we only have ONE rotation, broadcast it to match the peaks
                    if num_rot == 1:
                        self.static_R = jnp.tile(self.static_R, (max_run + 1, 1, 1))
                    else:
                        # Major mismatch: Force everything to run 0 to prevent crash, but warn
                        # (JAX doesn't warn easily in JIT, so we'll just clamp later)
                        pass
        # Default heuristic:
        # 1. If R is a stack of N rotations and we have N peaks, assume 1-to-1 mapping.
        elif self.static_R.ndim == 3:
            num_rotations = self.static_R.shape[0]
            if num_rotations == num_peaks:
                self.peak_run_indices = jnp.arange(num_peaks, dtype=jnp.int32)
            else:
                # Fallback: everything to run 0 if we can't decide
                self.peak_run_indices = jnp.zeros(num_peaks, dtype=jnp.int32)
        else:
            self.peak_run_indices = jnp.zeros(num_peaks, dtype=jnp.int32)

        # Final safety: Clamp run indices to R stack bounds to prevent UB in JAX
        if self.static_R.ndim == 3:
            self.peak_run_indices = jnp.clip(
                self.peak_run_indices, 0, self.static_R.shape[0] - 1
            )

        if peak_xyz_lab is not None:
            # peak_xyz_lab is (N, 3) or (3, N). We want (3, N).
            p_xyz = jnp.array(peak_xyz_lab)
            if p_xyz.shape[0] != 3 and p_xyz.shape[1] == 3:
                p_xyz = p_xyz.T
            self.peak_xyz = p_xyz
        else:
            self.peak_xyz = None

        self.refine_sample = refine_sample
        self.sample_bound = sample_bound_meters
        if sample_nominal is None:
            self.sample_nominal = jnp.zeros(3)
        else:
            self.sample_nominal = jnp.array(sample_nominal)

        self.refine_beam = refine_beam
        self.beam_bound_deg = beam_bound_deg
        if beam_nominal is None:
            self.beam_nominal = jnp.array([0.0, 0.0, 1.0])
        else:
            self.beam_nominal = jnp.array(beam_nominal)

        # Reconstruct kf from Q (kf = Q + ki)
        self.kf_lab_fixed = None
        if self.peak_xyz is not None:
            # Always calculate kf from physical detector positions and sample offset
            # peak_xyz is (3, N), sample_nominal is (3,)
            v = self.peak_xyz - self.sample_nominal[:, None]
            dist = jnp.linalg.norm(v, axis=0)
            self.kf_lab_fixed = v / jnp.where(dist == 0, 1.0, dist[None, :])
            # Input is in LAB frame, so it is NOT yet rotated to Sample frame.

        if kf_lab_fixed_vectors is not None and self.kf_lab_fixed is None:
            # Input was Lab Frame. Q_lab = kf_lab - ki_lab.
            q_vecs = jnp.array(kf_lab_fixed_vectors)
            if q_vecs.shape[0] != 3 and q_vecs.shape[1] == 3:
                q_vecs = q_vecs.T
            self.kf_lab_fixed = q_vecs + self.beam_nominal[:, None]
            self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(
                self.kf_lab_fixed, axis=0
            )

        if self.kf_lab_fixed is None:
            # Fallback
            q_vecs = self.kf_ki_dir_init
            if q_vecs.shape[0] != 3 and q_vecs.shape[1] == 3:
                q_vecs = q_vecs.T
            self.kf_lab_fixed = q_vecs + self.beam_nominal[:, None]
            self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(
                self.kf_lab_fixed, axis=0
            )
            # FIX: Lab angles (two_theta, azimuthal) are ALWAYS in Lab frame.
            # We must ensure the optimizer
            # applies the Lab -> Sample rotation (R^T) during objective evaluation.

        self.tolerance_deg = tolerance_deg
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
            if cell_params is None:
                raise ValueError("cell_params required")
            self.cell_init = jnp.array(cell_params)
            if self.lattice_system == "Cubic":
                self.free_params_init = self.cell_init[0:1]
            elif (
                self.lattice_system == "Hexagonal"
                or self.lattice_system == "Tetragonal"
            ):
                self.free_params_init = jnp.array(
                    [self.cell_init[0], self.cell_init[2]]
                )
            elif self.lattice_system == "Rhombohedral":
                self.free_params_init = jnp.array(
                    [self.cell_init[0], self.cell_init[3]]
                )
            elif self.lattice_system == "Orthorhombic":
                self.free_params_init = self.cell_init[0:3]
            elif self.lattice_system == "Monoclinic":
                self.free_params_init = jnp.array(
                    [
                        self.cell_init[0],
                        self.cell_init[1],
                        self.cell_init[2],
                        self.cell_init[4],
                    ]
                )
            else:
                self.free_params_init = self.cell_init

        if goniometer_axes is not None:
            axes = jnp.array(goniometer_axes)
            if axes.ndim == 2 and axes.shape[1] == 3:
                # Fallback for 3-component axes: add 1.0 orientation (CCW)
                axes = jnp.concatenate([axes, jnp.ones((axes.shape[0], 1))], axis=1)
            self.gonio_axes = axes

            angles = jnp.array(goniometer_angles)
            if angles.ndim == 2:
                # Expecting (num_axes, num_runs). If (num_runs, num_axes), transpose.
                if (
                    angles.shape[0] != self.gonio_axes.shape[0]
                    and angles.shape[1] == self.gonio_axes.shape[0]
                ):
                    angles = angles.T
            self.gonio_angles = angles
            self.num_gonio_axes = self.gonio_axes.shape[0]

            # CRITICAL: If gonio_angles is per-peak, force per-peak run mapping
            # to ensure R_per_peak = R[peak_run_indices] works correctly.
            if self.gonio_angles.shape[1] == num_peaks:
                self.peak_run_indices = jnp.arange(num_peaks, dtype=jnp.int32)

            if goniometer_refine_mask is not None:
                self.gonio_mask = np.array(goniometer_refine_mask, dtype=bool)
            else:
                self.gonio_mask = np.ones(self.num_gonio_axes, dtype=bool)
            self.num_active_gonio = np.sum(self.gonio_mask)

            if goniometer_nominal_offsets is None:
                self.gonio_nominal_offsets = jnp.zeros(self.num_gonio_axes)
            else:
                self.gonio_nominal_offsets = jnp.array(goniometer_nominal_offsets)

            self.gonio_min = jnp.full(self.num_gonio_axes, -goniometer_bound_deg)
            self.gonio_max = jnp.full(self.num_gonio_axes, goniometer_bound_deg)
        else:
            self.gonio_axes = None
            self.num_gonio_axes = 0

        wavelength = jnp.array(wavelength)
        self.wl_min_val = wavelength[0]
        self.wl_max_val = wavelength[1]
        self.num_candidates = 64

        if weights is None:
            self.weights = jnp.ones(num_peaks)
        else:
            self.weights = jnp.array(weights).flatten()
            if self.weights.shape[0] != num_peaks:
                raise ValueError(
                    f"Weights shape {self.weights.shape} does not match num_peaks {num_peaks}"
                )

        if peak_radii is None:
            self.peak_radii = jnp.zeros(num_peaks)
        else:
            self.peak_radii = jnp.array(peak_radii).flatten()
            if self.peak_radii.shape[0] != num_peaks:
                raise ValueError(
                    f"Peak radii shape {self.peak_radii.shape} does not match num_peaks {num_peaks}"
                )

        self.max_score = jnp.sum(self.weights)
        self.d_min = d_min if d_min is not None else 0.0
        self.d_max = d_max if d_max is not None else 1000.0
        self.search_window_size = search_window_size
        self.window_batch_size = window_batch_size

        self.chunk_size = chunk_size
        self.num_iters = num_iters
        self.top_k = top_k

        # --- Search Window Heuristic Warning ---
        if self.loss_method == "forward":
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
                        f"RECOMMENDED SIZE: >= {heuristic_win}\n",
                        stacklevel=2,
                    )

        # --- HKL Mask Generation ---
        # Robustly determine search range from cell and observed resolution
        # inv(B @ B.T) = inv(G*) = G (Real space metric tensor)
        # sqrt(diag(G)) = [a, b, c]
        a_real, b_real, c_real = jnp.sqrt(jnp.diag(jnp.linalg.inv(self.B @ self.B.T)))

        # Calculate resolution of observed peaks
        q_obs_max = jnp.max(jnp.linalg.norm(self.kf_ki_dir_init, axis=0))
        d_min_obs = 1.0 / (q_obs_max + 1e-9)

        # Determine pool resolution limit
        d_limit = self.d_min if self.d_min > 0 else d_min_obs

        # h_max = a / d_min
        h_max_res = int(jnp.ceil(a_real / d_limit))
        k_max_res = int(jnp.ceil(b_real / d_limit))
        l_max_res = int(jnp.ceil(c_real / d_limit))
        h_max = max(hkl_search_range, h_max_res)
        k_max = max(hkl_search_range, k_max_res)
        l_max = max(hkl_search_range, l_max_res)

        # Clamp to a reasonable maximum to prevent OOM
        h_max = min(h_max, 64)
        k_max = min(k_max, 64)
        l_max = min(l_max, 64)

        print(
            f"Generating HKL pool for Space Group: {self.space_group} (Range: {h_max},{k_max},{l_max})"
        )

        r_h = jnp.arange(-h_max, h_max + 1)
        r_k = jnp.arange(-k_max, k_max + 1)
        r_l = jnp.arange(-l_max, l_max + 1)
        h, k, l = jnp.meshgrid(r_h, r_k, r_l, indexing="ij")  # noqa: E741
        hkl_pool = jnp.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)

        # Apply Symmetry Mask to Pool
        mask_cpu = generate_hkl_mask(h_max, k_max, l_max, self.space_group)
        self.valid_hkl_mask = jnp.array(mask_cpu)
        self.mask_range_h = h_max
        self.mask_range_k = k_max
        self.mask_range_l = l_max
        self.mask_range = (
            h_max  # Alias for backward compatibility with cubic-assumption tests
        )

        idx_h = hkl_pool[0] + h_max
        idx_k = hkl_pool[1] + k_max
        idx_l = hkl_pool[2] + l_max
        allowed_pool = self.valid_hkl_mask[idx_h, idx_k, idx_l]

        hkl_pool = hkl_pool[:, allowed_pool]
        q_cart = self.B @ hkl_pool

        # NOTE: For Sinkhorn, we keep the flat pool available directly
        self.pool_hkl_flat = hkl_pool

        phis = jnp.arctan2(q_cart[1], q_cart[0])
        sort_idx = jnp.argsort(phis)
        self.pool_phi_sorted = phis[sort_idx]
        self.pool_hkl_sorted = hkl_pool[:, sort_idx]

        # --- PINNING INITIALIZATION ---
        # Pre-calculate reference HKL and Q magnitudes to prevent lattice bias
        # in derivative-free optimization. Assume identity orientation for pinning.
        B_inv_init = jnp.linalg.inv(self.B)
        h_init = B_inv_init @ self.kf_ki_dir_init
        self.hkl_mag_sq_pinned = jnp.sum(h_init**2, axis=0, keepdims=True)
        self.hkl_mag_sq_pinned = jnp.maximum(self.hkl_mag_sq_pinned, 1e-6)

        # Reference Lambda for soft/binary kernels
        # lambda = |k|^2 / (k . Q)
        k_dot_q_init = jnp.sum(self.kf_ki_dir_init * self.kf_ki_dir_init, axis=0)
        self.safe_lamb_pinned = jnp.clip(
            self.k_sq_init / jnp.maximum(k_dot_q_init, 1e-9),
            self.wl_min_val,
            self.wl_max_val,
        )[None, :]

        # Reference Pool Norms for Sinkhorn
        # Use padded pool to match chunked indexing in sinkhorn
        pad_len = (chunk_size - (hkl_pool.shape[1] % chunk_size)) % chunk_size
        hkl_pool_padded = (
            jnp.pad(hkl_pool, ((0, 0), (0, pad_len)), constant_values=0)
            if pad_len > 0
            else hkl_pool
        )
        q_pool_init = self.B @ hkl_pool_padded
        self.pool_norm_q_pinned = jnp.sqrt(jnp.sum(q_pool_init**2, axis=0) + 1e-9)

    def _get_physical_params_jax(self, x):
        """Reconstruct physical parameters (Base + Delta) for a batch of solutions x."""
        idx = 0
        rot_params = x[:, idx : idx + 3]
        U = self.orientation_U_jax(rot_params)
        idx += 3

        if self.refine_lattice:
            n_lat = self.free_params_init.size
            cell_params_norm = x[:, idx : idx + n_lat]
            B = self.compute_B_jax(cell_params_norm)
            idx += n_lat
            # Broadcase UB calculation: (S, 3, 3) @ (S, 3, 3) -> (S, 3, 3)
            UB = jnp.matmul(U, B)
        else:
            B = self.B
            # (S, 3, 3) @ (3, 3) -> (S, 3, 3)
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
                gonio_norm = jnp_update_set(
                    gonio_norm,
                    (slice(None), self.gonio_mask),
                    x[:, idx : idx + self.num_active_gonio],
                )
                idx += self.num_active_gonio

            offsets_delta = _forward_map_param(gonio_norm, self.goniometer_bound_deg)
            offsets_total = self.gonio_nominal_offsets + offsets_delta
            R = self.compute_goniometer_R_jax(
                gonio_norm
            )  # Helper assumes input is norm
        elif self.gonio_axes is not None:
            offsets_total = self.gonio_nominal_offsets[None, :].repeat(
                x.shape[0], axis=0
            )
            # To calculate R from Nominal (fixed), we pass 0.5 to helper
            gonio_norm = jnp.full((x.shape[0], self.num_gonio_axes), 0.5)
            R = self.compute_goniometer_R_jax(gonio_norm)
        else:
            offsets_total = None
            R = None

        return UB, B, sample_total, ki_vec, offsets_total, R

    def reconstruct_cell_params(self, params_norm):
        p_free = _forward_map_lattice(
            params_norm, self.free_params_init, self.lattice_bound_frac
        )
        S = params_norm.shape[0]
        deg90 = jnp.full((S,), 90.0)
        deg120 = jnp.full((S,), 120.0)
        if self.lattice_system == "Cubic":
            a = p_free[:, 0]
            return jnp.stack([a, a, a, deg90, deg90, deg90], axis=1)
        if self.lattice_system == "Hexagonal":
            a, c = p_free[:, 0], p_free[:, 1]
            return jnp.stack([a, a, c, deg90, deg90, deg120], axis=1)
        if self.lattice_system == "Tetragonal":
            a, c = p_free[:, 0], p_free[:, 1]
            return jnp.stack([a, a, c, deg90, deg90, deg90], axis=1)
        if self.lattice_system == "Rhombohedral":
            a, alpha = p_free[:, 0], p_free[:, 1]
            return jnp.stack([a, a, a, alpha, alpha, alpha], axis=1)
        if self.lattice_system == "Orthorhombic":
            a, b, c = p_free[:, 0], p_free[:, 1], p_free[:, 2]
            return jnp.stack([a, b, c, deg90, deg90, deg90], axis=1)
        if self.lattice_system == "Monoclinic":
            a, b, c, beta = (
                p_free[:, 0],
                p_free[:, 1],
                p_free[:, 2],
                p_free[:, 3],
            )
            return jnp.stack([a, b, c, deg90, beta, deg90], axis=1)
        return p_free

    def compute_B_jax(self, cell_params_norm):
        p = self.reconstruct_cell_params(cell_params_norm)
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        deg2rad = jnp.pi / 180.0
        alpha, beta, gamma = (
            p[:, 3] * deg2rad,
            p[:, 4] * deg2rad,
            p[:, 5] * deg2rad,
        )
        g11, g22, g33 = a**2, b**2, c**2
        g12, g13, g23 = (
            a * b * jnp.cos(gamma),
            a * c * jnp.cos(beta),
            b * c * jnp.cos(alpha),
        )
        row1 = jnp.stack([g11, g12, g13], axis=-1)
        row2 = jnp.stack([g12, g22, g23], axis=-1)
        row3 = jnp.stack([g13, g23, g33], axis=-1)
        G = jnp.stack([row1, row2, row3], axis=-2)
        G_star = jnp.linalg.inv(G)
        B = jscipy_linalg.cholesky(G_star, lower=False)
        return B

    def compute_goniometer_R_jax(self, gonio_offsets_norm):
        # NOTE: This helper uses Norm to calc Delta, then adds Nominal from self.
        offsets_delta = _forward_map_param(
            gonio_offsets_norm, self.goniometer_bound_deg
        )
        total_offsets = self.gonio_nominal_offsets + offsets_delta

        angles_deg = total_offsets[:, :, None] + self.gonio_angles[None, :, :]
        S, M = total_offsets.shape[0], self.gonio_angles.shape[1]
        R = jnp.eye(3)[None, None, ...].repeat(S, axis=0).repeat(M, axis=1)
        deg2rad = jnp.pi / 180.0
        for i in range(self.num_gonio_axes):
            axis_spec = self.gonio_axes[i]
            direction = axis_spec[0:3]
            sign = axis_spec[3]
            theta = sign * angles_deg[:, i, :] * deg2rad
            Ri = rotation_matrix_from_axis_angle_jax(direction, theta)
            # Mantid SetGoniometer: R = R0 @ R1 @ R2
            # Each Ri should be multiplied on the RIGHT of the current accumulated matrix.
            # Batched matmul: (S, M, 3, 3) @ (S, M, 3, 3) -> (S, M, 3, 3)
            R = jnp.matmul(R, Ri)
        return R

    def orientation_U_jax(self, param):
        U = jax.vmap(rotation_matrix_from_rodrigues_jax)(param)
        return U

    def indexer_dynamic_soft_jax(
        self, ub_mat, kf_ki_sample, k_sq_override=None, tolerance_rad=0.002
    ):
        ub_inv = jnp.linalg.inv(ub_mat)
        # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
        v = jnp.matmul(ub_inv, kf_ki_sample)
        abs_v = jnp.abs(v)
        max_v_val = jnp.max(abs_v, axis=1)
        n_start = max_v_val / self.wl_max_val
        start_int = jnp.ceil(n_start)
        k_sq = k_sq_override if k_sq_override is not None else self.k_sq_init[None, :]
        k_norm = jnp.sqrt(k_sq)

        initial_carry = (
            jnp.zeros(max_v_val.shape),
            jnp.zeros(max_v_val.shape),
            jnp.zeros((v.shape[0], 3, v.shape[2]), dtype=jnp.int32),
            jnp.zeros(max_v_val.shape),
        )

        def scan_body(carry, i):
            curr_sum, curr_max, curr_best_hkl, curr_best_lamb = carry
            n = start_int + i
            n_safe = jnp.where(n == 0, 1e-9, n)
            lamda_cand = max_v_val / n_safe
            hkl_float = v / lamda_cand[:, None, :]
            hkl_int = jnp.round(hkl_float).astype(jnp.int32)
            # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
            q_int = jnp.matmul(ub_mat, hkl_int.astype(jnp.float32))
            k_dot_q = jnp.sum(kf_ki_sample * q_int, axis=1)
            safe_dot = jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
            lambda_opt = jnp.clip(k_sq / safe_dot, self.wl_min_val, self.wl_max_val)
            q_obs = kf_ki_sample / lambda_opt[:, None, :]
            dist_sq = jnp.sum((q_obs - q_int) ** 2, axis=1)

            effective_sigma = (tolerance_rad + self.peak_radii[None, :]) * (
                k_norm / lambda_opt
            )
            # Robust Multi-Scale Kernel
            # 1. Narrow (High precision)
            log_p_narrow = -dist_sq / (2 * effective_sigma**2 + 1e-9)

            # 2. Wide (Capture range: 5 degrees)
            sigma_wide = jnp.deg2rad(5.0) * (k_norm / lambda_opt)
            log_p_wide = -dist_sq / (2 * sigma_wide**2 + 1e-9)

            # Combine via LogSumExp with 1% weight on wide kernel
            log_prob = jax.nn.logsumexp(
                jnp.stack([log_p_narrow, log_p_wide - 4.605]), axis=0
            )
            prob = jnp.exp(log_prob)

            # 1. Calc |Q|^2 for predicted HKL
            q_sq_pred = jnp.sum(q_int**2, axis=1)
            # 2. Convert to d = 1/|Q| (Crystallographic units)
            d_pred = 1.0 / jnp.sqrt(q_sq_pred + 1e-9)
            valid_res = (d_pred >= self.d_min) & (d_pred <= self.d_max)

            h, k, l = (  # noqa: E741
                hkl_int[:, 0, :],
                hkl_int[:, 1, :],
                hkl_int[:, 2, :],
            )
            is_allowed = self.is_allowed_jax(h, k, l)

            # Combine masks
            final_mask = is_allowed & valid_res

            prob = jnp.where(final_mask, prob, 0.0)

            new_sum = curr_sum + prob
            update_mask = prob > curr_max
            new_max = jnp.where(update_mask, prob, curr_max)
            new_best_hkl = jnp.where(update_mask[:, None, :], hkl_int, curr_best_hkl)
            new_best_lamb = jnp.where(update_mask, lambda_opt, curr_best_lamb)
            return (new_sum, new_max, new_best_hkl, new_best_lamb), None

        final_carry, _ = lax.scan(
            scan_body, initial_carry, jnp.arange(self.num_candidates)
        )
        accum_probs, prob_max, best_hkl, best_lamb = final_carry
        # score = -jnp.sum(self.weights * accum_probs, axis=1) # Original sum
        score = -jnp.sum(self.weights * prob_max, axis=1)
        return score, prob_max, best_hkl.transpose((0, 2, 1)), best_lamb

    def indexer_dynamic_cosine_aniso_jax(
        self, ub_mat, kf_ki_sample, *, k_sq_override=None, tolerance_rad=0.002
    ):
        # Use solve for better precision than inv + matmul
        v = jnp.linalg.solve(ub_mat, kf_ki_sample)
        abs_v = jnp.abs(v)
        max_v_val = jnp.max(abs_v, axis=1)
        n_start = max_v_val / self.wl_max_val
        start_int = jnp.ceil(n_start)

        k_sq = k_sq_override if k_sq_override is not None else self.k_sq_init[None, :]

        # kappa for von Mises-Fisher-like concentration in HKL space
        # Uniform angular tolerance: sigma_h approx tolerance_rad * h.

        initial_carry = (
            jnp.full(max_v_val.shape, -1e12),
            jnp.full(max_v_val.shape, -1e12),
            jnp.zeros((v.shape[0], 3, v.shape[2]), dtype=jnp.int32),
            jnp.zeros(max_v_val.shape),
        )

        def scan_body(carry, i):
            curr_sum, curr_max, curr_best_hkl, curr_best_lamb = carry
            n = start_int + i
            n_safe = jnp.where(n == 0, 1e-9, n)
            lamda_cand = max_v_val / n_safe

            # --- DYNAMIC WAVELENGTH OPTIMIZATION ---
            # Instead of just using lamda_cand, we find the lambda that best
            # satisfies the Laue condition for the nearest integer HKL.
            hkl_int = jnp.round(v / lamda_cand[:, None, :]).astype(jnp.int32)
            # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
            q_int = jnp.matmul(ub_mat, hkl_int.astype(jnp.float32))
            k_dot_q = jnp.sum(kf_ki_sample * q_int, axis=1)
            safe_dot = jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
            lambda_opt = jnp.clip(k_sq / safe_dot, self.wl_min_val, self.wl_max_val)

            # Recalculate HKL float at the optimal wavelength for the cosine kernel
            hkl_float = v / lambda_opt[:, None, :]

            # Robust Multi-Scale Kernel: Mixture of Narrow + Wide peaks
            # We use an isotropic tolerance based on the total HKL magnitude
            # to represent a uniform angular tolerance (sigma_h approx tolerance_rad * |h|).
            # This prevents the 'delta function' behavior for components near zero.
            hkl_mag_sq = jnp.sum(hkl_float**2, axis=1, keepdims=True)
            # Use 1.0 as floor to ensure low-order reflections don't have infinite precision
            hkl_mag_sq_safe = jnp.maximum(hkl_mag_sq, 1.0)

            kappa_scaled = 1.0 / (
                hkl_mag_sq_safe * (tolerance_rad + 1e-9) ** 2 * 4 * jnp.pi**2
            )
            # Use stable sin^2 form: cos(2pi x) - 1 = -2 sin^2(pi x)
            cos_diff_stable = -2.0 * jnp.sin(jnp.pi * hkl_float) ** 2
            log_p_narrow = jnp.sum(kappa_scaled * cos_diff_stable, axis=1)

            kappa_wide_scaled = 1.0 / (
                hkl_mag_sq_safe * jnp.deg2rad(5.0) ** 2 * 4 * jnp.pi**2
            )
            log_p_wide = jnp.sum(kappa_wide_scaled * cos_diff_stable, axis=1)

            # Combine via LogSumExp with 1% weight on wide kernel
            log_prob = jax.nn.logsumexp(
                jnp.stack([log_p_narrow, log_p_wide - 4.605]), axis=0
            )

            # --- VALIDATION LOGIC ---
            # Resolution Filter
            q_sq = jnp.sum(q_int**2, axis=1)
            d_est = 1.0 / jnp.sqrt(q_sq + 1e-9)
            valid_res = (d_est >= self.d_min) & (d_est <= self.d_max)

            # Symmetry Mask
            h, k, l = (  # noqa: E741
                hkl_int[:, 0, :],
                hkl_int[:, 1, :],
                hkl_int[:, 2, :],
            )
            is_allowed = self.is_allowed_jax(h, k, l)

            # Combine all masks
            final_mask = is_allowed & valid_res

            # Use LogSumExp style accumulation for robustness
            log_prob_masked = jnp.where(final_mask, log_prob, -1e12)

            # Update carry
            # curr_sum will now store the logsumexp of valid candidates
            new_sum = jax.nn.logsumexp(jnp.stack([curr_sum, log_prob_masked]), axis=0)

            update_mask = log_prob_masked > curr_max
            new_max = jnp.where(update_mask, log_prob_masked, curr_max)
            new_best_hkl = jnp.where(update_mask[:, None, :], hkl_int, curr_best_hkl)
            new_best_lamb = jnp.where(update_mask, lambda_opt, curr_best_lamb)
            return (new_sum, new_max, new_best_hkl, new_best_lamb), None

        final_carry, _ = lax.scan(
            scan_body, initial_carry, jnp.arange(self.num_candidates)
        )
        accum_probs, log_prob_max, best_hkl, best_lamb = final_carry
        # score = -jnp.sum(self.weights * jnp.exp(accum_probs), axis=1)
        # Use max probability per peak for the score
        score = -jnp.sum(self.weights * jnp.exp(log_prob_max), axis=1)
        return (
            score,
            jnp.exp(log_prob_max),
            best_hkl.transpose((0, 2, 1)),
            best_lamb,
        )

    def indexer_dynamic_binary_jax(
        self,
        ub_mat,
        kf_ki_sample,
        k_sq_override=None,
        tolerance_rad=0.002,
        window_batch_size=32,
    ):
        k_sq = k_sq_override if k_sq_override is not None else self.k_sq_init[None, :]
        k_norm = jnp.sqrt(k_sq)
        ub_inv = jnp.linalg.inv(ub_mat)
        # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
        hkl_float = jnp.matmul(ub_inv, kf_ki_sample)
        # Broadcasted matmul: (3, 3) @ (S, 3, N) -> (S, 3, N)
        hkl_cart_approx = jnp.matmul(self.B[None, ...], hkl_float)
        phi_obs = jnp.arctan2(hkl_cart_approx[:, 1, :], hkl_cart_approx[:, 0, :])
        idx_centers = jnp.searchsorted(self.pool_phi_sorted, phi_obs)
        half_win = self.search_window_size // 2
        raw_offsets = jnp.arange(-half_win, half_win + 1)
        pad_len = (
            window_batch_size - (raw_offsets.shape[0] % window_batch_size)
        ) % window_batch_size
        offsets_padded = jnp.pad(
            raw_offsets, (0, pad_len), constant_values=raw_offsets[-1]
        )
        offset_batches = offsets_padded.reshape(-1, window_batch_size)
        init_min_dist = jnp.full(idx_centers.shape, 1e9)
        init_best_hkl = jnp.zeros((*idx_centers.shape, 3))
        init_best_lamb = jnp.zeros(idx_centers.shape)
        init_carry = (init_min_dist, init_best_hkl, init_best_lamb)

        def scan_body(carry, batch_offsets):
            curr_min_dist, curr_best_hkl, curr_best_lamb = carry
            gather_idx = idx_centers[..., None] + batch_offsets[None, None, :]
            pool_T = self.pool_hkl_sorted.T
            hkl_cands = jnp.take(pool_T, gather_idx, axis=0, mode="wrap")
            # Broadcasted matmul: (S, 3, 3) @ (S, M, W, 3, 1) -> (S, M, W, 3, 1)
            # hkl_cands is (S, M, W, 3)
            q_pred = jnp.matmul(
                ub_mat[:, None, None, ...], hkl_cands[..., None]
            ).squeeze(-1)
            k_obs = jnp.transpose(kf_ki_sample, (0, 2, 1))[:, :, None, :]
            k_dot_q = jnp.sum(k_obs * q_pred, axis=3)
            lambda_opt = k_sq[..., None] / jnp.where(
                jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q
            )
            valid_lamb = (lambda_opt >= self.wl_min_val) & (
                lambda_opt <= self.wl_max_val
            )
            q_sq = jnp.sum(q_pred**2, axis=3)
            d_spacings = 1.0 / jnp.sqrt(q_sq + 1e-9)  # crystallographic convention
            valid_res = (d_spacings >= self.d_min) & (d_spacings <= self.d_max)
            h, k, l = (  # noqa: E741
                hkl_cands[..., 0],
                hkl_cands[..., 1],
                hkl_cands[..., 2],
            )
            valid_sym = self.is_allowed_jax(h, k, l)
            valid_mask = valid_lamb & valid_res & valid_sym
            q_obs_opt = k_obs / jnp.where(lambda_opt == 0, 1.0, lambda_opt)[..., None]
            diff = q_obs_opt - q_pred
            dist_sq = jnp.sum(diff**2, axis=3)
            dist_sq_masked = jnp.where(valid_mask, dist_sq, 1e9)
            batch_min_dist = jnp.min(dist_sq_masked, axis=2)
            batch_best_local_idx = jnp.argmin(dist_sq_masked, axis=2)
            batch_best_hkl = jnp.take_along_axis(
                hkl_cands, batch_best_local_idx[..., None, None], axis=2
            ).squeeze(axis=2)
            batch_best_lamb = jnp.take_along_axis(
                lambda_opt, batch_best_local_idx[..., None], axis=2
            ).squeeze(axis=2)
            improve_mask = batch_min_dist < curr_min_dist
            new_min_dist = jnp.where(improve_mask, batch_min_dist, curr_min_dist)
            new_best_hkl = jnp.where(
                improve_mask[..., None], batch_best_hkl, curr_best_hkl
            )
            new_best_lamb = jnp.where(improve_mask, batch_best_lamb, curr_best_lamb)
            return (new_min_dist, new_best_hkl, new_best_lamb), None

        final_carry, _ = lax.scan(scan_body, init_carry, offset_batches)
        best_dist_sq, best_hkl, best_lamb = final_carry

        # Use dynamic lambda for accurate physical tolerance scaling
        effective_sigma = (tolerance_rad + self.peak_radii[None, :]) * (
            k_norm / best_lamb
        )
        probs = jnp.exp(-best_dist_sq / (2 * effective_sigma**2 + 1e-9))
        score = -jnp.sum(self.weights * probs, axis=1)
        return score, probs, best_hkl, best_lamb

    # ==========================================================================
    # OPTIMIZED SINKHORN-EM INDEXER (Memory Efficient + Rotation Trick)
    # ==========================================================================
    def indexer_sinkhorn_jax(
        self,
        ub_mat,
        kf_ki_sample,
        k_sq_override=None,
        tolerance_rad=0.002,
        num_iters=20,
        epsilon=1.0,
        top_k=32,
        chunk_size=256,
    ):
        """
        Robust Memory-Efficient Sinkhorn with Soft-Masking and Log-Stability.
        """
        # 1. Setup Data
        hkl_pool = self.pool_hkl_flat  # (3, N_hkl)

        # Normalize Obs
        norm_obs = jnp.linalg.norm(kf_ki_sample, axis=1, keepdims=True)
        r_obs_unit = kf_ki_sample / (norm_obs + 1e-9)

        # Re-project Unit Obs into Crystal Frame: (Batch, 3, N_obs) @ (Batch, 3, 3) -> (Batch, 3, N_obs)
        # We need r_obs_unit_crystal = U^T @ r_obs_unit_lab
        # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
        r_obs_proj_unit = jnp.matmul(ub_mat.transpose(0, 2, 1), r_obs_unit)

        k_sq_obs = (
            k_sq_override if k_sq_override is not None else self.k_sq_init[None, :]
        )

        batch_size, _, n_obs = kf_ki_sample.shape
        _, n_hkl = hkl_pool.shape

        # Bandwidth and Resolution Constants for Penalties
        wl_mid = 0.5 * (self.wl_min_val + self.wl_max_val)
        wl_half_width = 0.5 * (self.wl_max_val - self.wl_min_val)
        res_mid = 0.5 * (self.d_min + self.d_max)
        res_half_width = 0.5 * (self.d_max - self.d_min)

        # Pad pool for chunking
        pad_len = (chunk_size - (n_hkl % chunk_size)) % chunk_size
        hkl_pool_padded = (
            jnp.pad(hkl_pool, ((0, 0), (0, pad_len)), constant_values=0)
            if pad_len > 0
            else hkl_pool
        )
        n_hkl_padded = n_hkl + pad_len

        num_chunks = n_hkl_padded // chunk_size

        def scan_topk(carry, i):
            curr_vals, curr_idxs = carry
            idx_start = i * chunk_size
            hkl_chunk = jax.lax.dynamic_slice(
                hkl_pool_padded, (0, idx_start), (3, chunk_size)
            )

            # dot_raw = (r_obs @ U) . h
            # r_obs_proj_unit is (Batch, 3, N_obs), hkl_chunk is (3, Chunk)
            # Result (Batch, N_obs, Chunk)
            # (S, 3, N).T @ (3, C) -> (S, N, C)
            dot_raw = jnp.matmul(r_obs_proj_unit.transpose(0, 2, 1), hkl_chunk)

            # cosine = dot_raw / |UB h|

            # Use pinned norms to prevent the optimizer from 'cheating' by
            # enlarging the lattice to reduce the predicted |Q|.
            norm_q_chunk_pinned = jax.lax.dynamic_slice(
                self.pool_norm_q_pinned, (idx_start,), (chunk_size,)
            )
            dots_chunk = dot_raw / (norm_q_chunk_pinned[None, None, :] + 1e-9)

            # --- FIX: Resolution & Wavelength Aware Top-K ---
            # lambda = k_sq_obs / (norm_obs * dot_raw)
            # norm_obs is (batch, 1, n_obs), dot_raw is (batch, n_obs, chunk)
            safe_dot = jnp.maximum(dot_raw, 1e-6)
            est_lambda = k_sq_obs[:, :, None] / (
                norm_obs.transpose(0, 2, 1) * safe_dot + 1e-9
            )

            wl_penalty = -jnp.abs(est_lambda - wl_mid) / (wl_half_width + 1e-9)

            # norm_q_chunk_pinned is (chunk,)
            d_chunk = 1.0 / (norm_q_chunk_pinned + 1e-9)
            res_penalty = -jnp.abs(d_chunk - res_mid) / (res_half_width + 1e-9)

            selection_metric = (
                dots_chunk + 0.1 * wl_penalty + 0.1 * res_penalty[None, None, :]
            )

            # Handle padded 0,0,0 vectors (norm 0) by ensuring they have low metrics
            selection_metric = jnp.where(
                norm_q_chunk_pinned[None, None, :] < 1e-6, -1e9, selection_metric
            )

            global_idxs = (jnp.arange(chunk_size) + idx_start).astype(jnp.int32)
            combined_vals = jnp.concatenate([curr_vals, selection_metric], axis=2)
            combined_idxs = jnp.concatenate(
                [
                    curr_idxs,
                    jnp.tile(global_idxs[None, None, :], (batch_size, n_obs, 1)),
                ],
                axis=2,
            )

            vals, top_k_indices = jax.lax.top_k(combined_vals, top_k)
            idxs = jnp.take_along_axis(combined_idxs, top_k_indices, axis=2)
            return (vals, idxs), None

        (top_vals, top_idxs), _ = lax.scan(
            scan_topk,
            (
                jnp.full((batch_size, n_obs, top_k), -1e9),
                jnp.zeros((batch_size, n_obs, top_k), dtype=jnp.int32),
            ),
            jnp.arange(num_chunks),
        )

        # 3. Log-Kernel with Soft Penalties
        # Gather HKL vectors and re-calculate full geometry for top-k
        hkl_selected = jnp.take(hkl_pool_padded.T, top_idxs, axis=0)
        # ub_mat: (S, 3, 3), hkl_selected: (S, N, K, 3)
        # We want (S, N, K, 3)
        q_selected = jnp.matmul(
            ub_mat[:, None, None, ...], hkl_selected[..., None]
        ).squeeze(-1)
        q_sq_selected = jnp.sum(q_selected**2, axis=3)
        norm_q_selected = jnp.sqrt(q_sq_selected + 1e-9)

        # Actual cosines for top-k HKLs
        # (Batch, 3, N_obs) -> (Batch, N_obs, 3)
        k_obs_unit = r_obs_unit.transpose(0, 2, 1)[:, :, None, :]
        dot_selected = jnp.sum(k_obs_unit * q_selected, axis=3)
        top_cosines = dot_selected / (norm_q_selected + 1e-9)

        # Wavelength penalty
        k_obs_aligned = kf_ki_sample.transpose(0, 2, 1)[:, :, None, :]
        k_dot_q = jnp.sum(k_obs_aligned * q_selected, axis=-1)
        lambda_sparse = k_sq_obs[:, :, None] / (k_dot_q + 1e-9)

        # Soft Lambda Penalty (Gaussian penalty for being outside bandwidth)
        # Using a broader width (10% of bandwidth) to prevent numerical
        # drowning of angular signal
        bw_width = self.wl_max_val - self.wl_min_val
        dist_wl = jnp.maximum(0.0, self.wl_min_val - lambda_sparse) + jnp.maximum(
            0.0, lambda_sparse - self.wl_max_val
        )
        # Scale: lambda penalty should be comparable to angular penalty (order of 1-10)
        # dist_wl of 0.1A should not give 1e5 cost.
        log_P_wl = -0.5 * (dist_wl / (0.1 * bw_width + 1e-9)) ** 2

        # Soft Resolution Penalty
        d_sparse = 1.0 / norm_q_selected
        dist_res = jnp.maximum(0.0, self.d_min - d_sparse) + jnp.maximum(
            0.0, d_sparse - self.d_max
        )
        log_P_res = -0.5 * (dist_res / (0.1 * self.d_min + 1e-9)) ** 2

        # Angular kernel (Multi-scale: Peak + Wide Background)
        # Using a mixture of a narrow peak and a heavy-tailed background ensures
        # we have a strong gradient near the peak and a stable signal far away.
        dist_ang = 1.0 - top_cosines

        # log_K_peak = -dist_ang / tolerance^2
        # log_K_wide = -log(1 + dist_ang / (wide_tol^2))
        # We use LogSumExp to smoothly combine them
        log_K_peak = -dist_ang / (tolerance_rad**2 + 1e-9)

        wide_tol = jnp.deg2rad(5.0)  # Always have a 5 degree capture range
        log_K_wide = -jnp.log(1.0 + dist_ang / (wide_tol**2 + 1e-9))

        # Combine (mixing weight 0.5 implicitly via LogSumExp if we don't scale)
        log_K = jax.nn.logsumexp(jnp.stack([log_K_peak, log_K_wide]), axis=0)

        # Combine into robust log-likelihood
        log_K_robust = log_K + log_P_wl + log_P_res

        # --- TIE-BREAKER PENALTIES ---
        # If multiple HKLs have identical orientation error (cosines),
        # prefer the one that matches the expected wavelength and resolution center.
        # This breaks ties caused by the regularizer (1e-9) favoring larger vectors.
        log_P_wl_tie = -1e-4 * jnp.abs(lambda_sparse - wl_mid) / (wl_half_width + 1e-9)
        log_P_res_tie = -1e-4 * jnp.abs(d_sparse - res_mid) / (res_half_width + 1e-9)
        log_K_robust += log_P_wl_tie + log_P_res_tie

        # 5. Dustbin & Softmax
        # Dustbin represents the "null" HKL match
        # Match to dustbin if outside the wide capture range (e.g. 3 * wide_tol)
        outlier_threshold_rad = jnp.minimum(jnp.deg2rad(45.0), 3.0 * wide_tol)
        dist_outlier = 1.0 - jnp.cos(outlier_threshold_rad)
        log_K_dustbin_peak = -dist_outlier / (tolerance_rad**2 + 1e-9)
        log_K_dustbin_wide = -jnp.log(1.0 + dist_outlier / (wide_tol**2 + 1e-9))
        log_K_dustbin = jax.nn.logsumexp(
            jnp.stack([log_K_dustbin_peak, log_K_dustbin_wide])
        )
        log_K_dustbin = jnp.full((batch_size, n_obs, 1), log_K_dustbin)

        log_K_extended = jnp.concatenate([log_K_robust, log_K_dustbin], axis=2)
        log_P_softmax = log_K_extended - jax.nn.logsumexp(
            log_K_extended, axis=2, keepdims=True
        )

        # 6. Score
        # We want to maximize the probability of matching ANY valid HKL (non-outlier).
        # Optimization is MINIMIZATION, so we return the negative probability sum.
        log_P_match = log_P_softmax[:, :, :-1]
        log_prob_any = jax.nn.logsumexp(log_P_match, axis=2)
        score = -jnp.sum(self.weights * jnp.exp(log_prob_any), axis=1)

        # Metrics for reporting
        best_k_idx = jnp.argmax(log_P_match, axis=2)
        best_hkl_idx = jnp.take_along_axis(
            top_idxs, best_k_idx[:, :, None], axis=2
        ).squeeze(2)
        best_hkl = jnp.take(hkl_pool_padded.T, best_hkl_idx, axis=0)
        best_lamb = jnp.take_along_axis(
            lambda_sparse, best_k_idx[:, :, None], axis=2
        ).squeeze(2)

        return score, jnp.exp(log_prob_any), best_hkl, best_lamb

    def is_allowed_jax(self, h, k, l):  # noqa: E741
        """
        Robust symmetry check in JAX. Uses pre-computed mask for speed,
        and falls back to centring parity checks for out-of-bounds HKLs.
        """
        rh, rk, rl = self.mask_range_h, self.mask_range_k, self.mask_range_l
        idx_h = jnp.clip(h + rh, 0, 2 * rh).astype(jnp.int32)
        idx_k = jnp.clip(k + rk, 0, 2 * rk).astype(jnp.int32)
        idx_l = jnp.clip(l + rl, 0, 2 * rl).astype(jnp.int32)

        in_bounds = (
            (h >= -rh) & (h <= rh) & (k >= -rk) & (k <= rk) & (l >= -rl) & (l <= rl)
        )

        # Parity checks for centring
        h_even = h % 2 == 0
        k_even = k % 2 == 0
        l_even = l % 2 == 0

        if self.centering == "F":
            # All odd or all even
            allowed_out = (h_even == k_even) & (k_even == l_even)
        elif self.centering == "I":
            # h+k+l is even
            allowed_out = (h + k + l) % 2 == 0
        elif self.centering == "A":
            # k+l is even
            allowed_out = (k + l) % 2 == 0
        elif self.centering == "B":
            # h+l is even
            allowed_out = (h + l) % 2 == 0
        elif self.centering == "C":
            # h+k is even
            allowed_out = (h + k) % 2 == 0
        elif self.centering == "R":
            # -h+k+l is divisible by 3
            allowed_out = (-h + k + l) % 3 == 0
        else:
            # P or other: Assume allowed unless we have a specific reason to reject
            allowed_out = True

        return jnp.where(
            in_bounds, self.valid_hkl_mask[idx_h, idx_k, idx_l], allowed_out
        )

    @partial(jax.jit, static_argnames="self")
    def get_results(self, x):
        """Full physical model and indexing pipeline for a batch of solutions x."""
        # --- WORKAROUND for JAX/ROCm S=1 bug ---
        # On some AMD backends (e.g. MI200), JITted functions with lax.scan
        # can produce incorrect results when the leading batch dimension is exactly 1.
        # We force a minimum batch size of 2 by duplicating the input if necessary.
        original_S = x.shape[0]
        pad_size = max(0, 2 - original_S)
        x_pad = jnp.pad(x, ((0, pad_size), (0, 0)), mode="edge")

        UB, _, sample_total, ki_vec, _, R = self._get_physical_params_jax(x_pad)

        # Determine current rotations (Lab -> Sample)
        R_curr = R  # (S, N_runs, 3, 3) or None
        if R_curr is None:
            # Fallback to static rotations
            R_curr = self.static_R  # (N_runs, 3, 3) or (3, 3)

        # Expand rotations to per-observation if mapping is provided
        if R_curr is not None:
            if R_curr.ndim == 4:
                # (S, N_runs, 3, 3) -> (S, N_peaks, 3, 3)
                R_per_peak = R_curr[:, self.peak_run_indices, :, :]
            elif R_curr.ndim == 3:
                # (N_runs, 3, 3) -> (N_peaks, 3, 3)
                R_per_peak = R_curr[self.peak_run_indices, :, :]
            else:
                # (3, 3)
                R_per_peak = R_curr
        else:
            R_per_peak = None

        # Determine current scattered beam directions (kf) in Lab frame
        if self.peak_xyz is not None:
            # Recalculate kf for every run/peak because the sample position s_lab
            # depends on the goniometer rotation R and the refined sample offset.
            if R_per_peak is not None:
                if R_per_peak.ndim == 4:
                    # (S, N, 3, 3) @ (S, 3, 1) -> (S, N, 3, 1)
                    s_lab = jnp.matmul(
                        R_per_peak, sample_total[:, None, :, None]
                    ).squeeze(-1)
                    s = s_lab.transpose(0, 2, 1)  # (S, 3, N)
                elif R_per_peak.ndim == 3:
                    # Broadcasted Matmul: (1, N, 3, 3) @ (S, 1, 3, 1) -> (S, N, 3, 1)
                    s_lab = jnp.matmul(
                        R_per_peak[None, ...], sample_total[:, None, :, None]
                    ).squeeze(-1)
                    s = s_lab.transpose(0, 2, 1)
                else:
                    # (3, 3) @ (S, 3, 1) -> (S, 3, 1)
                    s_lab = jnp.matmul(
                        R_per_peak[None, ...], sample_total[:, :, None]
                    ).squeeze(-1)
                    s = s_lab[:, :, None]
            else:
                s = sample_total[:, :, None]

            p = self.peak_xyz[None, :, :]  # (1, 3, N)
            v = p - s  # (S, 3, N)
            dist = jnp.sqrt(jnp.sum(v**2, axis=1, keepdims=True))
            kf = v / jnp.where(dist == 0, 1.0, dist)
            ki = ki_vec[:, :, None]  # (S, 3, 1)
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)
        else:
            kf = self.kf_lab_fixed[None, :, :].repeat(x.shape[0], axis=0)
            ki = ki_vec[:, :, None]
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)

        # Rotate to SAMPLE FRAME: q_sample = R^T * q_lab
        # (Inputs are always in Lab frame and require rotation)
        if R_per_peak is not None:
            # q_lab is (S, 3, N). We want (S, N, 3, 1) for matmul
            q_lab_vec = q_lab.transpose(0, 2, 1)[..., None]
            if R_per_peak.ndim == 4:
                # (S, N, 3, 3) and (S, N, 3, 1)
                # q_sample = R.T @ q_lab
                RT = R_per_peak.transpose(0, 1, 3, 2)
                kf_ki_vec_T = jnp.matmul(RT, q_lab_vec).squeeze(-1)
            elif R_per_peak.ndim == 3:
                # (1, N, 3, 3) and (S, N, 3, 1)
                RT = R_per_peak.transpose(0, 2, 1)[None, ...]
                kf_ki_vec_T = jnp.matmul(RT, q_lab_vec).squeeze(-1)
            else:
                # (1, 3, 3) and (S, N, 3, 1)
                RT = R_per_peak.T[None, None, ...]
                kf_ki_vec_T = jnp.matmul(RT, q_lab_vec).squeeze(-1)
            kf_ki_vec = kf_ki_vec_T.transpose(0, 2, 1)
        else:
            kf_ki_vec = q_lab

        if self.loss_method == "forward":
            res = self.indexer_dynamic_binary_jax(
                UB,
                kf_ki_vec,
                k_sq_override=k_sq_dyn,
                tolerance_rad=self.tolerance_rad,
                window_batch_size=self.window_batch_size,
            )
        elif self.loss_method == "cosine":
            res = self.indexer_dynamic_cosine_aniso_jax(
                UB,
                kf_ki_vec,
                k_sq_override=k_sq_dyn,
                tolerance_rad=self.tolerance_rad,
            )
        elif self.loss_method == "sinkhorn":
            res = self.indexer_sinkhorn_jax(
                UB,
                kf_ki_vec,
                k_sq_override=k_sq_dyn,
                tolerance_rad=self.tolerance_rad,
                chunk_size=self.chunk_size,
                num_iters=self.num_iters,
                top_k=self.top_k,
            )
        else:
            res = self.indexer_dynamic_soft_jax(
                UB,
                kf_ki_vec,
                k_sq_override=k_sq_dyn,
                tolerance_rad=self.tolerance_rad,
            )

        # Slice results back to original batch size (Workaround cleanup)
        return jax.tree.map(
            lambda arr: (
                arr[:original_S] if hasattr(arr, "shape") and arr.ndim > 0 else arr
            ),
            res,
        )

    @partial(jax.jit, static_argnames="self")
    def __call__(self, x):
        score, _, _, _ = self.get_results(x)
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
        self.R = data.get("goniometer/R")
        self.two_theta = data["peaks/two_theta"]
        self.az_phi = data["peaks/azimuthal"]
        self.intensity = data["peaks/intensity"]
        self.sigma_intensity = data["peaks/sigma"]
        self.radii = data["peaks/radius"]

        # Robust run_index resolution
        # Preference:
        # 1. If R is provided, find indices that match R's length.
        # 2. Otherwise, prefer peaks/run_index.
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
            # Fallback to single run
            num_peaks = len(data["peaks/two_theta"])
            self.run_indices = np.zeros(num_peaks, dtype=int)

        # Handle bytes vs string for Space Group
        sg = data["sample/space_group"]
        if isinstance(sg, bytes):
            self.space_group = sg.decode("utf-8")
        else:
            self.space_group = str(sg)

        if "sample/offset" in data:
            self.base_sample_offset = data["sample/offset"]

        if "peaks/xyz" in data:
            self.peak_xyz = data["peaks/xyz"]
        if "goniometer/axes" in data:
            self.goniometer_axes = data["goniometer/axes"]
        if "goniometer/angles" in data:
            self.goniometer_angles = data["goniometer/angles"]

        if "goniometer/names" in data:
            names = data["goniometer/names"]
            # Handle list of bytes vs list of strings
            self.goniometer_names = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names
            ]

        if "beam/ki_vec" in data:
            self.ki_vec = data["beam/ki_vec"]
        else:
            self.ki_vec = np.array([0.0, 0.0, 1.0])

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
            if "peaks/run_index" in f:
                data["peaks/run_index"] = f["peaks/run_index"][()]
            if "peaks/image_index" in f:
                data["peaks/image_index"] = f["peaks/image_index"][()]
            if "bank" in f:
                data["bank"] = f["bank"][()]
            if "bank_ids" in f:
                data["bank_ids"] = f["bank_ids"][()]
            data["sample/space_group"] = f["sample/space_group"][()]

            if "peaks/xyz" in f:
                data["peaks/xyz"] = f["peaks/xyz"][()]
            if "goniometer/axes" in f:
                data["goniometer/axes"] = f["goniometer/axes"][()]
            if "goniometer/angles" in f:
                data["goniometer/angles"] = f["goniometer/angles"][()]
            if "goniometer/names" in f:
                data["goniometer/names"] = f["goniometer/names"][()]
            if "beam/ki_vec" in f:
                data["beam/ki_vec"] = f["beam/ki_vec"][()]

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
            raw_x = None
            if "optimization/best_params" in f:
                raw_x = f["optimization/best_params"][()]
            b_a = f["sample/a"][()]
            b_b = f["sample/b"][()]
            b_c = f["sample/c"][()]
            b_alpha = f["sample/alpha"][()]
            b_beta = f["sample/beta"][()]
            b_gamma = f["sample/gamma"][()]
            b_offset = f["sample/offset"][()] if "sample/offset" in f else np.zeros(3)
            b_ki = (
                f["beam/ki_vec"][()]
                if "beam/ki_vec" in f
                else np.array([0.0, 0.0, 1.0])
            )
            b_gonio_offsets = None
            if "optimization/goniometer_offsets" in f:
                b_gonio_offsets = f["optimization/goniometer_offsets"][()]

        new_params = []
        if raw_x is not None:
            new_params.append(raw_x[:3])
        else:
            new_params.append(np.zeros(3))

        if refine_lattice:
            self.a, self.b, self.c = b_a, b_b, b_c
            self.alpha, self.beta, self.gamma = b_alpha, b_beta, b_gamma
            print(
                f"  > Recentered Lattice: {self.a:.2f}, {self.b:.2f}, {self.c:.2f}..."
            )
            lat_sys, _ = get_lattice_system(
                self.a,
                self.b,
                self.c,
                self.alpha,
                self.beta,
                self.gamma,
                self.space_group,
            )
            active_indices = _get_active_lattice_indices(lat_sys)
            new_params.append(np.full(len(active_indices), 0.5))

        if b_offset is not None:
            print(f"  > Setting Base Sample Offset: {b_offset}")
            self.base_sample_offset = b_offset  # Store always

        if refine_sample:
            new_params.append(np.full(3, 0.5))

        if b_ki is not None:
            print(f"  > Recentered Beam Vector: {b_ki}")
            self.ki_vec = b_ki  # Store always

        if refine_beam:
            new_params.append(np.full(2, 0.5))

        if b_gonio_offsets is not None:
            print(f"  > Setting Base Goniometer Offsets: {b_gonio_offsets}")
            self.base_gonio_offset = b_gonio_offsets  # Store always
        else:
            self.base_gonio_offset = (
                np.zeros(len(self.goniometer_axes))
                if self.goniometer_axes is not None
                else None
            )

        if refine_goniometer:
            active_mask = []
            if refine_goniometer_axes is not None and self.goniometer_names is not None:
                for name in self.goniometer_names:
                    is_active = any(req in name for req in refine_goniometer_axes)
                    active_mask.append(is_active)
            else:
                active_mask = [True] * len(self.goniometer_axes)

            n_active = sum(active_mask)
            new_params.append(np.full(n_active, 0.5))

        return np.concatenate([np.atleast_1d(p) for p in new_params])

    def _minimize_scipy(
        self,
        population_size: int = 1000,
        num_generations: int = 100,
        n_runs: int = 1,
        seed: int = 0,
        tolerance_deg: float = 0.1,
        softness: float = 0.01,
        loss_method: str = "gaussian",
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
        B_sharpen: float = 50,
    ):
        """
        SciPy-based fallback for minimize when JAX is not available.
        Uses scipy.optimize.differential_evolution with VectorizedObjective.
        """
        from scipy.optimize import differential_evolution

        # 1. Prepare Metadata & Parameters
        if goniometer_axes is None and self.goniometer_axes is not None:
            goniometer_axes = self.goniometer_axes
        if goniometer_angles is None and self.goniometer_angles is not None:
            # SciPy path usually expects (num_axes, num_runs)
            goniometer_angles = self.goniometer_angles.T
        if goniometer_names is None and self.goniometer_names is not None:
            goniometer_names = self.goniometer_names

        lattice_system = "Triclinic"
        num_lattice_params = 6
        if refine_lattice:
            lattice_system, num_lattice_params = get_lattice_system(
                self.a,
                self.b,
                self.c,
                self.alpha,
                self.beta,
                self.gamma,
                self.space_group,
            )
            print(
                f"Lattice System: {lattice_system} ({num_lattice_params} free params)"
            )

        goniometer_refine_mask = None
        if refine_goniometer:
            if refine_goniometer_axes is not None and goniometer_names is not None:
                mask = []
                for name in goniometer_names:
                    should_refine = any(req in name for req in refine_goniometer_axes)
                    mask.append(should_refine)
                goniometer_refine_mask = np.array(mask, dtype=bool)
            else:
                goniometer_refine_mask = np.ones(len(goniometer_axes), dtype=bool)

        # Determine number of dimensions for bounds setup
        num_dims = 3  # orientation
        if refine_lattice:
            num_dims += num_lattice_params
        if refine_sample:
            num_dims += 3
        if refine_beam:
            num_dims += 2
        if refine_goniometer:
            num_dims += np.sum(goniometer_refine_mask)

        # 2. Prepare Objective Input
        kf_ki_dir_lab = scattering_vector_from_angles(self.two_theta, self.az_phi)

        # Use per-observation rotation if no refinement
        static_R_input = self.R if self.R is not None else np.eye(3)

        # Prepare weights
        snr = self.intensity / (self.sigma_intensity + 1e-6)
        if B_sharpen is not None:
            theta_rad = np.deg2rad(self.two_theta) / 2.0
            sin_sq_theta = np.sin(theta_rad) ** 2
            wilson_correction = np.exp(B_sharpen * sin_sq_theta)
            weights = snr * wilson_correction
            weights = weights / (np.mean(weights) + 1e-9)
        else:
            weights = snr
        weights = np.clip(weights, 0, 10.0)

        # Initialize VectorizedObjective (works with NumPy shim)
        objective_v = VectorizedObjective(
            B=self.reciprocal_lattice_B(),
            kf_ki_dir=kf_ki_dir_lab,
            peak_xyz_lab=self.peak_xyz.T if self.peak_xyz is not None else None,
            wavelength=self.wavelength,
            angle_cdf=self._angle_cdf,
            angle_t=self._angle_t,
            weights=weights,
            tolerance_deg=tolerance_deg,
            cell_params=[self.a, self.b, self.c, self.alpha, self.beta, self.gamma],
            refine_lattice=refine_lattice,
            lattice_bound_frac=lattice_bound_frac,
            lattice_system=lattice_system,
            goniometer_axes=goniometer_axes,
            goniometer_angles=goniometer_angles,
            refine_goniometer=refine_goniometer,
            goniometer_bound_deg=goniometer_bound_deg,
            goniometer_refine_mask=goniometer_refine_mask,
            goniometer_nominal_offsets=self.base_gonio_offset,
            refine_sample=refine_sample,
            sample_bound_meters=sample_bound_meters,
            sample_nominal=self.base_sample_offset,
            refine_beam=refine_beam,
            beam_bound_deg=beam_bound_deg,
            beam_nominal=self.ki_vec,
            loss_method=loss_method,
            hkl_search_range=hkl_search_range,
            d_min=d_min,
            d_max=d_max if d_max is not None else 100.0,
            space_group=self.space_group,
            centering=get_centering(self.space_group),
            static_R=static_R_input,
            peak_run_indices=self.run_indices,
        )

        # 3. Setup Bounds
        bounds = [(-np.pi, np.pi)] * 3  # Orientation (Rodrigues)
        if refine_lattice:
            bounds += [(0.0, 1.0)] * num_lattice_params
        if refine_sample:
            bounds += [(0.0, 1.0)] * 3
        if refine_beam:
            bounds += [(0.0, 1.0)] * 2
        if refine_goniometer:
            bounds += [(0.0, 1.0)] * np.sum(goniometer_refine_mask)

        # Prepare initial guess
        x0 = None
        if init_params is not None:
            x0 = init_params
            if len(x0) < num_dims:
                x0 = np.concatenate([x0, np.full(num_dims - len(x0), 0.5)])
            elif len(x0) > num_dims:
                x0 = x0[:num_dims]

        # 4. Run Optimization
        print(f"\n--- Starting SciPy Differential Evolution ({n_runs} runs) ---")
        print(f"Population: {population_size}, Generations: {num_generations}")

        def objective_scipy(x):
            """Wrapper for SciPy to call VectorizedObjective."""
            if x.ndim == 1:
                return float(objective_v(x[None, :])[0])
            # SciPy's vectorized mode passes (dims, members)
            return np.array(objective_v(x.T))

        best_overall_fun = np.inf
        best_overall_x = None

        for i_run in range(n_runs):
            curr_seed = seed + i_run
            if n_runs > 1:
                print(f"Run {i_run + 1}/{n_runs} (Seed: {curr_seed})...")

            result = differential_evolution(
                objective_scipy,
                bounds,
                maxiter=num_generations,
                popsize=max(1, population_size // num_dims),
                seed=curr_seed,
                x0=x0 if i_run == 0 else None,
                vectorized=True,
                atol=0,
                tol=0.01,
            )

            if result.fun < best_overall_fun:
                best_overall_fun = result.fun
                best_overall_x = result.x

        print("\n--- Optimization Complete ---")
        print(f"Best DE score: {-best_overall_fun:.2f}")

        # --- Local Refinement (BFGS) ---
        print("Polishing solution with BFGS refinement...")
        from scipy.optimize import minimize as scipy_minimize

        res_ref = scipy_minimize(
            objective_scipy,
            best_overall_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50},
        )

        if res_ref.success:
            if res_ref.fun < best_overall_fun:
                print(f"Refinement successful. Final score: {-res_ref.fun:.2f}")
                self.x = res_ref.x
            else:
                print(
                    f"Refinement increased cost from {best_overall_fun:.4f} "
                    f"to {res_ref.fun:.4f}. Reverting to best DE solution."
                )
                self.x = best_overall_x
        else:
            print(
                f"Refinement did not converge: {res_ref.message}. Keeping best DE solution."
            )
            self.x = best_overall_x

        # 5. Store Results
        best_member = self.x[None, :]

        # Extract physical parameters and update self
        phys_results = objective_v._get_physical_params_jax(best_member)

        (
            UB_batch,
            B_batch,
            sample_batch,
            ki_batch,
            offsets_batch,
            R_batch,
        ) = phys_results

        # Reconstruct lattice parameters
        if B_batch.ndim == 3:
            B_final = np.array(B_batch[0])
        else:
            B_final = np.array(B_batch)

        cell_params = self._cell_from_B_numpy(B_final)
        self.a, self.b, self.c = cell_params[:3]
        self.alpha, self.beta, self.gamma = cell_params[3:]

        if refine_lattice:
            print("--- Refined Lattice Parameters ---")
            print(f"a: {self.a:.4f}, b: {self.b:.4f}, c: {self.c:.4f}")
            print(
                f"alpha: {self.alpha:.4f}, beta: {self.beta:.4f}, gamma: {self.gamma:.4f}"
            )

        if refine_sample:
            self.sample_offset = np.array(sample_batch[0])
            print(f"--- Refined Sample Offset: {self.sample_offset} ---")

        if refine_beam:
            self.ki_vec = np.array(ki_batch[0])
            print(f"--- Refined Beam Vector: {self.ki_vec} ---")

        if refine_goniometer:
            self.goniometer_offsets = np.array(offsets_batch[0])
            print(
                f"--- Refined Goniometer Offsets (deg): {self.goniometer_offsets} ---"
            )

        if R_batch is not None:
            self.R = np.array(R_batch[0])

        # 6. Final Result Generation
        _, accum_probs, hkl_final, lamda_final = objective_v.get_results(best_member)

        # Squeeze batch dim and convert to array
        hkl_final = np.array(hkl_final[0])
        lamda_final = np.array(lamda_final[0])
        accum_probs = np.array(accum_probs[0])

        # Calculate number of indexed peaks for logging
        mask = accum_probs > 0.5
        num_indexed = np.sum(mask)

        # Set non-indexed peaks to 0,0,0 to match JAX behavior
        hkl_final[~mask] = 0

        # Orientation
        rot_best = self.x[:3]
        U_final = np.array(self._rotation_matrix_from_rodrigues_numpy(rot_best))

        return int(num_indexed), hkl_final, lamda_final, U_final

    def _rotation_matrix_from_rodrigues_numpy(self, w):
        """NumPy version of Rodrigues rotation."""
        theta = np.linalg.norm(w) + 1e-9
        k = w / theta
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        I = np.eye(3)  # noqa: E741
        R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

    def _compute_B_numpy(self, params_norm, lattice_system, lattice_bound_frac):
        """NumPy version of B matrix computation."""
        # Reconstruct cell params
        cell_init = np.array(
            [self.a, self.b, self.c, self.alpha, self.beta, self.gamma]
        )
        active_indices = _get_active_lattice_indices(lattice_system)
        free_params_init = cell_init[active_indices]

        p_free = _forward_map_lattice(params_norm, free_params_init, lattice_bound_frac)

        # Reconstruct full cell params based on lattice system
        if lattice_system == "Cubic":
            a = p_free[0]
            cell_params = np.array([a, a, a, 90.0, 90.0, 90.0])
        elif lattice_system == "Hexagonal":
            a, c = p_free[0], p_free[1]
            cell_params = np.array([a, a, c, 90.0, 90.0, 120.0])
        elif lattice_system == "Tetragonal":
            a, c = p_free[0], p_free[1]
            cell_params = np.array([a, a, c, 90.0, 90.0, 90.0])
        elif lattice_system == "Rhombohedral":
            a, alpha = p_free[0], p_free[1]
            cell_params = np.array([a, a, a, alpha, alpha, alpha])
        elif lattice_system == "Orthorhombic":
            a, b, c = p_free[0], p_free[1], p_free[2]
            cell_params = np.array([a, b, c, 90.0, 90.0, 90.0])
        elif lattice_system == "Monoclinic":
            a, b, c, beta = p_free[0], p_free[1], p_free[2], p_free[3]
            cell_params = np.array([a, b, c, 90.0, beta, 90.0])
        else:
            cell_params = p_free

        # Compute B matrix
        a, b, c = cell_params[:3]
        alpha, beta, gamma = np.deg2rad(cell_params[3:])

        g11, g22, g33 = a**2, b**2, c**2
        g12 = a * b * np.cos(gamma)
        g13 = a * c * np.cos(beta)
        g23 = b * c * np.cos(alpha)

        G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])
        G_star = np.linalg.inv(G)
        B = scipy.linalg.cholesky(G_star, lower=False)
        return B

    def _cell_from_B_numpy(self, B):
        """Extract cell parameters from B matrix."""
        G_star = B.T @ B
        G = np.linalg.inv(G_star)

        a = np.sqrt(G[0, 0])
        b = np.sqrt(G[1, 1])
        c = np.sqrt(G[2, 2])

        alpha = np.rad2deg(np.arccos(G[1, 2] / (b * c)))
        beta = np.rad2deg(np.arccos(G[0, 2] / (a * c)))
        gamma = np.rad2deg(np.arccos(G[0, 1] / (a * b)))

        return np.array([a, b, c, alpha, beta, gamma])

    def _rotation_matrix_from_axis_angle_numpy(self, axis, angle_rad):
        """NumPy version of axis-angle rotation."""
        u = axis / np.linalg.norm(axis)
        ux, uy, uz = u
        K = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
        return R

    def _compute_goniometer_R_numpy(self, axes, angles, offsets_norm, bound_deg):
        """NumPy version of goniometer rotation matrices."""
        # offsets_norm is (num_axes,)
        offsets_delta = _forward_map_param(offsets_norm, bound_deg)
        total_offsets = (
            self.base_gonio_offset + offsets_delta
            if self.base_gonio_offset is not None
            else offsets_delta
        )

        # angles is (num_axes, num_runs) or (num_axes, num_obs)
        angles_deg = total_offsets[:, np.newaxis] + angles
        num_runs = angles.shape[1]
        R_stack = np.tile(np.eye(3)[np.newaxis, ...], (num_runs, 1, 1))

        deg2rad = np.pi / 180.0
        for i in range(len(axes)):
            axis_spec = axes[i]
            direction = axis_spec[:3]
            sign = axis_spec[3]
            theta = sign * angles_deg[i, :] * deg2rad

            for r in range(num_runs):
                Ri = self._rotation_matrix_from_axis_angle_numpy(direction, theta[r])
                # Mantid SetGoniometer: R = R0 @ R1 @ R2
                R_stack[r] = R_stack[r] @ Ri

        return R_stack

    def minimize(
        self,
        strategy_name: str,
        population_size: int = 1000,
        num_generations: int = 100,
        n_runs: int = 1,
        seed: int = 0,
        tolerance_deg: float = 0.1,
        loss_method: str = "gaussian",
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
        d_min: float | None = None,
        d_max: float | None = None,
        hkl_search_range: int = 20,
        search_window_size: int = 256,
        window_batch_size: int = 32,
        chunk_size: int = 2048,
        num_iters: int = 20,
        top_k: int = 32,
        batch_size: int | None = None,
        sigma_init: float | None = None,
        softness: float = 0.01,
        B_sharpen: float = 50,
    ):
        """
        Minimize the objective using evolutionary strategies.

        When JAX is not available, falls back to SciPy's differential_evolution.
        """
        if not HAS_JAX:
            # Fall back to SciPy-based optimization
            print("JAX not available - using SciPy-based optimization")
            return self._minimize_scipy(
                population_size=population_size,
                num_generations=num_generations,
                n_runs=n_runs,
                seed=seed,
                tolerance_deg=tolerance_deg,
                softness=softness,
                loss_method=loss_method,
                init_params=init_params,
                refine_lattice=refine_lattice,
                lattice_bound_frac=lattice_bound_frac,
                goniometer_axes=goniometer_axes,
                goniometer_angles=goniometer_angles,
                refine_goniometer=refine_goniometer,
                goniometer_bound_deg=goniometer_bound_deg,
                goniometer_names=goniometer_names,
                refine_goniometer_axes=refine_goniometer_axes,
                refine_sample=refine_sample,
                sample_bound_meters=sample_bound_meters,
                refine_beam=refine_beam,
                beam_bound_deg=beam_bound_deg,
                d_min=d_min,
                d_max=d_max,
                hkl_search_range=hkl_search_range,
                B_sharpen=B_sharpen,
            )

        # JAX-based optimization (original code follows)
        if goniometer_axes is None and self.goniometer_axes is not None:
            goniometer_axes = self.goniometer_axes
        if goniometer_angles is None and self.goniometer_angles is not None:
            goniometer_angles = self.goniometer_angles.T
        if goniometer_names is None and self.goniometer_names is not None:
            goniometer_names = self.goniometer_names

        kf_ki_dir_lab = scattering_vector_from_angles(self.two_theta, self.az_phi)
        num_obs = kf_ki_dir_lab.shape[1]

        # --- Gonio Mapping Fix ---
        # If goniometer data is per-peak, reduce it to per-run (image) IF AND ONLY IF
        # all peaks in a run share the same geometry. This saves memory in the
        # optimizer. If they differ, we MUST use per-peak indexing.
        static_R_input = self.R if self.R is not None else np.eye(3)
        if self.run_indices is not None:
            max_run_id = int(np.max(self.run_indices))
            num_runs_range = max_run_id + 1
            unique_runs, first_indices = np.unique(self.run_indices, return_index=True)

            # Check for intra-run variations
            def has_variation(data, indices):
                if data is None:
                    return False
                for r in unique_runs:
                    mask = indices == r
                    if np.sum(mask) <= 1:
                        continue
                    subset = data[mask] if data.ndim == 2 else data[mask, ...]
                    if not np.allclose(subset, subset[0:1], atol=1e-7):
                        return True
                return False

            can_reduce_angles = (
                goniometer_angles is not None
                and goniometer_angles.shape[1] == num_obs
                and not has_variation(goniometer_angles.T, self.run_indices)
            )
            can_reduce_R = (
                self.R is not None
                and self.R.ndim == 3
                and self.R.shape[0] == num_obs
                and not has_variation(self.R, self.run_indices)
            )

            if can_reduce_angles:
                # We have per-peak angles. We can reduce them to per-run.
                new_angles = np.zeros((goniometer_angles.shape[0], num_runs_range))
                new_angles[:] = goniometer_angles[:, first_indices[0:1]]
                new_angles[:, unique_runs] = goniometer_angles[:, first_indices]
                goniometer_angles = new_angles

            if can_reduce_R:
                # We have per-peak rotations. Reduce to per-run.
                new_R = np.zeros((num_runs_range, 3, 3))
                new_R[:] = self.R[first_indices[0:1]]
                new_R[unique_runs] = self.R[first_indices]
                static_R_input = new_R
            elif self.R is not None and self.R.ndim == 3 and self.R.shape[0] == num_obs:
                # Per-peak variation detected. Use per-peak mapping (peak_run_indices = 0..N)
                static_R_input = self.R
                # This will trigger VectorizedObjective's per-peak mode (arange)
                self.run_indices = np.arange(num_obs, dtype=np.int32)

            # NEW: If gonio_angles is per-peak, also force per-peak mapping
            elif (
                goniometer_angles is not None and goniometer_angles.shape[1] == num_obs
            ):
                self.run_indices = np.arange(num_obs, dtype=np.int32)

        # Always use Lab frame vectors for Objective initialization.
        kf_ki_input = kf_ki_dir_lab

        goniometer_refine_mask = None
        if refine_goniometer and refine_goniometer_axes is not None:
            if self.goniometer_names is None:
                print(
                    "Warning: refine_goniometer_axes provided but goniometer_names not found. Refining ALL."
                )
            else:
                mask = []
                print(f"Refining specific goniometer axes: {refine_goniometer_axes}")
                for name in self.goniometer_names:
                    should_refine = any(req in name for req in refine_goniometer_axes)
                    mask.append(should_refine)
                goniometer_refine_mask = np.array(mask, dtype=bool)
                print(
                    f"Goniometer Mask: {goniometer_refine_mask} (Names: {self.goniometer_names})"
                )

        snr = self.intensity / (self.sigma_intensity + 1e-6)

        if B_sharpen is not None:
            theta_rad = np.deg2rad(self.two_theta) / 2.0
            sin_sq_theta = np.sin(theta_rad) ** 2
            wilson_correction = np.exp(B_sharpen * sin_sq_theta)
            weights = snr * wilson_correction
            weights = weights / np.mean(weights)
        else:
            weights = snr

        weights = np.clip(weights, 0, 10.0)

        cell_params_init = np.array(
            [self.a, self.b, self.c, self.alpha, self.beta, self.gamma]
        )
        lattice_system, num_lattice_params = get_lattice_system(
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
            self.space_group,
        )

        if refine_lattice:
            print("Lattice Refinement Enabled.")
            print(
                f"Detected System: {lattice_system} ({num_lattice_params} free parameters)."
            )

        if loss_method == "forward" and (d_min is None or d_max is None):
            raise ValueError(
                "Need to supply --d_min and --d_max for loss_method=='forward'"
            )

        objective = VectorizedObjective(
            self.reciprocal_lattice_B(),
            kf_ki_input,
            self.peak_xyz,
            np.array(self.wavelength),
            self._angle_cdf,
            self._angle_t,
            weights=weights,
            tolerance_deg=tolerance_deg,
            space_group=self.space_group,
            centering=get_centering(self.space_group),
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
            static_R=static_R_input,
            kf_lab_fixed_vectors=kf_ki_dir_lab,  # Pass raw Lab vectors
            peak_run_indices=self.run_indices,
        )
        print(
            f"Objective initialized with {loss_method} loss. Tolerance: {tolerance_deg} deg"
        )

        num_dims = 3
        if refine_lattice:
            num_dims += num_lattice_params
        if refine_sample:
            if self.peak_xyz is None:
                refine_sample = False
            else:
                num_dims += 3
        if refine_beam:
            if self.peak_xyz is None:
                refine_beam = False
            else:
                num_dims += 2
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
                    n_new = num_dims - start_sol.shape[0]
                    start_sol_processed = jnp.concatenate(
                        [start_sol, jnp.full((n_new,), 0.5)]
                    )
                else:
                    start_sol_processed = start_sol[:num_dims]
            else:
                start_sol_processed = start_sol

        sample_solution = jnp.zeros(num_dims)
        target_sigma = sigma_init or (0.01 if start_sol_processed is not None else 3.14)
        print(f"Strategy: {strategy_name.upper()} | Target Sigma: {target_sigma}")

        if strategy_name.lower() == "de":
            strategy = DifferentialEvolution(
                solution=sample_solution, population_size=population_size
            )
            strategy_type = "population_based"
        elif strategy_name.lower() == "pso":
            strategy = PSO(solution=sample_solution, population_size=population_size)
            strategy_type = "population_based"
        elif strategy_name.lower() == "cma_es":
            strategy = CMA_ES(solution=sample_solution, population_size=population_size)
            strategy_type = "distribution_based"
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        es_params = strategy.default_params

        def init_single_run(rng, start_sol):
            rng, rng_pop, rng_init = jax.random.split(rng, 3)

            if start_sol is not None:
                if strategy_type == "population_based":
                    noise = (
                        jax.random.normal(rng_pop, (population_size, num_dims))
                        * target_sigma
                    )
                    p_orient = start_sol[:3] + noise[:, :3]
                    p_rest = jnp.clip(start_sol[3:] + noise[:, 3:], 0.0, 1.0)
                    population_init = jnp.concatenate([p_orient, p_rest], axis=1)
                    fitness_init = objective(population_init)
                    state = strategy.init(
                        rng_init, population_init, fitness_init, es_params
                    )
                else:
                    state = strategy.init(rng_init, start_sol, es_params)
                    state = state.replace(std=target_sigma)
            elif strategy_type == "population_based":
                pop_orient = (
                    jax.random.normal(rng_pop, (population_size, 3)) * target_sigma
                )
                rng_rest, _ = jax.random.split(rng_pop)
                pop_rest = jax.random.uniform(
                    rng_rest, (population_size, max(0, num_dims - 3))
                )
                population_init = jnp.concatenate([pop_orient, pop_rest], axis=1)
                fitness_init = objective(population_init)
                state = strategy.init(
                    rng_init, population_init, fitness_init, es_params
                )
            else:
                mean_orient = jnp.zeros(3)
                mean_rest = jnp.full((max(0, num_dims - 3),), 0.5)
                solution_init = jnp.concatenate([mean_orient, mean_rest])
                state = strategy.init(rng_init, solution_init, es_params)
                state = state.replace(std=target_sigma)
            return state

        mesh = Mesh(np.array(jax.devices()), ("i"))

        def step_single_run(rng, state):
            rng, rng_ask, rng_tell = jax.random.split(rng, 3)
            x, state_ask = strategy.ask(rng_ask, state, es_params)
            x_orient = x[:, :3]
            x_rest = jnp.clip(x[:, 3:], 0.0, 1.0)
            x_valid = jnp.concatenate([x_orient, x_rest], axis=1)
            fitness = objective(x_valid)

            # parallelize population across GPUs
            x_valid = jax.lax.with_sharding_constraint(
                x_valid, NamedSharding(mesh, P("i"))
            )

            state_tell, metrics = strategy.tell(
                rng_tell, x_valid, fitness, state_ask, es_params
            )
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
                current_gen_best = min(current_gen_best, b_min)
            if trange is not None:
                if loss_method == "sinkhorn":
                    pbar.set_description(
                        f"Gen {gen + 1} | Cost: {current_gen_best:.4f}"
                    )
                else:
                    pbar.set_description(
                        f"Gen {gen + 1} | Best: {-current_gen_best:.1f}/{num_obs}"
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

        # --- Local Refinement (BFGS) ---
        # Polishing the best member using JAX gradients for sub-arcsec precision
        print("Polishing solution with BFGS refinement...")

        # Use a small subset of generations for JAX JIT warm-up if needed,
        # but here we can just use scipy.optimize
        from scipy.optimize import minimize as scipy_minimize

        def ref_func(x_flat):
            # Objective returns a scalar (minimized)
            return float(objective(x_flat[None, :])[0])

        def ref_grad(x_flat):
            # Use jax.grad for the objective
            grad_fn = jax.grad(lambda x: objective(x[None, :])[0])
            return np.array(grad_fn(x_flat))

        res_ref = scipy_minimize(
            ref_func,
            np.array(best_overall_member),
            jac=ref_grad,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0) if i >= 3 else (None, None) for i in range(num_dims)],
            options={"maxiter": 50},
        )

        if res_ref.success:
            if res_ref.fun < best_overall_fitness:
                print(f"Refinement successful. Final cost: {res_ref.fun:.4f}")
                best_overall_member = res_ref.x
                best_overall_fitness = res_ref.fun
            else:
                print(
                    f"Refinement increased cost from {best_overall_fitness:.4f} "
                    f"to {res_ref.fun:.4f}. Reverting to best DE solution."
                )
        else:
            print(
                f"Refinement did not converge: {res_ref.message}. Keeping best DE solution."
            )

        print("\n--- Optimization Complete ---")
        if loss_method == "sinkhorn":
            print(f"Best overall cost: {best_overall_fitness:.4f}")
        else:
            print(f"Best overall peaks: {-best_overall_fitness:.2f}")

        self.x = np.array(best_overall_member)

        x_batch = jnp.array(self.x[None, :])
        (
            UB_final_batch,
            B_new_batch,
            s_total_batch,
            ki_vec_batch,
            offsets_total_batch,
            R_batch,
        ) = objective._get_physical_params_jax(x_batch)

        np.array(UB_final_batch[0])
        self.sample_offset = np.array(s_total_batch[0])
        self.ki_vec = np.array(ki_vec_batch[0]).flatten()
        if offsets_total_batch is not None:
            self.goniometer_offsets = np.array(offsets_total_batch[0])
        if R_batch is not None:
            # If R_batch is (S, N_runs, 3, 3), we want (N_runs, 3, 3) for the best member
            self.R = np.array(R_batch[0])

        idx = 0
        rot_params = self.x[idx : idx + 3]
        idx += 3
        U = objective.orientation_U_jax(rot_params[None])[0]

        if refine_lattice:
            print("--- Refined Lattice Parameters ---")
            idx_lat = 3
            cell_norm = jnp.array(self.x[None, idx_lat : idx_lat + num_lattice_params])
            p_full = np.array(objective.reconstruct_cell_params(cell_norm)[0])
            print(f"a: {p_full[0]:.4f}, b: {p_full[1]:.4f}, c: {p_full[2]:.4f}")
            print(
                f"alpha: {p_full[3]:.4f}, beta: {p_full[4]:.4f}, gamma: {p_full[5]:.4f}"
            )
            self.a, self.b, self.c = p_full[0], p_full[1], p_full[2]
            self.alpha, self.beta, self.gamma = p_full[3], p_full[4], p_full[5]

        if refine_sample:
            print("--- Refined Sample Offset (mm) ---")
            print(
                f"X: {1000 * self.sample_offset[0]:.4f}, "
                f"Y: {1000 * self.sample_offset[1]:.4f}, "
                f"Z: {1000 * self.sample_offset[2]:.4f}"
            )

        if refine_beam:
            print("-- Refined Beam Direction ---")
            print(
                f"(ki_x, ki_y, ki_z): ({self.ki_vec[0]:.3f}, {self.ki_vec[1]:.3f}, {self.ki_vec[2]:.3f})"
            )

        if self.goniometer_offsets is not None:
            print("--- Refined Goniometer Offsets (deg) ---")
            if goniometer_names is not None:
                for name, val in zip(
                    goniometer_names, self.goniometer_offsets, strict=True
                ):
                    print(f"{name}: {val:.4f}")
            else:
                print(self.goniometer_offsets)

        # Final Score Recalculation using the unified pipeline
        score, accum_probs, hkl, lamb = objective.get_results(x_batch)

        num_peaks_soft = float(np.sum(accum_probs[0]))
        print(
            f"Final Solution indexed {num_peaks_soft:.2f}/{num_obs} "
            "peaks (unweighted count)."
        )

        return num_peaks_soft, np.array(hkl[0]), np.array(lamb[0]), np.array(U)
