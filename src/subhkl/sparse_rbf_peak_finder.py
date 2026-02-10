from functools import partial

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# Import JAX with fallback from utils (centralized)
from subhkl.utils import HAS_JAX, jax, jit, jnp, lax, vmap

if HAS_JAX:
    import jax.scipy.optimize
    import jax.scipy.signal


class SparseRBFPeakFinder:
    """
    JAX-Native Recursive Window Sparse RBF Peak Finder.

    Theoretical Basis:
    - Solves the regression problem (0th-order PDE) in a Besov RKBS.
    - Uses 'Gradient Boosting' (Greedy Selection) and 'Gauss-Newton' (BFGS) optimization.
    - [cite_start]Implements 'Besov Scaling' (sigma^gamma) to penalize singularities[cite: 147].
    - Enforces Dirichlet Boundary Conditions by restricting the measure support.

    Requires JAX to be installed.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 2.0,  # Shape prior (2.0 recommended)
        min_sigma: float = 0.2,  # Search sub-pixel to identify artifacts
        max_sigma: float = 10.0,
        max_peaks: int = 500,
        chunk_size: int = 1024,
        show_steps: bool = False,
        show_scale: str = "linear",
    ):
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for SparseRBFPeakFinder. "
                'Install with: pip install -e ".[jax]" or pip install jax jaxlib'
            )
        self.alpha = alpha
        self.gamma = gamma
        self.ref_sigma = 1.0  # Reference unit (1 pixel)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.max_peaks = max_peaks
        self.chunk_size = chunk_size
        self.show_steps = show_steps

        # Settings
        self.base_window_size = 32
        self.refine_patch_size = 15
        self.max_seeds_per_window = 20

        # Geometric grid for scale search
        self.candidate_sigmas = jnp.geomspace(min_sigma, max_sigma, num=10)

    @staticmethod
    def _rbf_basis(x_grid, y, sigma):
        dist_sq = (x_grid[0] - y[0]) ** 2 + (x_grid[1] - y[1]) ** 2
        return jnp.exp(-dist_sq / (2.0 * sigma**2 + 1e-6))

    @staticmethod
    def _to_physical(params_raw, H, W, min_s, max_s):
        params_reshaped = params_raw.reshape((-1, 4))
        c_raw, r_raw, c_col_raw, s_raw = params_reshaped.T
        c = jax.nn.softplus(c_raw)
        r = jax.nn.sigmoid(r_raw) * H
        col = jax.nn.sigmoid(c_col_raw) * W
        sigma = min_s + jax.nn.sigmoid(s_raw) * (max_s - min_s)
        return jnp.stack([c, r, col, sigma], axis=1)

    @staticmethod
    def _to_unconstrained(params_phys, H, W, min_s, max_s):
        c, r, col, sigma = params_phys.T
        c_safe = jnp.maximum(c, 1e-9)
        c_raw = jnp.log(jnp.expm1(c_safe))
        r_safe = jnp.clip(r / H, 1e-6, 1.0 - 1e-6)
        r_raw = jax.scipy.special.logit(r_safe)
        c_safe = jnp.clip(col / W, 1e-6, 1.0 - 1e-6)
        c_col_raw = jax.scipy.special.logit(c_safe)
        s_norm = (sigma - min_s) / (max_s - min_s)
        s_safe = jnp.clip(s_norm, 1e-6, 1.0 - 1e-6)
        s_raw = jax.scipy.special.logit(s_safe)
        return jnp.stack([c_raw, r_raw, c_col_raw, s_raw], axis=1).ravel()

    @staticmethod
    def _predict_batch_physical(params_phys, x_grid):
        c, r, c_col, sigma = params_phys.T

        def eval_one(ci, ri, ci_col, si):
            return ci * SparseRBFPeakFinder._rbf_basis(
                x_grid, jnp.array([ri, ci_col]), si
            )

        basis_stack = vmap(eval_one)(c, r, c_col, sigma)
        return jnp.sum(basis_stack, axis=0)

    @staticmethod
    def _loss_fn(params_flat, x_grid, target, alpha, gamma, ref_s, bounds_tuple):
        H, W, min_s, max_s = bounds_tuple
        params_phys = SparseRBFPeakFinder._to_physical(params_flat, H, W, min_s, max_s)
        recon = SparseRBFPeakFinder._predict_batch_physical(params_phys, x_grid)

        mse = 0.5 * jnp.sum((recon - target) ** 2)

        intensities = jnp.abs(params_phys[:, 0])
        sigmas = params_phys[:, 3]

        # Besov Norm Regularization:
        # Penalize intensity based on width. Sharp peaks (small sigma) are penalized heavily.
        # This corresponds to the L1 norm of coefficients in the scaled basis.
        sigma_ratio = sigmas / ref_s
        reg_weight = 1.0 / (sigma_ratio**gamma + 1e-6)

        reg = alpha * jnp.sum(intensities * reg_weight)
        return mse + reg

    # =========================================================================
    # KERNEL 1: DENSE SOLVER
    # =========================================================================
    @partial(jit, static_argnames=["self", "H", "W", "max_peaks_local"])
    def _solve_dense(self, image, H, W, max_peaks_local):
        bounds = (float(H), float(W), self.min_sigma, self.max_sigma)
        yy, xx = jnp.indices((H, W))
        x_grid = jnp.array([yy, xx])

        max_k_rad = int(3.0 * self.max_sigma)
        max_k_rad = min(max_k_rad, H // 2)
        k_grid = jnp.arange(-max_k_rad, max_k_rad + 1)
        ky, kx = jnp.meshgrid(k_grid, k_grid)

        init_params = jnp.zeros((max_peaks_local, 4))
        init_state = (init_params, 0)

        def step_fn(state, _):
            params, idx = state
            recon = self._predict_batch_physical(params, x_grid)
            residual = image - recon

            def check_sigma(s):
                kernel = jnp.exp(-(kx**2 + ky**2) / (2 * s**2))
                corr = jax.scipy.signal.correlate2d(residual, kernel, mode="same")
                flat_idx = jnp.argmax(jnp.abs(corr))
                r_idx, c_idx = jnp.unravel_index(flat_idx, corr.shape)
                raw_val = jnp.abs(corr[r_idx, c_idx])

                # Selection Score (Dual Variable)
                # Score ~ <Resid, Phi> ~ Intensity * Volume * Sigma^Gamma
                # Weight increases with sigma to prefer broad peaks.
                weight = (s / self.ref_sigma) ** self.gamma
                weighted_score = raw_val * weight

                c_init = jnp.maximum(residual[r_idx, c_idx], 0.0)
                return weighted_score, jnp.array([c_init, r_idx, c_idx, s])

            vals, candidates = vmap(check_sigma)(self.candidate_sigmas)
            best_idx = jnp.argmax(vals)
            best_score = vals[best_idx]
            new_peak = candidates[best_idx]

            # Admission Threshold
            is_strong = best_score > self.alpha
            new_peak = jnp.where(is_strong, new_peak, jnp.zeros(4))

            params = params.at[idx].set(new_peak)

            def run_opt(p):
                p_raw = self._to_unconstrained(p, *bounds)
                res = jax.scipy.optimize.minimize(
                    fun=self._loss_fn,
                    x0=p_raw,
                    args=(
                        x_grid,
                        image,
                        self.alpha,
                        self.gamma,
                        self.ref_sigma,
                        bounds,
                    ),
                    method="BFGS",
                    options={"maxiter": 5},
                )
                return self._to_physical(res.x, *bounds)

            params = run_opt(params)
            return (params, idx + 1), None

        final_state, _ = lax.scan(step_fn, init_state, None, length=max_peaks_local)
        final_params, _ = final_state
        return final_params

    # =========================================================================
    # KERNEL 2: PATCHY SOLVER w/ HALO
    # =========================================================================
    @partial(jit, static_argnames=["self", "Win_H", "Win_W", "Halo_P", "Refine_P"])
    def _solve_patchy_window(
        self, window_with_halo, seeds, Win_H, Win_W, Halo_P, Refine_P
    ):
        seeds_halo_shifted = seeds.copy()
        seeds_halo_shifted = seeds_halo_shifted.at[:, 1].add(Halo_P)
        seeds_halo_shifted = seeds_halo_shifted.at[:, 2].add(Halo_P)

        half_p = Refine_P // 2

        def process_one_seed(seed):
            valid = seed[0] > 1e-6
            r_c = seed[1].astype(int)
            c_c = seed[2].astype(int)

            start_r = r_c - half_p
            start_c = c_c - half_p

            patch = lax.dynamic_slice(
                window_with_halo, (start_r, start_c), (Refine_P, Refine_P)
            )
            res = self._solve_dense(patch, Refine_P, Refine_P, 2)

            shift_r = (r_c - half_p) - Halo_P
            shift_c = (c_c - half_p) - Halo_P

            res = res.at[:, 1].add(shift_r)
            res = res.at[:, 2].add(shift_c)

            # Filter invalid results (weak or diverged)
            # We use the intensity check here; full Besov check happens in Merge
            mask = valid & (res[:, 0] > 1e-9)
            return jnp.where(mask[:, None], res, jnp.zeros_like(res))

        results = vmap(process_one_seed)(seeds_halo_shifted)
        return results.reshape(-1, 4)

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    def find_peaks_batch(self, images_batch):
        B, H, W = images_batch.shape

        # 1. Pre-process
        medians = np.median(images_batch, axis=(1, 2), keepdims=True)
        images_bg_corr = np.maximum(images_batch - medians, 0)
        global_max = images_bg_corr.max() + 1e-9
        images_norm = images_bg_corr / global_max

        # Note: We do NOT apply a spatial mask to the data here.
        # Masking the data smooths the artifacts, making them survive the Besov filter.
        # Instead, we enforce the boundary condition by filtering the *Results*.

        print(f"  > Pre-processing: Bg Subtracted. Global Max={global_max:.1f}")

        img_jax = jnp.array(images_norm)

        PAD_GLOBAL = 32
        img_jax_padded = jnp.pad(
            img_jax, ((0, 0), (PAD_GLOBAL, PAD_GLOBAL), (PAD_GLOBAL, PAD_GLOBAL))
        )

        current_peaks = [np.zeros((0, 4)) for _ in range(B)]

        win_size = self.base_window_size
        max_dim = max(H, W)
        loop_sizes = []
        w = win_size
        while w < max_dim:
            loop_sizes.append(w)
            w *= 2
        loop_sizes.append(max_dim)
        loop_sizes = sorted(list(set(loop_sizes)))

        for level_idx, w_curr in enumerate(loop_sizes):
            w_h = min(w_curr, H)
            w_w = min(w_curr, W)
            stride_h = w_h // 2
            stride_w = w_w // 2

            print(
                f"\n[Level {level_idx}] Processing Windows: {w_h}x{w_w} (Stride {stride_h})"
            )

            # --- Grid Generation ---
            grid_h_indices = list(range(0, H - w_h + 1, stride_h))
            if grid_h_indices[-1] + w_h < H:
                grid_h_indices.append(H - w_h)

            grid_w_indices = list(range(0, W - w_w + 1, stride_w))
            if grid_w_indices[-1] + w_w < W:
                grid_w_indices.append(W - w_w)

            window_coords = []
            for b in range(B):
                for r in grid_h_indices:
                    for c in grid_w_indices:
                        window_coords.append((b, r, c))

            total_wins_all = len(window_coords)
            window_coords_arr = np.array(window_coords, dtype=np.int32)

            # --- Solver Setup ---
            chunk_size = self.chunk_size
            all_results = []

            if level_idx == 0:
                # BASE CASE: DENSE
                @jit
                def extract_chunk_exact(b_idx, r_idx, c_idx):
                    r_pad = r_idx + PAD_GLOBAL
                    c_pad = c_idx + PAD_GLOBAL

                    def slice_one(bi, ri, ci):
                        return lax.dynamic_slice(
                            img_jax_padded[bi], (ri, ci), (w_h, w_w)
                        )

                    return vmap(slice_one)(b_idx, r_pad, c_pad)

                print(f"  > Dense Search on {total_wins_all} windows...")
                solver = jit(vmap(lambda w: self._solve_dense(w, w_h, w_w, 10)))

                with tqdm(
                    total=total_wins_all, desc="Dense Search", unit="win"
                ) as pbar:
                    for i in range(0, total_wins_all, chunk_size):
                        chunk_coords = window_coords_arr[i : i + chunk_size]
                        c_win = extract_chunk_exact(
                            chunk_coords[:, 0], chunk_coords[:, 1], chunk_coords[:, 2]
                        )

                        res = solver(c_win)
                        # Force blocking to ensure timing is accurate for the progress bar
                        res.block_until_ready()

                        all_results.append(np.array(res))
                        pbar.update(len(c_win))

            else:
                # RECURSIVE: PATCHY
                print(f"  > Refining seeds on {total_wins_all} windows (w/ Halo)...")
                M_S = self.max_seeds_per_window
                P = self.refine_patch_size
                HALO = P // 2 + 1

                @jit
                def extract_chunk_with_halo(b_idx, r_idx, c_idx):
                    r_pad = r_idx + PAD_GLOBAL - HALO
                    c_pad = c_idx + PAD_GLOBAL - HALO
                    h_extract = w_h + 2 * HALO
                    w_extract = w_w + 2 * HALO

                    def slice_one(bi, ri, ci):
                        return lax.dynamic_slice(
                            img_jax_padded[bi], (ri, ci), (h_extract, w_extract)
                        )

                    return vmap(slice_one)(b_idx, r_pad, c_pad)

                solver = jit(
                    vmap(
                        lambda w, s: self._solve_patchy_window(w, s, w_h, w_w, HALO, P)
                    )
                )

                with tqdm(total=total_wins_all, desc="Refinement", unit="win") as pbar:
                    for i in range(0, total_wins_all, chunk_size):
                        chunk_coords = window_coords_arr[i : i + chunk_size]

                        chunk_seeds = np.zeros(
                            (len(chunk_coords), M_S, 4), dtype=np.float32
                        )
                        for k, (b, r, c) in enumerate(chunk_coords):
                            cp = current_peaks[b]
                            if len(cp) == 0:
                                continue
                            valid_cp = cp[cp[:, 0] > 1e-6]
                            r_end = r + w_h
                            c_end = c + w_w

                            in_r = (valid_cp[:, 1] >= r) & (valid_cp[:, 1] < r_end)
                            in_c = (valid_cp[:, 2] >= c) & (valid_cp[:, 2] < c_end)
                            seeds = valid_cp[in_r & in_c]

                            if len(seeds) > 0:
                                seeds_local = seeds.copy()
                                seeds_local[:, 1] -= r
                                seeds_local[:, 2] -= c
                                if len(seeds_local) > M_S:
                                    order = np.argsort(seeds_local[:, 0])[::-1]
                                    seeds_local = seeds_local[order[:M_S]]
                                chunk_seeds[k, : len(seeds_local), :] = seeds_local

                        c_win_halo = extract_chunk_with_halo(
                            chunk_coords[:, 0], chunk_coords[:, 1], chunk_coords[:, 2]
                        )
                        res = solver(c_win_halo, jnp.array(chunk_seeds))
                        res.block_until_ready()  # Ensure accurate ETA
                        all_results.append(np.array(res))
                        pbar.update(len(c_win_halo))

            # --- Merge & Deduplicate ---
            print("\n  > Merging & Deduplicating...")
            flat_results = np.concatenate(all_results, axis=0)
            new_current_peaks = [[] for _ in range(B)]

            # --- BOUNDARY MARGIN ---
            # Paper Sec 3.4: Enforce u=0 on boundary.
            # We strictly reject peaks within 'margin' of the edge.
            MARGIN = 10

            for k, (b, r, c) in enumerate(window_coords):
                local_p = flat_results[k]

                intensities = local_p[:, 0]
                row_local = local_p[:, 1]
                col_local = local_p[:, 2]
                sigmas = local_p[:, 3]

                # Global coordinates for boundary check
                row_global = row_local + r
                col_global = col_local + c

                # 1. KKT Condition (Besov Pruning)
                vol_factor = (sigmas / self.ref_sigma) ** 2
                besov_factor = (sigmas / self.ref_sigma) ** self.gamma
                score = intensities * vol_factor * besov_factor

                # 2. Boundary Condition (Support Restriction)
                # Reject if center is within MARGIN of image boundary
                in_bounds = (
                    (row_global > MARGIN)
                    & (row_global < H - MARGIN)
                    & (col_global > MARGIN)
                    & (col_global < W - MARGIN)
                )

                mask = (score > self.alpha) & in_bounds
                valid = local_p[mask]

                if len(valid) > 0:
                    global_p = valid.copy()
                    global_p[:, 1] += r
                    global_p[:, 2] += c
                    new_current_peaks[b].append(global_p)

            final_peaks_next = []
            for b in range(B):
                if len(new_current_peaks[b]) > 0:
                    merged = np.vstack(new_current_peaks[b])
                    order = np.argsort(merged[:, 0])[::-1]
                    sorted_p = merged[order]
                    keep_mask = np.ones(len(sorted_p), dtype=bool)
                    coords = sorted_p[:, 1:3]
                    sigmas = sorted_p[:, 3]
                    if len(coords) > 1:
                        dists = squareform(pdist(coords))
                        np.fill_diagonal(dists, 9999.0)
                        for idx in range(len(coords)):
                            if keep_mask[idx]:
                                radius = max(2.0, sigmas[idx])
                                neighbors = np.where(dists[idx] < radius)[0]
                                neighbors = neighbors[neighbors > idx]
                                keep_mask[neighbors] = False
                    final_peaks_next.append(sorted_p[keep_mask])
                else:
                    final_peaks_next.append(np.zeros((0, 4)))
            current_peaks = final_peaks_next
            print(
                f"  > Level Complete. Found {sum(len(x) for x in current_peaks)} peaks total."
            )

        final_output = []
        for b in range(B):
            cp = current_peaks[b]
            if len(cp) > 0:
                final_output.append(cp[:, 1:3])
            else:
                final_output.append(np.empty((0, 2)))
        return final_output
