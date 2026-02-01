import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import jax.scipy.optimize
import jax.scipy.signal
import sys

class SparseRBFPeakFinder:
    """
    JAX-Native Sparse RBF Peak Finder (Overlapping Window Pursuit).
    
    Strategy:
    1. Divide image into OVERLAPPING windows (e.g. 32x32, stride 16).
    2. Run Robust Sequential Greedy Pursuit on all windows in parallel.
    3. Merge results to remove duplicates (peaks found in multiple windows).
    
    Benefits:
    - No downsampling artifacts (Full Resolution).
    - No edge artifacts (Overlapping windows cover edges).
    - Massive parallelism.
    """
    def __init__(
        self,
        alpha: float = 0.05,            # Min intensity threshold
        gamma: float = 0.0,             # Unused
        min_sigma: float = 1.0,
        max_sigma: float = 10.0,
        max_peaks: int = 500,           # Max peaks per bank (output buffer)
        tiles: tuple = None,            # Unused
        show_steps: bool = False,
        show_scale: str = "linear"
    ):
        self.alpha = alpha
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.max_peaks = max_peaks
        self.show_steps = show_steps
        
        # Window Settings
        self.window_size = 32           # Size of sliding window
        self.stride = 16                # Overlap factor (Stride < Size)
        self.max_peaks_per_window = 5   # Peaks to find per window
        
        # Pre-calc sigmas
        self.candidate_sigmas = jnp.geomspace(min_sigma, max_sigma, num=5)

    @staticmethod
    def _rbf_basis(x_grid, y, sigma):
        dist_sq = (x_grid[0] - y[0])**2 + (x_grid[1] - y[1])**2
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
            return ci * SparseRBFPeakFinder._rbf_basis(x_grid, jnp.array([ri, ci_col]), si)
        basis_stack = vmap(eval_one)(c, r, c_col, sigma)
        return jnp.sum(basis_stack, axis=0)

    @staticmethod
    def _loss_fn(params_flat, x_grid, target, alpha, bounds_tuple):
        H, W, min_s, max_s = bounds_tuple
        params_phys = SparseRBFPeakFinder._to_physical(params_flat, H, W, min_s, max_s)
        recon = SparseRBFPeakFinder._predict_batch_physical(params_phys, x_grid)
        diff = recon - target
        mse = 0.5 * jnp.sum(diff**2)
        reg = alpha * jnp.sum(jnp.abs(params_phys[:, 0])) 
        return mse + reg

    # =========================================================================
    # THE SOLVER (Sequential Greedy on a Window)
    # =========================================================================
    @partial(jit, static_argnames=['self', 'H', 'W'])
    def _solve_window(self, window, H, W):
        """
        Runs Robust Sequential Greedy Pursuit on a single window.
        """
        bounds = (float(H), float(W), self.min_sigma, self.max_sigma)
        
        yy, xx = jnp.indices((H, W))
        x_grid = jnp.array([yy, xx])
        
        # Static Kernel
        max_k_rad = int(3.0 * self.max_sigma)
        max_k_rad = min(max_k_rad, H // 2)
        k_grid = jnp.arange(-max_k_rad, max_k_rad + 1)
        ky, kx = jnp.meshgrid(k_grid, k_grid)
        
        init_params = jnp.zeros((self.max_peaks_per_window, 4))
        init_state = (init_params, 0)

        def step_fn(state, _):
            params, idx = state
            
            # 1. Residual
            recon = self._predict_batch_physical(params, x_grid)
            residual = window - recon
            
            # 2. Greedy Search
            def check_sigma(s):
                kernel = jnp.exp(-(kx**2 + ky**2) / (2 * s**2))
                corr = jax.scipy.signal.correlate2d(residual, kernel, mode='same')
                
                flat_idx = jnp.argmax(jnp.abs(corr))
                r_idx, c_idx = jnp.unravel_index(flat_idx, corr.shape)
                val = jnp.abs(corr[r_idx, c_idx])
                
                c_init = jnp.maximum(residual[r_idx, c_idx], 0.0)
                # Return array directly (JAX promotes int->float automatically)
                return val, jnp.array([c_init, r_idx, c_idx, s])

            vals, candidates = vmap(check_sigma)(self.candidate_sigmas)
            best_idx = jnp.argmax(vals)
            new_peak = candidates[best_idx]
            
            # 3. Threshold
            is_strong = new_peak[0] > self.alpha
            new_peak = jnp.where(is_strong, new_peak, jnp.zeros(4))
            
            # 4. Insert
            params = params.at[idx].set(new_peak)
            
            # 5. Quick Relax
            def run_opt(p):
                p_raw = self._to_unconstrained(p, *bounds)
                res = jax.scipy.optimize.minimize(
                    fun=self._loss_fn,
                    x0=p_raw,
                    args=(x_grid, window, self.alpha, bounds),
                    method='BFGS',
                    options={'maxiter': 5}
                )
                return self._to_physical(res.x, *bounds)
            
            params = run_opt(params)
            return (params, idx + 1), None

        final_state, _ = lax.scan(step_fn, init_state, None, length=self.max_peaks_per_window)
        final_params, _ = final_state
        
        # Final Polish
        params_raw = self._to_unconstrained(final_params, *bounds)
        res_final = jax.scipy.optimize.minimize(
            fun=self._loss_fn,
            x0=params_raw,
            args=(x_grid, window, self.alpha, bounds),
            method='BFGS',
            options={'maxiter': 20}
        )
        return self._to_physical(res_final.x, *bounds)

    def _progress_callback(self, current_step, total_steps):
        percent = (current_step / total_steps) * 100
        sys.stdout.write(f"\rSparseRBF (Window): {percent:.1f}%")
        sys.stdout.flush()

    def find_peaks_batch(self, images_batch):
        B, H, W = images_batch.shape

        # 1. Pre-process
        medians = np.median(images_batch, axis=(1, 2), keepdims=True)
        images_bg_corr = np.maximum(images_batch - medians, 0)
        global_max = images_bg_corr.max() + 1e-9
        images_norm = images_bg_corr / global_max

        print(f"  > Pre-processing: Bg Subtracted. Global Max={global_max:.1f}")

        img_jax = jnp.array(images_norm)

        # --- 2. Extract Overlapping Windows ---
        img_nchw = img_jax[:, None, :, :]

        win_size = self.window_size
        stride = self.stride

        # Output Shape: (B, Depth=1024, Out_H, Out_W)
        patches_conv = lax.conv_general_dilated_patches(
            lhs=img_nchw,
            filter_shape=(win_size, win_size),
            window_strides=(stride, stride),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )

        # --- FIX: Correct Shape Unpacking ---
        B, Depth, Out_H, Out_W = patches_conv.shape

        # We need to move Depth (1024) to the end: (B, Out_H, Out_W, Depth)
        patches_t = patches_conv.transpose(0, 2, 3, 1)

        total_wins = B * Out_H * Out_W
        print(f"  > Extracted {total_wins} overlapping windows (Grid: {Out_H}x{Out_W}).")

        # Now reshape to (Total_Windows, 32, 32)
        patches_reshaped = patches_t.reshape(total_wins, win_size, win_size)

        # --- 3. Parallel Solve ---
        solver = jit(vmap(
            lambda w: self._solve_window(w, win_size, win_size)
        ))

        chunk_size = 512
        all_results = []

        print("Running Window Pursuit...")
        for i in range(0, total_wins, chunk_size):
            chunk = patches_reshaped[i:i+chunk_size]
            res = solver(chunk)
            all_results.append(np.array(res))
            self._progress_callback(i + len(chunk), total_wins)

        print("\nMerging results...")

        flat_results = np.concatenate(all_results, axis=0) # (Total_Win, Peaks_Per_Win, 4)

        # Reshape back to grid structure to recover coordinates
        # (B, Out_H, Out_W, Peaks, 4)
        grid_results = flat_results.reshape(B, Out_H, Out_W, self.max_peaks_per_window, 4)

        final_peaks_list = []
        total_kept = 0

        # Import scipy distance for deduplication
        from scipy.spatial.distance import pdist, squareform

        for b in range(B):
            bank_peaks = []
            for hg in range(Out_H):
                for wg in range(Out_W):
                    local_peaks = grid_results[b, hg, wg]

                    # Filter
                    mask = (local_peaks[:, 0] > self.alpha) & \
                           (local_peaks[:, 3] > (self.min_sigma * 0.95))
                    valid = local_peaks[mask]

                    if len(valid) > 0:
                        # Convert Local to Global
                        start_r = hg * stride
                        start_c = wg * stride

                        shifted = valid.copy()
                        shifted[:, 1] += start_r
                        shifted[:, 2] += start_c

                        bank_peaks.append(shifted)

            if len(bank_peaks) > 0:
                merged = np.vstack(bank_peaks)

                # --- DEDUPLICATION ---
                # 1. Sort descending intensity (Keep brightest)
                order = np.argsort(merged[:, 0])[::-1]
                sorted_peaks = merged[order]

                keep_mask = np.ones(len(sorted_peaks), dtype=bool)
                r_coords = sorted_peaks[:, 1]
                c_coords = sorted_peaks[:, 2]
                sigmas   = sorted_peaks[:, 3]

                coords = np.column_stack([r_coords, c_coords])

                if len(coords) > 1:
                    dists = squareform(pdist(coords))
                    np.fill_diagonal(dists, 9999.0)

                    for i in range(len(coords)):
                        if keep_mask[i]:
                            # If neighbor is closer than its sigma radius, kill it
                            radius = max(2.0, sigmas[i])
                            neighbors = np.where(dists[i] < radius)[0]
                            neighbors = neighbors[neighbors > i] # Only kill strictly fainter ones
                            keep_mask[neighbors] = False

                final_peaks = sorted_peaks[keep_mask]

                final_peaks_list.append(final_peaks[:, 1:3])
                total_kept += len(final_peaks)
            else:
                final_peaks_list.append(np.empty((0, 2)))

        print(f"  > Total peaks found: {total_kept}")
        return final_peaks_list
