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
    JAX-Native Sparse RBF Peak Finder.
    
    Fixed for stability:
    - Uses Unit-Height Gaussians so 'c' represents actual intensity.
    - Alpha now corresponds directly to a minimum intensity threshold (e.g. 0.05 = 5%).
    """
    def __init__(
        self,
        alpha: float = 0.02,            # Threshold: peaks < 2% intensity are pruned
        gamma: float = 0.0,             # Unused (kept for API compatibility)
        min_sigma: float = 1.0,
        max_sigma: float = 10.0,
        max_peaks: int = 500,
        tiles: tuple = (2, 2),
        show_steps: bool = False,
        show_scale: str = "linear"
    ):
        self.alpha = alpha
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.max_peaks = max_peaks
        self.tiles = tiles
        self.show_steps = show_steps
        
        self.candidate_sigmas = jnp.geomspace(min_sigma, max_sigma, num=5)

    @staticmethod
    def _rbf_basis(x_grid, y, sigma):
        # Unit Height Gaussian: exp(-r^2 / 2sigma^2)
        # Peak value is always 1.0, regardless of sigma.
        dist_sq = (x_grid[0] - y[0])**2 + (x_grid[1] - y[1])**2
        return jnp.exp(-dist_sq / (2.0 * sigma**2 + 1e-6))

    # --- Parameter Transformations ---
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
        
        # L1 on Intensity 'c'
        reg = alpha * jnp.sum(jnp.abs(params_phys[:, 0])) 
        
        return mse + reg

    @partial(jit, static_argnames=['self', 'H', 'W'])
    def _run_tile_pursuit(self, image_tile, x_grid, H, W):
        bounds = (float(H), float(W), self.min_sigma, self.max_sigma)
        
        tile_max_peaks = max(5, self.max_peaks // (self.tiles[0] * self.tiles[1]))
        init_params = jnp.zeros((tile_max_peaks, 4))

        def step_fn(current_params, iter_idx):
            recon = self._predict_batch_physical(current_params, x_grid)
            residual = image_tile - recon
            
            def check_sigma(s):
                k_rad = int(4.0 * self.max_sigma) 
                grid_range = jnp.arange(-k_rad, k_rad + 1)
                yy, xx = jnp.meshgrid(grid_range, grid_range)
                
                # Match unit-height Gaussian shape
                kernel_shape = jnp.exp(-(xx**2 + yy**2) / (2 * s**2))
                
                # Cross-Correlation
                corr = jax.scipy.signal.correlate2d(residual, kernel_shape, mode='same')
                
                flat_idx = jnp.argmax(jnp.abs(corr))
                r_idx, c_idx = jnp.unravel_index(flat_idx, corr.shape)
                
                # Since basis has unit height, c_init is simply the residual intensity
                res_val = residual[r_idx, c_idx]
                c_init = jnp.maximum(0.0, res_val)
                
                return res_val, jnp.array([c_init, r_idx, c_idx, s])

            vals, candidates = vmap(check_sigma)(self.candidate_sigmas)
            best_idx = jnp.argmax(vals)
            best_val = vals[best_idx]
            new_peak = candidates[best_idx]
            
            # Threshold: Intensity > alpha * 0.5 (Safety margin)
            is_strong = new_peak[0] > (self.alpha * 0.5)
            new_peak = jnp.where(is_strong, new_peak, jnp.zeros(4))
            
            is_empty = (current_params[:, 0] == 0)
            first_empty = jnp.argmax(is_empty)
            updated_params = current_params.at[first_empty].set(new_peak)
            
            params_raw = self._to_unconstrained(updated_params, *bounds)
            res = jax.scipy.optimize.minimize(
                fun=self._loss_fn,
                x0=params_raw,
                args=(x_grid, image_tile, self.alpha, bounds),
                method='BFGS',
                options={'maxiter': 5}
            )
            return self._to_physical(res.x, *bounds), best_val

        final_params, debug_vals = lax.scan(step_fn, init_params, jnp.arange(tile_max_peaks))
        
        # Final Polish
        params_raw = self._to_unconstrained(final_params, *bounds)
        res_final = jax.scipy.optimize.minimize(
            fun=self._loss_fn,
            x0=params_raw,
            args=(x_grid, image_tile, self.alpha, bounds),
            method='BFGS',
            options={'maxiter': 20}
        )
        return self._to_physical(res_final.x, *bounds), debug_vals

    def _progress_callback(self, current_step, total_steps):
        percent = (current_step / total_steps) * 100
        sys.stdout.write(f"\rSparseRBF Progress: {percent:.1f}%")
        sys.stdout.flush()


    def find_peaks_batch(self, images_batch):
        B, H, W = images_batch.shape

        # Robust Normalization
        dmin = images_batch.min(axis=(1,2), keepdims=True)
        dmax = images_batch.max(axis=(1,2), keepdims=True) + 1e-9
        images_norm = (images_batch - dmin) / (dmax - dmin)

        # Tiling Logic
        TR, TC = self.tiles
        pad_h = (TR - H % TR) % TR
        pad_w = (TC - W % TC) % TC
        if pad_h > 0 or pad_w > 0:
            images_norm = np.pad(images_norm, ((0,0), (0, pad_h), (0, pad_w)))

        H_pad, W_pad = images_norm.shape[1], images_norm.shape[2]
        TH, TW = H_pad // TR, W_pad // TC

        img_jax = jnp.array(images_norm)
        img_tiled = img_jax.reshape(B, TR, TH, TC, TW).transpose(0, 1, 3, 2, 4).reshape(-1, TH, TW)

        # Tile Grid
        yy, xx = jnp.indices((TH, TW))
        x_grid = jnp.array([yy, xx])

        chunk_size = 32
        num_tiles = img_tiled.shape[0]
        all_results = []
        all_debug = []

        batch_worker = jit(vmap(
            lambda img: self._run_tile_pursuit(img, x_grid, TH, TW)
        ))

        print(f"Running pursuit (Alpha={self.alpha})...")
        for i in range(0, num_tiles, chunk_size):
            chunk = img_tiled[i:i+chunk_size]
            res, dbg = batch_worker(chunk)
            all_results.append(np.array(res))
            all_debug.append(np.array(dbg))
            self._progress_callback(i + len(chunk), num_tiles)

        print("\nMerging tiles...")

        # Debug Stats
        flat_debug = np.concatenate(all_debug, axis=0)
        print(f"  > Max Correlation Found: {np.max(flat_debug):.4f}")

        all_params = np.concatenate(all_results, axis=0)
        structured_params = all_params.reshape(B, TR, TC, -1, 4)

        final_peaks_list = []

        total_found = 0
        total_kept = 0

        for b in range(B):
            bank_peaks = []
            for tr in range(TR):
                for tc in range(TC):
                    p_tile = structured_params[b, tr, tc]

                    # --- FIX: Relaxed Filtering ---
                    # 1. Intensity: Check against alpha
                    intensity_mask = p_tile[:, 0] > (self.alpha * 0.5)

                    # 2. Sigma: Allow touching the bound ( >= instead of > )
                    # We only reject if it somehow went BELOW min (numerical noise)
                    sigma_mask = p_tile[:, 3] >= (self.min_sigma * 0.99)

                    valid_mask = intensity_mask & sigma_mask
                    valid_p = p_tile[valid_mask]

                    # Stats
                    total_found += len(p_tile)
                    total_kept += len(valid_p)

                    if len(valid_p) > 0:
                        offset_r = tr * TH
                        offset_c = tc * TW
                        shifted_p = valid_p.copy()
                        shifted_p[:, 1] += offset_r
                        shifted_p[:, 2] += offset_c
                        bank_peaks.append(shifted_p)

            if len(bank_peaks) > 0:
                merged = np.vstack(bank_peaks)
                final_peaks_list.append(merged[:, 1:3])
            else:
                final_peaks_list.append(np.empty((0, 2)))

        print(f"  > Filtering Stats: Found {total_found} candidates, Kept {total_kept}.")
        if total_found > 0 and total_kept == 0:
            print("  > WARNING: All peaks rejected. Check if Alpha is too high or Min Sigma is too large.")

        return final_peaks_list
