import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import jax.scipy.optimize
import jax.scipy.signal
import sys
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# Import JAX with fallback from utils (centralized)
from subhkl.utils.shim import (
    HAS_JAX,
    jax,
    jit,
    jnp,
    jnp_update_add,
    jnp_update_set,
    lax,
    vmap,
)

if HAS_JAX:
    import jax.scipy.optimize
    import jax.scipy.signal

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

class SparseRBFPeakFinder:
    """
    JAX-Native 2-Stage Sparse RBF Peak Finder.
    
    Features:
    - Scout-Sniper 2-stage detection.
    - L2-Normalized Greedy Selection (Corrects sigma bias).
    - Chunked Reconstruction (Fixes compiler hang & OOM).
    - Besov Regularization (sigma^gamma penalty).
    """
    def __init__(
        self,
        alpha: float = 0.05,            
        gamma: float = 2.0,             
        min_sigma: float = 0.2,         
        max_sigma: float = 10.0,
        max_peaks: int = 500,           
        chunk_size: int = 1024,
        loss: str = 'gaussian',
        show_steps: bool = False,
        show_scale: str = "linear",
        tiles: tuple = None 
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.ref_sigma = 1.0            
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.max_peaks = max_peaks
        self.chunk_size = chunk_size
        self.loss = loss
        self.show_steps = show_steps
        
        self.base_window_size = 32      
        self.refine_patch_size = 15     
        
        # Log-spaced sigmas for search
        self.candidate_sigmas = jnp.geomspace(min_sigma, max_sigma, num=10)

    @staticmethod
    def _rbf_basis(x_grid, y, sigma):
        """Gaussian Radial Basis Function (Height=1)."""
        dist_sq = (x_grid[0] - y[0])**2 + (x_grid[1] - y[1])**2
        return jnp.exp(-dist_sq / (2.0 * sigma**2 + 1e-6))

    @staticmethod
    def _to_physical(params_raw, H, W, min_s, max_s):
        """Unconstrained -> Physical Parameters."""
        params_reshaped = params_raw.reshape((-1, 4))
        c_raw, r_raw, c_col_raw, s_raw = params_reshaped.T
        c = jax.nn.softplus(c_raw)
        r = jax.nn.sigmoid(r_raw) * H
        col = jax.nn.sigmoid(c_col_raw) * W
        sigma = min_s + jax.nn.sigmoid(s_raw) * (max_s - min_s)
        return jnp.stack([c, r, col, sigma], axis=1)

    @staticmethod
    def _to_unconstrained(params_phys, H, W, min_s, max_s):
        """Physical -> Unconstrained Parameters."""
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
        """Reconstruction for small batches (Solver)."""
        c, r, c_col, sigma = params_phys.T
        def eval_one(ci, ri, ci_col, si):
            return ci * SparseRBFPeakFinder._rbf_basis(x_grid, jnp.array([ri, ci_col]), si)
        basis_stack = vmap(eval_one)(c, r, c_col, sigma)
        return jnp.sum(basis_stack, axis=0)

    @staticmethod
    def _predict_batch_scan(params_phys, x_grid):
        """Memory-efficient scan reconstruction (for full image metrics)."""
        def body(carry, param):
            c, r, col, sigma = param
            term = c * SparseRBFPeakFinder._rbf_basis(x_grid, jnp.array([r, col]), sigma)
            return carry + term, None
        H, W = x_grid.shape[1], x_grid.shape[2]
        init = jnp.zeros((H, W), dtype=params_phys.dtype)
        final_image, _ = lax.scan(body, init, params_phys)
        return final_image

    @staticmethod
    def _loss_fn(params_flat, x_grid, target, alpha, gamma, ref_s, bounds_tuple, loss_type):
        H, W, min_s, max_s = bounds_tuple
        params_phys = SparseRBFPeakFinder._to_physical(params_flat, H, W, min_s, max_s)
        recon = SparseRBFPeakFinder._predict_batch_physical(params_phys, x_grid)
        
        if loss_type == 1: # Poisson
            recon_safe = jnp.maximum(recon, 1e-9)
            nll = jnp.sum(recon_safe - target * jnp.log(recon_safe))
        else: # Gaussian
            nll = 0.5 * jnp.sum((recon - target)**2)
        
        intensities = jnp.abs(params_phys[:, 0])
        sigmas = params_phys[:, 3]
        
        # Besov Norm (L1 of coeffs in normalized basis)
        sigma_ratio = sigmas / ref_s
        reg_weight = 1.0 / (sigma_ratio ** gamma + 1e-6)
        reg = alpha * jnp.sum(intensities * reg_weight)
        
        return nll + reg

    # =========================================================================
    # KERNEL: DENSE SOLVER
    # =========================================================================
    @partial(jit, static_argnames=['self', 'H', 'W', 'max_peaks_local'])
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
        loss_code = 1 if self.loss == 'poisson' else 0

        def step_fn(state, _):
            params, idx = state
            recon = self._predict_batch_physical(params, x_grid)
            residual = image - recon
            
            # --- GREEDY SELECTION (OMP Style) ---
            def check_sigma(s):
                # 1. Unnormalized kernel (matches basis function shape)
                kernel_raw = jnp.exp(-(kx**2 + ky**2) / (2 * s**2))
                
                # 2. Correlate (Compute inner product <Resid, Phi>)
                corr = jax.scipy.signal.correlate2d(residual, kernel_raw, mode='same')
                
                # 3. Find Max Correlation
                flat_idx = jnp.argmax(jnp.abs(corr))
                r_idx, c_idx = jnp.unravel_index(flat_idx, corr.shape)
                raw_dot = jnp.abs(corr[r_idx, c_idx])
                
                # 4. Normalize Projection (L2 Norm of Atom)
                # Norm_L2 = s * sqrt(pi) (approx for 2D Gaussian)
                # This ensures we pick the atom that best explains the residual *energy*
                # irrespective of scale. Removes bias towards large sigma.
                atom_norm = s * jnp.sqrt(jnp.pi)
                proj_score = raw_dot / (atom_norm + 1e-9)
                
                # 5. Apply Prior Weighting
                # Adjust selection order based on Besov prior preference
                prior_weight = (s / self.ref_sigma) ** self.gamma
                final_score = proj_score * prior_weight

                c_init = jnp.maximum(residual[r_idx, c_idx], 0.0)
                return final_score, jnp.array([c_init, r_idx, c_idx, s])

            vals, candidates = vmap(check_sigma)(self.candidate_sigmas)
            best_idx = jnp.argmax(vals)
            best_score = vals[best_idx]
            new_peak = candidates[best_idx]
            
            # Admission Gate
            is_strong = best_score > self.alpha
            new_peak = jnp.where(is_strong, new_peak, jnp.zeros(4))
        
            # Coordinate Descent / BFGS Refinement
            params = jnp_update_set(params, idx, new_peak)

            def run_opt(p):
                p_raw = self._to_unconstrained(p, *bounds)
                res = jax.scipy.optimize.minimize(
                    fun=self._loss_fn,
                    x0=p_raw,
                    args=(x_grid, image, self.alpha, self.gamma, self.ref_sigma, bounds, loss_code),
                    method='BFGS',
                    options={'maxiter': 5}
                )
                return self._to_physical(res.x, *bounds)

            params = run_opt(params)
            return (params, idx + 1), None

        final_state, _ = lax.scan(step_fn, init_state, None, length=max_peaks_local)
        final_params, _ = final_state
        return final_params

    # =========================================================================
    # METRICS
    # =========================================================================
    def compute_metrics(self, images_norm, peaks_list, global_max):
        """
        Computes metrics using a Python loop over the batch, running a JIT 
        kernel per image. This prevents the XLA input_reduce_fusion compilation hang.
        """
        B, H, W = images_norm.shape
        yy, xx = np.indices((H, W))
        x_grid = jnp.array([yy, xx])
        
        print("\n  [Metrics] Calculating goodness-of-fit...")

        max_k = max([len(p) for p in peaks_list] + [1])
        peaks_padded = np.zeros((B, max_k, 4), dtype=np.float32)
        counts_per_image = np.zeros(B, dtype=np.float32)
        
        for b in range(B):
            n = len(peaks_list[b])
            if n > 0:
                peaks_padded[b, :n, :] = peaks_list[b]
                peaks_padded[b, :n, 0] *= global_max
                if n < max_k:
                    peaks_padded[b, n:, 3] = 1.0 # Safe sigma
            counts_per_image[b] = n

        loss_code = 1 if self.loss == 'poisson' else 0

        # JIT compile for a SINGLE image. Extremely fast compilation.
        @jit
        def process_one_image(peaks, target, k_val):
            recon = self._predict_batch_scan(peaks, x_grid) 
            
            recon = jnp.maximum(recon, 1e-9)
            target = jnp.maximum(target, 1e-9)
            
            if loss_code == 1: 
                nll = jnp.sum(recon - target * jnp.log(recon))
                term = target * jnp.log(target / recon) - (target - recon)
                dev = 2 * jnp.sum(term)
            else: 
                diff = recon - target
                nll = 0.5 * jnp.sum(diff**2)
                dev = jnp.sum(diff**2)
            
            n_pix = target.size
            n_params = k_val * 4
            bic = n_params * jnp.log(n_pix) + 2 * nll
            return nll, bic, dev

        nll_total = 0.0
        bic_total = 0.0
        deviance_total = 0.0
        
        # Python loop completely isolates the batch dimension from XLA fusion
        for b in range(B):
            nll, bic, dev = process_one_image(
                jnp.array(peaks_padded[b]), 
                jnp.array(images_norm[b] * global_max), 
                jnp.array(counts_per_image[b])
            )
            # Casting to float auto-blocks until ready
            nll_total += float(nll)
            bic_total += float(bic)
            deviance_total += float(dev)

        pixels_total = B * H * W
        params_total = float(np.sum(counts_per_image)) * 4
        dof = max(pixels_total - params_total, 1)
        dev_per_dof = deviance_total / dof
        
        print(f"  > Total NLL: {nll_total:.2e}")
        print(f"  > Total BIC: {bic_total:.2e}")
        print(f"  > Deviance/DoF: {dev_per_dof:.4f} (Target ~ 1.0)")
        
        return {"nll": nll_total, "bic": bic_total, "deviance_nu": dev_per_dof}

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    def find_peaks_batch(self, images_batch):
        B, H, W = images_batch.shape
        
        # Pre-process
        medians = np.median(images_batch, axis=(1, 2), keepdims=True)
        images_bg_corr = np.maximum(images_batch - medians, 0)
        global_max = images_bg_corr.max() + 1e-9
        images_norm = images_bg_corr / global_max
        img_jax = jnp.array(images_norm)
        
        print(f"  > Pre-processing: Bg Subtracted. Global Max={global_max:.1f}")
        print(f"  > Loss Function: {self.loss.upper()}")

        PAD_GLOBAL = 32
        img_jax_padded = jnp.pad(img_jax, ((0,0), (PAD_GLOBAL, PAD_GLOBAL), (PAD_GLOBAL, PAD_GLOBAL)))

        # Phase 1: Scout
        w_scout = self.base_window_size
        stride = w_scout // 2
        
        grid_h = list(range(0, H - w_scout + 1, stride))
        if grid_h[-1] + w_scout < H: grid_h.append(H - w_scout)
        grid_w = list(range(0, W - w_scout + 1, stride))
        if grid_w[-1] + w_scout < W: grid_w.append(W - w_scout)
        
        window_coords = []
        for b in range(B):
            for r in grid_h:
                for c in grid_w:
                    window_coords.append((b, r, c))
        
        window_coords_arr = np.array(window_coords, dtype=np.int32)
        total_scout_wins = len(window_coords)
        print(f"\n[Phase 1] Scout: Scanning {total_scout_wins} windows ({w_scout}x{w_scout})...")

        @jit
        def extract_scout_window(img, b_idx, r_idx, c_idx):
            r_pad = r_idx + PAD_GLOBAL
            c_pad = c_idx + PAD_GLOBAL
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (w_scout, w_scout))
            return vmap(slice_one)(b_idx, r_pad, c_pad)

        scout_solver = jit(vmap(lambda w: self._solve_dense(w, w_scout, w_scout, 5)))
        
        scout_results = []
        with tqdm(total=total_scout_wins, desc="Scout", unit="win") as pbar:
            for i in range(0, total_scout_wins, self.chunk_size):
                chunk = window_coords_arr[i:i+self.chunk_size]
                wins = extract_scout_window(img_jax_padded, chunk[:, 0], chunk[:, 1], chunk[:, 2])
                res = scout_solver(wins)
                res.block_until_ready()
                
                global_res = np.array(res)
                offsets_r = chunk[:, 1][:, None]
                offsets_c = chunk[:, 2][:, None]
                
                valid_mask = global_res[:, :, 0] > 1e-9
                for k in range(len(chunk)):
                    if np.any(valid_mask[k]):
                        peaks = global_res[k][valid_mask[k]]
                        peaks[:, 1] += offsets_r[k]
                        peaks[:, 2] += offsets_c[k]
                        bank_idx = chunk[k, 0]
                        peaks_with_bank = np.column_stack([np.full(len(peaks), bank_idx), peaks])
                        scout_results.append(peaks_with_bank)
                pbar.update(len(chunk))

        if not scout_results:
            print("  > No candidates found.")
            return [np.empty((0, 3)) for _ in range(B)]

        all_candidates = np.vstack(scout_results)
        unique_candidates = []
        print("  > Deduplicating candidates...")
        
        for b in range(B):
            bank_mask = (all_candidates[:, 0] == b)
            if not np.any(bank_mask): continue
            cands = all_candidates[bank_mask, 2:4]
            vals = all_candidates[bank_mask, 1]
            order = np.argsort(vals)[::-1]
            cands_sorted = cands[order]
            keep = np.ones(len(cands_sorted), dtype=bool)
            if len(cands_sorted) > 1:
                dists = squareform(pdist(cands_sorted))
                np.fill_diagonal(dists, 9999.0)
                radius = 3.0
                for i in range(len(cands_sorted)):
                    if keep[i]:
                        neighbors = np.where(dists[i] < radius)[0]
                        neighbors = neighbors[neighbors > i] 
                        keep[neighbors] = False
            valid_seeds = cands_sorted[keep]
            bank_col = np.full((len(valid_seeds), 1), b)
            unique_candidates.append(np.hstack([bank_col, valid_seeds]))

        if not unique_candidates:
            return [np.empty((0, 3)) for _ in range(B)]
            
        all_seeds = np.vstack(unique_candidates)
        total_seeds = len(all_seeds)
        print(f"\n[Phase 2] Sniper: Refining {total_seeds} unique seeds...")

        P = self.refine_patch_size
        half_p = P // 2
       
        @jit
        def extract_patch_at_peak(img, centers):
            b_idx = centers[:, 0].astype(int)
            r_center = centers[:, 1].astype(int)
            c_center = centers[:, 2].astype(int)
            r_start = r_center + PAD_GLOBAL - half_p
            c_start = c_center + PAD_GLOBAL - half_p
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (P, P))
            return vmap(slice_one)(b_idx, r_start, c_start)

        sniper_solver = jit(vmap(lambda w: self._solve_dense(w, P, P, 1)))
        
        refined_peaks_by_bank = [[] for _ in range(B)]
        
        with tqdm(total=total_seeds, desc="Sniper", unit="seed") as pbar:
            for i in range(0, total_seeds, self.chunk_size):
                chunk = all_seeds[i:i+self.chunk_size]
                patches = extract_patch_at_peak(img_jax_padded, jnp.array(chunk))
                res = sniper_solver(patches) 
                res.block_until_ready()
                res_cpu = np.array(res[:, 0, :])
                
                intensities = res_cpu[:, 0]
                local_r = res_cpu[:, 1]
                local_c = res_cpu[:, 2]
                sigmas = res_cpu[:, 3]
                
                vol_factor = (sigmas / self.ref_sigma) ** 2
                besov_factor = (sigmas / self.ref_sigma) ** self.gamma
                score = intensities * vol_factor * besov_factor
                
                global_r = chunk[:, 1] - half_p + local_r
                global_c = chunk[:, 2] - half_p + local_c
                
                MARGIN = 10
                in_bounds = (global_r > MARGIN) & (global_r < H - MARGIN) & \
                            (global_c > MARGIN) & (global_c < W - MARGIN)
                valid = (score > self.alpha) & in_bounds
                
                valid_indices = np.where(valid)[0]
                for idx in valid_indices:
                    b_id = int(chunk[idx, 0])
                    peak_data = np.array([intensities[idx], global_r[idx], global_c[idx], sigmas[idx]])
                    refined_peaks_by_bank[b_id].append(peak_data)
                pbar.update(len(chunk))

        final_coords_output = []
        final_peaks_full = [] 
        
        for b in range(B):
            peaks = np.array(refined_peaks_by_bank[b])
            if len(peaks) > 0:
                order = np.argsort(peaks[:, 0])[::-1]
                peaks_sorted = peaks[order]
                keep = np.ones(len(peaks_sorted), dtype=bool)
                coords = peaks_sorted[:, 1:3]
                sigmas = peaks_sorted[:, 3]
                if len(coords) > 1:
                    dists = squareform(pdist(coords))
                    np.fill_diagonal(dists, 9999.0)
                    for i in range(len(coords)):
                        if keep[i]:
                            r = max(2.0, sigmas[i])
                            neighbors = np.where(dists[i] < r)[0]
                            neighbors = neighbors[neighbors > i]
                            keep[neighbors] = False
                unique_peaks = peaks_sorted[keep]
                final_peaks_full.append(unique_peaks)
                final_coords_output.append(unique_peaks[:, 1:4])
            else:
                final_peaks_full.append(np.empty((0, 4)))
                final_coords_output.append(np.empty((0, 3)))
        
        self.compute_metrics(img_jax, final_peaks_full, global_max)
        
        return final_coords_output

class SparseLaueIntegrator(SparseRBFPeakFinder):
    """
    Physics-Informed Sniper.
    Takes predicted spot coordinates, extracts patches, and uses Volume-Penalized
    Sparse RBF to simultaneously refine sub-pixel position and integrate intensity.
    """
    def __init__(self, alpha=0.05, patch_size=15, min_sigma=0.5, max_sigma=5.0):
        # Inherit the exact same loss functions and utilities
        super().__init__(
            alpha=alpha, min_sigma=min_sigma, max_sigma=max_sigma,
            loss='gaussian', show_steps=False
        )
        self.refine_patch_size = patch_size

    def integrate_reflections(self, images_batch, frames, rs, cs):
        """
        images_batch: (B, H, W) array of images
        frames: (N,) array of image indices for each reflection
        rs, cs: (N,) arrays of predicted row and column coordinates
        """
        B, H, W = images_batch.shape
        N_spots = len(frames)

        # 1. Pre-process (Background Subtraction)
        medians = np.median(images_batch, axis=(1, 2), keepdims=True)
        images_bg_corr = np.maximum(images_batch - medians, 0)
        global_max = images_bg_corr.max() + 1e-9
        images_norm = images_bg_corr / global_max
        img_jax = jnp.array(images_norm)

        PAD = self.refine_patch_size
        img_jax_padded = jnp.pad(img_jax, ((0,0), (PAD, PAD), (PAD, PAD)))

        P = self.refine_patch_size
        half_p = P // 2

        # 2. Patch Extractor
        @jit
        def extract_patches(img, f_idx, r_idx, c_idx):
            r_start = r_idx.astype(int) + PAD - half_p
            c_start = c_idx.astype(int) + PAD - half_p
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (P, P))
            return vmap(slice_one)(f_idx, r_start, c_start)

        # 3. Dedicated 1-Atom Patch Solver
        bounds = (float(P), float(P), self.min_sigma, self.max_sigma)
        yy, xx = jnp.indices((P, P))
        x_grid = jnp.array([yy, xx])

        @jit
        def solve_patches(patches):
            def process_patch(patch):
                # Initialize exactly at center of patch
                c_init = jnp.max(patch) + 1e-6
                init_phys = jnp.array([[c_init, float(half_p), float(half_p), 2.0]])
                init_raw = self._to_unconstrained(init_phys, *bounds)

                res = jax.scipy.optimize.minimize(
                    fun=self._loss_fn,
                    x0=init_raw.ravel(),
                    args=(x_grid, patch, self.alpha, bounds, 0), # 0 = Gaussian Loss
                    method='BFGS',
                    options={'maxiter': 10}
                )
                return self._to_physical(res.x, *bounds)[0] # Return single atom
            return vmap(process_patch)(patches)

        # 4. Execute in Chunks
        refined_peaks = []
        with tqdm(total=N_spots, desc="Sparse Laue Integration") as pbar:
            for i in range(0, N_spots, self.chunk_size):
                chunk_f = jnp.array(frames[i:i+self.chunk_size])
                chunk_r = jnp.array(rs[i:i+self.chunk_size])
                chunk_c = jnp.array(cs[i:i+self.chunk_size])

                patches = extract_patches(img_jax_padded, chunk_f, chunk_r, chunk_c)
                res = solve_patches(patches)
                res.block_until_ready()

                res_cpu = np.array(res)
                # Map back to global coordinates
                # res_cpu: [Int, Local_R, Local_C, Sigma]
                res_cpu[:, 1] += rs[i:i+self.chunk_size] - half_p
                res_cpu[:, 2] += cs[i:i+self.chunk_size] - half_p
                # Scale intensity back to counts
                res_cpu[:, 0] *= global_max

                refined_peaks.append(res_cpu)
                pbar.update(len(chunk_f))

        if len(refined_peaks) == 0:
            return np.empty((0, 4))

        return np.vstack(refined_peaks)

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Dict
from functools import partial
from tqdm import tqdm

# =====================================================================
# 1. HYPER-OPTIMIZED GPU KERNELS
# =====================================================================

@partial(jax.jit, static_argnames=['N_shapes', 'max_peaks'])
def build_and_reduce_gpu(image: jnp.ndarray, padded_centers: jnp.ndarray, sigmas: jnp.ndarray,
                         gamma: float, N_shapes: int, max_peaks: int):
    """
    Builds the dense matrix virtually, computes A^T A, A^T y, and Fisher Info,
    then immediately discards A. Returns tiny 9MB matrices instead of 6GB.
    """
    H, W = image.shape
    K = H * W
    N_c = max_peaks * N_shapes

    y_idx = jnp.arange(H, dtype=jnp.float32)
    x_idx = jnp.arange(W, dtype=jnp.float32)
    y_coords, x_coords = jnp.meshgrid(y_idx, x_idx, indexing='ij')

    dx = x_coords.flatten()[:, None] - padded_centers[None, :, 0]
    dy = y_coords.flatten()[:, None] - padded_centers[None, :, 1]
    r2 = dx**2 + dy**2

    sig_arr = sigmas[None, None, :]
    phi = jnp.exp(-r2[:, :, None] / (2.0 * sig_arr**2))
    w = sig_arr**gamma

    A_peaks = (phi / w).reshape((K, N_c))
    weights = jnp.broadcast_to(w, (1, max_peaks, N_shapes)).reshape((N_c,))
    volumes = jnp.float32(2.0 * jnp.pi) * (jnp.broadcast_to(sig_arr, (1, max_peaks, N_shapes)).reshape((N_c,))**2)

    x_norm = (x_coords.flatten() - W / 2.0) / W
    y_norm = (y_coords.flatten() - H / 2.0) / H
    A_bg = jnp.column_stack([jnp.ones(K, dtype=jnp.float32), x_norm, y_norm])

    A = jnp.hstack([A_peaks, A_bg])
    y = image.flatten()

    # Condense into tiny matrices to save memory and transport
    Ht = (A.T @ A).astype(jnp.float32)
    At_y = (A.T @ y).astype(jnp.float32)
    y_sq_norm = jnp.float32(jnp.sum(y**2) / 2.0)

    # Compute Fisher Info instantly
    variances = jnp.maximum(y, 1.0)
    W_var_diag = 1.0 / variances
    I_fisher = (A.T @ (W_var_diag[:, None] * A)).astype(jnp.float32)

    return Ht, At_y, y_sq_norm, I_fisher, weights, volumes

@partial(jax.jit, static_argnames=['N_c', 'max_iter'])
def solve_ssn_gpu(Ht: jnp.ndarray, At_y: jnp.ndarray, y_sq_norm: jnp.float32,
                  N_c: int, alpha: float, max_iter: int = 50, tol: float = 1e-5):
    """
    Newton solve completely decoupled from A. Uses Conjugate Gradient (CG)
    to avoid O(N^3) LU factorizations, making it 150x faster.
    """
    N = Ht.shape[0]
    alpha = jnp.float32(alpha)
    tol = jnp.float32(tol)

    def obj(u):
        quad = jnp.float32(0.5) * jnp.dot(u, Ht @ u) - jnp.dot(u, At_y) + y_sq_norm
        return quad + alpha * jnp.sum(jnp.abs(u[:N_c]))

    u_init = jnp.zeros(N, dtype=jnp.float32)
    gf0 = -At_y
    gf0_c = gf0[:N_c]
    mask = jnp.abs(gf0_c) > alpha
    gf0_c = jnp.where(mask, jnp.float32(1.0 - 1e-14) * alpha * jnp.sign(gf0_c), gf0_c).astype(jnp.float32)
    q_init = (u_init - jnp.concatenate([gf0_c, gf0[N_c:]])).astype(jnp.float32)

    def cond_fun(state):
        step, _, _, Gq_norm, _ = state
        return (step < max_iter) & (Gq_norm > tol)

    def body_fun(state):
        step, q, u, Gq_norm, _ = state

        Gq = (q - u) + (Ht @ u) - At_y
        II_c = jnp.abs(q[:N_c]) > alpha
        II_b = jnp.ones(N - N_c, dtype=jnp.bool_)
        D = jnp.concatenate([II_c, II_b]).astype(jnp.float32)

        epsi = jnp.float32(1e-4)

        # 150x Faster Conjugate Gradient Solve for the Active Sub-system
        def matvec(v):
            return D * jnp.dot(Ht, D * v) + (1.0 - D) * v + epsi * v

        x, _ = jax.scipy.sparse.linalg.cg(matvec, -Gq, maxiter=30)

        dq_I = D * x
        dq_Ic = (1.0 - D) * (-Gq - jnp.dot(Ht, dq_I))
        dq = dq_I + dq_Ic

        j_baseline = obj(u)

        def bt_cond(bt_state):
            bt_i, tau, _, _, j_test, j_curr = bt_state
            return (bt_i < 10) & (j_test > j_curr * jnp.float32(1.0 + 1e-10))

        def bt_body(bt_state):
            bt_i, tau, _, _, _, j_curr = bt_state
            tau = jnp.float32(tau * 0.5)
            q_test = q + tau * dq
            c_test = jnp.sign(q_test[:N_c]) * jnp.maximum(0.0, jnp.abs(q_test[:N_c]) - alpha)
            u_test = jnp.concatenate([c_test, q_test[N_c:]]).astype(jnp.float32)
            j_test = obj(u_test)
            return (bt_i + 1, tau, q_test, u_test, j_test, j_curr)

        q_init_test = q + dq
        c_init_test = jnp.sign(q_init_test[:N_c]) * jnp.maximum(0.0, jnp.abs(q_init_test[:N_c]) - alpha)
        u_init_test = jnp.concatenate([c_init_test, q_init_test[N_c:]]).astype(jnp.float32)

        bt_init = (jnp.int32(0), jnp.float32(1.0), q_init_test, u_init_test, obj(u_init_test), j_baseline)
        bt_final = jax.lax.while_loop(bt_cond, bt_body, bt_init)
        _, _, q_final, u_final, _, _ = bt_final

        return (step + 1, q_final, u_final, jnp.linalg.norm(Gq).astype(jnp.float32), D > 0.5)

    init_state = (jnp.int32(0), q_init, u_init, jnp.float32(1e9), jnp.zeros(N, dtype=jnp.bool_))
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, _, u_prime, _, active_set = final_state

    return u_prime, active_set

# =====================================================================
# 2. CPU POST-PROCESSING & PYTEST BRIDGES
# =====================================================================

def evaluate_fisher_sigi_cpu(I_fisher_active: np.ndarray, c_active: np.ndarray, active_volumes: np.ndarray):
    """Calculates robust SIGI with NaN guards to prevent OpenBLAS infinite loops."""
    if np.any(np.isnan(I_fisher_active)) or np.any(np.isinf(I_fisher_active)):
        return 0.0, 0.0

    intensity = np.dot(c_active, active_volumes[:len(c_active)])
    try:
        U, S, Vh = np.linalg.svd(I_fisher_active, full_matrices=False)
        tol = 1e-6 * np.max(S)
        S_inv = np.where(S > tol, 1.0 / S, 0.0)
        covariance_matrix = (Vh.T * S_inv) @ U.T
        variance_I = np.dot(active_volumes.T, np.dot(covariance_matrix, active_volumes))
        return float(intensity), float(np.sqrt(max(0.0, variance_I)))
    except np.linalg.LinAlgError:
        return float(intensity), 0.0

def build_dense_padded_matrix(image: np.ndarray, peak_centers: np.ndarray, sigmas: List[float], gamma: float, max_peaks: int):
    """Bridge for Pytest Compatibility to reconstruct A on CPU."""
    H, W = image.shape
    y_idx = np.arange(H, dtype=np.float32)
    x_idx = np.arange(W, dtype=np.float32)
    y_coords, x_coords = np.meshgrid(y_idx, x_idx, indexing='ij')
    actual_peaks = len(peak_centers)
    padded = np.pad(peak_centers, ((0, max_peaks - actual_peaks), (0, 0)), constant_values=-10000.0)

    r2 = (x_coords.flatten()[:, None] - padded[None, :, 0])**2 + (y_coords.flatten()[:, None] - padded[None, :, 1])**2
    sig_arr = np.array(sigmas, dtype=np.float32)[None, None, :]
    phi = np.exp(-r2[:, :, None] / (2.0 * sig_arr**2))
    w = sig_arr**gamma

    A_peaks = (phi / w).reshape((H*W, max_peaks * len(sigmas)))
    A_bg = np.column_stack([np.ones(H*W), (x_coords.flatten() - W/2)/W, (y_coords.flatten() - H/2)/H])
    weights = np.broadcast_to(w, (1, max_peaks, len(sigmas))).reshape((max_peaks * len(sigmas),))
    volumes = 2.0 * np.pi * (np.broadcast_to(sig_arr, (1, max_peaks, len(sigmas))).reshape((max_peaks * len(sigmas),))**2)

    return jnp.array(np.hstack([A_peaks, A_bg])), jnp.array(weights), jnp.array(volumes), actual_peaks

def solve_ssn(A: np.ndarray, y: jnp.ndarray, N_c: int, alpha: float):
    """Bridge for Pytest Compatibility."""
    y = jnp.array(y)
    Ht = (A.T @ A).astype(jnp.float32)
    At_y = (A.T @ y).astype(jnp.float32)
    y_sq_norm = jnp.float32(jnp.sum(y**2) / 2.0)
    u_prime, active_set = solve_ssn_gpu(Ht, At_y, y_sq_norm, N_c, alpha)
    return np.array(u_prime), np.array(active_set)


# =====================================================================
# 3. SYNCHRONOUS ORCHESTRATOR
# =====================================================================

def integrate_peaks_rbf_ssn(peak_dict: Dict, image_handler, sigmas: List[float],
                            alpha: float, gamma: float, max_peaks: int, show_progress: bool):
    class RBFResult:
        def __init__(self):
            self.h, self.k, self.l = [], [], []
            self.intensity, self.sigma = [], []
            self.tt, self.run_id, self.xyz = [], [], []

    res = RBFResult()
    sigmas_jnp = jnp.array(sigmas, dtype=jnp.float32)
    N_shapes = len(sigmas)
    N_c = max_peaks * N_shapes

    for img_key, p_data in tqdm(peak_dict.items(), disable=not show_progress, desc="RBF Integration (Dense GPU)"):
        peak_centers = np.column_stack([p_data[0], p_data[1]])
        actual_peaks_count = len(peak_centers)

        if actual_peaks_count == 0:
            continue
        if actual_peaks_count > max_peaks:
            raise ValueError(f"Actual peaks ({actual_peaks_count}) exceeds max_peaks ({max_peaks}).")

        padded_centers = np.pad(peak_centers, ((0, max_peaks - actual_peaks_count), (0, 0)), constant_values=-10000.0)
        padded_centers_jnp = jnp.array(padded_centers, dtype=jnp.float32)

        image_raw = np.nan_to_num(image_handler.ims[img_key], nan=0.0, posinf=0.0, neginf=0.0)
        image_jnp = jnp.array(image_raw, dtype=jnp.float32)

        # 1. Condense Memory (Compute Ht and Fisher directly)
        Ht, At_y, y_sq_norm, I_fisher, weights, volumes = build_and_reduce_gpu(
            image_jnp, padded_centers_jnp, sigmas_jnp, gamma, N_shapes, max_peaks
        )
        Ht = Ht.block_until_ready()

        # 2. Conjugate Gradient SSN Solve
        u_prime, active_set = solve_ssn_gpu(Ht, At_y, y_sq_norm, N_c, alpha)
        u_prime = u_prime.block_until_ready()

        # 3. Fast CPU Extraction
        u_prime_cpu = np.array(u_prime)
        active_set_cpu = np.array(active_set)
        I_fisher_cpu = np.array(I_fisher)
        weights_cpu = np.array(weights)
        volumes_cpu = np.array(volumes)

        c_unscaled = u_prime_cpu[:N_c] / weights_cpu

        for p_idx in range(actual_peaks_count):
            start_idx = p_idx * N_shapes
            end_idx = start_idx + N_shapes

            p_intensity = np.dot(c_unscaled[start_idx:end_idx], volumes_cpu[start_idx:end_idx])
            p_sigi = 0.0

            if p_intensity > 0:
                p_mask = np.zeros_like(active_set_cpu)
                p_mask[start_idx:end_idx] = active_set_cpu[start_idx:end_idx]
                p_mask[N_c:] = True

                if np.any(p_mask[:N_c]):
                    active_indices = np.where(p_mask)[0]
                    I_fisher_active = I_fisher_cpu[np.ix_(active_indices, active_indices)]
                    c_active = u_prime_cpu[active_indices] / np.concatenate([weights_cpu, np.ones(3)])[active_indices]
                    vols_active = np.concatenate([volumes_cpu, np.zeros(3)])[active_indices]

                    _, p_sigi = evaluate_fisher_sigi_cpu(I_fisher_active, c_active, vols_active)

            res.h.append(p_data[2][p_idx])
            res.k.append(p_data[3][p_idx])
            res.l.append(p_data[4][p_idx])
            res.intensity.append(float(p_intensity))
            res.sigma.append(float(p_sigi))
            res.run_id.append(img_key)

    return res
