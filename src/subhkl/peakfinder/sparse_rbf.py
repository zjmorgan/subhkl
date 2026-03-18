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

@partial(jax.jit, static_argnames=['N_c', 'K'])
def ssn_step(q: jnp.ndarray, A: jnp.ndarray, y: jnp.ndarray, alpha: float, N_c: int, K: int) -> Tuple:
    """A single Semi-Smooth Newton step. Fully JIT-able."""
    N = q.shape[0]

    # 1. Proximal operator mapping q -> u
    c_part = jnp.sign(q[:N_c]) * jnp.maximum(0.0, jnp.abs(q[:N_c]) - alpha)
    b_part = q[N_c:]
    u = jnp.concatenate([c_part, b_part])

    # 2. Robinson Normal Map G(q, u)
    residual = A @ u - y
    Gq = (q - u) + A.T @ residual / K

    # 3. Active set mask
    II_c = jnp.abs(q[:N_c]) > alpha
    II_b = jnp.ones(N - N_c, dtype=bool)
    II = jnp.concatenate([II_c, II_b])

    # 4. Generalized Jacobian DG (using masking)
    DPc_mat = jnp.diag(II.astype(A.dtype))
    I = jnp.eye(N, dtype=A.dtype)
    DG = (I - DPc_mat) + (A.T @ A / K) @ DPc_mat

    epsi = 1e-12 * jnp.max(jnp.sum(jnp.abs(DG), axis=1))
    DG = DG + epsi * DPc_mat

    dq = jnp.linalg.solve(DG, -Gq)
    return u, Gq, dq, II

@partial(jax.jit, static_argnames=['N_c','max_iter'])
def solve_ssn(A: jnp.ndarray, y: jnp.ndarray, N_c: int, alpha: float, max_iter: int = 100, tol: float = 1e-6):
    """
    Solves the L1 regularized system using jax.lax.while_loop to prevent
    unrolling and massive compile times.
    """
    K_val, N = A.shape

    def obj(u):
        return jnp.sum((A @ u - y)**2) / (2 * K_val) + alpha * jnp.sum(jnp.abs(u[:N_c]))

    u_init = jnp.zeros(N, dtype=jnp.float32)
    gf0 = A.T @ (-y) / K_val

    gf0_c = gf0[:N_c]
    mask = jnp.abs(gf0_c) > alpha
    gf0_c = jnp.where(mask, (1 - 1e-14) * alpha * jnp.sign(gf0_c), gf0_c)
    q_init = u_init - jnp.concatenate([gf0_c, gf0[N_c:]])

    def cond_fun(state):
        step, _, _, Gq_norm, _ = state
        return (step < max_iter) & (Gq_norm > tol)

    def body_fun(state):
        step, q, u, Gq_norm, active_set = state
        u_new, Gq, dq, active_set_new = ssn_step(q, A, y, alpha, N_c, K_val)

        # Backtracking Line Search (Inner while loop)
        j_baseline = obj(u)

        def bt_cond(bt_state):
            bt_i, tau, _, _, j_test, j_curr = bt_state
            return (bt_i < 10) & (j_test > j_curr * (1.0 + 1e-10))

        def bt_body(bt_state):
            bt_i, tau, _, _, _, j_curr = bt_state
            tau = tau * 0.5
            q_test = q + tau * dq
            c_test = jnp.sign(q_test[:N_c]) * jnp.maximum(0.0, jnp.abs(q_test[:N_c]) - alpha)
            u_test = jnp.concatenate([c_test, q_test[N_c:]])
            j_test = obj(u_test)
            return (bt_i + 1, tau, q_test, u_test, j_test, j_curr)

        q_init_test = q + dq
        c_init_test = jnp.sign(q_init_test[:N_c]) * jnp.maximum(0.0, jnp.abs(q_init_test[:N_c]) - alpha)
        u_init_test = jnp.concatenate([c_init_test, q_init_test[N_c:]])

        bt_init = (0, 1.0, q_init_test, u_init_test, obj(u_init_test), j_baseline)
        bt_final = jax.lax.while_loop(bt_cond, bt_body, bt_init)
        _, _, q_final, u_final, _, _ = bt_final

        return (step + 1, q_final, u_final, jnp.linalg.norm(Gq), active_set_new)

    init_state = (0, q_init, u_init, 1e9, jnp.zeros(N, dtype=bool))
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

    _, _, final_u, _, final_active_set = final_state
    return final_u, final_active_set

def build_dense_padded_matrix(image: np.ndarray, peak_centers: np.ndarray, sigmas: List[float], gamma: float, max_peaks: int):
    """Builds the flattened, padded dense matrix A and volume constraints."""
    H, W = image.shape
    K = H * W
    N_shapes = len(sigmas)

    # Meshgrid for panel
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    actual_peaks = len(peak_centers)
    if actual_peaks > max_peaks:
        raise ValueError(f"Actual peaks ({actual_peaks}) exceeds max_peaks_per_panel ({max_peaks}). Increase parameter.")

    # Pad peak centers with dummy values extremely far off the detector
    padded_centers = np.pad(peak_centers, ((0, max_peaks - actual_peaks), (0, 0)), constant_values=-10000.0)

    # 1. Build A_peaks
    A_peaks = np.zeros((K, max_peaks * N_shapes), dtype=np.float32)
    weights = np.zeros(max_peaks * N_shapes, dtype=np.float32)
    volumes = np.zeros(max_peaks * N_shapes, dtype=np.float32)

    idx = 0
    for cx, cy in padded_centers:
        for sig in sigmas:
            r2 = (x_flat - cx)**2 + (y_flat - cy)**2
            # For dummies, distance is huge, so exp() -> exactly 0.0 in float32
            phi = np.exp(-r2 / (2 * sig**2))
            w = sig**gamma

            A_peaks[:, idx] = phi / w  # Apply Besov scaling
            weights[idx] = w
            volumes[idx] = 2 * np.pi * sig**2
            idx += 1

    # 2. Linear Background
    A_bg = np.column_stack([np.ones(K), x_flat, y_flat]).astype(np.float32)

    A = np.hstack([A_peaks, A_bg])
    return jnp.array(A), jnp.array(weights), jnp.array(volumes), actual_peaks

def evaluate_fisher_sigi(A: jnp.ndarray, u_prime: jnp.ndarray, active_set: jnp.ndarray,
                         intensities: jnp.ndarray, volumes: jnp.ndarray, weights: jnp.ndarray, N_c: int):
    """Calculates robust SIGI strictly on the active set using SVD."""
    # Ensure variances don't cause division by zero for zero-count background pixels
    variances = jnp.maximum(intensities, 1.0)
    W_var = jnp.diag(1.0 / variances)

    A_active = A[:, active_set]
    I_fisher = A_active.T @ W_var @ A_active

    # Robust Pseudo-Inverse
    U, S, Vh = jnp.linalg.svd(I_fisher, full_matrices=False)
    tol = 1e-12 * jnp.max(S)
    S_inv = jnp.where(S > tol, 1.0 / S, 0.0)
    covariance_matrix = (Vh.T * S_inv) @ U.T

    # Reconstruct unscaled intensity
    active_peaks_mask = active_set[:N_c]
    c_active = u_prime[:N_c][active_peaks_mask] / weights[active_peaks_mask]

    active_vols = jnp.concatenate([volumes[active_peaks_mask], jnp.zeros(3)]) # 0 vol for BG
    intensity = jnp.dot(c_active, active_vols[:len(c_active)])

    variance_I = jnp.dot(active_vols.T, jnp.dot(covariance_matrix, active_vols))
    return intensity, jnp.sqrt(variance_I)

def integrate_peaks_rbf_ssn(peak_dict: Dict, image_handler, sigmas: List[float],
                            alpha: float, gamma: float, max_peaks: int, show_progress: bool):
    """Iterates dense solves over all panels/images."""
    class RBFResult:
        def __init__(self):
            self.h, self.k, self.l = [], [], []
            self.intensity, self.sigma = [], []
            self.tt, self.run_id, self.xyz = [], [], []

    res = RBFResult()
    N_shapes = len(sigmas)
    N_c = max_peaks * N_shapes

    # Trigger JIT compilation on the first iteration
    for img_key, p_data in tqdm(peak_dict.items(), disable=not show_progress, desc="RBF Integration (Dense GPU)"):
        # p_data = [i, j, h, k, l, wl]
        peak_centers = np.column_stack([p_data[0], p_data[1]])
        actual_peaks_count = len(peak_centers)

        if actual_peaks_count == 0:
            continue

        image = image_handler.ims[str(img_key)]
        intensities = jnp.array(image.flatten(), dtype=jnp.float32)

        # Build Dense Padded Matrix
        A, weights, volumes, _ = build_dense_padded_matrix(image, peak_centers, sigmas, gamma, max_peaks)

        # Execute JIT compiled solver
        u_prime, active_set = solve_ssn(A, intensities, N_c, alpha)

        # Process and un-pad results
        active_peaks_mask = active_set[:N_c]
        c_unscaled = u_prime[:N_c] / weights

        # Calculate isolated variance for EACH active peak (ignoring crosstalk in this step for export speed,
        # or you can run SVD jointly as shown in the function above).
        # For full rigorous UQ, we loop SVD extraction over actual peaks:
        for p_idx in range(actual_peaks_count):
            start_idx = p_idx * N_shapes
            end_idx = start_idx + N_shapes

            p_intensity = jnp.dot(c_unscaled[start_idx:end_idx], volumes[start_idx:end_idx])

            p_sigi = 0.0
            if p_intensity > 0:
                # Build isolated active mask for just this peak + background
                p_mask = jnp.zeros_like(active_set)
                p_mask = p_mask.at[start_idx:end_idx].set(active_set[start_idx:end_idx])
                p_mask = p_mask.at[N_c:].set(True) # Keep BG

                if jnp.any(p_mask[:N_c]):
                    _, p_sigi = evaluate_fisher_sigi(A, u_prime, p_mask, intensities, volumes, weights, N_c)

            res.h.append(p_data[2][p_idx])
            res.k.append(p_data[3][p_idx])
            res.l.append(p_data[4][p_idx])
            res.intensity.append(float(p_intensity))
            res.sigma.append(float(p_sigi))

            # Additional metadata padding handling (tt, run_id, xyz) would be extracted from `peaks` object here.
            # Assuming you pass run_id or extract tt via det.pixel_to_angles
            res.run_id.append(img_key)

    return res
