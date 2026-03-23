import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import jax.scipy.optimize
import jax.scipy.signal
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

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


class SparseRBFPeakFinder:
    """
    Hierarchical Sparse RBF Peak Finder with Symmetric V-Cycle Basis Pursuit.
    
    Features:
    - Dyadic Scale Hierarchy (Powers of 2)
    - Halo Context Extraction (Prevents boundary truncation).
    - Upward Macro-Merge (Besov Pursuit via SSN L1 Projection).
    - Unified Semi-Smooth Newton Solver with Fisher Scoring for Poisson NLL.
    """
    def __init__(
        self,
        alpha: float = 0.05,            
        gamma: float = 2.0,             
        min_sigma: float = 0.5,         
        max_sigma: float = 8.0,
        max_peaks: int = 500,           
        chunk_size: int = 1024,
        loss: str = 'gaussian',
        show_steps: bool = True
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
        self.halo = 5  
        self.max_local_peaks = 3  
        
        dyadic_scales = []
        current_s = max_sigma
        while current_s >= min_sigma:
            dyadic_scales.append(current_s)
            current_s /= 2.0
        self.candidate_sigmas = jnp.array(dyadic_scales)

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
    def _predict_batch_scan(params_phys, x_grid):
        def body(carry, param):
            c, r, col, sigma = param
            term = c * SparseRBFPeakFinder._rbf_basis(x_grid, jnp.array([r, col]), sigma)
            return carry + term, None
        H, W = x_grid.shape[1], x_grid.shape[2]
        init = jnp.zeros((H, W), dtype=params_phys.dtype)
        final_image, _ = lax.scan(body, init, params_phys)
        return final_image

    @staticmethod
    def _loss_fn(params_flat, x_grid, target, alpha, gamma, ref_s, bounds_tuple):
        H, W, min_s, max_s = bounds_tuple
        params_phys = SparseRBFPeakFinder._to_physical(params_flat, H, W, min_s, max_s)
        recon = SparseRBFPeakFinder._predict_batch_physical(params_phys, x_grid)
        nll = 0.5 * jnp.sum((recon - target)**2)
        
        intensities = jnp.abs(params_phys[:, 0])
        sigmas = params_phys[:, 3]
        
        # DIRECT PENALTY
        reg_weight = (sigmas / ref_s) ** gamma + 1e-6
        reg = alpha * jnp.sum(intensities * reg_weight)
        
        return nll + reg

    @staticmethod
    @partial(jit, static_argnames=['max_iter', 'loss_type'])
    def _solve_ssn_unified(A, y, alpha_vec, loss_type, c_warm, max_iter=20):
        if loss_type == 1: 
            A_use = jnp.hstack([A, jnp.ones((A.shape[0], 1))])
            alpha_pad = jnp.append(alpha_vec, 0.0) 
            c_init = jnp.append(c_warm, jnp.maximum(jnp.mean(y), 1e-2))
        else:
            A_use = A
            alpha_pad = alpha_vec
            c_init = c_warm

        N = A_use.shape[1]
        q_init = c_init

        def get_loss_grad_hess(c):
            u = A_use @ c
            if loss_type == 1: 
                u_safe = jnp.maximum(u, 1e-5)
                nll = jnp.sum(u_safe - y * jnp.log(u_safe))
                grad = A_use.T @ (1.0 - y / u_safe)
                W = 1.0 / jnp.maximum(u_safe, 1e-5)
                hess = A_use.T @ (W[:, None] * A_use)
            else: 
                nll = 0.5 * jnp.sum((u - y)**2)
                grad = A_use.T @ (u - y)
                hess = A_use.T @ A_use
                
            reg = jnp.sum(alpha_pad * c)
            return nll + reg, grad, hess

        def cond_fn(state):
            step, _, _, Gq_norm = state
            return (step < max_iter) & (Gq_norm > 1e-4)

        def body_fn(state):
            step, q, c, _ = state
            obj_val, grad, hess = get_loss_grad_hess(c)

            Gq = (q - c) + grad
            D = (q > alpha_pad).astype(jnp.float32)
            DP_mat = jnp.diag(D)
            I = jnp.eye(N)
            
            DG = (I - DP_mat) + hess @ DP_mat + 1e-3 * I
            dq = jnp.linalg.solve(DG, -Gq)

            def bt_cond(bt_state):
                bt_i, tau, _, _, j_test, j_curr = bt_state
                is_valid = jnp.isfinite(j_test)
                return (bt_i < 8) & ((j_test > j_curr) | ~is_valid)

            def bt_body(bt_state):
                bt_i, tau, _, _, _, j_curr = bt_state
                tau = tau * 0.5
                q_test = q + tau * dq
                c_test = jnp.maximum(0.0, q_test - alpha_pad) 
                j_test, _, _ = get_loss_grad_hess(c_test)
                return (bt_i + 1, tau, q_test, c_test, j_test, j_curr)

            q_test = q + dq
            c_test = jnp.maximum(0.0, q_test - alpha_pad)
            j_test, _, _ = get_loss_grad_hess(c_test)

            bt_init = (0, 1.0, q_test, c_test, j_test, obj_val)
            bt_final = lax.while_loop(bt_cond, bt_body, bt_init)
            _, _, q_final, c_final, _, _ = bt_final

            return (step + 1, q_final, c_final, jnp.linalg.norm(Gq))

        init_state = (0, q_init, c_init, 1e9)
        final_state = lax.while_loop(cond_fn, body_fn, init_state)
        _, _, c_final, _ = final_state

        if loss_type == 1:
            return c_final[:-1] 
        else:
            return c_final      

    @partial(jit, static_argnames=['self', 'H', 'W', 'max_peaks_local', 'loss_code', 'do_merge'])
    def _solve_dense(self, patch_geom, patch_stat, global_max, eff_alpha_scout, eff_alpha_stat, H, W, max_peaks_local, loss_code, do_merge):
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
            residual = patch_geom - recon 
            
            def check_sigma(s):
                kernel_raw = jnp.exp(-(kx**2 + ky**2) / (2 * s**2))
                corr = jax.scipy.signal.correlate2d(residual, kernel_raw, mode='same')
                flat_idx = jnp.argmax(jnp.abs(corr))
                r_idx, c_idx = jnp.unravel_index(flat_idx, corr.shape)
                raw_dot = jnp.abs(corr[r_idx, c_idx])
                
                # CORRECT HEURISTIC INVERSE: Divides by sigma**gamma to match L1 admission requirement
                weight = 1.0 / ((s / self.ref_sigma) ** self.gamma + 1e-6)
                final_score = raw_dot * weight
                
                c_init = jnp.maximum(residual[r_idx, c_idx], 0.0)
                return final_score, jnp.array([c_init, r_idx, c_idx, s])

            vals, candidates = vmap(check_sigma)(self.candidate_sigmas)
            best_idx = jnp.argmax(vals)
            best_score = vals[best_idx]
            new_peak = candidates[best_idx]
            
            is_strong = best_score > eff_alpha_scout
            new_peak = jnp.where(is_strong, new_peak, jnp.zeros(4))
            params = jnp_update_set(params, idx, new_peak)

            def run_opt(p):
                p_raw = self._to_unconstrained(p, *bounds)
                res = jax.scipy.optimize.minimize(
                    fun=self._loss_fn,
                    x0=p_raw,
                    args=(x_grid, patch_geom, eff_alpha_scout, self.gamma, self.ref_sigma, bounds),
                    method='BFGS',
                    options={'maxiter': 5}
                )
                p_refined = self._to_physical(res.x, *bounds)
                c_norm, r, col, sigma = p_refined.T
                
                def eval_one(ri, ci_col, si):
                    return self._rbf_basis(x_grid, jnp.array([ri, ci_col]), si).flatten()
                
                A = vmap(eval_one)(r, col, sigma).T
                
                c_warm = jnp.where(loss_code == 1, c_norm * global_max, c_norm)
                
                # DIRECT PENALTY
                weights = (sigma / self.ref_sigma)**self.gamma + 1e-6
                alpha_vec_stat = eff_alpha_stat * weights
                
                c_sparse_stat = self._solve_ssn_unified(A, patch_stat.flatten(), alpha_vec_stat, loss_code, c_warm)
                c_sparse_norm = jnp.where(loss_code == 1, c_sparse_stat / global_max, c_sparse_stat)
                return jnp.stack([c_sparse_norm, r, col, sigma], axis=1)

            params = run_opt(params)
            return (params, idx + 1), None

        final_state, _ = lax.scan(step_fn, init_state, None, length=max_peaks_local)
        final_params, _ = final_state
        
        # --- ALL-IN-ONE JAX V-CYCLE (Macro Merge inside the GPU Kernel) ---
        if do_merge:
            c, r, col, sigma = final_params.T
            active_mask = c > 1e-9
            num_active = jnp.sum(active_mask)
            
            total_amp = jnp.sum(c) + 1e-12
            com_r = jnp.sum(c * r) / total_amp
            com_c = jnp.sum(c * col) / total_amp
            var_r = jnp.sum(c * (r - com_r)**2) / total_amp
            var_c = jnp.sum(c * (col - com_c)**2) / total_amp
            macro_sigma = jnp.sqrt(var_r + var_c) + jnp.sum(c * sigma) / total_amp
            
            macro_atom = jnp.stack([jnp.sum(c), com_r, com_c, macro_sigma])
            macro_atom = jnp.where(num_active > 1, macro_atom, jnp.zeros(4))
            augmented_dict = jnp.vstack([final_params, macro_atom])
            
            c_warm_norm, r_aug, col_aug, sigma_aug = augmented_dict.T
            c_warm_stat = jnp.where(loss_code == 1, c_warm_norm * global_max, c_warm_norm)
            
            def eval_one_aug(ri, ci_col, si):
                return self._rbf_basis(x_grid, jnp.array([ri, ci_col]), si).flatten()
            
            A_aug = vmap(eval_one_aug)(r_aug, col_aug, sigma_aug).T
            weights_aug = (sigma_aug / self.ref_sigma)**self.gamma + 1e-6
            alpha_vec_stat_aug = eff_alpha_stat * weights_aug
            
            c_sparse_stat_aug = self._solve_ssn_unified(A_aug, patch_stat.flatten(), alpha_vec_stat_aug, loss_code, c_warm_stat)
            c_sparse_norm_aug = jnp.where(loss_code == 1, c_sparse_stat_aug / global_max, c_sparse_stat_aug)
            
            return jnp.stack([c_sparse_norm_aug, r_aug, col_aug, sigma_aug], axis=1)
        else:
            return final_params

    def compute_metrics(self, images_norm, peaks_list, global_max):
        B, H, W = images_norm.shape
        yy, xx = np.indices((H, W))
        x_grid = jnp.array([yy, xx])
        
        if self.show_steps:
            print("\n  [Metrics] Calculating goodness-of-fit...")

        max_k = max([len(p) for p in peaks_list] + [1])
        peaks_padded = np.zeros((B, max_k, 4), dtype=np.float32)
        counts_per_image = np.zeros(B, dtype=np.float32)
        
        for b in range(B):
            n = len(peaks_list[b])
            if n > 0:
                peaks_padded[b, :n, :] = peaks_list[b]
                if n < max_k:
                    peaks_padded[b, n:, 3] = 1.0 
            counts_per_image[b] = n

        if self.loss == 'poisson':
            loss_code = 1
        elif self.loss == 'gaussian':
            loss_code = 0
        else:
            raise ValueError("Unsupported loss. Not 'gaussian' or 'poisson'")

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

        nll_total, bic_total, deviance_total = 0.0, 0.0, 0.0
        
        for b in range(B):
            nll, bic, dev = process_one_image(
                jnp.array(peaks_padded[b]), 
                jnp.array(images_norm[b] * global_max), 
                jnp.array(counts_per_image[b])
            )
            nll_total += float(nll)
            bic_total += float(bic)
            deviance_total += float(dev)

        pixels_total = B * H * W
        params_total = float(np.sum(counts_per_image)) * 4
        dof = max(pixels_total - params_total, 1)
        dev_per_dof = deviance_total / dof
        
        if self.show_steps:
            print(f"  > Total NLL: {nll_total:.2e}")
            print(f"  > Total BIC: {bic_total:.2e}")
            print(f"  > Deviance/DoF: {dev_per_dof:.4f} (Target ~ 1.0)")
        
        return {"nll": nll_total, "bic": bic_total, "deviance_nu": dev_per_dof}

    def find_peaks_batch(self, images_batch):
        B, H, W = images_batch.shape
        
        # 1. Background subtraction
        medians = np.median(images_batch, axis=(1, 2), keepdims=True)
        images_bg_corr = np.maximum(images_batch - medians, 0)
        global_max = images_bg_corr.max() + 1e-9
        images_norm = images_bg_corr / global_max
        
        if self.show_steps:
            print(f"  > Pre-processing: Bg Subtracted. Global Max={global_max:.1f}")

        PAD_GLOBAL = 32
        img_jax_scout = jnp.array(images_norm)
        img_jax_scout_padded = jnp.pad(img_jax_scout, ((0,0), (PAD_GLOBAL, PAD_GLOBAL), (PAD_GLOBAL, PAD_GLOBAL)))

        # 2. Target Routing and Alpha scaling
        eff_alpha_norm = self.alpha / global_max
        eff_alpha_scout = eff_alpha_norm * 0.1 

        if self.loss == 'poisson':
            img_jax_sniper_padded = jnp.pad(jnp.array(images_batch), ((0,0), (PAD_GLOBAL, PAD_GLOBAL), (PAD_GLOBAL, PAD_GLOBAL)))
            eff_alpha_stat = self.alpha 
        else:
            img_jax_sniper_padded = img_jax_scout_padded
            eff_alpha_stat = eff_alpha_norm 

        # =====================================================================
        # PHASE 1: SCOUT (Generic Seed Discovery)
        # =====================================================================
        w_scout = self.base_window_size
        stride = w_scout // 2
        
        grid_h = list(range(0, H - w_scout + 1, stride))
        if grid_h[-1] + w_scout < H: grid_h.append(H - w_scout)
        grid_w = list(range(0, W - w_scout + 1, stride))
        if grid_w[-1] + w_scout < W: grid_w.append(W - w_scout)
        
        window_coords = [(b, r, c) for b in range(B) for r in grid_h for c in grid_w]
        window_coords_arr = np.array(window_coords, dtype=np.int32)
        total_scout_wins = len(window_coords)

        @jit
        def extract_scout_window(img, b_idx, r_idx, c_idx):
            r_pad = r_idx + PAD_GLOBAL
            c_pad = c_idx + PAD_GLOBAL
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (w_scout, w_scout))
            return vmap(slice_one)(b_idx, r_pad, c_pad)

        scout_solver = jit(vmap(lambda wg, ws: self._solve_dense(wg, ws, global_max, eff_alpha_scout, eff_alpha_norm, w_scout, w_scout, 5, 0, False)))
        
        scout_results = []
        scout_pbar = tqdm(range(0, total_scout_wins, self.chunk_size), desc="Scout Phase", disable=not self.show_steps)
        
        for i in scout_pbar:
            chunk = window_coords_arr[i:i+self.chunk_size]
            wins_geom = extract_scout_window(img_jax_scout_padded, chunk[:, 0], chunk[:, 1], chunk[:, 2])
            res = scout_solver(wins_geom, wins_geom)
            res.block_until_ready()
            
            global_res = np.array(res)
            
            valid_mask = global_res[:, :, 0] > 1e-9
            b_indices, peak_indices = np.where(valid_mask)
            if len(b_indices) > 0:
                valid_peaks = global_res[b_indices, peak_indices]
                valid_banks = chunk[b_indices, 0]
                valid_peaks[:, 1] += chunk[b_indices, 1]
                valid_peaks[:, 2] += chunk[b_indices, 2]
                
                peaks_with_bank = np.column_stack([valid_banks, valid_peaks])
                scout_results.append(peaks_with_bank)

        if not scout_results:
            return [np.empty((0, 4)) for _ in range(B)]

        all_candidates = np.vstack(scout_results)
        unique_candidates = []
        
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
            return [np.empty((0, 4)) for _ in range(B)]
            
        all_seeds = np.vstack(unique_candidates)
        total_seeds = len(all_seeds)

        # =====================================================================
        # PHASE 2: SYMMETRIC V-CYCLE (Multi-Atom Sniper + Halo + Macro-Merge)
        # =====================================================================
        P = self.refine_patch_size
        P_EXT = P + 2 * self.halo
       
        @jit
        def extract_patch_with_halo(img, centers):
            b_idx = centers[:, 0].astype(int)
            r_center = centers[:, 1].astype(int)
            c_center = centers[:, 2].astype(int)
            r_start = r_center + PAD_GLOBAL - (P // 2) - self.halo
            c_start = c_center + PAD_GLOBAL - (P // 2) - self.halo
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (P_EXT, P_EXT))
            return vmap(slice_one)(b_idx, r_start, c_start)

        loss_code_sniper = 1 if self.loss == 'poisson' else 0
        sniper_solver = jit(vmap(lambda wg, ws: self._solve_dense(wg, ws, global_max, eff_alpha_scout, eff_alpha_stat, P_EXT, P_EXT, self.max_local_peaks, loss_code_sniper, True)))
        refined_peaks_by_bank = [[] for _ in range(B)]
        
        sniper_pbar = tqdm(range(0, total_seeds, self.chunk_size), desc="Sniper V-Cycle", disable=not self.show_steps)
        
        for i in sniper_pbar:
            chunk = all_seeds[i:i+self.chunk_size]
            
            patches_geom = extract_patch_with_halo(img_jax_scout_padded, jnp.array(chunk))
            patches_stat = extract_patch_with_halo(img_jax_sniper_padded, jnp.array(chunk))
            
            res = sniper_solver(patches_geom, patches_stat) 
            res.block_until_ready()
            res_cpu = np.array(res)
            
            # Fast vectorized mapping
            valid_mask = res_cpu[:, :, 0] > 1e-9
            b_indices, peak_indices = np.where(valid_mask)
            
            if len(b_indices) > 0:
                valid_peaks = res_cpu[b_indices, peak_indices]
                valid_b_ids = chunk[b_indices, 0]
                valid_r_centers = chunk[b_indices, 1]
                valid_c_centers = chunk[b_indices, 2]
                
                global_rs = valid_r_centers - (P // 2) - self.halo + valid_peaks[:, 1]
                global_cs = valid_c_centers - (P // 2) - self.halo + valid_peaks[:, 2]
                
                MARGIN = 10
                in_bounds = (global_rs > MARGIN) & (global_rs < H - MARGIN) & \
                            (global_cs > MARGIN) & (global_cs < W - MARGIN)
                            
                # SSN implicitly enforces sparsity; we just drop boundary artifacts
                final_mask = (valid_peaks[:, 0] > 1e-5) & in_bounds
                
                for k in range(len(final_mask)):
                    if final_mask[k]:
                        b_id = int(valid_b_ids[k])
                        refined_peaks_by_bank[b_id].append(
                            np.array([valid_peaks[k, 0], global_rs[k], global_cs[k], valid_peaks[k, 3]])
                        )

        final_coords_output = []
        final_peaks_full = [] 
        
        for b in range(B):
            peaks = np.array(refined_peaks_by_bank[b])
            if len(peaks) > 0:
                peaks[:, 0] *= global_max 
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
                final_coords_output.append(unique_peaks) 
            else:
                final_peaks_full.append(np.empty((0, 4)))
                final_coords_output.append(np.empty((0, 4)))
        
        self.compute_metrics(img_jax_scout, final_peaks_full, global_max)
        
        return final_coords_output


class SparseLaueIntegrator(SparseRBFPeakFinder):
    """
    Physics-Informed Sniper.
    Takes predicted spot coordinates, extracts patches, and uses Volume-Penalized
    Sparse RBF to simultaneously refine sub-pixel position and integrate intensity.
    """
    def __init__(self, alpha=0.05, patch_size=15, min_sigma=0.5, max_sigma=5.0):
        super().__init__(
            alpha=alpha, min_sigma=min_sigma, max_sigma=max_sigma,
            loss='gaussian', show_steps=False
        )
        self.refine_patch_size = patch_size

    def integrate_reflections(self, images_batch, frames, rs, cs):
        B, H, W = images_batch.shape
        N_spots = len(frames)

        medians = np.median(images_batch, axis=(1, 2), keepdims=True)
        images_bg_corr = np.maximum(images_batch - medians, 0)
        global_max = images_bg_corr.max() + 1e-9
        images_norm = images_bg_corr / global_max
        img_jax = jnp.array(images_norm)

        PAD = self.refine_patch_size
        img_jax_padded = jnp.pad(img_jax, ((0,0), (PAD, PAD), (PAD, PAD)))

        P = self.refine_patch_size
        half_p = P // 2

        @jit
        def extract_patches(img, f_idx, r_idx, c_idx):
            r_start = r_idx.astype(int) + PAD - half_p
            c_start = c_idx.astype(int) + PAD - half_p
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (P, P))
            return vmap(slice_one)(f_idx, r_start, c_start)

        bounds = (float(P), float(P), self.min_sigma, self.max_sigma)
        yy, xx = jnp.indices((P, P))
        x_grid = jnp.array([yy, xx])

        @jit
        def solve_patches(patches):
            def process_patch(patch):
                c_init = jnp.max(patch) + 1e-6
                init_phys = jnp.array([[c_init, float(half_p), float(half_p), 2.0]])
                init_raw = self._to_unconstrained(init_phys, *bounds)

                res = jax.scipy.optimize.minimize(
                    fun=self._loss_fn,
                    x0=init_raw.ravel(),
                    args=(x_grid, patch, self.alpha, self.gamma, self.ref_sigma, bounds, 0), 
                    method='BFGS',
                    options={'maxiter': 10}
                )
                return self._to_physical(res.x, *bounds)[0] 
            return vmap(process_patch)(patches)

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
                res_cpu[:, 1] += rs[i:i+self.chunk_size] - half_p
                res_cpu[:, 2] += cs[i:i+self.chunk_size] - half_p
                res_cpu[:, 0] *= global_max

                refined_peaks.append(res_cpu)
                pbar.update(len(chunk_f))

        if len(refined_peaks) == 0:
            return np.empty((0, 4))

        return np.vstack(refined_peaks)

# =====================================================================
# [Below: Original GPU integration codes for integrate_peaks_rbf_ssn]
# =====================================================================
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Dict
from functools import partial
from tqdm import tqdm
from subhkl.instrument.detector import Detector

@partial(jax.jit, static_argnames=['N_shapes', 'max_peaks'])
def build_and_reduce_gpu(image: jnp.ndarray, padded_centers: jnp.ndarray, sigmas: jnp.ndarray,
                         gamma: float, N_shapes: int, max_peaks: int):
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

    Ht = (A.T @ A).astype(jnp.float32)
    At_y = (A.T @ y).astype(jnp.float32)
    y_sq_norm = jnp.float32(jnp.sum(y**2) / 2.0)

    variances = jnp.maximum(y, 1.0)
    W_var_diag = 1.0 / variances
    I_fisher = (A.T @ (W_var_diag[:, None] * A)).astype(jnp.float32)

    return Ht, At_y, y_sq_norm, I_fisher, weights, volumes

@partial(jax.jit, static_argnames=['N_c', 'max_iter'])
def solve_ssn_gpu(Ht: jnp.ndarray, At_y: jnp.ndarray, y_sq_norm: jnp.float32, 
                  N_c: int, alpha: float, max_iter: int = 50, tol: float = 1e-5):
    N = Ht.shape[0]
    alpha = jnp.float32(alpha)
    tol = jnp.float32(tol)

    def obj(u):
        quad = jnp.float32(0.5) * jnp.dot(u, Ht @ u) - jnp.dot(u, At_y) + y_sq_norm
        return quad + alpha * jnp.sum(u[:N_c])

    u_init = jnp.zeros(N, dtype=jnp.float32)
    q_init = At_y
    q_c = q_init[:N_c]
    q_c_clamped = jnp.minimum(q_c, jnp.float32(1.0 - 1e-14) * alpha)
    q_init = jnp.concatenate([q_c_clamped, q_init[N_c:]]).astype(jnp.float32)

    def cond_fun(state):
        step, _, _, Gq_norm, _ = state
        return (step < max_iter) & (Gq_norm > tol)

    def body_fun(state):
        step, q, u, Gq_norm, _ = state
        Gq = (q - u) + (Ht @ u) - At_y
        II_c = q[:N_c] > alpha
        II_b = jnp.ones(N - N_c, dtype=jnp.bool_)
        D = jnp.concatenate([II_c, II_b]).astype(jnp.float32)
        epsi = jnp.float32(1e-5)
        
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
            c_test = jnp.maximum(0.0, q_test[:N_c] - alpha)
            u_test = jnp.concatenate([c_test, q_test[N_c:]]).astype(jnp.float32)
            j_test = obj(u_test)
            return (bt_i + 1, tau, q_test, u_test, j_test, j_curr)
            
        q_init_test = q + dq
        c_init_test = jnp.maximum(0.0, q_init_test[:N_c] - alpha)
        u_init_test = jnp.concatenate([c_init_test, q_init_test[N_c:]]).astype(jnp.float32)
        
        bt_init = (jnp.int32(0), jnp.float32(1.0), q_init_test, u_init_test, obj(u_init_test), j_baseline)
        bt_final = jax.lax.while_loop(bt_cond, bt_body, bt_init)
        _, _, q_final, u_final, _, _ = bt_final
        
        return (step + 1, q_final, u_final, jnp.linalg.norm(Gq).astype(jnp.float32), D > 0.5)

    init_state = (jnp.int32(0), q_init, u_init, jnp.float32(1e9), jnp.zeros(N, dtype=jnp.bool_))
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, _, u_prime, _, active_set = final_state

    return u_prime, active_set

def evaluate_fisher_sigi_cpu(I_fisher_active: np.ndarray, c_active: np.ndarray, active_volumes: np.ndarray):
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
    y = jnp.array(y)
    Ht = (A.T @ A).astype(jnp.float32)
    At_y = (A.T @ y).astype(jnp.float32)
    y_sq_norm = jnp.float32(jnp.sum(y**2) / 2.0)
    u_prime, active_set = solve_ssn_gpu(Ht, At_y, y_sq_norm, N_c, alpha)
    return np.array(u_prime), np.array(active_set)

def integrate_peaks_rbf_ssn(peak_dict: Dict, peaks_obj, sigmas: List[float],
                            alpha: float, gamma: float, max_peaks: int, show_progress: bool,
                            all_R: np.ndarray = None, sample_offset: np.ndarray = None,
                            create_visualizations: bool = False):
    class RBFResult:
        def __init__(self):
            self.h, self.k, self.l = [], [], []
            self.intensity, self.sigma = [], []
            self.tt, self.az, self.wavelength = [], [], []
            self.run_id, self.bank, self.xyz = [], [], []

    res = RBFResult()
    sigmas_jnp = jnp.array(sigmas, dtype=jnp.float32)
    N_shapes = len(sigmas)
    N_c = max_peaks * N_shapes

    if sample_offset is None:
        sample_offset = np.zeros(3)

    for img_key, p_data in tqdm(peak_dict.items(), disable=not show_progress, desc="RBF Integration (Dense GPU)"):

        i_arr, j_arr, h_arr, k_arr, l_arr, wl_arr = p_data
        initial_peaks_count = len(i_arr)
        
        if initial_peaks_count == 0:
            continue
            
        hkl_sq = h_arr**2 + k_arr**2 + l_arr**2
        unique_peaks = {}
        
        for idx in range(initial_peaks_count):
            h, k, l = int(h_arr[idx]), int(k_arr[idx]), int(l_arr[idx])
            
            if h == 0 and k == 0 and l == 0:
                continue 
                
            g = np.gcd.reduce([abs(h), abs(k), abs(l)])
            fund_hkl = (h//g, k//g, l//g)
            
            if fund_hkl not in unique_peaks or hkl_sq[idx] < unique_peaks[fund_hkl]['hkl_sq']:
                unique_peaks[fund_hkl] = {'idx': idx, 'hkl_sq': hkl_sq[idx]}
                
        keep_indices = sorted([v['idx'] for v in unique_peaks.values()])
        p_data = [arr[keep_indices] for arr in p_data]
        i_arr, j_arr, h_arr, k_arr, l_arr, wl_arr = p_data
        
        peak_centers = np.column_stack([i_arr, j_arr])
        actual_peaks_count = len(peak_centers)

        if actual_peaks_count == 0:
            continue
        if actual_peaks_count > max_peaks:
            raise ValueError(f"Actual peaks ({actual_peaks_count}) exceeds max_peaks ({max_peaks}).")

        padded_centers = np.pad(peak_centers, ((0, max_peaks - actual_peaks_count), (0, 0)), constant_values=-10000.0)
        padded_centers_jnp = jnp.array(padded_centers, dtype=jnp.float32)

        image_raw = np.nan_to_num(peaks_obj.image.ims[img_key], nan=0.0, posinf=0.0, neginf=0.0)
        image_jnp = jnp.array(image_raw, dtype=jnp.float32)

        physical_bank = peaks_obj.image.bank_mapping.get(img_key, img_key)
        det = peaks_obj.get_detector(img_key)

        run_id = peaks_obj.image.get_run_id(img_key)
        if all_R is not None and all_R.ndim == 3:
            current_R_val = all_R[run_id] if run_id < len(all_R) else all_R[0]
        else:
            current_R_val = all_R

        s_lab = current_R_val @ sample_offset if current_R_val is not None else sample_offset
        bank_tt, bank_az = det.pixel_to_angles(p_data[0], p_data[1], sample_offset=s_lab)

        Ht, At_y, y_sq_norm, I_fisher, weights, volumes = build_and_reduce_gpu(
            image_jnp, padded_centers_jnp, sigmas_jnp, gamma, N_shapes, max_peaks
        )
        Ht = Ht.block_until_ready()

        u_prime, active_set = solve_ssn_gpu(Ht, At_y, y_sq_norm, N_c, alpha)
        u_prime = u_prime.block_until_ready()

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
            res.wavelength.append(p_data[5][p_idx])
            res.intensity.append(float(p_intensity))
            res.sigma.append(float(p_sigi))
            res.tt.append(float(bank_tt[p_idx]))
            res.az.append(float(bank_az[p_idx]))
            res.run_id.append(run_id)
            res.bank.append(physical_bank)

        if create_visualizations:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            import matplotlib.lines as mlines
            import matplotlib.cm as cm
            
            if plt.get_backend().lower() != "agg":
                plt.switch_backend("Agg")
                
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(1 + image_raw, norm="log", cmap="binary", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.02, 0.98, f"Bank {physical_bank} (Run {run_id})", 
                    transform=ax.transAxes, ha='left', va='top', 
                    fontsize=16, color='black',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=3))
            
            ax.scatter(peak_centers[:, 1], peak_centers[:, 0], marker='+', color='blue', s=60, label="Predicted")
            color_map = cm.rainbow(np.linspace(0, 1, max(2, N_shapes)))
            
            for p_idx in range(actual_peaks_count):
                start_idx = p_idx * N_shapes
                end_idx = start_idx + N_shapes
                
                active_shapes = active_set_cpu[start_idx:end_idx]
                if np.any(active_shapes):
                    cx = peak_centers[p_idx, 0]
                    cy = peak_centers[p_idx, 1]
                    
                    for s_idx, is_active in enumerate(active_shapes):
                        if is_active:
                            active_sig = sigmas[s_idx]
                            color = color_map[s_idx]
                            circle = Circle((cy, cx), 2.0 * active_sig, edgecolor=color, facecolor='none', lw=1.5)
                            ax.add_patch(circle)
            
            handles, labels = ax.get_legend_handles_labels()
            for s_idx in range(N_shapes):
                color = color_map[s_idx]
                active_sig = sigmas[s_idx]
                circle_key = mlines.Line2D([], [], color=color, marker='o', fillstyle='none', ls='', markersize=8)
                handles.append(circle_key)
                labels.append(rf'$2\sigma={2.0 * active_sig}$')
                
            ax.legend(
                handles=handles, labels=labels, loc='lower center', 
                ncol=len(handles), frameon=False, fontsize=12
            )
            out_name = f"rbf_viz_bank{physical_bank}_run{run_id}_img{img_key}.png"
            fig.savefig(out_name, bbox_inches="tight", dpi=150, pad_inches=0)
            plt.close(fig)

    return res
