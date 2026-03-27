import numpy as np
import scipy.special
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import jax.scipy.optimize
import jax.scipy.signal
import jax.scipy.special
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
    """
    def __init__(
        self,
        alpha: float = 0.05,            
        gamma: float = 2.0,             
        min_sigma: float = 0.5,         
        max_sigma: float = 8.0,
        max_peaks: int = 500,           
        chunk_size: int = 128,
        loss: str = 'gaussian',
        border_width: int = 0,
        num_sigmas: int = 32,
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
        self.border_width = border_width 
        self.show_steps = show_steps
        
        self.base_window_size = 64
        self.refine_patch_size = 15
        self.halo = 5  
        self.max_local_peaks = 5  
       
        self.candidate_sigmas = jnp.linspace(min_sigma, max_sigma, num_sigmas)

    @staticmethod
    def _rbf_basis(x_grid, y, sigma):
        """
        Analytic 2D pixel integral of the unnormalized continuous Gaussian (Error Function).
        This eliminates subpixel Grid Variance entirely.
        """
        sig_sq2 = sigma * jnp.sqrt(2.0) + 1e-6
        erf_r = jax.scipy.special.erf((x_grid[0] + 0.5 - y[0]) / sig_sq2) - jax.scipy.special.erf((x_grid[0] - 0.5 - y[0]) / sig_sq2)
        erf_c = jax.scipy.special.erf((x_grid[1] + 0.5 - y[1]) / sig_sq2) - jax.scipy.special.erf((x_grid[1] - 0.5 - y[1]) / sig_sq2)
        return (jnp.pi / 2.0) * (sigma**2) * erf_r * erf_c

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
        c_raw = jnp.where(c_safe > 20.0, c_safe, jnp.log(jnp.expm1(c_safe)))
        r_safe = jnp.clip(r / H, 1e-6, 1.0 - 1e-6)
        r_raw = jax.scipy.special.logit(r_safe)
        c_safe = jnp.clip(col / W, 1e-6, 1.0 - 1e-6)
        c_col_raw = jax.scipy.special.logit(c_safe)
        s_norm = (sigma - min_s) / (max_s - min_s)
        s_safe = jnp.clip(s_norm, 1e-6, 1.0 - 1e-6)
        s_raw = jax.scipy.special.logit(s_safe)
        return jnp.stack([c_raw, r_raw, c_col_raw, s_raw], axis=1).ravel()

    @staticmethod
    def _predict_batch_physical(params_phys, x_grid, mask=None):
        c, r, c_col, sigma = params_phys.T
        if mask is not None:
            c = c * mask
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
    def _loss_fn(params_flat, x_grid, target_raw, target_bg, eff_alpha, gamma, ref_s, bounds_tuple, opt_mask, loss_code):
        H, W, min_s, max_s = bounds_tuple
        params_phys = SparseRBFPeakFinder._to_physical(params_flat, H, W, min_s, max_s)
        recon = SparseRBFPeakFinder._predict_batch_physical(params_phys, x_grid, opt_mask) + target_bg
        
        # UNPENALIZED MLE GEOMETRY
        # Free from L1 drag, we can safely use the exact statistical metric.
        # This cures the L2 shape bias and expands sigma perfectly.
        recon_safe = jnp.maximum(recon, 1e-6)
        bg_med = jnp.maximum(jnp.median(target_bg), 1e-3)

        nll = jnp.where(
            loss_code == 1,
            bg_med * jnp.sum(recon_safe - target_raw * jnp.log(recon_safe)),
            0.5 * jnp.sum((recon - target_raw)**2)
        )

        intensities = jnp.abs(params_phys[:, 0]) * opt_mask
        sigmas = params_phys[:, 3]
        reg_weight = (sigmas / ref_s) ** gamma + 1e-6
        reg = eff_alpha * jnp.sum(intensities * reg_weight)

        return nll + reg

    @staticmethod
    @partial(jit, static_argnames=['max_iter', 'loss_type'])
    def _solve_ssn_unified(A, y, bg_flat, alpha_vec, loss_type, c_warm, max_iter=20):
        N_peaks = A.shape[1]
        N_params = N_peaks
        q_init = c_warm.astype(jnp.float32)

        # 1. CONSTANT SCALAR PRECONDITIONER FOR L1 (Maintains Proximal Geometry)
        bg_med = jnp.maximum(jnp.median(bg_flat), 1e-3).astype(jnp.float32)
        gamma_scale = jnp.where(loss_type == 1, bg_med, jnp.float32(1.0))
        scaled_alpha = gamma_scale * alpha_vec

        def get_loss_grad_hess(c):
            u = A @ c + bg_flat
            if loss_type == 1:
                u_safe = jnp.maximum(u, 1e-6)
                # PERFECT SCALING: nll, grad, and hess MUST all be multiplied by gamma_scale
                nll = gamma_scale * jnp.sum(u_safe - y * jnp.log(u_safe))
                grad = gamma_scale * (A.T @ (1.0 - y / u_safe))
                W_diag = 1.0 / jnp.maximum(u_safe, 1e-3)
                hess = gamma_scale * (A.T @ (W_diag[:, None] * A))
            else:
                nll = 0.5 * jnp.sum((u - y)**2)
                grad = A.T @ (u - y)
                hess = A.T @ A
            
            reg = jnp.sum(scaled_alpha * c)
            return nll + reg, grad, hess

        def cond_fn(state):
            step, _, _, dq_norm = state
            return (step < max_iter) & (dq_norm > 1e-3)

        def body_fn(state):
            step, q, c, _ = state
            obj_val, grad, hess = get_loss_grad_hess(c)

            Gq = (q - c) + grad

            D = (q > scaled_alpha).astype(jnp.float32)
            DP_mat = jnp.diag(D)
            I = jnp.eye(N_params, dtype=jnp.float32)

            DG = (I - DP_mat) + hess @ DP_mat + 1e-4 * I
            
            dq = jnp.linalg.solve(DG, -Gq).astype(jnp.float32)

            def bt_cond(bt_state):
                bt_i, tau, _, _, j_test, j_curr = bt_state
                is_valid = jnp.isfinite(j_test)
                return (bt_i < 8) & ((j_test > j_curr) | ~is_valid)

            def bt_body(bt_state):
                bt_i, tau, _, _, _, j_curr = bt_state
                tau = jnp.float32(tau * 0.5)
                q_test = (q + tau * dq).astype(jnp.float32)
                c_test = jnp.maximum(0.0, q_test - scaled_alpha).astype(jnp.float32)
                j_test, _, _ = get_loss_grad_hess(c_test)
                return (bt_i + 1, tau, q_test, c_test, j_test, j_curr)

            q_test = (q + dq).astype(jnp.float32)
            c_test = jnp.maximum(0.0, q_test - scaled_alpha).astype(jnp.float32)
            j_test, _, _ = get_loss_grad_hess(c_test)

            bt_init = (0, jnp.float32(1.0), q_test, c_test, j_test, obj_val)
            bt_final = lax.while_loop(bt_cond, bt_body, bt_init)
            _, _, q_final, c_final, _, _ = bt_final

            return (step + 1, q_final.astype(jnp.float32), c_final.astype(jnp.float32), jnp.linalg.norm(dq).astype(jnp.float32))

        init_state = (0, q_init.astype(jnp.float32), c_warm.astype(jnp.float32), jnp.float32(1e9))
        final_state = lax.while_loop(cond_fn, body_fn, init_state)
        _, _, c_l1, _ = final_state

        # DEBIASING PHASE
        active_mask = c_l1 > 1e-5

        def debias_cond(state):
            step, _, actual_step_norm = state
            return (step < 100) & (actual_step_norm > 1e-4)

        def debias_body(state):
            step, c, _ = state
            _, grad, hess = get_loss_grad_hess(c)

            H_diag = jnp.diag(hess)
            eta = 1.0 / jnp.maximum(H_diag, 1e-6)

            I = jnp.eye(N_params, dtype=jnp.float32)
            D_mat = jnp.diag(active_mask.astype(jnp.float32))

            F_c = (1.0 - active_mask) * c + active_mask * (eta * grad)
            DG = (I - D_mat) + (eta[:, None] * hess) @ D_mat + 1e-4 * I 

            dc = jnp.linalg.solve(DG, -F_c).astype(jnp.float32)

            tau = jnp.where(loss_type == 1, jnp.float32(0.8), jnp.float32(1.0))
            
            c_new_raw = c + tau * dc * active_mask
            c_new = jnp.maximum(0.0, c_new_raw) * active_mask

            actual_step = c_new - c
            return (step + 1, c_new.astype(jnp.float32), jnp.linalg.norm(actual_step).astype(jnp.float32))

        debias_state = lax.while_loop(debias_cond, debias_body, (0, c_l1.astype(jnp.float32), jnp.float32(1e9)))
        _, c_final, _ = debias_state

        return c_final.astype(jnp.float32)

    @partial(jit, static_argnames=['self', 'H', 'W', 'max_peaks_local', 'loss_code', 'do_merge'])
    def _solve_dense(self, patch_stat, patch_bg, alpha_z_score, H, W, max_peaks_local, loss_code, do_merge):
        
        local_bg_med = jnp.maximum(jnp.median(patch_bg), 1e-3)
        local_noise_floor = jnp.sqrt(local_bg_med)
       
        # Absolute Photons (For Scout and BFGS)
        eff_alpha = alpha_z_score * local_noise_floor

        # The continuous Newton solvers require precise statistical gradient mapping!
        if loss_code == 1:
            eff_alpha_stat = alpha_z_score / local_noise_floor # Poisson Mapped
        else:
            eff_alpha_stat = alpha_z_score * local_noise_floor # L2 Mapped
        
        bounds = (float(H), float(W), self.min_sigma, self.max_sigma)
        yy, xx = jnp.indices((H, W))
        x_grid = jnp.array([yy, xx])
        
        # Max K radius defines the valid correlation boundaries
        max_k_rad = int(3.0 * self.max_sigma)
        k_grid = jnp.arange(-max_k_rad, max_k_rad + 1)
        ky, kx = jnp.meshgrid(k_grid, k_grid, indexing='ij')
        
        init_params = jnp.zeros((max_peaks_local, 4))
        init_active = jnp.zeros(max_peaks_local, dtype=bool)
        init_state = (init_params, init_active, 0)

        def step_fn(state, _):
            params, active_mask, idx = state
            recon = self._predict_batch_physical(params, x_grid, active_mask)
            
            def check_sigma(s):
                sig_sq2 = s * jnp.sqrt(2.0) + 1e-6
                erf_y = jax.scipy.special.erf((ky + 0.5) / sig_sq2) - jax.scipy.special.erf((ky - 0.5) / sig_sq2)
                erf_x = jax.scipy.special.erf((kx + 0.5) / sig_sq2) - jax.scipy.special.erf((kx - 0.5) / sig_sq2)
                kernel_raw = (jnp.pi / 2.0) * (s**2) * erf_y * erf_x

                recon_total = jnp.maximum(recon + patch_bg, 1e-3)
                raw_grad = patch_stat - recon_total

                # VALID CORRELATION: Kernel never touches a boundary, eliminating padding artifacts
                dual_var = jax.scipy.signal.correlate2d(raw_grad, kernel_raw, mode='valid')

                flat_idx = jnp.argmax(dual_var)
                r_valid, c_valid = jnp.unravel_index(flat_idx, dual_var.shape)

                # Map from the smaller valid core back to the full extended patch
                r_idx = r_valid + max_k_rad
                c_idx = c_valid + max_k_rad

                kernel_sq_norm = jnp.sum(kernel_raw ** 2)

                # normalized Pearson cross-correlation (scale discriminator)
                # Peaks strictly at the true sigma, immune to amplitude bias
                scale_score = dual_var[r_valid, c_valid] / jnp.sqrt(kernel_sq_norm)

                # Extract the raw amplitude for the threshold check
                c_matched = dual_var[r_valid, c_valid] / kernel_sq_norm
                c_init = jnp.maximum(c_matched, 0.0)

                return scale_score, jnp.array([c_init, r_idx, c_idx, s])

            vals, candidates = vmap(check_sigma)(self.candidate_sigmas)
            best_idx = jnp.argmax(vals)
            new_peak = candidates[best_idx]
            
            # Extract the actual physical amplitude found in photons
            c_best = new_peak[0] 
            
            # Threshold the physical amplitude, NOT the scale-amplified SNR score!
            # This instantly annihilates the 32 background ghost bumps.
            is_strong = c_best > eff_alpha
            
            dummy_peak = jnp.array([0.0, 0.0, 0.0, 1.0])
            new_peak = jnp.where(is_strong, new_peak, dummy_peak)
            
            params = jnp_update_set(params, idx, new_peak)
            active_mask = jnp_update_set(active_mask, idx, is_strong)

            def run_opt(operand):
                p, a_mask = operand
                
                # 1. PERMANENTLY LOCK SIGMA TO THE SCOUT'S NCC ESTIMATE
                locked_sigmas = p[:, 3]
                
                # 2. Define 3-Parameter Mappings (c, r, col)
                def to_unc3(p_phys):
                    c, r, col = p_phys[:, 0], p_phys[:, 1], p_phys[:, 2]
                    c_safe = jnp.maximum(c, 1e-9)
                    c_raw = jnp.where(c_safe > 20.0, c_safe, jnp.log(jnp.expm1(c_safe)))
                    r_raw = jax.scipy.special.logit(jnp.clip(r / H, 1e-6, 1.0 - 1e-6))
                    col_raw = jax.scipy.special.logit(jnp.clip(col / W, 1e-6, 1.0 - 1e-6))
                    return jnp.stack([c_raw, r_raw, col_raw], axis=1).ravel()
                
                def to_phys3(p3_raw):
                    p3_reshaped = p3_raw.reshape((-1, 3))
                    c_raw, r_raw, col_raw = p3_reshaped.T
                    c = jax.nn.softplus(c_raw)
                    r = jax.nn.sigmoid(r_raw) * H
                    col = jax.nn.sigmoid(col_raw) * W
                    # Re-inject the pristine locked sigmas
                    return jnp.stack([c, r, col, locked_sigmas], axis=1)

                p_raw = to_unc3(p)
                
                # 3. Pure L2 Unpenalized Geometry Loss (Immune to Neyman bias)
                def loss_fn_locked(p3_raw_flat, x_grid, target_raw, target_bg, opt_mask):
                    p_phys = to_phys3(p3_raw_flat)
                    recon = self._predict_batch_physical(p_phys, x_grid, opt_mask) + target_bg
                    return 0.5 * jnp.sum((recon - target_raw)**2)
                
                # 4. Run BFGS Strictly to find Subpixel (r, c)
                res = jax.scipy.optimize.minimize(
                    fun=loss_fn_locked,
                    x0=p_raw,
                    args=(x_grid, patch_stat, patch_bg, a_mask),
                    method='BFGS',
                    options={'maxiter': 20}
                )
                
                p_refined = to_phys3(res.x)
                c_phys, r, col, sigma = p_refined.T
                
                def eval_one(ri, ci_col, si):
                    return self._rbf_basis(x_grid, jnp.array([ri, ci_col]), si).flatten()
                
                A = vmap(eval_one)(r, col, sigma).T
                A_masked = A * a_mask
                
                weights = (sigma / self.ref_sigma)**self.gamma + 1e-6
                alpha_vec_stat = eff_alpha_stat * weights
                
                # STRICT MASKING: Kill ghost amplitudes
                c_phys_masked = c_phys * a_mask
                
                # 5. Hand the locked shapes to the true Poisson Integrator
                c_sparse_stat = self._solve_ssn_unified(A_masked, patch_stat.flatten(), patch_bg.flatten(), alpha_vec_stat, loss_code, c_phys_masked)
                
                c_sparse_norm = c_sparse_stat * a_mask
                return jnp.stack([c_sparse_norm, r, col, sigma], axis=1)

            def skip_opt(operand):
                p, _ = operand
                return p

            params = lax.cond(is_strong, run_opt, skip_opt, (params, active_mask))

            return (params, active_mask, idx + 1), None

        final_state, _ = lax.scan(step_fn, init_state, None, length=max_peaks_local)
        final_params, final_active, _ = final_state
        
        if do_merge:
            # intensity-weighted averaging of peak locations to subpixel accuracy
            c, r, col, sigma = final_params.T
            active_mask = final_active & (c > 1e-9)
            num_active = jnp.sum(active_mask)
            
            c_active = jnp.where(active_mask, c, 0.0)
            total_amp = jnp.sum(c_active) + 1e-12
            
            com_r = jnp.sum(c_active * r) / total_amp
            com_c = jnp.sum(c_active * col) / total_amp
            var_r = jnp.sum(c_active * (r - com_r)**2) / total_amp
            var_c = jnp.sum(c_active * (col - com_c)**2) / total_amp
            
            mean_sigma = jnp.sum(jnp.where(active_mask, sigma, 0.0)) / jnp.maximum(num_active, 1)
            macro_sigma = jnp.sqrt(var_r + var_c) + mean_sigma
            
            dummy_atom = jnp.array([0.0, -100.0, -100.0, 1.0])
            macro_atom = jnp.stack([total_amp, com_r, com_c, macro_sigma])
            macro_atom = jnp.where(num_active > 1, macro_atom, dummy_atom)
            
            augmented_dict = jnp.vstack([final_params, macro_atom])
            aug_mask = jnp.append(active_mask, num_active > 1)
            
            c_warm_raw, r_aug, col_aug, sigma_aug = augmented_dict.T
            
            def eval_one_aug(ri, ci_col, si):
                return self._rbf_basis(x_grid, jnp.array([ri, ci_col]), si).flatten()
            
            A_aug = vmap(eval_one_aug)(r_aug, col_aug, sigma_aug).T
            A_aug_masked = A_aug * aug_mask
            
            weights_aug = (sigma_aug / self.ref_sigma)**self.gamma + 1e-6
            alpha_vec_stat_aug = eff_alpha_stat * weights_aug
            
            c_sparse_stat_aug = self._solve_ssn_unified(A_aug_masked, patch_stat.flatten(), patch_bg.flatten(), alpha_vec_stat_aug, loss_code, c_warm_raw)
            
            return jnp.stack([c_sparse_stat_aug * aug_mask, r_aug, col_aug, sigma_aug], axis=1)
        else:
            return final_params

    def compute_metrics(self, images_raw, bg_map, peaks_list, global_max):
        B, H, W = images_raw.shape
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
            raise ValueError("Unsupported loss")

        @jit
        def process_one_image(peaks, target_raw, median_val, k_val):
            recon_peaks = self._predict_batch_scan(peaks, x_grid) 
            recon_total = jnp.maximum(recon_peaks + median_val, 1e-9)
            
            if loss_code == 1: 
                # 1. Exact Poisson NLL using xlogy
                nll = jnp.sum(recon_total - jax.scipy.special.xlogy(target_raw, recon_total))
                # 2. Exact Poisson Deviance (no more 1e-9 target clamping)
                term = jax.scipy.special.xlogy(target_raw, target_raw / recon_total) - (target_raw - recon_total)
                dev = 2 * jnp.sum(term)
            else: 
                diff = recon_total - target_raw
                nll = 0.5 * jnp.sum(diff**2)
                dev = jnp.sum((diff**2) / recon_total)
            
            n_pix = target_raw.size
            n_params = k_val * 4
            bic = n_params * jnp.log(n_pix) + 2 * nll
            return nll, bic, dev

        nll_total, bic_total, deviance_total = 0.0, 0.0, 0.0
        
        for b in range(B):
            nll, bic, dev = process_one_image(
                jnp.array(peaks_padded[b]), 
                jnp.array(images_raw[b]), 
                jnp.array(bg_map[b]),
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
            target_str = "(Target ~ 1.0)" if loss_code == 1 else "(MSE/Variance of noise)"
            print(f"  > Total NLL: {nll_total:.2e}")
            print(f"  > Total BIC: {bic_total:.2e}")
            print(f"  > Deviance/DoF: {dev_per_dof:.4f} {target_str}")

        return {"nll": nll_total, "bic": bic_total, "deviance_nu": dev_per_dof}

    def find_peaks_batch(self, images_batch):
        B, H, W = images_batch.shape
        
        # alpha is strictly interpreted as the Z-score (SNR) threshold
        alpha_z_score = self.alpha

        from scipy.ndimage import median_filter, gaussian_filter
        bg_med = median_filter(images_batch, size=(1, 15, 15))
        bg_map = gaussian_filter(bg_med, sigma=(0, 3.0, 3.0))
        bg_map = np.clip(bg_map, 1e-3, None)
        self._last_bg_map = bg_map
        
        valid_bg = bg_map[bg_map > 1e-2]
        if valid_bg.size == 0:
            median_bg_level = 1.0
        else:
            median_bg_level = float(np.median(valid_bg))
            
        if np.isnan(median_bg_level) or median_bg_level <= 0:
            median_bg_level = 1.0
            
        poisson_noise_floor = np.maximum(np.sqrt(median_bg_level), 1.0)

        if self.show_steps:
            print(f"  > Pre-processing: Morphological Bg Evaluated.")
            print(f"  > Autotuning: Median BG={median_bg_level:.1f}, Noise Floor=~{poisson_noise_floor:.1f}")

        img_jax_stat_np = np.copy(images_batch)
        if self.border_width > 0:
            bw = self.border_width
            valid_interior = np.zeros((H, W), dtype=bool)
            valid_interior[bw:-bw, bw:-bw] = True
            valid_mask_batch = np.broadcast_to(valid_interior, (B, H, W))
            img_jax_stat_np = np.where(valid_mask_batch, img_jax_stat_np, bg_map)
            
        # NO GLOBAL PADDING. We strictly use the raw physical bounds.
        img_jax_stat = jnp.array(img_jax_stat_np)
        img_jax_bg = jnp.array(bg_map)

        loss_code_sniper = 1 if self.loss == 'poisson' else 0

        # Dynamically size the valid boundary exclusion zone
        max_k_rad = int(3.0 * self.max_sigma)
        
        w_scout_core = self.base_window_size
        w_ext = w_scout_core + 2 * max_k_rad
        stride = w_scout_core // 2

        min_required_patch = 2 * max_k_rad + 1
        P_core = max(self.refine_patch_size, min_required_patch)
        P_EXT = P_core + 2 * max_k_rad

        pad_size = P_core // 2 + max_k_rad

        # pad symmetrically
        img_jax_stat = jnp.pad(img_jax_stat, ((0,0), (pad_size, pad_size), (pad_size, pad_size)), mode='symmetric')
        img_jax_bg = jnp.pad(img_jax_bg, ((0,0), (pad_size, pad_size), (pad_size, pad_size)), mode='symmetric')

        start_h, end_h = pad_size, pad_size + H
        start_w, end_w = pad_size, pad_size + W
        
        grid_h = list(range(start_h, end_h - w_scout_core + 1, stride))
        if not grid_h or grid_h[-1] + w_scout_core < end_h: 
            grid_h.append(max(start_h, end_h - w_scout_core))
            
        grid_w = list(range(start_w, end_w - w_scout_core + 1, stride))
        if not grid_w or grid_w[-1] + w_scout_core < end_w: 
            grid_w.append(max(start_w, end_w - w_scout_core))
        
        window_coords = [(b, r, c) for b in range(B) for r in grid_h for c in grid_w]
        window_coords_arr = np.array(window_coords, dtype=np.int32)
        total_scout_wins = len(window_coords)

        @jit
        def extract_scout_window(img, b_idx, r_idx, c_idx):
            # r_idx, c_idx are the top-left of the CORE window.
            # Shift back to extract the expanded valid halo (Guaranteed to be >= 0 by grid bounds)
            r_start = r_idx - max_k_rad
            c_start = c_idx - max_k_rad
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (w_ext, w_ext))
            return vmap(slice_one)(b_idx, r_start, c_start)

        scout_solver = jit(vmap(lambda ws, wb: self._solve_dense(ws, wb, alpha_z_score, w_ext, w_ext, 5, loss_code_sniper, False)))
        
        scout_results = []
        scout_pbar = tqdm(range(0, total_scout_wins, self.chunk_size), desc="Scout Phase", disable=not self.show_steps)
        
        for i in scout_pbar:
            chunk = window_coords_arr[i:i+self.chunk_size]
            wins_stat = extract_scout_window(img_jax_stat, chunk[:, 0], chunk[:, 1], chunk[:, 2])
            wins_bg = extract_scout_window(img_jax_bg, chunk[:, 0], chunk[:, 1], chunk[:, 2])
            res = scout_solver(wins_stat, wins_bg)
            res.block_until_ready()
            
            global_res = np.array(res)
            
            valid_mask = global_res[:, :, 0] > 1e-9
            b_indices, peak_indices = np.where(valid_mask)
            if len(b_indices) > 0:
                valid_peaks = global_res[b_indices, peak_indices]
                valid_banks = chunk[b_indices, 0]
                
                # Map the returned w_ext coordinates perfectly back to the global image grid
                valid_peaks[:, 1] += chunk[b_indices, 1] - max_k_rad
                valid_peaks[:, 2] += chunk[b_indices, 2] - max_k_rad
                
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
                radius = 1.5 
                for i in range(len(cands_sorted)):
                    if keep[i]:
                        neighbors = np.where(dists[i] < radius)[0]
                        neighbors = neighbors[neighbors > i] 
                        keep[neighbors] = False
            valid_seeds = cands_sorted[keep]

            if len(valid_seeds) > 0:
                bank_col = np.full((len(valid_seeds), 1), b)
                unique_candidates.append(np.hstack([bank_col, valid_seeds]))

        if not unique_candidates:
            return [np.empty((0, 4)) for _ in range(B)]
            
        all_seeds = np.vstack(unique_candidates)
        total_seeds = len(all_seeds)

        @jit
        def extract_patch_with_halo(img, centers):
            b_idx = centers[:, 0].astype(int)
            r_center = centers[:, 1].astype(int)
            c_center = centers[:, 2].astype(int)
            
            r_start = r_center - pad_size
            c_start = c_center - pad_size
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (P_EXT, P_EXT))
            return vmap(slice_one)(b_idx, r_start, c_start)

        sniper_solver = jit(vmap(lambda ws, wb: self._solve_dense(ws, wb, alpha_z_score, P_EXT, P_EXT, self.max_local_peaks, loss_code_sniper, True)))
        refined_peaks_by_bank = [[] for _ in range(B)]
        
        sniper_pbar = tqdm(range(0, total_seeds, self.chunk_size), desc="Sniper V-Cycle", disable=not self.show_steps)
        
        for i in sniper_pbar:
            chunk = all_seeds[i:i+self.chunk_size]
            
            patches_stat = extract_patch_with_halo(img_jax_stat, jnp.array(chunk))
            patches_bg = extract_patch_with_halo(img_jax_bg, jnp.array(chunk))
            
            res = sniper_solver(patches_stat, patches_bg) 
            res.block_until_ready()
            res_cpu = np.array(res)
            
            valid_mask = res_cpu[:, :, 0] > 1e-9
            b_indices, peak_indices = np.where(valid_mask)
            
            if len(b_indices) > 0:
                valid_peaks = res_cpu[b_indices, peak_indices]
                valid_b_ids = chunk[b_indices, 0]
                valid_r_centers = chunk[b_indices, 1]
                valid_c_centers = chunk[b_indices, 2]
                
                # Recover PADDED coordinates
                global_rs_padded = valid_r_centers.astype(int) - pad_size + valid_peaks[:, 1]
                global_cs_padded = valid_c_centers.astype(int) - pad_size + valid_peaks[:, 2]
                
                # SHIFT BACK TO PHYSICAL SENSOR COORDINATES
                global_rs = global_rs_padded - pad_size
                global_cs = global_cs_padded - pad_size 

                MARGIN = max(3, self.border_width)
                in_bounds = (global_rs >= MARGIN) & (global_rs < H - MARGIN) & \
                            (global_cs >= MARGIN) & (global_cs < W - MARGIN)
                            
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
                order = np.argsort(peaks[:, 0])[::-1]
                peaks_sorted = peaks[order]
                keep = np.ones(len(peaks_sorted), dtype=bool)
                coords = peaks_sorted[:, 1:3]
                
                if len(coords) > 1:
                    dists = squareform(pdist(coords))
                    np.fill_diagonal(dists, 9999.0)
                    r = 1.5 
                    for i in range(len(coords)):
                        if keep[i]:
                            neighbors = np.where(dists[i] < r)[0]
                            neighbors = neighbors[neighbors > i]
                            keep[neighbors] = False
                            
                unique_peaks = peaks_sorted[keep]
                final_peaks_full.append(unique_peaks)
                final_coords_output.append(unique_peaks) 
            else:
                final_peaks_full.append(np.empty((0, 4)))
                final_coords_output.append(np.empty((0, 4)))
        
        self.compute_metrics(images_batch, bg_map, final_peaks_full, 1.0)
        
        return final_coords_output

class SparseLaueIntegrator(SparseRBFPeakFinder):
    """
    Physics-Informed Sniper.
    Takes predicted spot coordinates, extracts patches, and uses Volume-Penalized
    Sparse RBF to accurately integrate intensity using the Preconditioned SSN Engine.
    """
    def __init__(self, alpha=0.05, min_sigma=0.5, max_sigma=5.0, gamma=2.0, loss='poisson', num_sigmas=64):
        super().__init__(
            alpha=alpha, gamma=gamma, min_sigma=min_sigma, max_sigma=max_sigma,
            loss=loss, border_width=0, show_steps=False, num_sigmas=num_sigmas
        )

    def integrate_reflections(self, images_batch, frames, rs, cs):
        B, H, W = images_batch.shape
        N_spots = len(frames)

        P = self.refine_patch_size
        half_p = P // 2
        PAD = P
        
        K_NEIGHBORS = min(4, N_spots) if N_spots > 0 else 1

        img_jax_padded = jnp.pad(jnp.array(images_batch), ((0,0), (PAD, PAD), (PAD, PAD)), mode='reflect')

        bounds = (float(P), float(P), self.min_sigma, self.max_sigma)
        yy, xx = jnp.indices((P, P))
        x_grid = jnp.array([yy, xx])
        loss_code = 1 if self.loss == 'poisson' else 0

        @jit
        def extract_patches(img, f_idx, r_idx, c_idx):
            r_start = jnp.clip(jnp.int32(jnp.round(r_idx)) - half_p, 0, img.shape[1] - P)
            c_start = jnp.clip(jnp.int32(jnp.round(c_idx)) - half_p, 0, img.shape[2] - P)
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (P, P))
            return vmap(slice_one)(f_idx, r_start, c_start), r_start, c_start

        @jit
        def solve_patches(patches, rs_global_chunk, cs_global_chunk, r_starts, c_starts, all_rs_jnp, all_cs_jnp):
            N_shapes = len(self.candidate_sigmas)
            
            def process_patch(patch, r_global, c_global, r_start, c_start):
                bg_med = jnp.maximum(jnp.median(patch), 1e-3)
                patch_bg = jnp.full_like(patch, bg_med)
                noise_floor = jnp.sqrt(bg_med)
                
                eff_alpha_stat = jnp.where(loss_code == 1, self.alpha / noise_floor, self.alpha * noise_floor)
                
                dists = (all_rs_jnp - r_global)**2 + (all_cs_jnp - c_global)**2
                _, nbr_idxs = jax.lax.top_k(-dists, K_NEIGHBORS)
                
                nbr_rs = all_rs_jnp[nbr_idxs]
                nbr_cs = all_cs_jnp[nbr_idxs]
                
                local_rs = nbr_rs - r_start
                local_cs = nbr_cs - c_start
                
                def eval_neighbor(nr, nc):
                    def eval_shape(si):
                        return self._rbf_basis(x_grid, jnp.array([nr, nc]), si).flatten()
                    return vmap(eval_shape)(self.candidate_sigmas).T
                
                A_all = vmap(eval_neighbor)(local_rs, local_cs)
                A_joint = jnp.transpose(A_all, (1, 0, 2)).reshape(P*P, K_NEIGHBORS * N_shapes)
                
                y_sub = (patch - patch_bg).flatten()
                
                pixel_dists_k = (yy.flatten()[:, None] - local_rs[None, :])**2 + (xx.flatten()[:, None] - local_cs[None, :])**2
                closest_k = jnp.argmin(pixel_dists_k, axis=1)
                pixel_masks = jax.nn.one_hot(closest_k, K_NEIGHBORS) 
                
                A_k = A_joint.reshape(P*P, K_NEIGHBORS, N_shapes)
                
                y_sub_k = y_sub[:, None] * pixel_masks
                A_k_masked = A_k * pixel_masks[:, :, None]
                
                A_norms = jnp.sqrt(jnp.maximum(jnp.sum(A_k_masked**2, axis=0), 1e-6))
                ncc_k = jnp.sum(A_k_masked * y_sub_k[:, :, None], axis=0) / A_norms
                c_warm_proj_k = jnp.maximum(0.0, jnp.sum(A_k_masked * y_sub_k[:, :, None], axis=0) / jnp.maximum(jnp.sum(A_k_masked**2, axis=0), 1e-6)).astype(jnp.float32)
                
                best_idx_k = jnp.argmax(ncc_k, axis=1) # [K]
                
                # --- L1 GREEDY SWAP FIX ---
                # Slice out strictly the 1 best shape per neighbor using advanced JAX indexing
                indices = jnp.arange(K_NEIGHBORS)
                A_best = A_k[:, indices, best_idx_k]       # [961, K]
                c_warm_best = c_warm_proj_k[indices, best_idx_k] # [K]
                best_sigmas = self.candidate_sigmas[best_idx_k]  # [K]
                
                weights_best = (best_sigmas / self.ref_sigma)**self.gamma + 1e-6
                alpha_vec_best = eff_alpha_stat * weights_best
                
                # JOINT SSN SOLVE (With strict K columns)
                c_final_joint = self._solve_ssn_unified(
                    A_best, patch.flatten(), patch_bg.flatten(), alpha_vec_best, loss_code, c_warm_best, 20
                )
                
                # Target peak is guaranteed to be index 0 of the neighbors
                c_final_target = c_final_joint[0]
                best_sig_target = best_sigmas[0]
                
                volumes = jnp.float32(2.0 * jnp.pi) * (best_sig_target**2)
                intensity = c_final_target * volumes
                
                return jnp.array([intensity, local_rs[0], local_cs[0], jnp.where(intensity > 0, best_sig_target, 0.0)])
                
            return vmap(process_patch)(patches, rs_global_chunk, cs_global_chunk, r_starts, c_starts)

        refined_peaks = []
        rs_padded = np.array(rs) + PAD
        cs_padded = np.array(cs) + PAD
        
        PAD_N = max(N_spots, 4)
        rs_full = np.pad(rs_padded, (0, PAD_N - N_spots), constant_values=-10000.0)
        cs_full = np.pad(cs_padded, (0, PAD_N - N_spots), constant_values=-10000.0)
        all_rs_jnp = jnp.array(rs_full, dtype=jnp.float32)
        all_cs_jnp = jnp.array(cs_full, dtype=jnp.float32)
        
        from tqdm import tqdm
        with tqdm(total=N_spots, desc="Sparse Laue Integration", disable=not self.show_steps) as pbar:
            for i in range(0, N_spots, self.chunk_size):
                chunk_f = jnp.array(frames[i:i+self.chunk_size])
                chunk_r = jnp.array(rs_padded[i:i+self.chunk_size])
                chunk_c = jnp.array(cs_padded[i:i+self.chunk_size])

                patches, r_starts, c_starts = extract_patches(img_jax_padded, chunk_f, chunk_r, chunk_c)

                res = solve_patches(patches, chunk_r, chunk_c, r_starts, c_starts, all_rs_jnp, all_cs_jnp)
                res.block_until_ready()

                res_cpu = np.array(res)
                res_cpu[:, 1] = res_cpu[:, 1] + r_starts - PAD
                res_cpu[:, 2] = res_cpu[:, 2] + c_starts - PAD

                refined_peaks.append(res_cpu)
                pbar.update(len(chunk_f))

        if len(refined_peaks) == 0:
            return np.empty((0, 4))

        return np.vstack(refined_peaks)

# =====================================================================
# API WRAPPER FOR BACKWARD COMPATIBILITY
# =====================================================================
from typing import Dict, List
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
    if sample_offset is None:
        sample_offset = np.zeros(3)

    integrator = SparseLaueIntegrator(
        alpha=alpha, min_sigma=min(sigmas), max_sigma=max(sigmas), gamma=gamma, loss='poisson'
    )
    integrator.candidate_sigmas = jnp.array(sigmas, dtype=jnp.float32)
    integrator.show_steps = show_progress

    from tqdm import tqdm
    for img_key, p_data in tqdm(peak_dict.items(), disable=not show_progress, desc="RBF Integration Wrapper"):
        i_arr, j_arr, h_arr, k_arr, l_arr, wl_arr = p_data
        if len(i_arr) == 0: continue

        hkl_sq = h_arr**2 + k_arr**2 + l_arr**2
        unique_peaks = {}
        for idx in range(len(i_arr)):
            h, k, l = int(h_arr[idx]), int(k_arr[idx]), int(l_arr[idx])

            if h == 0 and k == 0 and l == 0:
                fund_hkl = (0, 0, 0)
            else:
                g = np.gcd.reduce([abs(h), abs(k), abs(l)])
                fund_hkl = (h//g, k//g, l//g)

            # --- SPATIAL DEDUPLICATION FIX ---
            # Append an approximate detector bin (5 pixels) to the dictionary key.
            # This ensures spatially separated harmonics (or test arrays) are preserved!
            loc_key = (int(np.round(i_arr[idx]/5.0)), int(np.round(j_arr[idx]/5.0)))
            unique_key = (fund_hkl, loc_key)

            if unique_key not in unique_peaks or hkl_sq[idx] < unique_peaks[unique_key]['hkl_sq']:
                unique_peaks[unique_key] = {'idx': idx, 'hkl_sq': hkl_sq[idx]}

        keep_indices = sorted([v['idx'] for v in unique_peaks.values()])
        p_data = [arr[keep_indices] for arr in p_data]
        i_arr, j_arr, h_arr, k_arr, l_arr, wl_arr = p_data

        image_raw = np.nan_to_num(peaks_obj.image.ims[img_key], nan=0.0, posinf=0.0, neginf=0.0)
        images_batch = image_raw[np.newaxis, ...]
        frames = np.zeros(len(i_arr), dtype=int)

        integrated_results = integrator.integrate_reflections(images_batch, frames, i_arr, j_arr)

        physical_bank = peaks_obj.image.bank_mapping.get(img_key, img_key)
        det = peaks_obj.get_detector(img_key)
        run_id = peaks_obj.image.get_run_id(img_key)

        if all_R is not None and all_R.ndim == 3:
            current_R_val = all_R[run_id] if run_id < len(all_R) else all_R[0]
        else:
            current_R_val = all_R

        s_lab = current_R_val @ sample_offset if current_R_val is not None else sample_offset
        bank_tt, bank_az = det.pixel_to_angles(p_data[0], p_data[1], sample_offset=s_lab)

        for p_idx in range(len(i_arr)):
            res.h.append(p_data[2][p_idx])
            res.k.append(p_data[3][p_idx])
            res.l.append(p_data[4][p_idx])
            res.wavelength.append(p_data[5][p_idx])
            res.intensity.append(float(integrated_results[p_idx, 0]))
            res.sigma.append(float(integrated_results[p_idx, 3]))
            res.tt.append(float(bank_tt[p_idx]))
            res.az.append(float(bank_az[p_idx]))
            res.run_id.append(run_id)
            res.bank.append(physical_bank)

    return res
