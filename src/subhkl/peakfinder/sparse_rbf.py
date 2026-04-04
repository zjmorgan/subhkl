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

# auxiliary functions
@partial(jit, static_argnames=['window_size'])
def jax_median_2d(img, window_size):
    """
    Args:
        img: [photons/Pixel]
        window_size: [Pixel^0.5]
    Returns:
        [photons/Pixel]
    """
    pad_w = window_size // 2  # [Pixel^0.5]
    padded = jnp.pad(img, pad_w, mode='reflect')  # [photons/Pixel]
    im_4d = padded[None, None, :, :]  # [photons/Pixel]

    patches = lax.conv_general_dilated_patches(
        im_4d,
        filter_shape=(window_size, window_size),
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    return jnp.median(patches[0], axis=0)  # [photons/Pixel]

@jit
def jax_gaussian_blur_2d(img):
    """
    Args:
        img: [photons/Pixel]
    Returns:
        [photons/Pixel]
    """
    sigma = 3.0  # [Pixel^0.5]
    radius = int(4.0 * sigma + 0.5)  # [Pixel^0.5]
    x = jnp.arange(-radius, radius + 1)  # [Pixel^0.5]
    k_1d = jnp.exp(-0.5 * (x / sigma) ** 2)  # [-]
    k_1d = k_1d / jnp.sum(k_1d)  # [-]

    k_col = k_1d[:, None]  # [-]
    k_row = k_1d[None, :]  # [-]

    padded = jnp.pad(img, radius, mode='reflect')  # [photons/Pixel]
    temp = jax.scipy.signal.correlate2d(padded, k_col, mode='valid')  # [photons/Pixel]
    blurred = jax.scipy.signal.correlate2d(temp, k_row, mode='valid')  # [photons/Pixel]
    return blurred  # [photons/Pixel]

@partial(jit, static_argnames=['filter_size'])
def compute_bg_batch(imgs, filter_size):
    """
    Args:
        imgs: [photons/Pixel]
        filter_size: [Pixel^0.5]
    Returns:
        [photons/Pixel]
    """
    def process_one(img):
        med = jax_median_2d(img, filter_size)  # [photons/Pixel]
        blur = jax_gaussian_blur_2d(med)  # [photons/Pixel]
        return jnp.maximum(blur, 1e-3)  # [photons/Pixel]
    return lax.map(process_one, imgs)  # [photons/Pixel]


class SparseRBFPeakFinder:
    """
    Hierarchical Sparse RBF Peak Finder with Symmetric V-Cycle Basis Pursuit.
    
    Units:
        alpha: [-] (Z-score threshold)
        gamma: [-] (Besov weight power)
        min_sigma / max_sigma: [Pixel^0.5]
        ref_sigma: [Pixel^0.5]
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
        self.alpha = alpha  # [-]
        self.gamma = gamma  # [-]
        self.ref_sigma = 1.0  # [Pixel^0.5]         
        self.min_sigma = min_sigma  # [Pixel^0.5]
        self.max_sigma = max_sigma  # [Pixel^0.5]
        self.max_peaks = max_peaks  # [-]
        self.chunk_size = chunk_size  # [-]
        self.loss = loss  # [-]
        self.border_width = border_width  # [Pixel^0.5]
        self.show_steps = show_steps
        
        self.base_window_size = 64  # [Pixel^0.5]
        self.refine_patch_size = 15  # [Pixel^0.5]
        self.halo = 5  # [Pixel^0.5]
        self.max_local_peaks = 5  # [-]
       
        self.candidate_sigmas = jnp.linspace(min_sigma, max_sigma, num_sigmas)  # [Pixel^0.5]

    @staticmethod
    def _rbf_basis(x_grid, y, sigma_long, theta=0.0, phi=0.0, anisotropic=False, sigma_short=1.5):
        if not anisotropic:
            # ORIGINAL ISOTROPIC ERF LOGIC
            sig_sq2 = sigma_long * jnp.sqrt(2.0) + 1e-6
            erf_r = jax.scipy.special.erf((x_grid[0] + 0.5 - y[0]) / sig_sq2) - jax.scipy.special.erf((x_grid[0] - 0.5 - y[0]) / sig_sq2)
            erf_c = jax.scipy.special.erf((x_grid[1] + 0.5 - y[1]) / sig_sq2) - jax.scipy.special.erf((x_grid[1] - 0.5 - y[1]) / sig_sq2)
            return (jnp.pi / 2.0) * (sigma_long**2) * erf_r * erf_c
        else:
            # ANISOTROPIC 4x4 QUADRATURE LOGIC
            # 1. Build the Precision Matrix (Sigma^-1)
            cos_p = jnp.cos(phi)
            sin_p = jnp.sin(phi)

            var_l = jnp.maximum(sigma_long**2, 1e-6)
            var_s = jnp.maximum(sigma_short**2, 1e-6)

            a = (cos_p**2) / var_l + (sin_p**2) / var_s
            b = sin_p * cos_p * (1.0 / var_l - 1.0 / var_s)
            c = (sin_p**2) / var_l + (cos_p**2) / var_s

            # Setup 4x4 Sub-pixel Offsets (-0.375, -0.125, 0.125, 0.375)
            sub_offsets = jnp.array([-0.375, -0.125, 0.125, 0.375])

            # meshgrid 'xy' indexing: ox varies across cols (X), oy varies across rows (Y)
            ox, oy = jnp.meshgrid(sub_offsets, sub_offsets)

            # Distance from predicted center to pixel centers
            # x_grid[0] is yy (rows), x_grid[1] is xx (cols)
            dr_center = x_grid[0] - y[0]
            dc_center = x_grid[1] - y[1]

            # 3. Evaluate Gaussian at sub-points
            def eval_subpoint(ox_i, oy_i):
                dr = dr_center + oy_i  # row diff (Y) gets oy_i
                dc = dc_center + ox_i  # col diff (X) gets ox_i

                # 'a' applies to X (cols, dc), 'c' applies to Y (rows, dr)
                return jnp.exp(-0.5 * (a * dc**2 + 2.0 * b * dr * dc + c * dr**2))

            sub_evals = vmap(vmap(eval_subpoint))(ox, oy)

            # 4. Average the 16 evaluations and scale by analytic volume
            area_scalar = 2.0 * jnp.pi * sigma_long * sigma_short
            return jnp.mean(sub_evals, axis=(0, 1)) * area_scalar

    @staticmethod
    def _to_physical(params_raw, H, W, min_s, max_s):
        """
        Returns:
            [c: [photons/Pixel^2], r: [Pixel^0.5], col: [Pixel^0.5], sigma: [Pixel^0.5]]
        """
        params_reshaped = params_raw.reshape((-1, 4))
        c_raw, r_raw, c_col_raw, s_raw = params_reshaped.T
        c = jax.nn.softplus(c_raw)  # [photons/Pixel^2]
        r = jax.nn.sigmoid(r_raw) * H  # [Pixel^0.5]
        col = jax.nn.sigmoid(c_col_raw) * W  # [Pixel^0.5]
        sigma = min_s + jax.nn.sigmoid(s_raw) * (max_s - min_s)  # [Pixel^0.5]
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
        """
        Returns:
            [photons/Pixel]
        """
        c, r, c_col, sigma = params_phys.T
        if mask is not None:
            c = c * mask  # [photons/Pixel^2]
        def eval_one(ci, ri, ci_col, si):
            # [photons/Pixel^2] * [Pixel] = [photons/Pixel]
            return ci * SparseRBFPeakFinder._rbf_basis(x_grid, jnp.array([ri, ci_col]), si)
        basis_stack = vmap(eval_one)(c, r, c_col, sigma)  # [photons/Pixel]
        return jnp.sum(basis_stack, axis=0)  # [photons/Pixel]

    @staticmethod
    def _predict_batch_scan(params_phys, x_grid):
        """
        Returns: [photons/Pixel]
        """
        def body(carry, param):
            c, r, col, sigma = param
            # [photons/Pixel^2] * [Pixel] = [photons/Pixel]
            term = c * SparseRBFPeakFinder._rbf_basis(x_grid, jnp.array([r, col]), sigma)
            return carry + term, None
        H, W = x_grid.shape[1], x_grid.shape[2]
        init = jnp.zeros((H, W), dtype=params_phys.dtype)  # [photons/Pixel]
        final_image, _ = lax.scan(body, init, params_phys)
        return final_image  # [photons/Pixel]

    @staticmethod
    @partial(jit, static_argnames=['max_iter', 'loss_type', 'force_target'])
    def _solve_ssn_unified(A, y, bg_flat, alpha_vec, loss_type, c_warm, max_iter=20, force_target=False):
        """
        Solves the L1-Regularized Poisson/Gaussian optimization using a Semi-Smooth Newton (SSN) method.
        
        ========================================================================================
        MATHEMATICAL DERIVATION: PROXIMAL GRADIENT & SEMI-SMOOTH NEWTON
        ========================================================================================
        Objective: Minimize J(c) = f(c) + h(c)
                   where f(c) is the smooth Negative Log-Likelihood (NLL)
                   and   h(c) is the non-smooth volume penalty subject to c >= 0.
        
        1. The Proximal Gradient Step (Forward-Backward Splitting)
        ----------------------------------------------------------
        A standard gradient descent step on the smooth part f(c) gives an intermediate variable q:
            q = c_k - τ ∇f(c_k)    <-- [Code: q_test = q + tau * dq]
            
        To handle the non-smooth penalty h(c), we apply the Proximal Operator:
            c_{k+1} = prox_{τh}(q) = argmin_x [ h(x) + (1/2τ) ||x - q||_2^2 ]
            
        For a non-negative soft threshold, the analytical solution is:
            c_{k+1} = max(0, q - Threshold)
            
        2. The Fisher Information Threshold (The Dimensional Magic)
        ----------------------------------------------------------
        How do we define the Threshold? In dimensional analysis, the coefficient c is a 
        photon density [photons / Pixel^2]. 
        
        The Hessian (∇²f(c)) represents the Fisher Information Matrix. Its inverse (τ) 
        is the exact statistical variance of the coefficient (Cramer-Rao bound)!
        Therefore, sqrt(τ) is the true standard deviation of the peak density.
        
        By setting the threshold to: Z_score * Besov_Weight * sqrt(τ)
        The threshold dimensionally and mathematically maps perfectly into [photons / Pixel^2],
        making the solver inherently immune to arbitrary background or detector scaling.
            
        3. The Semi-Smooth Newton (SSN) Acceleration
        ----------------------------------------------------------
        Standard proximal gradient descent is slow. SSN accelerates this by finding the 
        root of the "Proximal Residual" mapping F(c):
            F(c) = (1/τ)(c - prox_{τh}(c - τ ∇f(c))) = 0
            
        We solve F(c) = 0 using a Newton step: c_{k+1} = c_k - J_F^{-1} F(c_k)
        Because the 'max' operator is not strictly differentiable at exactly 0, we use a 
        Generalized Jacobian (Clarke Subdifferential).
        
        The Active Set matrix D indicates which variables survived the proximal threshold:
            D_ii = 1 if q_i > Threshold  (Active Peak)
            D_ii = 0 if q_i <= Threshold (Crushed Halo/Noise)
            
        The generalized Hessian (DG) for the Newton step incorporates this active set:
            DG = (1/τ)(I - D) + H @ D    where H is the NLL Hessian (∇²f(c))
            
        This flawlessly partitions the linear system:
        - Active peaks (D=1) get solved via the true 2nd-order Hessian (H).
        - Crushed peaks (D=0) get clamped via the metric (1/τ) and forced to 0.
        ========================================================================================
        
        Input Units:
            A: [Pixel]
            y, bg_flat: [photons / Pixel]
            alpha_vec: [-] (Z-score * Besov shape weight)
            c_warm: [photons / Pixel^2]
        """
        N_peaks = A.shape[1]
        N_params = N_peaks
        q_init = c_warm.astype(jnp.float32)  # [photons / Pixel^2]

        bg_med = jnp.maximum(jnp.median(bg_flat), 1e-3).astype(jnp.float32)  # [photons / Pixel]

        def get_loss_grad_hess(c):
            u = A @ c + bg_flat  # [Pixel] * [photons/Pixel^2] + [photons/Pixel] = [photons/Pixel]
            
            if loss_type == 1:
                # POISSON Loss
                u_safe = jnp.maximum(u, 1e-6)  # [photons / Pixel]
                nll = jnp.sum(u_safe - y * jnp.log(u_safe))  # [photons / Pixel]
                
                grad = A.T @ (1.0 - y / u_safe)  # [Pixel] * [-] = [Pixel]
                
                W_diag = 1.0 / jnp.maximum(u_safe, 1e-3)  # [Pixel / photons]
                hess = A.T @ (W_diag[:, None] * A)  # [Pixel] * [Pixel/photons] * [Pixel] = [Pixel^3 / photons]
            else:
                # GAUSSIAN (OLS) Loss
                nll = 0.5 * jnp.sum((u - y)**2)  # [photons^2 / Pixel^2]
                
                grad = A.T @ (u - y)  # [Pixel] * [photons/Pixel] = [photons]
                
                hess = A.T @ A  # [Pixel] * [Pixel] = [Pixel^2]
                
            return nll, grad, hess

        def cond_fn(state):
            step, _, _, dq_norm = state
            return (step < max_iter) & (dq_norm > 1e-3)

        def body_fn(state):
            step, q, c, _ = state
            nll, grad, hess = get_loss_grad_hess(c)

            # 1. Compute physical metric tau (Inverse of Max Fisher Information)
            L = jnp.max(jnp.diag(hess)) + 1e-4
            tau = 1.0 / L  # Poisson: [photons / Pixel^3], Gaussian: [Pixel^-2]

            # 2. Extract Exact Statistical Variance (Cramer-Rao Lower Bound)
            # Both paths mathematically converge to var_c = [photons / Pixel^3]
            var_c = jnp.where(loss_type == 1, tau, bg_med * tau)
            
            # 3. Compute Dimensionally Pure Statistical Threshold 
            # tau_alpha = [-] * sqrt([photons / Pixel^3]) -> [photons / Pixel^2] (Statistical equivalent)
            tau_alpha = alpha_vec * jnp.sqrt(var_c)

            # 4. Proximal Residual Mapping
            # Poisson: [photons/Pixel^2] / [photons/Pixel^3] + [Pixel] = [Pixel] -> mapped via metric
            # Gaussian: [photons/Pixel^2] / [Pixel^-2] + [photons] = [photons]
            Gq = (q - c) / tau + grad  

            # 5. Active Set Discovery (Shape/Halo Trap)
            D = (q > tau_alpha).astype(jnp.float32)  # [-]
            DP_mat = jnp.diag(D)  # [-]
            I = jnp.eye(N_params, dtype=jnp.float32)  # [-]

            # 6. Clarke Subdifferential Generalized Hessian
            DG = (I - DP_mat) / tau + hess @ DP_mat + 1e-4 * I  
            
            # dq mathematically evaluates perfectly to [photons / Pixel^2] in both pathways
            dq = jnp.linalg.solve(DG, -Gq).astype(jnp.float32)

            def bt_cond(bt_state):
                bt_i, step_size, _, _, j_test, j_curr = bt_state
                is_valid = jnp.isfinite(j_test)
                return (bt_i < 8) & ((j_test > j_curr) | ~is_valid)

            def bt_body(bt_state):
                bt_i, step_size, _, _, _, j_curr = bt_state
                step_size = jnp.float32(step_size * 0.5)  # [-]
                
                q_test = (q + step_size * dq).astype(jnp.float32)  # [photons / Pixel^2]
                c_test = jnp.maximum(0.0, q_test - tau_alpha).astype(jnp.float32)  # [photons / Pixel^2]
                
                j_test, _, _ = get_loss_grad_hess(c_test)
                
                # Re-add the exact threshold penalty to the objective space to check descent
                reg_penalty = jnp.sum((tau_alpha / tau) * c_test)
                return (bt_i + 1, step_size, q_test, c_test, j_test + reg_penalty, j_curr)

            q_test = (q + dq).astype(jnp.float32)
            c_test = jnp.maximum(0.0, q_test - tau_alpha).astype(jnp.float32)
            j_test, _, _ = get_loss_grad_hess(c_test)
            
            reg_penalty = jnp.sum((tau_alpha / tau) * c_test)
            obj_val = nll + jnp.sum((tau_alpha / tau) * c)

            bt_init = (0, jnp.float32(1.0), q_test, c_test, j_test + reg_penalty, obj_val)
            bt_final = lax.while_loop(bt_cond, bt_body, bt_init)
            _, _, q_final, c_final, _, _ = bt_final

            return (step + 1, q_final.astype(jnp.float32), c_final.astype(jnp.float32), jnp.linalg.norm(dq).astype(jnp.float32))

        init_state = (0, q_init.astype(jnp.float32), c_warm.astype(jnp.float32), jnp.float32(1e9))
        final_state = lax.while_loop(cond_fn, body_fn, init_state)
        _, _, c_l1, _ = final_state

        # DEBIASING PHASE
        active_mask = c_l1 > 1e-5  # [-]

        if force_target:
            # Guarantee the target peak (index 0) is measured unbiasedly
            active_mask = active_mask.at[0].set(True)

        def debias_cond(state):
            step, _, actual_step_norm = state
            return (step < 100) & (actual_step_norm > 1e-4)

        def debias_body(state):
            step, c, _ = state
            _, grad, hess = get_loss_grad_hess(c)

            H_diag = jnp.diag(hess)
            eta = 1.0 / jnp.maximum(H_diag, 1e-6)  # Acts exactly as tau

            I = jnp.eye(N_params, dtype=jnp.float32)
            D_mat = jnp.diag(active_mask.astype(jnp.float32))

            F_c = (1.0 - active_mask) * c + active_mask * (eta * grad)  # [photons / Pixel^2]
            DG = (I - D_mat) + (eta[:, None] * hess) @ D_mat + 1e-4 * I  

            dc = jnp.linalg.solve(DG, -F_c).astype(jnp.float32)  # [photons / Pixel^2]

            tau_debias = jnp.where(loss_type == 1, jnp.float32(0.8), jnp.float32(1.0))  # [-]
            
            c_new_raw = c + tau_debias * dc * active_mask  # [photons / Pixel^2]
            c_new = jnp.maximum(0.0, c_new_raw) * active_mask  # [photons / Pixel^2]

            actual_step = c_new - c  # [photons / Pixel^2]
            return (step + 1, c_new.astype(jnp.float32), jnp.linalg.norm(actual_step).astype(jnp.float32))

        debias_state = lax.while_loop(debias_cond, debias_body, (0, c_l1.astype(jnp.float32), jnp.float32(1e9)))
        _, c_final, _ = debias_state

        return c_final.astype(jnp.float32)  # [photons / Pixel^2]

    @partial(jit, static_argnames=['self', 'H', 'W', 'max_peaks_local', 'loss_code', 'do_merge'])
    def _solve_dense(self, patch_stat, patch_bg, alpha_z_score, H, W, max_peaks_local, loss_code, do_merge):
        
        local_bg_med = jnp.maximum(jnp.median(patch_bg), 1e-3)  # [photons/Pixel]
        local_noise_floor = jnp.sqrt(local_bg_med)  # [photons^0.5 / Pixel^0.5]
       
        bounds = (float(H), float(W), self.min_sigma, self.max_sigma)  # [Pixel^0.5]
        yy, xx = jnp.indices((H, W))  # [Pixel^0.5]
        x_grid = jnp.array([yy, xx])  # [Pixel^0.5]
        
        max_k_rad = int(3.0 * self.max_sigma)  # [Pixel^0.5]
        k_grid = jnp.arange(-max_k_rad, max_k_rad + 1)  # [Pixel^0.5]
        
        init_params = jnp.zeros((max_peaks_local, 4))
        init_active = jnp.zeros(max_peaks_local, dtype=bool)
        init_state = (init_params, init_active, 0)

        def step_fn(state, _):
            params, active_mask, idx = state
            recon = self._predict_batch_physical(params, x_grid, active_mask)  # [photons/Pixel]
            
            def check_sigma(s):
                sig_sq2 = s * jnp.sqrt(2.0) + 1e-6  # [Pixel^0.5]
                
                # separable kernels
                k_1d = jax.scipy.special.erf((k_grid + 0.5) / sig_sq2) - jax.scipy.special.erf((k_grid - 0.5) / sig_sq2)  # [-]
                k_col = k_1d[:, None]  # [-]
                k_row = k_1d[None, :]  # [-]

                recon_total = jnp.maximum(recon + patch_bg, 1e-3)  # [photons/Pixel]
                raw_grad = patch_stat - recon_total  # [photons/Pixel]

                temp = jax.scipy.signal.correlate2d(raw_grad, k_col, mode='valid')  # [photons/Pixel]
                dual_var_unscaled = jax.scipy.signal.correlate2d(temp, k_row, mode='valid')  # [photons/Pixel]
                
                area_scalar = (jnp.pi / 2.0) * (s**2)  # [Pixel]
                dual_var = dual_var_unscaled * area_scalar  # [photons]

                flat_idx = jnp.argmax(dual_var)
                r_valid, c_valid = jnp.unravel_index(flat_idx, dual_var.shape)  # [Pixel^0.5]

                # --- EXACT LOG-PARABOLIC INTERPOLATION ---
                padded_dv = jnp.pad(dual_var, 1, mode='edge')
                r_p, c_p = r_valid + 1, c_valid + 1  # [Pixel^0.5]
                
                safe_dv = jnp.maximum(padded_dv, 1e-6)
                val    = jnp.log(safe_dv[r_p, c_p])  # [ln(photons)]
                val_up = jnp.log(safe_dv[r_p - 1, c_p])
                val_dn = jnp.log(safe_dv[r_p + 1, c_p])
                val_lf = jnp.log(safe_dv[r_p, c_p - 1])
                val_rt = jnp.log(safe_dv[r_p, c_p + 1])
                
                den_r = val_up - 2.0 * val + val_dn  # [-]
                den_r = jnp.minimum(den_r, -1e-6) 
                dr = 0.5 * (val_up - val_dn) / den_r  # [Pixel^0.5]
                
                den_c = val_lf - 2.0 * val + val_rt  # [-]
                den_c = jnp.minimum(den_c, -1e-6)
                dc = 0.5 * (val_lf - val_rt) / den_c  # [Pixel^0.5]

                dr = jnp.clip(dr, -0.5, 0.5)  # [Pixel^0.5]
                dc = jnp.clip(dc, -0.5, 0.5)  # [Pixel^0.5]

                r_idx = r_valid + max_k_rad + dr  # [Pixel^0.5]
                c_idx = c_valid + max_k_rad + dc  # [Pixel^0.5]

                k_1d_sq_sum = jnp.sum(k_1d ** 2)  # [-]
                kernel_sq_norm = (area_scalar ** 2) * (k_1d_sq_sum ** 2)  # [Pixel^2]

                scale_score = dual_var[r_valid, c_valid] / jnp.sqrt(kernel_sq_norm)  # [photons / Pixel]

                # Exact dimensional recovery of volumetric density:
                c_matched = dual_var[r_valid, c_valid] / kernel_sq_norm  # [photons / Pixel^2]
                c_init = jnp.maximum(c_matched, 0.0)  # [photons/Pixel^2]

                return scale_score, jnp.array([c_init, r_idx, c_idx, s])

            vals, candidates = vmap(check_sigma)(self.candidate_sigmas)
            best_idx = jnp.argmax(vals)
            new_peak = candidates[best_idx]
            
            s_best = new_peak[3]  # [Pixel^0.5]
           
            best_scale_score = vals[best_idx]  # [photons]
            z_score = best_scale_score / local_noise_floor  # [photons^0.5 * Pixel^0.5] (dimensionally scales as pure SNR)
            weight_best = (s_best / self.ref_sigma) ** self.gamma  # [-]
            
            is_strong = z_score > (alpha_z_score * weight_best)  # True dimensionless threshold check

            dummy_peak = jnp.array([0.0, 0.0, 0.0, 1.0])
            new_peak = jnp.where(is_strong, new_peak, dummy_peak)
            
            params = jnp_update_set(params, idx, new_peak)
            active_mask = jnp_update_set(active_mask, idx, is_strong)

            def run_opt(operand):
                p, a_mask = operand
                
                c_init = p[:, 0]  # [photons/Pixel^2]
                r = p[:, 1]  # [Pixel^0.5]
                col = p[:, 2]  # [Pixel^0.5]
                sigma = p[:, 3]  # [Pixel^0.5]
                
                def eval_one(ri, ci_col, si):
                    return self._rbf_basis(x_grid, jnp.array([ri, ci_col]), si).flatten()  # [Pixel]
                
                A = vmap(eval_one)(r, col, sigma).T  # [Pixel]
                A_masked = A * a_mask

                volumes = jnp.where(
                    getattr(self, 'anisotropic', False),
                    jnp.float32(2.0 * jnp.pi) * best_sig_target * jnp.minimum(getattr(self, 'sigma_short', 1.5), best_sig_target),
                    jnp.float32(2.0 * jnp.pi) * (best_sig_target**2)
                )  # [Pixel]
                weights = (sigma / self.ref_sigma) ** self.gamma  # [-]
                alpha_vec_stat = alpha_z_score * weights  # [Pixel]
                
                c_phys_masked = c_init * a_mask  # [photons/Pixel^2]
                
                c_sparse_stat = self._solve_ssn_unified(A_masked, patch_stat.flatten(), patch_bg.flatten(), alpha_vec_stat, loss_code, c_phys_masked)  # [photons/Pixel^2]
                
                c_sparse_norm = c_sparse_stat * a_mask  # [photons/Pixel^2]
                return jnp.stack([c_sparse_norm, r, col, sigma], axis=1)

            def skip_opt(operand):
                p, _ = operand
                return p

            params = lax.cond(is_strong, run_opt, skip_opt, (params, active_mask))

            return (params, active_mask, idx + 1), None

        final_state, _ = lax.scan(step_fn, init_state, None, length=max_peaks_local)
        final_params, final_active, _ = final_state
        
        if do_merge:
            c, r, col, sigma = final_params.T
            active_mask = final_active & (c > 1e-9)
            num_active = jnp.sum(active_mask)
            
            c_active = jnp.where(active_mask, c, 0.0)  # [photons/Pixel^2]
            total_amp = jnp.sum(c_active) + 1e-12  # [photons/Pixel^2]
            
            com_r = jnp.sum(c_active * r) / total_amp  # [Pixel^0.5]
            com_c = jnp.sum(c_active * col) / total_amp  # [Pixel^0.5]
            var_r = jnp.sum(c_active * (r - com_r)**2) / total_amp  # [Pixel]
            var_c = jnp.sum(c_active * (col - com_c)**2) / total_amp  # [Pixel]
            
            mean_sigma = jnp.sum(jnp.where(active_mask, sigma, 0.0)) / jnp.maximum(num_active, 1)  # [Pixel^0.5]
            macro_sigma = jnp.sqrt(var_r + var_c) + mean_sigma  # [Pixel^0.5]
            
            dummy_atom = jnp.array([0.0, -100.0, -100.0, 1.0])
            macro_atom = jnp.stack([total_amp, com_r, com_c, macro_sigma])
            macro_atom = jnp.where(num_active > 1, macro_atom, dummy_atom)
            
            augmented_dict = jnp.vstack([final_params, macro_atom])
            aug_mask = jnp.append(active_mask, num_active > 1)
            
            c_warm_raw, r_aug, col_aug, sigma_aug = augmented_dict.T  # [photons/Pixel^2], [Pixel^0.5], [Pixel^0.5], [Pixel^0.5]
            
            def eval_one_aug(ri, ci_col, si):
                return self._rbf_basis(x_grid, jnp.array([ri, ci_col]), si).flatten()  # [Pixel]
            
            A_aug = vmap(eval_one_aug)(r_aug, col_aug, sigma_aug).T  # [Pixel]
            A_aug_masked = A_aug * aug_mask
           
            volumes_aug = (jnp.pi / 2.0) * (sigma_aug**2)  # [Pixel]
            weights_aug = (sigma_aug / self.ref_sigma) ** self.gamma  # [-]
            alpha_vec_stat_aug = alpha_z_score * weights_aug  # [Pixel]

            c_sparse_stat_aug = self._solve_ssn_unified(A_aug_masked, patch_stat.flatten(), patch_bg.flatten(), alpha_vec_stat_aug, loss_code, c_warm_raw)  # [photons/Pixel^2]
            
            return jnp.stack([c_sparse_stat_aug * aug_mask, r_aug, col_aug, sigma_aug], axis=1)
        else:
            return final_params

    def compute_metrics(self, images_raw, bg_map, peaks_list, global_max):
        """
        Args:
            images_raw, bg_map: [photons/Pixel]
            global_max: [photons/Pixel]
        """
        B, H, W = images_raw.shape
        yy, xx = np.indices((H, W))  # [Pixel^0.5]
        x_grid = jnp.array([yy, xx])  # [Pixel^0.5]
        
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
            recon_peaks = self._predict_batch_scan(peaks, x_grid)  # [photons/Pixel]
            recon_total = jnp.maximum(recon_peaks + median_val, 1e-9)  # [photons/Pixel]
            
            if loss_code == 1: 
                # 1. Exact Poisson NLL using xlogy
                nll = jnp.sum(recon_total - jax.scipy.special.xlogy(target_raw, recon_total))  # [photons/Pixel]
                # 2. Exact Poisson Deviance (no more 1e-9 target clamping)
                term = jax.scipy.special.xlogy(target_raw, target_raw / recon_total) - (target_raw - recon_total)
                dev = 2 * jnp.sum(term)  # [-] (Deviance acts as dimensionless chi-square equivalent)
            else: 
                diff = recon_total - target_raw  # [photons/Pixel]
                nll = 0.5 * jnp.sum(diff**2)  # [photons^2 / Pixel^2]
                dev = jnp.sum((diff**2) / recon_total)  # [-]
            
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
        """
        Args:
            images_batch: [photons/Pixel]
        """
        B, H, W = images_batch.shape
        
        # alpha is strictly interpreted as the Z-score (SNR) threshold
        alpha_z_score = self.alpha  # [-]

        filter_size = max(15, int(self.max_sigma * 5))  # [Pixel^0.5]
        if filter_size % 2 == 0:
            filter_size += 1

        bg_map = np.array(compute_bg_batch(jnp.array(images_batch, dtype=jnp.float32), filter_size))  # [photons/Pixel]
        self._last_bg_map = bg_map

        valid_bg = bg_map[bg_map > 1e-2]
        if valid_bg.size == 0:
            median_bg_level = 1.0  # [photons/Pixel]
        else:
            median_bg_level = float(np.median(valid_bg))  # [photons/Pixel]
            
        if np.isnan(median_bg_level) or median_bg_level <= 0:
            median_bg_level = 1.0
            
        poisson_noise_floor = np.maximum(np.sqrt(median_bg_level), 1.0)  # [photons^0.5 / Pixel^0.5]

        if self.show_steps:
            print(f"  > Pre-processing: Morphological Bg Evaluated.")
            print(f"  > Autotuning: Median BG={median_bg_level:.1f}, Noise Floor=~{poisson_noise_floor:.1f}")

        img_jax_stat_np = np.copy(images_batch)  # [photons/Pixel]
        if self.border_width > 0:
            bw = self.border_width  # [Pixel^0.5]
            valid_interior = np.zeros((H, W), dtype=bool)
            valid_interior[bw:-bw, bw:-bw] = True
            valid_mask_batch = np.broadcast_to(valid_interior, (B, H, W))
            img_jax_stat_np = np.where(valid_mask_batch, img_jax_stat_np, bg_map)
            
        img_jax_stat = jnp.array(img_jax_stat_np)  # [photons/Pixel]
        img_jax_bg = jnp.array(bg_map)  # [photons/Pixel]

        loss_code_sniper = 1 if self.loss == 'poisson' else 0

        max_k_rad = int(3.0 * self.max_sigma)  # [Pixel^0.5]
        
        w_scout_core = self.base_window_size  # [Pixel^0.5]
        w_ext = w_scout_core + 2 * max_k_rad  # [Pixel^0.5]
        stride = w_scout_core // 2  # [Pixel^0.5]

        min_required_patch = 2 * max_k_rad + 1  # [Pixel^0.5]
        P_core = max(self.refine_patch_size, min_required_patch)  # [Pixel^0.5]
        P_EXT = P_core + 2 * max_k_rad  # [Pixel^0.5]

        pad_size = P_core // 2 + max_k_rad  # [Pixel^0.5]

        img_jax_stat = jnp.pad(img_jax_stat, ((0,0), (pad_size, pad_size), (pad_size, pad_size)), mode='symmetric')  # [photons/Pixel]
        img_jax_bg = jnp.pad(img_jax_bg, ((0,0), (pad_size, pad_size), (pad_size, pad_size)), mode='symmetric')  # [photons/Pixel]

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
            r_start = r_idx - max_k_rad  # [Pixel^0.5]
            c_start = c_idx - max_k_rad  # [Pixel^0.5]
            def slice_one(bi, ri, ci):
                return lax.dynamic_slice(img[bi], (ri, ci), (w_ext, w_ext))  # [photons/Pixel]
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
                
                valid_peaks[:, 1] += chunk[b_indices, 1] - max_k_rad  # [Pixel^0.5]
                valid_peaks[:, 2] += chunk[b_indices, 2] - max_k_rad  # [Pixel^0.5]
                
                peaks_with_bank = np.column_stack([valid_banks, valid_peaks])
                scout_results.append(peaks_with_bank)

        if not scout_results:
            return [np.empty((0, 4)) for _ in range(B)]

        all_candidates = np.vstack(scout_results)
        unique_candidates = []
        
        for b in range(B):
            bank_mask = (all_candidates[:, 0] == b)
            if not np.any(bank_mask): continue
            cands = all_candidates[bank_mask, 2:4]  # [Pixel^0.5]
            vals = all_candidates[bank_mask, 1]  # [photons/Pixel^2]
            order = np.argsort(vals)[::-1]
            cands_sorted = cands[order]
            keep = np.ones(len(cands_sorted), dtype=bool)
            if len(cands_sorted) > 1:
                dists = squareform(pdist(cands_sorted))  # [Pixel^0.5]
                np.fill_diagonal(dists, 9999.0)
                radius = 1.5  # [Pixel^0.5]
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
            r_center = centers[:, 1].astype(int)  # [Pixel^0.5]
            c_center = centers[:, 2].astype(int)  # [Pixel^0.5]
            
            r_start = r_center - pad_size  # [Pixel^0.5]
            c_start = c_center - pad_size  # [Pixel^0.5]
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
                
                global_rs_padded = valid_r_centers.astype(int) - pad_size + valid_peaks[:, 1]  # [Pixel^0.5]
                global_cs_padded = valid_c_centers.astype(int) - pad_size + valid_peaks[:, 2]  # [Pixel^0.5]
                
                global_rs = global_rs_padded - pad_size  # [Pixel^0.5]
                global_cs = global_cs_padded - pad_size  # [Pixel^0.5]

                MARGIN = max(3, self.border_width)  # [Pixel^0.5]
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
                coords = peaks_sorted[:, 1:3]  # [Pixel^0.5]
                
                if len(coords) > 1:
                    dists = squareform(pdist(coords))  # [Pixel^0.5]
                    np.fill_diagonal(dists, 9999.0)
                    r = 1.5  # [Pixel^0.5]
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

@jit
def build_3d_cov(params):
    # params: [L11, L21, L22, L31, L32, L33]
    L = jnp.array([
        [params[0],       0.0,       0.0],
        [params[1], params[2],       0.0],
        [params[3], params[4], params[5]]
    ])
    # Ensure strict positive-definiteness on the diagonal
    L = L.at[0,0].set(jnp.abs(L[0,0]) + 0.1)
    L = L.at[1,1].set(jnp.abs(L[1,1]) + 0.1)
    L = L.at[2,2].set(jnp.abs(L[2,2]) + 0.1)
    return L @ L.T

@partial(jit, static_argnames=['patch_size'])
def global_shape_objective(params, patches, bgs, drs, dcs, P_mats, distances, patch_size):
    # Constant shape tensor
    Sigma_shape = build_3d_cov(params[:6])

    # Extract the isotropic mosaicity (radians)
    eta = jnp.abs(params[6]) + 1e-6
    Sigma_eta_base = jnp.eye(3) * (eta**2)

    yy, xx = jnp.indices((patch_size, patch_size))
    def fit_one_peak(patch, bg, dr, dc, P_true, D_i):
        # Add the distance-dependent mosaicity tensor
        Sigma_total_3D = Sigma_shape + (D_i**2) * Sigma_eta_base

        # Project using the central projection matrix
        Sigma_2D = P_true @ Sigma_total_3D @ P_true.T
        det_sigma = jnp.maximum(Sigma_2D[0,0] * Sigma_2D[1,1] - Sigma_2D[0,1]**2, 1e-6)

        # 2. Analytic Precision Matrix
        a = Sigma_2D[1,1] / det_sigma
        b = -Sigma_2D[0,1] / det_sigma
        c = Sigma_2D[0,0] / det_sigma

        dr_grid = yy - dr
        dc_grid = xx - dc

        # 3. Evaluate Unnormalized Template
        template = jnp.exp(-0.5 * (a * dc_grid**2 + 2.0 * b * dr_grid * dc_grid + c * dr_grid**2))

        # 4. Fast OLS Amplitude Fit
        y_sub = patch - bg
        amp = jnp.sum(y_sub * template) / jnp.maximum(jnp.sum(template * template), 1e-6)
        amp = jnp.maximum(amp, 0.0)

        # 5. Return Mean Squared Error
        residual = y_sub - amp * template
        return jnp.sum(residual**2)

    mses = vmap(fit_one_peak)(patches, bgs, drs, dcs, P_mats, distances)
    return jnp.mean(mses)

# Generate the fast gradient function
val_and_grad_fn = jit(jax.value_and_grad(global_shape_objective), static_argnames=['patch_size'])

def optimize_global_crystal(patches, bgs, drs, dcs, P_mats, distances):
    """Finds the optimal 6 global parameters for the 3D crystal tensor."""
    def scipy_objective(x_np):
        val, grad = val_and_grad_fn(jnp.array(x_np), patches, bgs, drs, dcs, P_mats, distances, patches.shape[-1])
        return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)

    # Initial guess: Isotropic 1.0 pixel sphere
    x0 = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.001])

    print("\n  > Optimizing 3D Global Crystal Tensor...")
    res = scipy.optimize.minimize(scipy_objective, x0, method='L-BFGS-B', jac=True)

    print(f"  > Global Optimization Complete. (Final MSE: {res.fun:.2f})")
    print(f"  > Recovered Mosaicity (eta): {abs(res.x[6])*1000:.3f} mrad")
    return res.x

class SparseLaueIntegrator(SparseRBFPeakFinder):
    """
    Physics-Informed Sniper.
    Takes predicted spot coordinates, extracts patches, and uses Volume-Penalized
    Sparse RBF to accurately integrate intensity using the Preconditioned SSN Engine.
    
    Units:
        alpha: [-] (Z-score threshold)
        min_sigma / max_sigma / nominal_sigma: [-]
        mosaicity_eta: [Pixel^0.5] (?)
        gamma: [-]
        Returns integrations containing sigI: [photons^0.5 / Pixel^0.5]
    """
    def __init__(self,
                 alpha=0.05,
                 min_sigma=0.1,    # [Pixels] at theta=45
                 max_sigma=15.0,   # [Pixels] at theta=45
                 gamma=2.0,
                 loss='poisson',
                 border_width=0,
                 num_sigmas=32,
                 nominal_sigma=2.0,
                 anisotropic=False,
                 chunk_size=1024):

        # 1. Initialize parent with safe dummy pixel values to keep it functional 
        # (in case you ever call super().find_peaks_batch)
        super().__init__(
            alpha=alpha, gamma=gamma, min_sigma=0.5, max_sigma=5.0,
            loss=loss, border_width=border_width, num_sigmas=num_sigmas, chunk_size=chunk_size,
            show_steps=False
        )

        self.nominal_sigma = nominal_sigma
        self.anisotropic = anisotropic

    def integrate_reflections(self, images_batch, frames, rs, cs, var_us, var_vs, cov_uvs):
        """
        Args:
            images_batch: [photons/Pixel]
            frames: [-]
            rs, cs: [Pixel^0.5]
        Returns:
            [intensity: [photons/Pixel], r: [Pixel^0.5], c: [Pixel^0.5], sigma: [Pixel^0.5], sigI: [photons^0.5 / Pixel^0.5]]
        """
        B, H, W = images_batch.shape
        N_spots = len(frames)

        P = self.refine_patch_size  # [Pixel^0.5]
        half_p = P // 2  # [Pixel^0.5]
        PAD = P  # [Pixel^0.5]

        K_NEIGHBORS = min(4, N_spots) if N_spots > 0 else 1

        filter_size = max(15, int(self.max_sigma * 5))  # [Pixel^0.5]
        if filter_size % 2 == 0: 
            filter_size += 1

        images_jax = jnp.array(images_batch, dtype=jnp.float32)  # [photons/Pixel]
        bg_maps_jax = compute_bg_batch(images_jax, filter_size)  # [photons/Pixel]

        img_jax_padded = jnp.pad(images_jax, ((0,0), (PAD, PAD), (PAD, PAD)), mode='reflect')  # [photons/Pixel]
        bg_jax_padded = jnp.pad(bg_maps_jax, ((0,0), (PAD, PAD), (PAD, PAD)), mode='reflect')  # [photons/Pixel]

        bounds = (float(P), float(P), self.min_sigma, self.max_sigma)  # [Pixel^0.5]
        yy, xx = jnp.indices((P, P))  # [Pixel^0.5]
        x_grid = jnp.array([yy, xx])  # [Pixel^0.5]
        loss_code = 1 if self.loss == 'poisson' else 0

        @jit
        def extract_patches(img_src, bg_src, f_idx, r_idx, c_idx):
            r_start = jnp.clip(jnp.int32(jnp.round(r_idx)) - half_p, 0, img_src.shape[1] - P)  # [Pixel^0.5]
            c_start = jnp.clip(jnp.int32(jnp.round(c_idx)) - half_p, 0, img_src.shape[2] - P)  # [Pixel^0.5]
            def slice_img(bi, ri, ci): return lax.dynamic_slice(img_src[bi], (ri, ci), (P, P))
            def slice_bg(bi, ri, ci):  return lax.dynamic_slice(bg_src[bi], (ri, ci), (P, P))
            return vmap(slice_img)(f_idx, r_start, c_start), vmap(slice_bg)(f_idx, r_start, c_start), r_start, c_start

        @jit
        def solve_patches(patches, patches_bg, fs_chunk, rs_global_chunk, cs_global_chunk, r_starts, c_starts,
                          all_fs_jnp, all_rs_jnp, all_cs_jnp,
                          var_us_jnp, var_vs_jnp, cov_uvs_jnp):
            alpha_z_score = self.alpha

            def process_patch(patch, patch_bg, f_global, r_global, c_global, r_start, c_start):
                # 1. Fetch exact pre-computed global geometry
                peak_var_u = var_us_jnp[f_global]
                peak_var_v = var_vs_jnp[f_global]
                peak_cov_uv = cov_uvs_jnp[f_global]

                bg_med = jnp.maximum(jnp.median(patch_bg), 1e-3)
                noise_floor = jnp.sqrt(bg_med)

                dists = (all_rs_jnp - r_global)**2 + (all_cs_jnp - c_global)**2
                frame_penalty = jnp.where(all_fs_jnp == f_global, 0.0, 1e9)
                _, nbr_idxs = jax.lax.top_k(-(dists + frame_penalty), K_NEIGHBORS)

                nbr_rs = all_rs_jnp[nbr_idxs]
                nbr_cs = all_cs_jnp[nbr_idxs]
                local_rs = nbr_rs - r_start
                local_cs = nbr_cs - c_start

                # SUBPIXEL RELAXATION (Log-Parabolic Target Snapping)
                # We use a static 1.0px blur just to smooth the noise for the center-of-mass finding
                y_sub_raw = patch - patch_bg
                nominal_sig_sq2 = 1.0 * jnp.sqrt(2.0) + 1e-6
                k_grid = jnp.arange(-2, 3)
                k_1d = jax.scipy.special.erf((k_grid + 0.5) / nominal_sig_sq2) - jax.scipy.special.erf((k_grid - 0.5) / nominal_sig_sq2)

                temp = jax.scipy.signal.correlate2d(y_sub_raw, k_1d[:, None], mode='same')
                dual_var_smooth = jax.scipy.signal.correlate2d(temp, k_1d[None, :], mode='same')

                r_int = jnp.clip(jnp.int32(jnp.round(local_rs[0])), 1, P - 2)
                c_int = jnp.clip(jnp.int32(jnp.round(local_cs[0])), 1, P - 2)

                safe_dv = jnp.maximum(dual_var_smooth, 1e-6)
                val    = jnp.log(safe_dv[r_int, c_int])
                val_up = jnp.log(safe_dv[r_int - 1, c_int])
                val_dn = jnp.log(safe_dv[r_int + 1, c_int])
                val_lf = jnp.log(safe_dv[r_int, c_int - 1])
                val_rt = jnp.log(safe_dv[r_int, c_int + 1])

                den_r = jnp.minimum(val_up - 2.0 * val + val_dn, -1e-6)
                dr = 0.5 * (val_up - val_dn) / den_r

                den_c = jnp.minimum(val_lf - 2.0 * val + val_rt, -1e-6)
                dc = 0.5 * (val_lf - val_rt) / den_c

                dr = jnp.clip(dr, -1.5, 1.5)
                dc = jnp.clip(dc, -1.5, 1.5)

                local_rs = local_rs.at[0].add(dr)
                local_cs = local_cs.at[0].add(dc)

                # =====================================================================
                # ANALYTIC GAUSSIAN EVALUATION (No Dictionary Search!)
                # =====================================================================
                def eval_neighbor(nr, nc):
                    det_sigma = jnp.maximum(peak_var_u * peak_var_v - peak_cov_uv**2, 1e-6)

                    a = peak_var_v / det_sigma
                    b = -peak_cov_uv / det_sigma
                    c = peak_var_u / det_sigma

                    dr_grid = x_grid[0] - nr # rows (V)
                    dc_grid = x_grid[1] - nc # cols (U)

                    gaussian = jnp.exp(-0.5 * (a * dc_grid**2 + 2.0 * b * dr_grid * dc_grid + c * dr_grid**2))
                    area_scalar = 2.0 * jnp.pi * jnp.sqrt(det_sigma)

                    # Flatten returns directly to (P*P) because N_shapes is gone!
                    return (gaussian * area_scalar).flatten() 

                A_all = vmap(eval_neighbor)(local_rs, local_cs) 
                y_sub = (patch - patch_bg).flatten() 

                pixel_dists_k = (yy.flatten()[:, None] - local_rs[None, :])**2 + (xx.flatten()[:, None] - local_cs[None, :])**2 
                closest_k = jnp.argmin(pixel_dists_k, axis=1)
                pixel_masks = jax.nn.one_hot(closest_k, K_NEIGHBORS) 

                A_k = A_all.T # (P*P, K_NEIGHBORS)
                A_k_masked = A_k * pixel_masks 

                # Calculate effective scalar size for the L1 threshold
                effective_sigma = jnp.sqrt(jnp.sqrt(jnp.maximum(peak_var_u * peak_var_v - peak_cov_uv**2, 1e-6)))
                weight = (effective_sigma / self.ref_sigma) ** self.gamma 
                alpha_vec_joint = jnp.full(K_NEIGHBORS, alpha_z_score * weight) 

                c_warm_joint = jnp.zeros(K_NEIGHBORS, dtype=jnp.float32) 

                # The SSN Solver runs ONCE per neighbor set. Extremely fast.
                c_ssn = self._solve_ssn_unified(
                    A_k_masked, patch.flatten(), patch_bg.flatten(), alpha_vec_joint, loss_code, c_warm_joint, 20, force_target=False
                )

                surviving_mask_strict = c_ssn > 1e-9
                is_target = jnp.arange(K_NEIGHBORS) == 0
                surviving_mask = surviving_mask_strict | is_target

                A_best_masked = A_k * surviving_mask[None, :] 

                # =====================================================================
                # STAGE 2: UNCONSTRAINED OLS 
                # =====================================================================
                A_tilde = jnp.hstack([A_best_masked, jnp.ones((P*P, 1))]) 
                w = 1.0 / jnp.maximum(patch.flatten(), 1.0) 

                I_mat = A_tilde.T @ (w[:, None] * A_tilde)  
                C_mat = jnp.linalg.inv(I_mat + 1e-6 * jnp.eye(K_NEIGHBORS + 1))  
                rhs = A_tilde.T @ (w * y_sub)  
                c_ols = C_mat @ rhs 

                c_final_target = c_ols[0] 
                volumes = jnp.float32(2.0 * jnp.pi) * jnp.sqrt(jnp.maximum(peak_var_u * peak_var_v - peak_cov_uv**2, 1e-6)) 

                intensity = c_final_target * volumes 
                var_c0 = C_mat[0, 0] 
                sigI = volumes * jnp.sqrt(jnp.maximum(var_c0, 0.0)) 

                return jnp.array([
                    intensity, 
                    local_rs[0], 
                    local_cs[0], 
                    effective_sigma, 
                    sigI
                ])

            return vmap(process_patch)(patches, patches_bg, fs_chunk, rs_global_chunk, cs_global_chunk, r_starts, c_starts)

        refined_peaks = []
        rs_padded = np.array(rs) + PAD  # [Pixel^0.5]
        cs_padded = np.array(cs) + PAD  # [Pixel^0.5]

        PAD_N = max(N_spots, 4)
        fs_full = np.pad(np.array(frames), (0, PAD_N - N_spots), constant_values=-1)
        rs_full = np.pad(rs_padded, (0, PAD_N - N_spots), constant_values=-10000.0)  # [Pixel^0.5]
        cs_full = np.pad(cs_padded, (0, PAD_N - N_spots), constant_values=-10000.0)  # [Pixel^0.5]

        all_fs_jnp = jnp.array(fs_full, dtype=jnp.int32)
        all_rs_jnp = jnp.array(rs_full, dtype=jnp.float32)  # [Pixel^0.5]
        all_cs_jnp = jnp.array(cs_full, dtype=jnp.float32)  # [Pixel^0.5]

        from tqdm import tqdm
        with tqdm(total=N_spots, desc="Sparse Laue Integration", disable=not self.show_steps) as pbar:
            for i in range(0, N_spots, self.chunk_size):
                chunk_f = jnp.array(frames[i:i+self.chunk_size])
                chunk_r = jnp.array(rs_padded[i:i+self.chunk_size])  # [Pixel^0.5]
                chunk_c = jnp.array(cs_padded[i:i+self.chunk_size])  # [Pixel^0.5]

                patches, patches_bg, r_starts, c_starts = extract_patches(img_jax_padded, bg_jax_padded, chunk_f, chunk_r, chunk_c)

                res = solve_patches(patches, patches_bg, chunk_f, chunk_r, chunk_c, r_starts, c_starts,
                                    all_fs_jnp, all_rs_jnp, all_cs_jnp, var_us, var_vs, cov_uvs)
                res.block_until_ready()

                res_cpu = np.array(res)
                res_cpu[:, 1] = res_cpu[:, 1] + r_starts - PAD  # [Pixel^0.5]
                res_cpu[:, 2] = res_cpu[:, 2] + c_starts - PAD  # [Pixel^0.5]

                refined_peaks.append(res_cpu)
                pbar.update(len(chunk_f))

        if len(refined_peaks) == 0:
            return np.empty((0, 4))

        return np.vstack(refined_peaks)

# =====================================================================
# API WRAPPER FOR BACKWARD COMPATIBILITY
# =====================================================================

from typing import Dict, List
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
import os

def _render_and_save_rbf_plot(args):
    """Standalone plotting function for multiprocessing."""
    (image_raw, physical_bank, run_id, img_key, img_rs, img_cs,
     ref_rs, ref_cs, img_intensities, img_var_us, img_var_vs, img_cov_uvs) = args

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.lines as mlines
    import numpy as np

    # Force non-interactive backend for thread safety
    if plt.get_backend().lower() != "agg":
        plt.switch_backend("Agg")

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(image_raw, cmap="viridis", origin="lower")

    ax.set_xlim(0, image_raw.shape[1])
    ax.set_ylim(0, image_raw.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.02, 0.98, f"Bank {physical_bank} (Run {run_id})",
            transform=ax.transAxes, ha='left', va='top', fontsize=16,
            color='white', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # Plot initial predictions
    ax.scatter(img_cs, img_rs, marker='+', color='cyan', s=60, alpha=0.5, label="Predicted Center")

    # Plot the analytically projected ellipses for active peaks
    for s_idx, (cx, cy, intensity) in enumerate(zip(ref_cs, ref_rs, img_intensities)):
        if intensity > 0:
            var_u = img_var_us[s_idx]
            var_v = img_var_vs[s_idx]
            cov_uv = img_cov_uvs[s_idx]

            # 1. Eigenvalue decomposition for major/minor axes
            diff = var_u - var_v
            sum_uv = var_u + var_v
            sq = np.sqrt(diff**2 + 4.0 * cov_uv**2)

            lambda_1 = (sum_uv + sq) / 2.0  # Major axis variance
            lambda_2 = (sum_uv - sq) / 2.0  # Minor axis variance

            sigma_1 = np.sqrt(max(lambda_1, 1e-6))
            sigma_2 = np.sqrt(max(lambda_2, 1e-6))

            # 2. Eigenvector angle calculation
            # arctan2(Y, X). X is along cols (u), Y is along rows (v)
            phi = 0.5 * np.arctan2(2.0 * cov_uv, diff)
            angle_deg = np.degrees(phi)

            # Matplotlib Ellipse width/height are full diameters (4 * sigma)
            w = 4.0 * sigma_1
            h = 4.0 * sigma_2

            patch = Ellipse((cx, cy), width=w, height=h, angle=angle_deg,
                            edgecolor='red', facecolor='none', lw=1.5, alpha=0.8)
            ax.add_patch(patch)

    # Clean legend
    handles = [
        mlines.Line2D([], [], color='cyan', marker='+', ls='', markersize=10, label='Prediction'),
        mlines.Line2D([], [], color='red', marker='o', fillstyle='none', ls='', markersize=10, label='Projected 3D Tensor')
    ]
    ax.legend(handles=handles, loc='lower center', ncol=2, frameon=True, fontsize=12,
              facecolor='black', edgecolor='none', labelcolor='white')
    
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_name = f"rbf_viz_bank{physical_bank}_run{run_id}_img{img_key}.png"
    fig.savefig(out_name, bbox_inches="tight", dpi=150, pad_inches=0.2)
    plt.close(fig)

def integrate_peaks_rbf_ssn(peak_dict: Dict, peaks_obj, sigmas: List[float],
                            alpha: float, gamma: float, max_peaks: int, show_progress: bool,
                            all_R: np.ndarray = None, sample_offset: np.ndarray = None,
                            nominal_sigma: float = 2.0, anisotropic: bool = False,
                            border_width: int = 0, chunk_size: int = 1024,
                            create_visualizations: bool = False, max_workers: int = None):
    """
    Args:
        peak_dict: Dictionary containing peak arrays
        peaks_obj: Instrument mapping object
        sigmas: List of unstretched peak radii
        nominal_sigma: Fallback sigma for crushed peaks
    Returns:
        res: RBFResult containing intensities and sigI
    """

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
        alpha=alpha,  min_sigma=min(sigmas), max_sigma=max(sigmas), gamma=gamma,
        loss='poisson', border_width=border_width, nominal_sigma=nominal_sigma,
        anisotropic=anisotropic, chunk_size=chunk_size,
    )
    # Ensure the solver dictionary perfectly matches the provided list
    integrator.candidate_sigmas = jnp.array(sigmas, dtype=jnp.float32)
    integrator.show_steps = show_progress

    # --- PHASE 1: GATHER AND BATCH ---
    images_list = []
    all_frames = []
    all_rs, all_cs = [], []
    meta_h, meta_k, meta_l, meta_wl = [], [], [], []
    meta_keys = []

    frame_counter = 0
    img_keys_ordered = sorted(peak_dict.keys())

    for img_key in tqdm(img_keys_ordered, disable=not show_progress, desc="Batching Images"):
        p_data = peak_dict[img_key]
        i_arr, j_arr, h_arr, k_arr, l_arr, wl_arr = p_data

        initial_peaks_count = len(i_arr)
        if initial_peaks_count == 0:
            continue

        hkl_sq = h_arr**2 + k_arr**2 + l_arr**2
        unique_peaks = {}

        # Exact crystallographic harmonic deduplication
        for idx in range(initial_peaks_count):
            h, k, l = int(h_arr[idx]), int(k_arr[idx]), int(l_arr[idx])

            if h == 0 and k == 0 and l == 0:
                continue

            g = np.gcd.reduce([abs(h), abs(k), abs(l)])
            fund_hkl = (h//g, k//g, l//g)

            if fund_hkl not in unique_peaks or hkl_sq[idx] < unique_peaks[fund_hkl]['hkl_sq']:
                unique_peaks[fund_hkl] = {'idx': idx, 'hkl_sq': hkl_sq[idx]}

        keep_indices = sorted([v['idx'] for v in unique_peaks.values()])
        actual_peaks_count = len(keep_indices)

        det = peaks_obj.get_detector(img_key)
        run_id = peaks_obj.image.get_run_id(img_key)

        if all_R is not None and all_R.ndim == 3:
            current_R_val = all_R[run_id] if run_id < len(all_R) else all_R[0]
        else:
            current_R_val = all_R

        s_lab = current_R_val @ sample_offset if current_R_val is not None else sample_offset

        batch_rs = np.array([i_arr[idx] for idx in keep_indices])
        batch_cs = np.array([j_arr[idx] for idx in keep_indices])

        xyz_lab = det.pixel_to_lab(batch_rs, batch_cs)
        bank_tt, bank_az = det.pixel_to_angles(batch_rs, batch_cs, sample_offset=s_lab)

        if show_progress and initial_peaks_count != actual_peaks_count:
            physical_b = peaks_obj.image.bank_mapping.get(img_key, img_key)
            comp_ratio = (1.0 - actual_peaks_count / initial_peaks_count) * 100
            tqdm.write(f"Bank {physical_b} [Run {run_id}]: "
                       f"Harmonics Filtered {initial_peaks_count} -> {actual_peaks_count} "
                       f"({comp_ratio:.1f}% compression)")

        image_raw = np.nan_to_num(peaks_obj.image.ims[img_key], nan=0.0, posinf=0.0, neginf=0.0)
        images_list.append(image_raw)

        for idx in keep_indices:
            all_frames.append(frame_counter)
            all_rs.append(i_arr[idx])
            all_cs.append(j_arr[idx])
            meta_h.append(h_arr[idx])
            meta_k.append(k_arr[idx])
            meta_l.append(l_arr[idx])
            meta_wl.append(wl_arr[idx])
            meta_keys.append(img_key)

        frame_counter += 1

    if not images_list:
        return res

    images_batch = np.stack(images_list)
    frames = np.array(all_frames, dtype=int)

    # --- PHASE 1.5: GLOBAL TENSOR OPTIMIZATION ---
    B, H, W = images_batch.shape
    bw = max(border_width, 5)

    # 1. Build the exact P_true matrices for all peaks
    all_P_mats = []
    all_distances = []
    
    for idx, img_key in enumerate(meta_keys):
        det = peaks_obj.get_detector(img_key)
        run_id = frames[idx] 
        
        # Get sample offset
        if all_R is not None and all_R.ndim == 3:
            current_R = all_R[run_id] if run_id < len(all_R) else all_R[0]
        else:
            current_R = all_R
        s_lab = current_R @ sample_offset if current_R is not None else sample_offset
        
        # 1a. Ray vector (k_f)
        pixel_xyz = det.pixel_to_lab(all_rs[idx], all_cs[idx])
        k_f = pixel_xyz - s_lab
        distance = np.linalg.norm(k_f)
        all_distances.append(distance)
        k_f_hat = k_f / distance
        
        # 1b. Detector Normal (n)
        n_det = np.cross(det.uhat, det.vhat)
        n_det_hat = n_det / np.linalg.norm(n_det)
        
        # 1c. Orthogonal Projection
        P_ortho = np.vstack([det.uhat, det.vhat]) # (2, 3)
        
        # 1d. The Central Projection Skew Matrix
        cos_alpha = np.dot(k_f_hat, n_det_hat)
        # Protect against rays parallel to the detector (edge cases)
        cos_alpha = np.sign(cos_alpha) * max(abs(cos_alpha), 0.01) 
        
        Skew = np.eye(3) - np.outer(k_f_hat, n_det_hat) / cos_alpha
        
        # 1e. The final P matrix for this specific peak
        P_true = P_ortho @ Skew
        all_P_mats.append(P_true)
        
    all_P_mats = np.array(all_P_mats)
    all_distances = np.array(all_distances)

    # 2. Identify the Top 500 strongest peaks using a rapid 3x3 pixel sum proxy
    P_proxy = 3
    pad_images = np.pad(images_batch, ((0,0), (P_proxy, P_proxy), (P_proxy, P_proxy)), mode='reflect')
    proxy_intensities = []
    for f, r, c in zip(frames, all_rs, all_cs):
        ri, ci = int(round(r)) + P_proxy, int(round(c)) + P_proxy
        proxy_intensities.append(np.sum(pad_images[f, ri-1:ri+2, ci-1:ci+2]))
    
    top_indices = np.argsort(proxy_intensities)[-500:]
    
    # 3. Extract exact patches for the optimizer
    opt_P = 15
    opt_half = opt_P // 2
    opt_patches, opt_bgs, opt_drs, opt_dcs, opt_Pmats, opt_dists = [], [], [], [], [], []
    for idx in top_indices:
        f, r, c = frames[idx], all_rs[idx], all_cs[idx]
        ri, ci = int(round(r)), int(round(c))
        
        if bw < ri < H - bw and bw < ci < W - bw:
            patch = images_batch[f, ri-opt_half:ri+opt_half+1, ci-opt_half:ci+opt_half+1]
            opt_patches.append(patch)
            opt_bgs.append(np.median(patch)) 
            opt_drs.append(r - (ri - opt_half))
            opt_dcs.append(c - (ci - opt_half))
            opt_Pmats.append(all_P_mats[idx])
            opt_dists.append(all_distances[idx])

    # 4. Run the Global Optimizer
    res_x = optimize_global_crystal(
        jnp.array(opt_patches), jnp.array(opt_bgs), 
        jnp.array(opt_drs), jnp.array(opt_dcs),
        jnp.array(opt_Pmats), jnp.array(opt_dists)
    )

    # 5. Project the EXACT 2D footprints for ALL peaks
    Sigma_shape_jnp = build_3d_cov(jnp.array(res_x[:6]))
    Sigma_eta_jnp = jnp.eye(3) * (abs(res_x[6])**2 + 1e-12)

    @jit
    def project_all_shapes(P_mats, dists):
        def project_one(P, D_i):
            # The convolution of shape and mosaicity!
            Sigma_total = Sigma_shape_jnp + (D_i**2) * Sigma_eta_jnp
            return P @ Sigma_total @ P.T
        return vmap(project_one)(P_mats, dists)

    all_Sigma_2D = project_all_shapes(jnp.array(all_P_mats), jnp.array(all_distances))
    all_var_u = all_Sigma_2D[:, 0, 0]
    all_var_v = all_Sigma_2D[:, 1, 1]
    all_cov_uv = all_Sigma_2D[:, 0, 1]

    # --- PHASE 2: GPU INTEGRATION ---
    integrated_results = integrator.integrate_reflections(
        images_batch, frames, all_rs, all_cs,
        var_us=all_var_u, var_vs=all_var_v, cov_uvs=all_cov_uv
    )

    # --- PHASE 3: GEOMETRY AND METADATA MAPPING ---
    results_by_img = defaultdict(list)
    for i in range(len(meta_keys)):
        results_by_img[meta_keys[i]].append(i)

    plot_tasks = []

    for img_key, indices in tqdm(results_by_img.items(), disable=not show_progress, desc="Mapping Geometry"):
        physical_bank = peaks_obj.image.bank_mapping.get(img_key, img_key)
        det = peaks_obj.get_detector(img_key)
        run_id = peaks_obj.image.get_run_id(img_key)

        image_raw = np.nan_to_num(peaks_obj.image.ims[img_key], nan=0.0, posinf=0.0, neginf=0.0)
        H, W = image_raw.shape
        bw = border_width

        if all_R is not None and all_R.ndim == 3:
            current_R_val = all_R[run_id] if run_id < len(all_R) else all_R[0]
        else:
            current_R_val = all_R

        s_lab = current_R_val @ sample_offset if current_R_val is not None else sample_offset

        img_rs = [all_rs[idx] for idx in indices]
        img_cs = [all_cs[idx] for idx in indices]
        bank_tt, bank_az = det.pixel_to_angles(np.array(img_rs), np.array(img_cs), sample_offset=s_lab)

        # --- EDGE MASKING FIX ---
        valid_global_indices = []
        valid_local_indices = []

        for local_idx, global_idx in enumerate(indices):
            r = float(integrated_results[global_idx, 1])
            c = float(integrated_results[global_idx, 2])
            if (bw <= r < H - bw) and (bw <= c < W - bw):
                valid_global_indices.append(global_idx)
                valid_local_indices.append(local_idx)

        if not valid_global_indices:
            continue

        img_intensities = [float(integrated_results[idx, 0]) for idx in valid_global_indices]
        img_spatial_sigmas = [float(integrated_results[idx, 3]) for idx in valid_global_indices]
        img_sigI = [float(integrated_results[idx, 4]) for idx in valid_global_indices]

        res.intensity += img_intensities
        res.sigma += img_sigI

        for local_idx, global_idx in zip(valid_local_indices, valid_global_indices):
            res.h.append(meta_h[global_idx])
            res.k.append(meta_k[global_idx])
            res.l.append(meta_l[global_idx])
            res.wavelength.append(meta_wl[global_idx])
            res.tt.append(float(bank_tt[local_idx]))
            res.az.append(float(bank_az[local_idx]))
            res.run_id.append(run_id)
            res.bank.append(physical_bank)

        # Defer plotting by storing the necessary static data
        if create_visualizations:
            filt_img_rs = [img_rs[i] for i in valid_local_indices]
            filt_img_cs = [img_cs[i] for i in valid_local_indices]
            filt_ref_rs = [float(integrated_results[idx, 1]) for idx in valid_global_indices]
            filt_ref_cs = [float(integrated_results[idx, 2]) for idx in valid_global_indices]

            # Extract the exact projected 2D covariance components for the valid peaks
            filt_var_us = [float(all_var_u[idx]) for idx in valid_global_indices]
            filt_var_vs = [float(all_var_v[idx]) for idx in valid_global_indices]
            filt_cov_uvs = [float(all_cov_uv[idx]) for idx in valid_global_indices]

            plot_tasks.append((
                image_raw, physical_bank, run_id, img_key,
                filt_img_rs, filt_img_cs, filt_ref_rs, filt_ref_cs,
                img_intensities, filt_var_us, filt_var_vs, filt_cov_uvs
            ))

    # --- PHASE 4: PARALLEL VISUALIZATION ---
    if create_visualizations and plot_tasks:
        # Render the global diagnostic plot in the main thread
        if max_workers is None:
            max_workers = os.cpu_count()

        max_workers = min(max_workers, len(plot_tasks))

        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(_render_and_save_rbf_plot, plot_tasks),
                total=len(plot_tasks),
                desc="Rendering Plots",
                disable=not show_progress
            ))

    return res
