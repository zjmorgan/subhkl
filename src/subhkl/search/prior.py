import numpy as np
import scipy.ndimage
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation

from subhkl.config import beamlines
from subhkl.optimization import rotation_matrix_from_rodrigues_jax

from functools import partial

@jax.jit
def batch_busing_levy(v_lab, u_calc):
    """
    Vectorized Busing-Levy Orthogonal Triad construction.
    Directly computes the Sample -> Crystal orientation matrix (U) from two matched vector pairs.
    
    v_lab: (batch, 2, 3) - Empirical Zone Axes (Lab Frame)
    u_calc: (batch, 2, 3) - Theoretical Zone Axes (Crystal Frame)
    """
    v1 = v_lab[:, 0, :]
    v2 = v_lab[:, 1, :]
    u1 = u_calc[:, 0, :]
    u2 = u_calc[:, 1, :]
    
    # --- Lab Frame Triads ---
    t1_lab = v1
    t2_lab_unnorm = jnp.cross(v1, v2)
    t2_lab = t2_lab_unnorm / (jnp.linalg.norm(t2_lab_unnorm, axis=1, keepdims=True) + 1e-9)
    t3_lab = jnp.cross(t1_lab, t2_lab)
    
    # --- Crystal Frame Triads ---
    t1_c = u1
    t2_c_unnorm = jnp.cross(u1, u2)
    t2_c = t2_c_unnorm / (jnp.linalg.norm(t2_c_unnorm, axis=1, keepdims=True) + 1e-9)
    t3_c = jnp.cross(t1_c, t2_c)
    
    # --- 3x3 Basis Matrices ---
    T_lab = jnp.stack([t1_lab, t2_lab, t3_lab], axis=-1)
    T_c = jnp.stack([t1_c, t2_c, t3_c], axis=-1)
    
    # Calculate U_sc (Sample -> Crystal)
    # T_c = U_sc @ T_lab  => U_sc = T_c @ T_lab^T
    U_sc = jnp.einsum('bij,bkj->bik', T_c, T_lab)
    
    return U_sc

class HoughPrior:
    """
    A robust prior generator for Laue crystallography.
    Fuses Multi-Run Combinatorial Hough Transforms with Busing-Levy Matrix Extrapolation
    and an exact physical forward-model filter to seed global evolutionary optimization.
    """
    def __init__(self, B_mat, R_stack, ki_vec=np.array([0.0, 0.0, 1.0])):
        self.B_mat = B_mat
        self.R_stack = R_stack
        self.ki_vec = ki_vec

    def compute_hough_accumulator(self, q_rays, grid_resolution=1024, min_pairs=3, n_hough=15,
                                  plot_filename=None, border_frac=0.1):
        """Executes the 3D Combinatorial Hough Transform using Lambert Azimuthal projection."""
        N = len(q_rays)
        if N < 2:
            return np.zeros((0, 3)), np.zeros(0)
            
        idx_i, idx_j = np.triu_indices(N, k=1)
        n_cands = np.cross(q_rays[idx_i], q_rays[idx_j])
        norms = np.linalg.norm(n_cands, axis=1)
        
        valid = norms > 1e-3
        n_cands = n_cands[valid] / norms[valid, None]
        
        # Upper hemisphere projection
        n_cands[n_cands[:, 2] < 0] *= -1
        
        # Lambert Projection
        factor = np.sqrt(2.0 / (1.0 + n_cands[:, 2]))
        x_proj = factor * n_cands[:, 0]
        y_proj = factor * n_cands[:, 1]
        
        limit = np.sqrt(2.0)
        H, xedges, yedges = np.histogram2d(
            x_proj, y_proj, bins=grid_resolution, range=[[-limit, limit], [-limit, limit]]
        )
       
        # Scale blur radius to bridge physical goniometer wobble
        blur_sigma = grid_resolution / 1024.0
        H_smooth = scipy.ndimage.gaussian_filter(H, sigma=blur_sigma)
        local_max = scipy.ndimage.maximum_filter(H_smooth, size=3) == H_smooth
        peak_mask = local_max & (H_smooth >= min_pairs)

        H, W = peak_mask.shape
        margin_r = int(H * border_frac)
        margin_c = int(W * border_frac)
        peak_mask[:margin_r, :] = False
        peak_mask[-margin_r:, :] = False
        peak_mask[:, :margin_c] = False
        peak_mask[:, -margin_c:] = False

        row_idx, col_idx = np.where(peak_mask)
        weights = H_smooth[row_idx, col_idx]
        
        if len(weights) == 0:
            return np.zeros((0, 3)), np.zeros(0)
            
        sort_order = np.argsort(weights)[::-1]

        # --- SPATIAL DIVERSITY NMS (min_distance) ---
        min_dist_deg = 10.0
        min_dist_px = int((min_dist_deg / 90.0) * (grid_resolution / 2.0))
        min_dist_sq = min_dist_px**2

        keep_rows, keep_cols, keep_weights = [], [], []

        for idx in sort_order:
            r, c, w = row_idx[idx], col_idx[idx], weights[idx]

            if len(keep_rows) > 0:
                dist_sq = (np.array(keep_rows) - r)**2 + (np.array(keep_cols) - c)**2
                if np.min(dist_sq) < min_dist_sq:
                    continue  

            keep_rows.append(r)
            keep_cols.append(c)
            keep_weights.append(w)

            if len(keep_rows) >= n_hough:
                break

        row_idx = np.array(keep_rows)
        col_idx = np.array(keep_cols)
        weights_obs = np.array(keep_weights)
        
        dx, dy = xedges[1] - xedges[0], yedges[1] - yedges[0]
        
        # --- THE SUB-PIXEL FIX: Continuous Center of Mass ---
        exact_x, exact_y = [], []
        
        for r, c in zip(row_idx, col_idx):
            x_min = xedges[max(0, r-1)]
            x_max = xedges[min(len(xedges)-1, r+2)]
            y_min = yedges[max(0, c-1)]
            y_max = yedges[min(len(yedges)-1, c+2)]
            
            mask = (x_proj >= x_min) & (x_proj <= x_max) & \
                   (y_proj >= y_min) & (y_proj <= y_max)
            
            x_local = x_proj[mask]
            y_local = y_proj[mask]
            
            if len(x_local) > 0:
                exact_x.append(np.mean(x_local))
                exact_y.append(np.mean(y_local))
            else:
                exact_x.append(xedges[r] + dx/2.0)
                exact_y.append(yedges[c] + dy/2.0)
                
        exact_x = np.array(exact_x)
        exact_y = np.array(exact_y)
        
        r_sq = exact_x**2 + exact_y**2
        factor_inv = np.sqrt(np.clip(1.0 - r_sq / 4.0, 0, 1))
        n_obs = np.column_stack([exact_x * factor_inv, exact_y * factor_inv, 1.0 - r_sq / 2.0])
        
        n_obs /= np.linalg.norm(n_obs, axis=1, keepdims=True)
        
        if plot_filename:
            self._plot_hough(x_proj, y_proj, n_obs, plot_filename, bins=grid_resolution)
            
        return n_obs, weights_obs

    def _plot_hough(self, x_proj, y_proj, n_obs, filename, bins=256):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))

        # Set bins, range, and get counts (H)
        h, xedges, yedges = np.histogram2d(x_proj, y_proj, bins=bins)

        # This ensures no zeros, making log normalization valid
        h_regularized = h + 1

        # 4. Plot using pcolormesh for custom normalization
        plt.pcolormesh(xedges, yedges, h_regularized.T, norm=matplotlib.colors.LogNorm(), cmap='viridis')
        plt.xlim(-1.414, 1.414)
        plt.ylim(-1.414, 1.414)
        plt.colorbar(label='Cross Product Intersections')

        if len(n_obs) > 0:
            f_obs = np.sqrt(2.0 / (1.0 + n_obs[:, 2]))
            x_obs, y_obs = f_obs * n_obs[:, 0], f_obs * n_obs[:, 1]
            plt.scatter(x_obs, y_obs, facecolors='none', edgecolors='red', marker='o', s=200, linewidths=2.5, label='Extracted Normals')
            for idx, (x, y) in enumerate(zip(x_obs, y_obs)):
                plt.text(x + 0.05, y + 0.05, str(idx), color='white', fontsize=12, fontweight='bold')

        plt.title("Lambert Azimuthal Accumulator (Sample Frame)")
        plt.legend(loc='upper right')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_theoretical_zones(self, L_max=None, top_k=600, max_uvw=25):
        """
        Generates the fundamental Zone Axes directly from Cartesian lattice translations
        using an isotropic, dynamically scaled spherical bounding box in real space.
        """
        # A_mat columns are the real-space basis vectors a, b, c
        A_mat = np.linalg.inv(self.B_mat).T
        A_norms = np.linalg.norm(A_mat, axis=0) # Lengths of a, b, c in Angstroms
        
        # Dynamic default: 1.5x the longest unit cell axis to ensure diagonals are captured
        if L_max is None:
            L_max = 1.5 * np.max(A_norms)
            
        # Calculate exact integer steps needed to reach the spherical boundary L_max
        u_max, v_max, w_max = np.ceil(L_max / A_norms).astype(int)

        # Hard cap to prevent memory explosion on pathological cells
        u_max, v_max, w_max = min(u_max, max_uvw), min(v_max, max_uvw), min(w_max, max_uvw)
        
        print(f"  -> Dynamic Real-Space Grid (L_max={L_max}Å): u(±{u_max}), v(±{v_max}), w(±{w_max})")
        
        u_idx, v_idx, w_idx = np.meshgrid(
            np.arange(-u_max, u_max + 1),
            np.arange(-v_max, v_max + 1),
            np.arange(-w_max, w_max + 1),
            indexing='ij'
        )
        uvw = np.vstack([u_idx.flatten(), v_idx.flatten(), w_idx.flatten()])
        uvw = uvw[:, np.any(uvw != 0, axis=0)] # Remove origin (0,0,0)
        
        # Map integer indices to Cartesian real-space coordinates
        r_cart = np.dot(A_mat, uvw)
        norms = np.linalg.norm(r_cart, axis=0)
        
        # Rigorously crop off the "corners" of the bounding box to make it a perfect sphere
        valid_sphere = norms <= L_max
        r_cart = r_cart[:, valid_sphere]
        norms = norms[valid_sphere]
        
        # The shortest physical translation vectors define the highest-density, most visible Zone Axes
        sort_idx = np.argsort(norms)
        limit = min(top_k, len(norms))
        
        n_calc_raw = r_cart[:, sort_idx[:limit]] / norms[sort_idx[:limit]]
        
        # Map to upper hemisphere and round to handle numeric floating-point fuzz
        n_calc_raw[:, n_calc_raw[2] < 0] *= -1
        n_calc_raw = np.round(n_calc_raw, 5)
        
        # Deduplicate parallel vectors
        return jnp.array(np.unique(n_calc_raw, axis=1).T)

    def solve_permutations(self, n_obs, weights_obs, n_calc, q_hat_sample, space_group="P 1",
                           angle_tol_deg=0.25, scoring_tol_deg=0.25, d_min=2.0, max_hkl=35):
        """
        Lifts the permutation problem from pairs to 3-cliques (triplets) to robustly reject bad geometry,
        then employs the Normalized Busing-Levy matrix extrapolation on the GPU.
        """
        n_obs_np = np.array(n_obs)
        n_calc_np = np.array(n_calc)
        
        S_obs = np.abs(np.dot(n_obs_np, n_obs_np.T))
        S_calc = np.abs(np.dot(n_calc_np, n_calc_np.T))
        
        ang_obs = np.degrees(np.arccos(np.clip(S_obs, -1.0, 1.0)))
        ang_calc = np.degrees(np.arccos(np.clip(S_calc, -1.0, 1.0)))
        
        valid_permutations = []
        num_obs = min(len(n_obs_np), 15)
        strict_tol = angle_tol_deg 
        
        print(f"  -> Scanning theoretical lattice for exact 3D triplet matches (Tol: {strict_tol} deg)...")
        
        for i in range(num_obs):
            for j in range(i + 1, num_obs):
                for k in range(j + 1, num_obs):
                    a_ij, a_ik, a_jk = ang_obs[i, j], ang_obs[i, k], ang_obs[j, k]
                    M_AB = np.abs(ang_calc - a_ij) <= strict_tol
                    M_AC = np.abs(ang_calc - a_ik) <= strict_tol
                    M_BC = np.abs(ang_calc - a_jk) <= strict_tol
                    
                    A_idx, B_idx = np.where(M_AB)
                    for A, B in zip(A_idx, B_idx):
                        if A == B: continue
                        valid_C = np.where(M_AC[A, :] & M_BC[B, :])[0]
                        for C in valid_C:
                            if C == A or C == B: continue
                            valid_permutations.append(([i, j, k], [A, B, C]))
                            
        print(f"  -> Triplet Graph Search yielded exactly {len(valid_permutations)} physical matches.")
        if len(valid_permutations) == 0:
            return None, None
            
        print(f"  -> Vectorizing {len(valid_permutations)} physical matches onto GPU...")
        
        obs_indices = np.array([p[0] for p in valid_permutations], dtype=np.int32)
        calc_indices = np.array([p[1] for p in valid_permutations], dtype=np.int32)
        v_base = jnp.array(n_obs_np)[obs_indices]      
        u_base = jnp.array(n_calc_np)[calc_indices]    
        
        import itertools
        signs = jnp.array(list(itertools.product([1, -1], repeat=3)), dtype=v_base.dtype)
        v_expanded = v_base[:, None, :, :] * signs[None, :, :, None]
        
        v_hyp_all = v_expanded.reshape(-1, 3, 3)  
        u_hyp_all = jnp.repeat(u_base, 8, axis=0) 
        
        print(f"  -> Polishing {len(v_hyp_all)} geometric hypotheses using Angular Laue Indexing...")
        
        B_norms = np.linalg.norm(self.B_mat, axis=0)
        h_max, k_max, l_max = np.ceil(1.0 / (d_min * B_norms)).astype(int)
        h_max, k_max, l_max = min(h_max, max_hkl), min(k_max, max_hkl), min(l_max, max_hkl)
        
        print(f"  -> Dynamic Reciprocal Grid Limits (d_min={d_min}Å): h(±{h_max}), k(±{k_max}), l(±{l_max})")

        u, v, w = np.meshgrid(
            np.arange(-h_max, h_max + 1), 
            np.arange(-k_max, k_max + 1), 
            np.arange(-l_max, l_max + 1), 
            indexing='ij'
        )
        hkls = np.vstack([u.flatten(), v.flatten(), w.flatten()])
        hkls = hkls[:, np.any(hkls != 0, axis=0)] 

        # --- EXACT SYMMETRY MASK ---
        from subhkl.core.spacegroup import generate_hkl_mask
        print(f"  -> Applying exact systematic absences for Space Group: {space_group}")
        mask_3d = generate_hkl_mask(h_max, k_max, l_max, space_group)
        
        idx_h = (hkls[0] + h_max).astype(int)
        idx_k = (hkls[1] + k_max).astype(int)
        idx_l = (hkls[2] + l_max).astype(int)
        
        valid_mask = mask_3d[idx_h, idx_k, idx_l]
        hkls = hkls[:, valid_mask]
        print(f"  -> Retained {np.sum(valid_mask)}/{len(valid_mask)} mathematically allowed reflections.")
        
        q_theor = np.dot(self.B_mat, hkls).T
        q_theor_norms = np.linalg.norm(q_theor, axis=1, keepdims=True)
        q_theor_hat = q_theor / q_theor_norms
        
        q_theor_hemi = np.where(q_theor_hat[:, 2:3] < 0, -q_theor_hat, q_theor_hat)
        q_theor_hemi = np.round(q_theor_hemi, 5)
        q_theor_unique = np.unique(q_theor_hemi, axis=0)
        
        q_theor_jax = jnp.array(q_theor_unique)
        q_sample_jax = jnp.array(q_hat_sample)
        
        cos_tol = jnp.cos(jnp.radians(scoring_tol_deg))

        @jax.jit
        def evaluate_chunk(v_hyp, u_hyp):
            U_hyp = batch_busing_levy(v_hyp[:, :2, :], u_hyp[:, :2, :])
            q_cryst_hat = jnp.einsum('bij,nj->bni', U_hyp, q_sample_jax)
            dots = jnp.einsum('bni,mi->bnm', q_cryst_hat, q_theor_jax)
            max_dots = jnp.max(jnp.abs(dots), axis=2)
            scores = jnp.sum(max_dots >= cos_tol, axis=1)
            return U_hyp, scores

        all_U_final, all_s_final = [], []
        bytes_per_hyp = len(q_hat_sample) * len(q_theor_unique) * 4
        target_vram_bytes = 10 * 1024**3
        chunk_size = max(1, int(target_vram_bytes / bytes_per_hyp))
        
        for start in range(0, len(v_hyp_all), chunk_size):
            end = start + chunk_size
            U_f, s_f = evaluate_chunk(v_hyp_all[start:end], u_hyp_all[start:end])
            all_U_final.append(U_f)
            all_s_final.append(s_f)
            
        U_gpu = jnp.concatenate(all_U_final)
        scores_gpu = jnp.concatenate(all_s_final)
        
        num_rays = len(q_hat_sample)
        num_lines = len(q_theor_unique)
        fraction_of_sphere_per_line = 1.0 - float(cos_tol)
        random_hit_prob = num_lines * fraction_of_sphere_per_line
        expected_random_hits = num_rays * random_hit_prob
        
        min_inliers = max(5, int(expected_random_hits + 5 * np.sqrt(expected_random_hits)))
        
        valid_mask = scores_gpu >= min_inliers
        U_valid = np.array(U_gpu[valid_mask])
        scores_valid = np.array(scores_gpu[valid_mask])

        if len(U_valid) == 0:
            return None, None

        rots = Rotation.from_matrix(U_valid).as_quat() 
        quats = np.column_stack([rots[:, 3], rots[:, 0], rots[:, 1], rots[:, 2]]) 
        quats_inv = quats * np.array([1.0, -1.0, -1.0, -1.0])

        sort_idx = np.argsort(scores_valid)[::-1]
        quats_sorted = quats_inv[sort_idx]
        scores_sorted = scores_valid[sort_idx]
        
        quats_hemi = np.where(quats_sorted[:, 0:1] < 0, -quats_sorted, quats_sorted)
        quats_rounded = np.round(quats_hemi, decimals=2)
        
        _, unique_indices = np.unique(quats_rounded, axis=0, return_index=True)
        q_unique = quats_sorted[unique_indices]
        s_unique = scores_sorted[unique_indices]
        
        final_sort = np.argsort(s_unique)[::-1]
        return jnp.array(q_unique[final_sort]), jnp.array(s_unique[final_sort])

    def physics_filter(self, prior_quats, objective_function, batch_size=4096, z_score_threshold=4.0):
        """Evaluates all macroscopic seeds against the strict physical forward-model to gather statistics."""
        if prior_quats is None or len(prior_quats) == 0:
            return None

        print("\n[Prior Validation] Computing Random Orientation Baseline...")
        rng = jax.random.PRNGKey(42)
        rand_q = jax.random.normal(rng, (4096, 4))
        rand_q = rand_q / jnp.linalg.norm(rand_q, axis=1, keepdims=True)
        rand_rots = jax.vmap(quaternion_to_rodrigues)(rand_q)

        import tqdm
        rand_losses = []
        for i in tqdm.tqdm(range(0, len(rand_rots), batch_size), desc="Forward Model Filter (random)"):
            batch = rand_rots[i:i+batch_size]
            losses = np.array(-objective_function(batch))
            rand_losses.append(losses)
        rand_losses = np.concatenate(rand_losses)

        r_mean = np.mean(rand_losses)
        r_std = np.std(rand_losses)
        r_max = np.max(rand_losses)

        print(f"  -> Random Background (N=4096) | Mean: {r_mean:.2f} | Max: {r_max:.2f} | Std: {r_std:.2f}")

        print(f"  -> Forward-Modeling all {len(prior_quats)} Prior seeds for statistical observation...")
        prior_rots = jax.vmap(quaternion_to_rodrigues)(prior_quats)

        physics_losses = []
        for i in tqdm.tqdm(range(0, len(prior_rots), batch_size), desc="Forward Model Filter (prior)"):
            batch = prior_rots[i:i+batch_size]
            losses = np.array(objective_function(batch))
            physics_losses.append(losses)

        physics_losses_np = np.concatenate(physics_losses)

        ranks = np.arange(len(physics_losses_np))

        physics_sort_idx = np.argsort(physics_losses_np)
        physics_ranks = np.empty_like(physics_sort_idx)
        physics_ranks[physics_sort_idx] = np.arange(len(physics_losses_np))

        import scipy.stats
        rho, p_val = scipy.stats.spearmanr(ranks, physics_ranks)

        print("\n[Observation Stats] RANSAC vs. Physical Ranking:")
        print(f"  -> Global Spearman Rho: {rho:.4f} (p={p_val:.2e})")

        top_1_dav_rank = physics_sort_idx[0]
        top_10_dav_ranks = physics_sort_idx[:10]
        top_100_dav_ranks = physics_sort_idx[:100]

        print(f"  -> Top 1 Physical Seed was RANSAC Rank: #{top_1_dav_rank}")
        print(f"  -> Top 10 Physical Seeds max RANSAC Rank: #{np.max(top_10_dav_ranks)}")
        print(f"  -> Top 100 Physical Seeds max RANSAC Rank: #{np.max(top_100_dav_ranks)}")

        for N in [1000, 10000, 50000]:
            if len(physics_losses_np) >= N:
                dav_top_n = set(ranks[:N])
                phys_top_n = set(physics_sort_idx[:N])
                overlap = len(dav_top_n.intersection(phys_top_n))
                print(f"  -> Target Overlap in Top {N}: {overlap}/{N} ({overlap/N*100:.1f}%)")

        best_loss = physics_losses_np[physics_sort_idx[0]]
        z_loss = (best_loss - r_mean) / (r_std + 1e-9)

        print(f"\n  -> RANSAC Top Score:       | Score: {best_loss:.2f} | Z-Score: {z_loss:.1f} sigma")
        print(f"  -> Top 5 Prior Scores: {physics_losses_np[physics_sort_idx[:5]]}")

        if z_loss >= z_loss_threshold:
            print(f"[Prior Validation] SUCCESS: Prior is statistically significant (+{z_loss:.1f} sigma). Proceeding to GA...")
            return prior_rots[physics_sort_idx]
        else:
            print(f"[Prior Validation] FAILED: Prior hallucinated (Z-Score {z_loss:.1f} < {z_loss_threshold}). Falling back to Uniform GA...")
            return None

@jax.jit
def jax_predict_reflections(U, B, hkl, R, ki_vec, sample_offset, center, uhat, vhat, width, height, m, n, wl_min, wl_max,
                            border_frac=0.1):
    """
    Pure JAX 3D-to-2D Laue Detector Projection.
    Strictly parallels predict_reflections_on_panel and lab_to_pixel.
    """
    # 1. Calculate Q vectors (Units: 1/d)
    q_local = jnp.dot(B, hkl)
    q_lab_direction = jnp.dot(R @ U, q_local)

    # Scale to 2pi/d to match predict_reflections_on_panel
    Q_vecs = 2.0 * jnp.pi * q_lab_direction

    ki_hat = ki_vec / (jnp.linalg.norm(ki_vec) + 1e-9)
    Q_sq = jnp.sum(Q_vecs**2, axis=0)
    Q_dot_ki = jnp.sum(Q_vecs * ki_hat[:, None], axis=0)

    # 2. Calculate Wavelength using the exact subhkl formula
    lamda = -4.0 * jnp.pi * Q_dot_ki / (Q_sq + 1e-9)

    valid_wl = (lamda >= wl_min) & (lamda <= wl_max)

    # 3. Calculate kf direction
    k_mag = 2.0 * jnp.pi / (lamda + 1e-9)
    kf_vecs = Q_vecs + k_mag * ki_hat[:, None]

    kf_hat = kf_vecs / jnp.linalg.norm(kf_vecs, axis=0)

    s_lab = jnp.dot(R, sample_offset)

    norm = jnp.cross(uhat, vhat)
    c_minus_s_dot_n = jnp.dot(center - s_lab, norm)
    d_dot_n = jnp.sum(norm[:, None] * kf_hat, axis=0)

    t = c_minus_s_dot_n / (d_dot_n + 1e-9)
    valid_t = t > 0

    intersection = s_lab[:, None] + t * kf_hat

    vec = intersection - center[:, None]

    dw = width / (m - 1)
    dh = height / (n - 1)

    dot_v = jnp.sum(vec * vhat[:, None], axis=0)
    row_f = dot_v / dh

    dot_u = jnp.sum(vec * uhat[:, None], axis=0)
    col_f = dot_u / dw

    margin_row = n * border_frac
    margin_col = m * border_frac

    valid_row = (row_f >= margin_row) & (row_f < n - margin_row)
    valid_col = (col_f >= margin_col) & (col_f < m - margin_col)

    valid = valid_wl & valid_t & valid_row & valid_col

    return row_f, col_f, lamda, valid

@jax.jit
def quaternion_to_rodrigues(q):
    """Maps a normalized quaternion [w, x, y, z] to a 3D Rodrigues vector safely."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # Safely handle the double-cover without zeroing out w=0
    sign = jnp.where(w < 0, -1.0, 1.0)
    w, x, y, z = w * sign, x * sign, y * sign, z * sign
    w_safe = jnp.clip(w, -1.0, 1.0)
    theta = 2.0 * jnp.arccos(w_safe)
    sin_half = jnp.sqrt(1.0 - w_safe**2)
    scale = jnp.where(sin_half > 1e-12, theta / sin_half, 2.0)

    return jnp.array([x * scale, y * scale, z * scale])

class ImageBasedObjective:
    def __init__(self, images_landscape, hkl_pool, B_mat, R_stack, wl_min, wl_max,
                 det_centers, uhats, vhats, widths, heights, ms, ns, ki_vec, sample_offset,
                 border_frac=0.1):
        self.images_landscape = jnp.array(images_landscape)
        self.hkl = jnp.array(hkl_pool)
        self.B_mat = jnp.array(B_mat)
        self.R_stack = jnp.array(R_stack)
        self.wl_min = wl_min
        self.wl_max = wl_max

        self.det_centers = jnp.array(det_centers)
        self.uhats = jnp.array(uhats)
        self.vhats = jnp.array(vhats)
        self.widths = jnp.array(widths)
        self.heights = jnp.array(heights)
        self.ms = jnp.array(ms)
        self.ns = jnp.array(ns)

        self.ki_vec = jnp.array(ki_vec)
        self.sample_offset = jnp.array(sample_offset)
        self.border_frac = border_frac

    @partial(jax.jit, static_argnames='self')
    def __call__(self, x):
        def evaluate_single_U(params):
            U = rotation_matrix_from_rodrigues_jax(params[:3])

            def scan_fn(carry, frame_data):
                R, center, uhat, vhat, w, h, m, n, img_land = frame_data
                row, col, lam, valid = jax_predict_reflections(
                    U, self.B_mat, self.hkl, R, self.ki_vec, self.sample_offset,
                    center, uhat, vhat, w, h, m, n, self.wl_min, self.wl_max, self.border_frac
                )
                coords = jnp.stack([row, col], axis=0)
                b_i = jax.scipy.ndimage.map_coordinates(img_land, coords, order=1, mode='constant', cval=0.0)

                return carry + jnp.sum((b_i**2) * valid), None

            frame_data = (self.R_stack, self.det_centers, self.uhats, self.vhats,
                          self.widths, self.heights, self.ms, self.ns, self.images_landscape)
            total_fitness, _ = jax.lax.scan(scan_fn, 0.0, frame_data)

            return -total_fitness

        return jax.vmap(evaluate_single_U)(x)
