import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_unrolled_detector(peaks, images, detectors, finder_peaks=None, out_name='unrolled_detector_peaks.png', instrument=None):
    fig, ax = plt.subplots(figsize=(16, 6))

    sample_offset = getattr(peaks, 'sample_offset', np.zeros(3))
    
    R_stack = getattr(peaks, 'R', None)
    if R_stack is not None:
        R_stack = np.array(R_stack)
        if R_stack.ndim == 2:
            R_stack = R_stack[np.newaxis, ...] 

    def get_s_lab_for_image(img_index_val):
        if R_stack is None:
            return sample_offset
            
        n_rotations = R_stack.shape[0]
        if n_rotations == 1:
            return R_stack[0] @ sample_offset
            
        idx = int(img_index_val)
        if idx < n_rotations:
            return R_stack[idx] @ sample_offset
                
        return R_stack[0] @ sample_offset

    # ==========================================
    # PRE-PASS: Find Wrap Bounds, Gaps & Radius
    # ==========================================
    wrapped_panels = set()
    panel_bounds = []
    radii = []

    for img_key, img in images.items():
        det = detectors.get(img_key)
        if det is None: continue
        
        s_lab = get_s_lab_for_image(img_key)
        
        c, r = np.meshgrid(np.arange(det.m + 1) - 0.5, np.arange(det.n + 1) - 0.5)
        xyz = det.pixel_to_lab(r, c)
        xyz = xyz - s_lab
        
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        
        # Strictly use absolute room frame roty
        roty = np.rad2deg(np.arctan2(X, Z))
        
        if np.ptp(roty) > 180:
            roty = np.where(roty < 0, roty + 360, roty)
            wrapped_panels.add(img_key)
            
        panel_bounds.append([np.min(roty), np.max(roty)])
        radii.append(np.mean(np.sqrt(X**2 + Z**2)))

    mean_radius = np.mean(radii) if radii else 1.0

    if panel_bounds:
        panel_bounds.sort(key=lambda x: x[0])
        merged = []
        for b in panel_bounds:
            if not merged:
                merged.append([b[0], b[1]])
            else:
                prev = merged[-1]
                if b[0] <= prev[1] + 5.0:
                    prev[1] = max(prev[1], b[1])
                else:
                    merged.append([b[0], b[1]])
    else:
        merged = []

    gaps = []
    visual_gap_size = 3.0 
    
    for i in range(len(merged) - 1):
        g_start = merged[i][1]
        g_end = merged[i+1][0]
        if g_end - g_start > 72:
            gaps.append((g_start, g_end))

    def compress_roty(r_orig):
        if not gaps: return r_orig
        r_out = np.copy(r_orig)
        for g_start, g_end in gaps:
            actual_gap = g_end - g_start
            shift = actual_gap - visual_gap_size
            r_out[r_orig >= g_end] -= shift
            inside = (r_orig > g_start) & (r_orig < g_end)
            if np.any(inside):
                fraction = (r_orig[inside] - g_start) / actual_gap
                r_out[inside] -= fraction * shift
        return r_out

    # ==========================================
    # PLOTTING
    # ==========================================
    global_vmax = 1
    if images:
        global_vmax = max(np.max(img) for img in images.values())
    global_norm = colors.LogNorm(vmin=1, vmax=global_vmax + 1)
    mesh_handle = None

    # 1. Plot the Images
    for img_key, img in images.items():
        det = detectors.get(img_key)
        if det is None: continue

        s_lab = get_s_lab_for_image(img_key)

        cols, rows = np.meshgrid(np.arange(det.m + 1) - 0.5, np.arange(det.n + 1) - 0.5)
        lab_xyz = det.pixel_to_lab(rows, cols)
        lab_xyz = lab_xyz - s_lab
        X, Y, Z = lab_xyz[..., 0], lab_xyz[..., 1], lab_xyz[..., 2]

        roty = np.rad2deg(np.arctan2(X, Z))
        if img_key in wrapped_panels:
            roty = np.where(roty < 0, roty + 360, roty)

        roty = compress_roty(roty)

        mesh = ax.pcolormesh(roty, Y, img, shading='auto', cmap='binary', norm=global_norm)
        if mesh_handle is None: mesh_handle = mesh

    # 2. Plot Finder Peaks
    if finder_peaks is not None:
        added_finder_label = False
        for img_key, coords in finder_peaks.items():
            if coords is None or len(coords) == 0: continue
            det = detectors.get(img_key)
            if det is None: continue

            s_lab = get_s_lab_for_image(img_key)

            coords = np.atleast_2d(coords)
            f_rows, f_cols = coords[:, 1], coords[:, 2]

            f_xyz = det.pixel_to_lab(f_rows, f_cols)
            f_xyz = f_xyz - s_lab
            if f_xyz.ndim == 1: f_xyz = f_xyz[np.newaxis, :]

            f_X, f_Y, f_Z = f_xyz[:, 0], f_xyz[:, 1], f_xyz[:, 2]
            f_roty = np.rad2deg(np.arctan2(f_X, f_Z))

            if img_key in wrapped_panels:
                f_roty = np.where(f_roty < 0, f_roty + 360, f_roty)
                
            f_roty = compress_roty(f_roty)

            label = 'Finder Candidates' if not added_finder_label else ""
            ax.scatter(f_roty, f_Y, marker='o', facecolors='none', edgecolors='blue', s=40, linewidths=0.25, label=label)
            added_finder_label = True

    # 3. Plot the Projected 3D Ellipsoids
    if getattr(peaks, 'var_u', None) is not None and getattr(peaks, 'peak_rows', None) is not None:
        is_isotropic = np.allclose(peaks.var_u, peaks.var_v) and np.allclose(peaks.cov_uv, 0)
        base_label = 'Isotropic Radius' if is_isotropic else 'Projected 3D Tensor'
        
        added_ellipse_label = False
        theta = np.linspace(0, 2 * np.pi, 50)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        for i in range(len(peaks.image_index)):
            img_key = peaks.image_index[i]
            det = detectors.get(img_key)
            if det is None: continue
            
            s_lab = get_s_lab_for_image(img_key)

            r_center, c_center = peaks.peak_rows[i], peaks.peak_cols[i]
            var_u, var_v, cov_uv = peaks.var_u[i], peaks.var_v[i], peaks.cov_uv[i]
            if var_u is None or var_v is None: continue

            diff = var_u - var_v
            sum_uv = var_u + var_v
            sq = np.sqrt(diff**2 + 4.0 * cov_uv**2)

            sigma_1 = np.sqrt(max((sum_uv + sq) / 2.0, 1e-6))
            sigma_2 = np.sqrt(max((sum_uv - sq) / 2.0, 1e-6))
            phi = 0.5 * np.arctan2(2.0 * cov_uv, diff)
            a, b = 2.0 * sigma_1, 2.0 * sigma_2

            u_ell = c_center + a * cos_t * np.cos(phi) - b * sin_t * np.sin(phi)
            v_ell = r_center + a * cos_t * np.sin(phi) + b * sin_t * np.cos(phi)

            ell_xyz = det.pixel_to_lab(v_ell, u_ell)
            ell_xyz = ell_xyz - s_lab
            e_X, e_Y, e_Z = ell_xyz[..., 0], ell_xyz[..., 1], ell_xyz[..., 2]
            e_roty = np.rad2deg(np.arctan2(e_X, e_Z))

            if img_key in wrapped_panels or np.ptp(e_roty) > 180:
                e_roty = np.where(e_roty < 0, e_roty + 360, e_roty)

            e_roty = compress_roty(e_roty)

            label = base_label if not added_ellipse_label else ""
            ax.plot(e_roty, e_Y, color='red', lw=0.25, alpha=0.8, label=label)
            added_ellipse_label = True

    # 4. Plot the peak centers
    if getattr(peaks, 'peak_rows', None) is not None and getattr(peaks, 'peak_cols', None) is not None and getattr(peaks, 'var_u', None) is None:
        p_rotys = []
        p_Ys = []
        
        for i in range(len(peaks.image_index)):
            img_key = peaks.image_index[i]
            det = detectors.get(img_key)
            if det is None: continue
            
            s_lab = get_s_lab_for_image(img_key)
            
            r_center = peaks.peak_rows[i]
            c_center = peaks.peak_cols[i]
            
            lab_xyz = det.pixel_to_lab(r_center, c_center)
            v = lab_xyz - s_lab
            
            p_X, p_Y, p_Z = v[0], v[1], v[2]
            p_roty = np.rad2deg(np.arctan2(p_X, p_Z))
            
            if img_key in wrapped_panels and p_roty < 0:
                p_roty += 360
                
            p_rotys.append(p_roty)
            p_Ys.append(p_Y)
            
        if p_rotys:
            p_rotys = compress_roty(np.array(p_rotys))
            ax.scatter(p_rotys, p_Ys, marker='o', facecolors='none', edgecolors='red', s=40, linewidths=0.25, label='Integrated Peaks')

    # ==========================================
    # FORMATTING & GAPS VISUALIZATION
    # ==========================================
    
    if merged:
        global_min = merged[0][0]
        global_max = merged[-1][1]
        c_min = compress_roty(np.array([global_min]))[0]
        c_max = compress_roty(np.array([global_max]))[0]
    else:
        c_min, c_max = ax.get_xlim()
        global_min, global_max = c_min, c_max

    if merged:
        start_tick = np.floor(global_min / 45.0) * 45
        end_tick = np.ceil(global_max / 45.0) * 45
        original_ticks = np.arange(start_tick, end_tick + 1, 45)
        
        valid_ticks = []
        for t in original_ticks:
            if t < global_min or t > global_max:
                continue
            if not any(g_start + 1 < t < g_end - 1 for g_start, g_end in gaps):
                valid_ticks.append(t)
                
        if valid_ticks:
            valid_ticks = np.array(valid_ticks)
            tick_pos = compress_roty(valid_ticks)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([f"{int(t) % 360}$^\circ$" for t in valid_ticks])

    if gaps:
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)

        valid_start = c_min
        trans = ax.get_xaxis_transform()
        
        d_y = 0.0075 
        d_x = 0.8     
        m_lw = 0.8   
        
        for g_start, g_end in gaps:
            t_start = compress_roty(np.array([g_start]))[0]
            t_end = compress_roty(np.array([g_end]))[0]
            
            ax.plot([valid_start, t_start], [0, 0], color='black', lw=1, transform=trans, clip_on=False)
            ax.plot([valid_start, t_start], [1, 1], color='black', lw=1, transform=trans, clip_on=False)
            
            ax.plot([t_start - d_x, t_start + d_x], [-d_y, d_y], color='black', transform=trans, clip_on=False, lw=m_lw)
            ax.plot([t_start - d_x, t_start + d_x], [1 - d_y, 1 + d_y], color='black', transform=trans, clip_on=False, lw=m_lw)
            
            ax.plot([t_end - d_x, t_end + d_x], [-d_y, d_y], color='black', transform=trans, clip_on=False, lw=m_lw)
            ax.plot([t_end - d_x, t_end + d_x], [1 - d_y, 1 + d_y], color='black', transform=trans, clip_on=False, lw=m_lw)
            
            valid_start = t_end
            
        ax.plot([valid_start, c_max], [0, 0], color='black', lw=1, transform=trans, clip_on=False)
        ax.plot([valid_start, c_max], [1, 1], color='black', lw=1, transform=trans, clip_on=False)

    ax.margins(0, 0)
    ax.set_xlim(c_min, c_max)

    aspect_ratio = 180.0 / (np.pi * mean_radius)
    ax.set_aspect(aspect_ratio, adjustable='box')

    ax.set_xlabel('Rotation Angle (roty) [degrees]')
    ax.set_ylabel('Lab Vertical (Y) [m]')
    if instrument is not None:
        ax.set_title(f'{instrument} cylindrical projection')

    if mesh_handle is not None:
        cbar = fig.colorbar(mesh_handle, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('Intensity (counts)')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=6, markerscale=0.5)

    plt.savefig(out_name, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
