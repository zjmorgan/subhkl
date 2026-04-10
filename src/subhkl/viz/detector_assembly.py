import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_unrolled_detector(peaks, images, detectors, finder_peaks=None, out_name='unrolled_detector_peaks.png', instrument=None):
    """
    Plots an unrolled cylindrical detector from a DetectorPeaks object and image dict,
    handling the 180-degree wrapping seam and dynamically cutting out large x-axis gaps.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # ==========================================
    # PRE-PASS: Find Wrap Bounds & Empty Gaps
    # ==========================================
    wrapped_panels = set()
    panel_bounds = []

    for img_key, img in images.items():
        det = detectors.get(img_key)
        if det is None: continue
        
        c, r = np.meshgrid(np.arange(det.m + 1) - 0.5, np.arange(det.n + 1) - 0.5)
        xyz = det.pixel_to_lab(r, c)
        roty = np.rad2deg(np.arctan2(xyz[..., 0], xyz[..., 2]))
        
        if np.ptp(roty) > 180:
            roty = np.where(roty < 0, roty + 360, roty)
            wrapped_panels.add(img_key)
            
        panel_bounds.append([np.min(roty), np.max(roty)])

    # Merge intervals (with 5-degree tolerance)
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

    # Find significant gaps (> 72 degrees, which is 20% of 360)
    gaps = []
    visual_gap_size = 15.0 # The visual width (in degrees) to draw the break indicator
    for i in range(len(merged) - 1):
        g_start = merged[i][1]
        g_end = merged[i+1][0]
        if g_end - g_start > 72:
            gaps.append((g_start, g_end))

    def compress_roty(r_orig):
        """Maps physical rotation angles to a compressed plotting axis."""
        if not gaps: return r_orig
        r_out = np.copy(r_orig)
        
        for g_start, g_end in gaps:
            actual_gap = g_end - g_start
            shift = actual_gap - visual_gap_size
            
            # Shift everything to the right of the gap
            r_out[r_orig >= g_end] -= shift
            
            # Linearly compress anything inside the gap
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

        cols, rows = np.meshgrid(np.arange(det.m + 1) - 0.5, np.arange(det.n + 1) - 0.5)
        lab_xyz = det.pixel_to_lab(rows, cols)
        X, Y, Z = lab_xyz[..., 0], lab_xyz[..., 1], lab_xyz[..., 2]

        roty = np.rad2deg(np.arctan2(X, Z))
        if img_key in wrapped_panels:
            roty = np.where(roty < 0, roty + 360, roty)

        # Apply spatial compression
        roty = compress_roty(roty)

        mesh = ax.pcolormesh(
            roty, Y, img,
            shading='auto', cmap='binary', norm=global_norm
        )
        if mesh_handle is None: mesh_handle = mesh

    # 2. Plot Finder Peaks
    if finder_peaks is not None:
        added_finder_label = False
        for img_key, coords in finder_peaks.items():
            if coords is None or len(coords) == 0: continue
            det = detectors.get(img_key)
            if det is None: continue

            coords = np.atleast_2d(coords)
            f_rows, f_cols = coords[:, 1], coords[:, 2]

            f_xyz = det.pixel_to_lab(f_rows, f_cols)
            if f_xyz.ndim == 1: f_xyz = f_xyz[np.newaxis, :]

            f_X, f_Y, f_Z = f_xyz[:, 0], f_xyz[:, 1], f_xyz[:, 2]
            f_roty = np.rad2deg(np.arctan2(f_X, f_Z))

            if img_key in wrapped_panels:
                f_roty = np.where(f_roty < 0, f_roty + 360, f_roty)
                
            f_roty = compress_roty(f_roty)

            label = 'Finder Candidates' if not added_finder_label else ""
            ax.scatter(f_roty, f_Y, marker='o', facecolors='none', edgecolors='blue',
                       s=40, linewidths=0.25, label=label)
            added_finder_label = True

    # 3. Plot the Projected 3D Ellipsoids
    if getattr(peaks, 'var_u', None) is not None and getattr(peaks, 'peak_rows', None) is not None:
        added_ellipse_label = False
        theta = np.linspace(0, 2 * np.pi, 50)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        for i in range(len(peaks.intensity)):
            img_key = peaks.image_index[i]
            det = detectors.get(img_key)
            if det is None: continue

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
            e_X, e_Y, e_Z = ell_xyz[..., 0], ell_xyz[..., 1], ell_xyz[..., 2]
            e_roty = np.rad2deg(np.arctan2(e_X, e_Z))

            if img_key in wrapped_panels or np.ptp(e_roty) > 180:
                e_roty = np.where(e_roty < 0, e_roty + 360, e_roty)
                
            e_roty = compress_roty(e_roty)

            label = 'Projected 3D Tensor' if not added_ellipse_label else ""
            ax.plot(e_roty, e_Y, color='red', lw=0.25, alpha=0.8, label=label)
            added_ellipse_label = True

    # 4. Plot the Integrated Peaks
    if peaks is not None and getattr(peaks, 'xyz', None) is not None and len(peaks.xyz) > 0:
        p_xyz = np.array(peaks.xyz)
        if p_xyz.ndim == 1: p_xyz = p_xyz[np.newaxis, :]

        p_X, p_Y, p_Z = p_xyz[:, 0], p_xyz[:, 1], p_xyz[:, 2]
        p_roty = np.rad2deg(np.arctan2(p_X, p_Z))

        if getattr(peaks, 'image_index', None) is not None:
            for i, img_key in enumerate(peaks.image_index):
                if img_key in wrapped_panels and p_roty[i] < 0:
                    p_roty[i] += 360
                    
        p_roty = compress_roty(p_roty)

        ax.scatter(p_roty, p_Y, marker='o', facecolors='none', edgecolors='red',
                       s=40, linewidths=0.25, label='Integrated Peaks')

    # ==========================================
    # FORMATTING & GAPS VISUALIZATION
    # ==========================================
    # Draw the Break Indicators (--//--)
    for g_start, g_end in gaps:
        t_start = compress_roty(np.array([g_start]))[0]
        t_end = compress_roty(np.array([g_end]))[0]
        
        # Shade the gap to clearly indicate the break
        ax.axvspan(t_start, t_end, color='whitesmoke', zorder=0)
        ax.axvspan(t_start, t_end, color='none', hatch='//', edgecolor='lightgray', zorder=1)
        ax.axvline(t_start, color='black', linestyle='--', lw=1.5, zorder=2)
        ax.axvline(t_end, color='black', linestyle='--', lw=1.5, zorder=2)

    # Format the X-Axis Ticks dynamically to reflect physical angles
    if merged:
        global_min = merged[0][0]
        global_max = merged[-1][1]
        
        start_tick = np.floor(global_min / 45.0) * 45
        end_tick = np.ceil(global_max / 45.0) * 45
        original_ticks = np.arange(start_tick, end_tick + 1, 45)
        
        valid_ticks = []
        for t in original_ticks:
            # Skip ticks that fall squarely inside a cut-out gap
            if not any(g_start + 1 < t < g_end - 1 for g_start, g_end in gaps):
                valid_ticks.append(t)
                
        if valid_ticks:
            valid_ticks = np.array(valid_ticks)
            tick_pos = compress_roty(valid_ticks)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([f"{int(t) % 360}$^\circ$" for t in valid_ticks])

    ax.set_xlabel('Rotation Angle (roty) [degrees]')
    ax.set_ylabel('Lab Vertical (Y) [m]')
    if instrument is not None:
        ax.set_title(f'{instrument} cylindrical projection')

    if mesh_handle is not None:
        cbar = fig.colorbar(mesh_handle, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('Intensity (counts)')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Prevent duplicate legend entries if multiple components added the label
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plt.savefig(out_name, dpi=600)
    plt.close(fig)
