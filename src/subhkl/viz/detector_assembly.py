import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_unrolled_detector(peaks, images, detectors, finder_peaks=None, out_name='unrolled_detector_peaks.png', instrument=None):
    """
    Plots an unrolled cylindrical detector from a DetectorPeaks object and image dict.

    peaks: DetectorPeaks dataclass instance.
    images: Dict of 2D numpy arrays, indexed by img_key (image_index).
    detectors: Dict mapping img_key to instantiated Detector objects.
    finder_peaks: Optional dict mapping img_key to numpy arrays of shape (N, 2)
                  containing [row, col] pixel coordinates from the peak finder.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Keep track of which panels crossed the seam and were unwrapped
    wrapped_panels = set()

    # Determine global max for consistent color scaling across all panels
    global_vmax = 1
    if images:
        global_vmax = max(np.max(img) for img in images.values())

    global_norm = colors.LogNorm(vmin=1, vmax=global_vmax + 1)
    mesh_handle = None

    # 1. Plot the Images
    for img_key, img in images.items():
        det = detectors.get(img_key)
        if det is None:
            continue

        # Generate cell *edges* (N+1, M+1) instead of centers (N, M).
        # We shift by -0.5 to get the physical boundary of the pixels.
        cols, rows = np.meshgrid(
            np.arange(det.m + 1) - 0.5,
            np.arange(det.n + 1) - 0.5
        )

        lab_xyz = det.pixel_to_lab(rows, cols)
        X = lab_xyz[..., 0]
        Y = lab_xyz[..., 1]
        Z = lab_xyz[..., 2]

        # Unroll cylinder: angle in XZ plane
        roty = np.rad2deg(np.arctan2(X, Z))

        # Handle the cylindrical seam for the image mesh
        if np.ptp(roty) > 180:
            roty = np.where(roty < 0, roty + 360, roty)
            wrapped_panels.add(img_key)

        mesh = ax.pcolormesh(
            roty,
            Y,
            img,
            shading='auto',
            cmap='binary',
            norm=global_norm
        )

        # Save the first mesh so we can attach a colorbar to it
        if mesh_handle is None:
            mesh_handle = mesh

    # 2. Plot Finder Peaks (if provided)
    if finder_peaks is not None:
        added_finder_label = False

        for img_key, coords in finder_peaks.items():
            if coords is None or len(coords) == 0:
                continue

            det = detectors.get(img_key)
            if det is None:
                continue

            coords = np.atleast_2d(coords)
            f_rows, f_cols = coords[:, 1], coords[:, 2]

            f_xyz = det.pixel_to_lab(f_rows, f_cols)

            if f_xyz.ndim == 1:
                f_xyz = f_xyz[np.newaxis, :]

            f_X, f_Y, f_Z = f_xyz[:, 0], f_xyz[:, 1], f_xyz[:, 2]
            f_roty = np.rad2deg(np.arctan2(f_X, f_Z))

            # Sync finder peaks with unwrapped panels
            if img_key in wrapped_panels:
                f_roty = np.where(f_roty < 0, f_roty + 360, f_roty)

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

        ax.scatter(p_roty, p_Y, marker='o', facecolors='none', edgecolors='red',
                       s=40, linewidths=0.25, label='Integrated Peaks')

    # 5. Formatting & Colorbar
    ax.set_xlabel('Rotation Angle (roty) [degrees]')
    ax.set_ylabel('Lab Vertical (Y) [m]')
    if instrument is not None:
        ax.set_title(f'{instrument} cylindrical projection')

    if mesh_handle is not None:
        cbar = fig.colorbar(mesh_handle, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('Intensity (counts)')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_name, dpi=600)
    plt.close(fig)
