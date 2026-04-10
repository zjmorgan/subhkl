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

        # Handle the cylindrical seam: If a panel wraps around the -Z axis 
        # (jumping from 179 to -179), shift the negative values to make it continuous.
        if np.ptp(roty) > 180:
            roty = np.where(roty < 0, roty + 360, roty)

        ax.pcolormesh(
            roty,
            Y,
            img,
            # 'auto' falls back to 'flat' when edges are provided, 
            # which is what we want for an (N+1, M+1) coordinate grid.
            shading='auto', 
            cmap='binary',
            norm=colors.LogNorm(vmin=1, vmax=np.max(img) + 1)
        )

    # 2. Plot Finder Peaks (if provided)
    if finder_peaks is not None:
        # Keep track of whether we've added the label for the legend
        added_finder_label = False

        for img_key, coords in finder_peaks.items():
            if coords is None or len(coords) == 0:
                continue

            det = detectors.get(img_key)
            if det is None:
                continue

            # Ensure coords is an array of shape (N, 2) representing [row, col]
            coords = np.atleast_2d(coords)
            # coords shape is [intensity, r, c, sigma]
            f_rows, f_cols = coords[:, 1], coords[:, 2]

            f_xyz = det.pixel_to_lab(f_rows, f_cols)

            if f_xyz.ndim == 1:
                f_xyz = f_xyz[np.newaxis, :]

            f_X, f_Y, f_Z = f_xyz[:, 0], f_xyz[:, 1], f_xyz[:, 2]
            f_roty = np.rad2deg(np.arctan2(f_X, f_Z))

            label = 'Finder Candidates' if not added_finder_label else ""
            ax.scatter(f_roty, f_Y, marker='o', facecolors='none', edgecolors='blue',
                       s=40, linewidths=1.2, label=label)
            added_finder_label = True

    # 3. Plot the Projected 3D Ellipsoids
    if getattr(peaks, 'var_u', None) is not None and getattr(peaks, 'peak_rows', None) is not None:
        added_ellipse_label = False
        
        # Generate points for a unit circle once
        theta = np.linspace(0, 2 * np.pi, 50)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        for i in range(len(peaks.intensity)):
            img_key = peaks.image_index[i]
            det = detectors.get(img_key)
            if det is None:
                continue
                
            # Extract peak center and tensor
            r_center = peaks.peak_rows[i]
            c_center = peaks.peak_cols[i]
            var_u = peaks.var_u[i]
            var_v = peaks.var_v[i]
            cov_uv = peaks.cov_uv[i]
            
            if var_u is None or var_v is None:
                continue
                
            # Eigenvalue decomposition for the major/minor axes
            diff = var_u - var_v
            sum_uv = var_u + var_v
            sq = np.sqrt(diff**2 + 4.0 * cov_uv**2)

            sigma_1 = np.sqrt(max((sum_uv + sq) / 2.0, 1e-6))
            sigma_2 = np.sqrt(max((sum_uv - sq) / 2.0, 1e-6))
            phi = 0.5 * np.arctan2(2.0 * cov_uv, diff)

            # Radii are 2*sigma (to match the 4*sigma full width/height in Matplotlib)
            a = 2.0 * sigma_1
            b = 2.0 * sigma_2

            # Sample the parametric ellipse in Pixel space (u=cols, v=rows)
            u_ell = c_center + a * cos_t * np.cos(phi) - b * sin_t * np.sin(phi)
            v_ell = r_center + a * cos_t * np.sin(phi) + b * sin_t * np.cos(phi)

            # Project the entire boundary directly to Lab XYZ
            ell_xyz = det.pixel_to_lab(v_ell, u_ell)
            
            # Unroll the boundary into cylindrical space
            e_X, e_Y, e_Z = ell_xyz[..., 0], ell_xyz[..., 1], ell_xyz[..., 2]
            e_roty = np.rad2deg(np.arctan2(e_X, e_Z))
            
            label = 'Projected 3D Tensor' if not added_ellipse_label else ""

            # Plot the closed loop. The projected shape will naturally warp
            # if the unrolling introduces significant non-linearity over the span of the peak.
            ax.plot(e_roty, e_Y, color='red', lw=0.25, alpha=0.8, label=label)
            added_ellipse_label = True

    # 4. Plot the Integrated Peaks
    if peaks is not None and peaks.xyz is not None and len(peaks.xyz) > 0:
        p_xyz = np.array(peaks.xyz)

        if p_xyz.ndim == 1 and len(p_xyz) == 3:
            p_xyz = p_xyz[np.newaxis, :]

        p_X, p_Y, p_Z = p_xyz[:, 0], p_xyz[:, 1], p_xyz[:, 2]
        p_roty = np.rad2deg(np.arctan2(p_X, p_Z))

        ax.scatter(p_roty, p_Y, marker='x', color='red', s=15, linewidths=1, label='Integrated Peaks')

    # 4. Formatting
    ax.set_xlabel('Rotation Angle (roty) [degrees]')
    ax.set_ylabel('Lab Vertical (Y) [m]')
    if instrument is not None:
        ax.set_title(f'{instrument} cylindrical projection')

    # Add a legend if we have peaks
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_name, dpi=600)
    plt.close(fig)
