import numpy as np
import numpy.typing as npt
from subhkl.instrument.detector import Detector
from subhkl.core.crystallography import get_q_lab


def scale_coordinates(xp, yp, scale_x, scale_y, nx, ny):
    """
    Scale pixel coordinates to physical coordinates.

    Parameters
    ----------
    xp : float
        Pixel x-coordinate
    yp : float
        Pixel y-coordinate
    scale_x : float
        Scale factor in x direction (m/pixel)
    scale_y : float
        Scale factor in y direction (m/pixel)
    nx : int
        Number of pixels in x direction
    ny : int
        Number of pixels in y direction

    Returns
    -------
    x, y : tuple of float
        Physical coordinates in meters
    """
    x = (xp - nx / 2) * scale_x
    y = (yp - ny / 2) * scale_y
    return x, y


def predict_reflections_on_panel(
    detector,
    h,
    k,
    l,
    RUB,
    wavelength_min,
    wavelength_max,
    sample_offset=None,
    ki_vec=None,
    R_all=None,
):
    """
    Predict reflection positions on a specific detector panel.
    Expects RUB to be a SINGLE (3,3) matrix for this specific panel's exposure.
    Expects h, k, l to be 1D arrays of length M (the theoretical reflection pool).
    """
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])
    
    ki_hat = ki_vec / np.linalg.norm(ki_vec)
    hkl = np.stack([h, k, l], axis=0)  # (3, M)

    # 1. Transform Miller indices to absolute Lab frame Q-vectors
    # RUB is strictly (3, 3). hkl is (3, M). Result Q is (3, M)
    q_lab = RUB @ hkl 
    Q_vec = 2 * np.pi * q_lab  # (3, M)
    
    Q_sq = np.sum(Q_vec**2, axis=0) # (M,)
    Q_dot_ki = np.sum(Q_vec * ki_hat[:, None], axis=0) # (M,)
    
    # 2. Solve Laue Condition for Wavelength
    with np.errstate(divide='ignore', invalid='ignore'):
        lamda = -4 * np.pi * Q_dot_ki / Q_sq # (M,)
        
    mask_wl = (lamda >= wavelength_min) & (lamda <= wavelength_max) & np.isfinite(lamda)
    
    if not np.any(mask_wl):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    lamda_v = lamda[mask_wl]
    Q_vec_v = Q_vec[:, mask_wl]
    h_v, k_v, l_v = h[mask_wl], k[mask_wl], l[mask_wl]
    
    # 3. Calculate Final Scattered Ray Direction (kf)
    k_mag = 2 * np.pi / lamda_v
    kf_vec = Q_vec_v + k_mag * ki_hat[:, None] # (3, M_v)
    kf_dir = kf_vec / np.linalg.norm(kf_vec, axis=0, keepdims=True) # (3, M_v)
    
    # 4. Resolve the True Sample Ray Origin (s_lab)
    s = sample_offset if sample_offset is not None else np.zeros(3)
    if R_all is not None:
        # R_all is strictly (3, 3) for this panel
        s_lab = R_all @ s
    else:
        s_lab = s

    # 5. Intersect with Panel
    mask_panel, row, col = detector.reflections_mask(
        kf_dir[0], kf_dir[1], kf_dir[2], sample_offset=s_lab
    )
    
    # Return strictly the peaks that fall within the physical detector boundary
    return row[mask_panel], col[mask_panel], h_v[mask_panel], k_v[mask_panel], l_v[mask_panel], lamda_v[mask_panel]

def calculate_angular_error(
    xyz_det: npt.NDArray,
    h: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,  # noqa: E741
    lam: npt.NDArray,
    RUB: npt.NDArray,
    sample_offset: npt.NDArray = None,
    ki_vec: npt.NDArray = None,
    R_all: npt.NDArray = None,
):
    """
    Calculate D-spacing and Angular errors for observed peaks vs predicted geometry.
    Uses the RUB matrix (R @ U @ B) for all coordinate transformations.
    """
    if sample_offset is None:
        sample_offset = np.zeros(3)
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])

    # 1. Calculate Q_calc (Lab Frame)
    q_lab_calc = get_q_lab(h, k, l, RUB)
    q_calc_norm = q_lab_calc / np.linalg.norm(q_lab_calc, axis=1, keepdims=True)

    # 2. Calculate Q_obs (Lab Frame) from Detector Pixel Position
    # v = Pixel_Position - Sample_Position
    if R_all is not None:
        if R_all.ndim == 3:
            s_lab = np.einsum("nij,j->ni", R_all, sample_offset)
        else:
            s_lab = R_all @ sample_offset
        v = xyz_det - s_lab
    else:
        v = xyz_det - sample_offset

    dist = np.linalg.norm(v, axis=1, keepdims=True)
    kf_dir = v / dist  # Unit vector pointing from sample to pixel

    ki_dir = ki_vec / np.linalg.norm(ki_vec)

    # Scattering vector direction matches delta_k = kf - ki
    delta_k = kf_dir - ki_dir[None, :]
    two_sin_theta = np.linalg.norm(delta_k, axis=1)

    # Q_obs direction
    with np.errstate(divide="ignore", invalid="ignore"):
        q_obs_norm = delta_k / two_sin_theta[:, None]
        # Handle 0-degree scattering where direction is undefined
        q_obs_norm = np.nan_to_num(q_obs_norm, nan=0.0)

    # 3. Angular Error (Angle between Q_calc direction and Q_obs direction)
    dot = np.sum(q_obs_norm * q_calc_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang_err = np.rad2deg(np.arccos(dot))
    # If q_obs_norm was zeroed out (0-degree scattering), arccos(0) = 90 deg.
    # This is a reasonable penalty for a 0-degree observation matching a finite Q prediction.

    # 4. D-Spacing Error
    # d_obs = lambda / 2sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_obs = np.divide(lam, two_sin_theta)

    # d_calc = 1 / |RUB * hkl|
    # Since R and U are rotations (unitary), they preserve length.
    # |R * U * B * h| == |B * h| == 1/d
    q_lab_mag = np.linalg.norm(q_lab_calc, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        d_calc = np.divide(1.0, q_lab_mag)

    d_err = np.abs(d_obs - d_calc)

    return d_err, ang_err
