import numpy as np
import numpy.typing as npt
import typing

if typing.TYPE_CHECKING:
    from subhkl.detector import Detector

def get_q_lab(
    h: npt.ArrayLike, 
    k: npt.ArrayLike, 
    l: npt.ArrayLike, 
    RUB: npt.ArrayLike
) -> npt.NDArray:
    """
    Calculate Q vectors in the Lab Frame.
    Q_lab = RUB * hkl
    
    RUB should be the composite matrix (R @ U @ B).
    """
    hkl = np.stack([h, k, l], axis=1) # (N, 3)
    
    # Handle RUB shape: (3,3) or (N,3,3)
    if RUB.ndim == 3:
        # Einsum: n=batch, i=row, j=col. RUB[n,i,j] * hkl[n,j] -> out[n,i]
        q_lab = np.einsum('nij,nj->ni', RUB, hkl)
    else:
        # Standard matmul: hkl @ RUB.T
        q_lab = hkl @ RUB.T
        
    return q_lab

def calculate_angular_error(
    xyz_det: npt.NDArray,
    h: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,
    lam: npt.NDArray,
    RUB: npt.NDArray,
    sample_offset: npt.NDArray = None,
    ki_vec: npt.NDArray = None
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
    v = xyz_det - sample_offset
    dist = np.linalg.norm(v, axis=1, keepdims=True)
    kf_dir = v / dist # Unit vector pointing from sample to pixel
    
    ki_dir = ki_vec / np.linalg.norm(ki_vec)
    
    # Scattering vector direction matches delta_k = kf - ki
    delta_k = kf_dir - ki_dir[None, :]
    two_sin_theta = np.linalg.norm(delta_k, axis=1)
    
    # Q_obs direction
    with np.errstate(divide='ignore', invalid='ignore'):
        q_obs_norm = delta_k / two_sin_theta[:, None]

    # 3. Angular Error (Angle between Q_calc direction and Q_obs direction)
    dot = np.sum(q_obs_norm * q_calc_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang_err = np.rad2deg(np.arccos(dot))

    # 4. D-Spacing Error
    # d_obs = lambda / 2sin(theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_obs = np.divide(lam, two_sin_theta)
    
    # d_calc = 1 / |RUB * hkl|
    # Since R and U are rotations (unitary), they preserve length.
    # |R * U * B * h| == |B * h| == 1/d
    q_lab_mag = np.linalg.norm(q_lab_calc, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d_calc = np.divide(1.0, q_lab_mag)

    d_err = np.abs(d_obs - d_calc)
    
    return d_err, ang_err

def predict_reflections_on_panel(
    detector: "Detector",
    h: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,
    RUB: npt.NDArray,
    wavelength_min: float,
    wavelength_max: float,
    sample_offset: npt.NDArray = None,
    ki_vec: npt.NDArray = None
):
    """
    Predicts which HKLs fall on a specific detector panel using the RUB matrix.
    Returns: (row, col, h, k, l, wavelength)
    """
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])
    ki_hat = ki_vec / np.linalg.norm(ki_vec)

    # 1. Calculate Q vectors (Units: 2pi/d)
    # get_q_lab returns 1/d units. Multiply by 2pi.
    q_lab_direction = get_q_lab(h, k, l, RUB) 
    Q_vecs = 2 * np.pi * q_lab_direction.T # Shape (3, N)
    
    Q_sq = np.sum(Q_vecs**2, axis=0)

    # 2. Calculate Wavelength (Generalized Laue)
    # lambda = -4pi * (Q . ki) / Q^2
    Q_dot_ki = np.sum(Q_vecs * ki_hat[:, None], axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
         lamda = -4 * np.pi * Q_dot_ki / Q_sq

    # 3. Filter Wavelength
    mask = (lamda > wavelength_min) & (lamda < wavelength_max)
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    Q_vecs = Q_vecs[:, mask]
    lamda = lamda[mask]
    h, k, l = h[mask], k[mask], l[mask]
    
    # 4. Calculate kf direction
    k_mag = 2 * np.pi / lamda
    kf_vecs = Q_vecs + k_mag * ki_hat[:, None]
    
    kf_norms = np.linalg.norm(kf_vecs, axis=0)
    kf_dirs = kf_vecs / kf_norms # Shape (3, N_filtered)

    x, y, z = kf_dirs[0], kf_dirs[1], kf_dirs[2]

    # 5. Ray Trace intersection with Panel
    mask_panel, row, col = detector.reflections_mask(x, y, z, sample_offset=sample_offset)
    
    return row[mask_panel], col[mask_panel], h[mask_panel], k[mask_panel], l[mask_panel], lamda[mask_panel]
