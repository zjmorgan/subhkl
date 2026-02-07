from enum import Enum
import numpy as np
import numpy.typing as npt


class DetectorShape(str, Enum):
    flat_panel = "flat"
    curved_panel = "curved"


def scattering_vector_from_angles(two_theta: npt.ArrayLike, az_phi: npt.ArrayLike) -> npt.NDArray:
    """
    Calculate the direction of the scattering vector (kf - ki) from scattering angles.
    """
    tt = np.deg2rad(two_theta)
    az = np.deg2rad(az_phi)

    # kf direction (unit vector)
    kx = np.sin(tt) * np.cos(az)
    ky = np.sin(tt) * np.sin(az)
    kz = np.cos(tt)

    # ki direction is (0, 0, 1)
    return np.array([kx, ky, kz - 1])

def angles_from_kf(kf_vectors):
    """
    Converts outgoing wavevectors (kf) to Lab Frame detector angles.
    """
    norms = np.linalg.norm(kf_vectors, axis=1, keepdims=True)
    kf_dir = kf_vectors / norms
    two_theta = np.arccos(kf_dir[:, 2])
    azimuth = np.arctan2(kf_dir[:, 1], kf_dir[:, 0])
    return two_theta, azimuth

def angles_from_scattering_vector(q_vectors, ki_vec=None):
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])
    if q_vectors.ndim == 2 and ki_vec.ndim == 1:
        ki = ki_vec[None, :]
    else:
        ki = ki_vec
    kf = q_vectors + ki
    return angles_from_kf(kf)

class Detector:
    def __init__(self, config: dict):
        self.config = config
        self.m = config["m"] # Rows
        self.n = config["n"] # Cols
        self.width = config["width"]
        self.height = config["height"]
        
        self.center = np.array(config["center"])
        # vhat is the direction from the lower left (0,0)
        # to the upper left (n-1, 0) corner (Y axis)
        self.vhat = np.array(config["vhat"])
        
        self.panel_type = DetectorShape(config["panel"])
        
        if self.panel_type == DetectorShape.flat_panel:
            # uhat is the direction from the lower left (0,0)
            # to the lower right (0,m-1) corner (X axis)
            self.uhat = np.array(config["uhat"])
        elif self.panel_type == DetectorShape.curved_panel:
            self.radius = config["radius"]
            self.rhat = np.array(config["rhat"])
        else:
            raise ValueError(f"Unknown panel type: {self.panel_type}")

    def pixel_to_lab(self, row: npt.ArrayLike, col: npt.ArrayLike) -> npt.NDArray:
        """
        Convert detector pixel coordinates (row, col) to lab frame (x, y, z).

        col (Dim 1) -> u (Width)
        row (Dim 0) -> v (Height)
        """
        row = np.asarray(row)
        col = np.asarray(col)

        # u scales with Width (m columns)
        u = col / (self.m - 1) * self.width

        # v scales with Height (n rows)
        v = row / (self.n - 1) * self.height

        dv = np.einsum("...,d->...d", v, self.vhat)

        if self.panel_type == DetectorShape.flat_panel:
            du = np.einsum("...,d->...d", u, self.uhat)
        else:
            # Curved panel logic
            w = np.cross(self.vhat, self.rhat)
            angle = u / self.radius
            sin_u = np.sin(angle)
            cos_u = np.cos(angle)
            dvr = np.einsum("...,d->...d", self.radius * sin_u, w)
            dr = np.einsum("...,d->...d", self.radius * (cos_u - 1), self.rhat)
            du = dr + dvr

        xyz = self.center + du + dv
        return xyz.T if xyz.ndim > 1 else xyz

    def lab_to_pixel(self, x: float, y: float, z: float, clip: bool = False) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Convert lab frame coordinates (x, y, z) to detector pixel coordinates.
        Returns (row, col) in Image Space.
        
        If clip=False (default), returns raw coordinates which may be negative 
        or larger than detector dimensions.
        """
        p = np.array([x, y, z])
        
        dw = self.width / (self.m - 1)  # Width / Cols (n)
        dh = self.height / (self.n - 1) # Height / Rows (m)

        vec = p.T - self.center
        
        # Projection onto vertical axis (v) -> Rows
        if vec.ndim == 1:
            dot_v = np.dot(vec, self.vhat)
        else:
            dot_v = np.dot(vec, self.vhat)

        # v -> Row Index (No Clipping)
        row_f = dot_v / dh

        if self.panel_type == DetectorShape.flat_panel:
            if vec.ndim == 1:
                dot_u = np.dot(vec, self.uhat)
            else:
                dot_u = np.dot(vec, self.uhat)
            
            # u -> Col Index (No Clipping)
            col_f = dot_u / dw

        else:
            # Curved logic
            if vec.ndim == 1: d_planar = vec - dot_v * self.vhat
            else: d_planar = vec - (dot_v[:, np.newaxis] * self.vhat)

            what = np.cross(self.vhat, self.rhat)
            
            if vec.ndim == 1:
                dot_r = np.dot(d_planar, self.rhat)
                dot_w = np.dot(d_planar, what)
            else:
                dot_r = np.dot(d_planar, self.rhat)
                dot_w = np.dot(d_planar, what)

            val = -dot_r / dot_w
            dt = 2 * np.arctan(val)
            dt = np.mod(dt, 2 * np.pi)

            col_f = dt * (self.radius / dw)

        if clip:
            row_f = np.clip(row_f, 0, self.n)
            col_f = np.clip(col_f, 0, self.m)

        return row_f, col_f

    def pixel_to_angles(self, row: npt.ArrayLike, col: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Calculate scattering angles (two_theta, az_phi) for pixels (row, col).
        """
        xyz = self.pixel_to_lab(row, col)
        X, Y, Z = xyz[0], xyz[1], xyz[2]

        R = np.sqrt(X**2 + Y**2 + Z**2)
        # Avoid division by zero
        two_theta = np.rad2deg(np.arccos(np.clip(Z / R, -1.0, 1.0)))
        az_phi = np.rad2deg(np.arctan2(Y, X))

        return two_theta, az_phi

    def reflections_mask(self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike, sample_offset: npt.ArrayLike = None) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Determine which reflections intersect this detector.
        Returns: (mask, row, col) in Image Coordinates.
        """
        dir_vec = np.array([x, y, z]) # Shape (3, N)
        if dir_vec.ndim == 1:
            dir_vec = dir_vec[:, np.newaxis]
            
        if sample_offset is None:
            s = np.zeros(3)
        else:
            s = np.array(sample_offset)

        if self.panel_type == DetectorShape.flat_panel:
            norm = np.cross(self.uhat, self.vhat)
            c_minus_s_dot_n = np.dot(self.center - s, norm)
            d_dot_n = np.einsum("i,in->n", norm, dir_vec)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                t = c_minus_s_dot_n / d_dot_n
                
        else:
            v = self.vhat
            B_vec = np.cross(s, v)
            D_vec = np.cross(dir_vec.T, v).T
            
            QA = np.sum(D_vec**2, axis=0)
            QB = 2 * np.dot(B_vec, D_vec)
            QC = np.dot(B_vec, B_vec) - self.radius**2
            
            delta = QB**2 - 4*QA*QC
            
            with np.errstate(invalid='ignore'):
                sqrt_delta = np.sqrt(delta)
                t1 = (-QB + sqrt_delta) / (2*QA)
                t2 = (-QB - sqrt_delta) / (2*QA)
                t = np.where((t2 > 0), t2, t1)
                t = np.where(delta < 0, -1.0, t)

        # Handle stacked sample offsets (N, 3) or single offset (3,)
        if s.ndim == 2:
            X = s[:, 0] + t * dir_vec[0]
            Y = s[:, 1] + t * dir_vec[1]
            Z = s[:, 2] + t * dir_vec[2]
        else:
            X = s[0] + t * dir_vec[0]
            Y = s[1] + t * dir_vec[1]
            Z = s[2] + t * dir_vec[2]
        
        row, col = self.lab_to_pixel(X, Y, Z)
        mask = (row > 0) & (col > 0) & (row < self.n - 1) & (col < self.m - 1) & (t > 0)
        
        return mask, row, col
