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


class Detector:
    def __init__(self, config: dict):
        self.config = config
        self.m = config["m"]
        self.n = config["n"]
        self.width = config["width"]
        self.height = config["height"]
        
        self.center = np.array(config["center"])
        self.vhat = np.array(config["vhat"])
        
        self.panel_type = DetectorShape(config["panel"])
        
        if self.panel_type == DetectorShape.flat_panel:
            self.uhat = np.array(config["uhat"])
        elif self.panel_type == DetectorShape.curved_panel:
            self.radius = config["radius"]
            self.rhat = np.array(config["rhat"])
        else:
            raise ValueError(f"Unknown panel type: {self.panel_type}")

    def pixel_to_lab(self, row: npt.ArrayLike, col: npt.ArrayLike) -> npt.NDArray:
        """
        Convert detector pixel coordinates (row, col) to lab frame (x, y, z).
        
        TRANSPOSED MAPPING:
        row (Dim 0) -> u (Width)
        col (Dim 1) -> v (Height)
        """
        row = np.asarray(row)
        col = np.asarray(col)

        phys_row_idx = row
        phys_col_idx = col

        # 2. Map to Physical Dimensions (TRANSPOSED)
        # Row Index -> u (Width)
        # Col Index -> v (Height)

        # Note: We use self.m (Row count) for u scaling
        u = phys_row_idx / (self.m - 1) * self.width

        # Note: We use self.n (Col count) for v scaling
        v = phys_col_idx / (self.n - 1) * self.height

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

    def lab_to_pixel(self, x: float, y: float, z: float) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Convert lab frame coordinates (x, y, z) to detector pixel coordinates.
        Returns (row, col) in Image Space.
        """
        p = np.array([x, y, z])
        
        # Scale factors match the transpose above
        dw = self.width / (self.m - 1)  # Width / Rows (m)
        dh = self.height / (self.n - 1) # Height / Cols (n)

        vec = p.T - self.center
        
        # Projection onto vertical axis (v)
        if vec.ndim == 1:
            dot_v = np.dot(vec, self.vhat)
        else:
            dot_v = np.dot(vec, self.vhat)

        # v -> Col Index (phys_col)
        phys_col = np.clip(dot_v / dh, 0, self.n)

        if self.panel_type == DetectorShape.flat_panel:
            if vec.ndim == 1:
                dot_u = np.dot(vec, self.uhat)
            else:
                dot_u = np.dot(vec, self.uhat)
            
            # u -> Row Index (phys_row)
            phys_row = np.clip(dot_u / dw, 0, self.m)

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

            # u -> Row Index
            phys_row = np.clip(dt * (self.radius / dw), 0, self.m)

        row = phys_row
        col = phys_col

        return row, col

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

    def reflections_mask(self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Determine which reflections intersect this detector.
        Returns (mask, row, col) in Image Coordinates.
        """
        dir_vec = np.array([x, y, z])
        
        if self.panel_type == DetectorShape.flat_panel:
            norm = np.cross(self.uhat, self.vhat)
            numer = np.dot(self.center, norm)
            denom = np.einsum("i,in->n", norm, dir_vec)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = numer / denom
        else:
            d_dot_v = np.einsum("i,in->n", self.vhat, dir_vec)
            d_perp_x = x - d_dot_v * self.vhat[0]
            d_perp_y = y - d_dot_v * self.vhat[1]
            d_perp_z = z - d_dot_v * self.vhat[2]
            norm_d_perp = np.sqrt(d_perp_x**2 + d_perp_y**2 + d_perp_z**2)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = self.radius / norm_d_perp

        X, Y, Z = t * x, t * y, t * z
        
        # Returns Image Coordinates (Row, Col)
        row, col = self.lab_to_pixel(X, Y, Z)
        
        mask = (row > 0) & (col > 0) & (row < self.m - 1) & (col < self.n - 1) & (t > 0)
        
        return mask, row, col
