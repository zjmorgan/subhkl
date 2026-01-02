from enum import Enum
import numpy as np
import numpy.typing as npt


class DetectorShape(str, Enum):
    flat_panel = "flat"
    curved_panel = "curved"


def scattering_vector_from_angles(two_theta: npt.ArrayLike, az_phi: npt.ArrayLike) -> npt.NDArray:
    """
    Calculate the direction of the scattering vector (kf - ki) from scattering angles.
    
    Assumes elastic scattering and that the incident beam ki is along (0, 0, 1).
    The returned vector is normalized by the wavenumber k (i.e., this returns Q/k).
    
    Parameters
    ----------
    two_theta : array_like
        Scattering angle (2theta) in degrees.
    az_phi : array_like
        Azimuthal angle (phi) in degrees.
        
    Returns
    -------
    kf_ki_dir : ndarray
        The vector kf_hat - ki_hat. Shape (3, ...).
    """
    tt = np.deg2rad(two_theta)
    az = np.deg2rad(az_phi)

    # kf direction (unit vector)
    kx = np.sin(tt) * np.cos(az)
    ky = np.sin(tt) * np.sin(az)
    kz = np.cos(tt)

    # ki direction is (0, 0, 1)
    # result is kf - ki
    return np.array([kx, ky, kz - 1])


class Detector:
    def __init__(self, config: dict):
        """
        Initialize detector geometry from configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Dictionary containing detector parameters:
            - m, n: pixel dimensions
            - width, height: physical dimensions
            - center: vector to center of detector [x, y, z]
            - vhat: unit vector along height axis
            - panel: 'flat' or 'curved'
            - uhat: unit vector along width axis (for flat panels)
            - radius, rhat: radius and normal vector (for curved panels)
        """
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

    def pixel_to_lab(self, i: npt.ArrayLike, j: npt.ArrayLike) -> npt.NDArray:
        """
        Convert detector pixel coordinates (i, j) to lab frame (x, y, z).
        """
        i = np.asarray(i)
        j = np.asarray(j)

        u = i / (self.m - 1) * self.width
        v = j / (self.n - 1) * self.height

        dv = np.einsum("...,d->...d", v, self.vhat)

        if self.panel_type == DetectorShape.flat_panel:
            du = np.einsum("...,d->...d", u, self.uhat)
        else:
            # Curved panel logic
            # For curved panels, u maps to an angle around the cylinder
            w = np.cross(self.vhat, self.rhat)
            
            angle = u / self.radius
            sin_u = np.sin(angle)
            cos_u = np.cos(angle)

            dvr = np.einsum("...,d->...d", self.radius * sin_u, w)
            dr = np.einsum("...,d->...d", self.radius * (cos_u - 1), self.rhat)

            du = dr + dvr

        # Result shape: (..., 3)
        xyz = self.center + du + dv
        return xyz.T if xyz.ndim > 1 else xyz

    def lab_to_pixel(self, x: float, y: float, z: float) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Convert lab frame coordinates (x, y, z) to detector pixel coordinates (i, j).
        """
        p = np.array([x, y, z])
        
        dw = self.width / (self.m - 1)
        dh = self.height / (self.n - 1)

        # Vector from center to point
        vec = p.T - self.center
        
        # Projection onto vertical axis
        if vec.ndim == 1:
            dot_v = np.dot(vec, self.vhat)
        else:
            dot_v = np.dot(vec, self.vhat)

        j = np.clip(dot_v / dh, 0, self.n)

        if self.panel_type == DetectorShape.flat_panel:
            if vec.ndim == 1:
                dot_u = np.dot(vec, self.uhat)
            else:
                dot_u = np.dot(vec, self.uhat)
                
            i = np.clip(dot_u / dw, 0, self.m)

        else:
            # Curved panel
            # Project out the vertical component to get component in the cylinder plane
            # d_planar = vec - (dot_v * vhat)
            if vec.ndim == 1:
                d_planar = vec - dot_v * self.vhat
            else:
                d_planar = vec - (dot_v[:, np.newaxis] * self.vhat)

            what = np.cross(self.vhat, self.rhat)

            # Calculate angle in the plane defined by rhat and what
            # rhat is the "origin" direction (u=0)
            if vec.ndim == 1:
                dot_r = np.dot(d_planar, self.rhat)
                dot_w = np.dot(d_planar, what)
            else:
                dot_r = np.dot(d_planar, self.rhat)
                dot_w = np.dot(d_planar, what)

            val = -dot_r / dot_w
            dt = 2 * np.arctan(val)
            dt = np.mod(dt, 2 * np.pi)

            i = np.clip(dt * (self.radius / dw), 0, self.m)

        return i.astype(int), j.astype(int)

    def pixel_to_angles(self, i: npt.ArrayLike, j: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Calculate scattering angles (two_theta, az_phi) for pixels (i, j).
        """
        xyz = self.pixel_to_lab(i, j)
        X, Y, Z = xyz[0], xyz[1], xyz[2]

        R = np.sqrt(X**2 + Y**2 + Z**2)
        # Avoid division by zero if R is 0 (unlikely for detector pixels)
        two_theta = np.rad2deg(np.arccos(np.clip(Z / R, -1.0, 1.0)))
        az_phi = np.rad2deg(np.arctan2(Y, X))

        return two_theta, az_phi

    def reflections_mask(self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Determine which reflections at lab coordinates (x, y, z) intersect this detector.
        
        Returns
        -------
        mask : ndarray (bool)
            True if the reflection intersects the detector.
        i, j : ndarray
            Pixel coordinates of the intersections.
        """
        # Calculate intersection parameter t such that point P = t * (x,y,z) lies on detector surface
        
        # x, y, z are unit vectors (or scaled vectors) from sample origin.
        # Detector surface equation:
        # Flat: (P - Center) . Norm = 0  => (t*dir - Center) . Norm = 0 => t = (Center . Norm) / (dir . Norm)
        # Curved: Distance from cylinder axis = Radius. 
        
        # We need to handle arrays
        dir_vec = np.array([x, y, z]) # Shape (3, N)
        
        if self.panel_type == DetectorShape.flat_panel:
            norm = np.cross(self.uhat, self.vhat)
            numer = np.dot(self.center, norm)
            denom = np.einsum("i,in->n", norm, dir_vec)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                t = numer / denom
        else:
            # Curved: Cylinder equation
            # Project direction onto plane perpendicular to cylinder axis (vhat)
            # D_perp = D - (D.vhat)vhat
            # P_perp = t * D_perp
            # We want |P_perp| = Radius => t = Radius / |D_perp|
            
            d_dot_v = np.einsum("i,in->n", self.vhat, dir_vec)
            
            # vhat is (3,)
            # dir_vec is (3, N)
            # d_dot_v is (N,)
            
            # D_perp component per point
            d_perp_x = x - d_dot_v * self.vhat[0]
            d_perp_y = y - d_dot_v * self.vhat[1]
            d_perp_z = z - d_dot_v * self.vhat[2]
            
            norm_d_perp = np.sqrt(d_perp_x**2 + d_perp_y**2 + d_perp_z**2)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                t = self.radius / norm_d_perp

        # Intersections
        X, Y, Z = t * x, t * y, t * z
        
        i, j = self.lab_to_pixel(X, Y, Z)
        
        # Check bounds
        mask = (i > 0) & (j > 0) & (i < self.m - 1) & (j < self.n - 1) & (t > 0)
        
        return mask, i, j
