from enum import Enum
import numpy as np
import numpy.typing as npt


class DetectorShape(str, Enum):
    flat_panel = "flat"
    curved_panel = "curved"


class Detector:
    def __init__(
        self,
        xp: npt.ArrayLike,
        yp: npt.ArrayLike,
        detector_distance: float,
        detector_height: float,
        image_orientation: float = 0,
        panel: DetectorShape = DetectorShape.curved_panel,
    ):
        """
        Parameters
        ----------
        xp, yp : array, float
            Image pixel position.
        detector_distance : float
            Horizontal detector distance.
        detector_height : float
            Vertical detector height.
        image_orientation : float, optional
            Image orientation. The default is 0.
        panel : str
            Detector panel geometry
        """
        self.x = xp
        self.y = yp
        self.panel = panel
        self.distance = detector_distance
        self.height = detector_height
        self.gamma = image_orientation

    def flat_panel(self):
        """
        Place a flat detector image into 3d-spatial coordinates.

        Returns
        -------
        X, Y, Z : array, float
            Spatial positions.

        """
        x = self.x
        y = self.y
        d = self.distance
        h = self.height
        gamma = self.gamma

        X = x * np.cos(gamma) + d * np.sin(gamma)
        Y = y + h
        Z = d * np.cos(gamma) - x * np.sin(gamma)

        return X, Y, Z

    def curved_panel(self):
        """
        Place a curved detector image into 3d-spatial coordinates.

        Returns
        -------
        X, Y, Z : array, float
            Spatial positions.

        """
        x = self.x
        y = self.y
        d = self.distance
        h = self.height

        X = d * np.sin(x / d)
        Y = y + h
        Z = d * np.cos(x / d)

        return X, Y, Z

    def detector_trajectories(self):
        """
        Detector image into 3d-spatial coordinates.

        Returns
        -------
        two_theta : array, float
            In-plane scattering angles
        az_phi : array, float
            Azimuthal scattering angles

        """

        if self.panel == DetectorShape.curved_panel:
            X, Y, Z = self.curved_panel()

        elif self.panel == DetectorShape.flat_panel:
            X, Y, Z = self.flat_panel()
        else:
            raise Exception(f"Detetor shape {self.panel} not supported!!!")

        R = np.sqrt(X**2 + Y**2 + Z**2)
        two_theta = np.rad2deg(np.arccos(Z / R))
        az_phi = np.rad2deg(np.arctan2(Y, X))

        return two_theta, az_phi
