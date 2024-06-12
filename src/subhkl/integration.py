import numpy as np

from PIL import Image

import skimage.feature


class FindPeaks:
    def __init__(self, filename):
        """
        Find peaks from an image.

        Parameters
        ----------
        filename : str
            Filename of detector image.

        """

        self.im = np.array(Image.open(filename))

    def harvest_peaks(self, min_pix=50, min_rel_intens=0.5):
        """
        Locate peak positions in pixel coordinates.

        Parameters
        ----------
        min_pix : int, optional
            Minimum pixel distance between peaks. The default is 50.
        min_rel_intens: float, optional
            Minimum intensity relative to maximum value. The default is 0.5

        Returns
        -------
        xp : array, int
            x-pixel coordinates.
        yp : array, int
            y-pixel coordinates.

        """

        coords = skimage.feature.peak_local_max(
            self.im, min_distance=min_pix, threshold_rel=min_rel_intens
        )

        return coords[:, 1], coords[:, 0]

    def scale_coordinates(self, xp, yp, scale_x, scale_y):
        """
        Scale from pixel coordinates to real positions

        Parameters
        ----------
        xp, yp : array, int
            Image coordinates.
        scale_x, scale_y : float
            Pixel scaling factors.

        Returns
        -------
        x, y : array, float
            Image pixel position.

        """

        ny, nx = self.im.shape

        return (xp - nx / 2) * scale_x, (yp - ny / 2) * scale_y

    def flat_panel(self, x, y, d, h, gamma=0):
        """
        Place a flat detector image into 3d-spatial coordinates.

        Parameters
        ----------
        x, y : array, float
            Image pixel position.
        d : float
            Horizontal detector distance.
        h : float
            Vertical detector height.
        gamma : float, optional
            Image orientation. The default is 0.

        Returns
        -------
        X, Y, Z : array, float
            Spatial positions.

        """

        X = x * np.cos(gamma) + d * np.sin(gamma)
        Y = y + h
        Z = d * np.cos(gamma) - x * np.sin(gamma)

        return X, Y, Z

    def curved_panel(self, x, y, d, h, gamma=0):
        """
        Place a curved detector image into 3d-spatial coordinates.

        Parameters
        ----------
        x, y : array, float
            Image pixel position.
        d : float
            Horizontal detector distance.
        h : float
            Vertical detector height.
        gamma : float, optional
            Image orientation. The default is 0.

        Returns
        -------
        X, Y, Z : array, float
            Spatial positions.

        """

        X = d * np.sin(x / d)
        Y = y + h
        Z = d * np.cos(x / d)

        return X, Y, Z

    def detector_trajectories(self, x, y, d, h, gamma, panel="curved"):
        if panel == "curved":
            X, Y, Z = self.curved_panel(x, y, d, h, gamma)
        else:
            X, Y, Z = self.flat_panel(x, y, d, h, gamma)

        R = np.sqrt(X**2 + Y**2 + Z**2)
        two_theta = np.rad2deg(np.arccos(Z / R))
        az_phi = np.rad2deg(np.arctan2(Y, X))

        return two_theta, az_phi
