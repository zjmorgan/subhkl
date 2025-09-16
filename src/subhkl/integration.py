import numpy as np

from PIL import Image

import skimage.feature


class FindPeaks:
    def __init__(self, filename, peak_integrator=None):
        """
        Find peaks from an image.

        Parameters
        ----------
        filename : str
            Filename of detector image.

        peak_integrator : PeakIntegrator, optional
            Peak integrator algorithm object

        """

        self.im = np.array(Image.open(filename))
        self.peak_integrator = peak_integrator

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

    def fit_convex_hull(self, xp, yp):
        """
        Refine preliminary peak detections and determine peak intensity and
        uncertainty.

        Parameters
        ----------
        xp : array, int
            x pixel coordinates of preliminary peaks
        yp : array, int
            y pixel coordinates of preliminary peaks

        Returns
        -------
        peak_dict : dictionary
            Refined peak geometry, peak intensities, and uncertainties.
            Dictionary keys are (x, y) pixel coordinates of corresponding
            preliminary peak. Values are (peak_hull, peak_intensity, sigma),
            where peak_hull is a ConvexHull describing the peak geometry.

        """
        assert self.peak_integrator is not None, \
            'Must set peak integrator to use fit_convex_hull'

        peak_centers = np.array([xp, yp]).T
        peak_data, peak_hulls = self.peak_integrator.integrate_peaks(
            0,  # Don't care about bank ID
            self.im,
            peak_centers,
            return_hulls=True
        )

        peak_dict = {}
        for (x, y, data, hulls) in zip(xp, yp, peak_data, peak_hulls):
            core_hull, peak_hull, inner_hull, outer_hull = hulls
            _, _, bg_density, peak_intensity, peak_bg_intensity, sigma = data
            peak_dict[(x, y)] = peak_hull, peak_intensity, sigma

        return peak_dict

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
