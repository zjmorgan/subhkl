import os
import re
import typing
from collections import namedtuple

import numpy as np
import numpy.typing as npt

from h5py import File
from PIL import Image

import skimage.feature
import scipy.optimize

from subhkl.config import (
    beamlines,
    reduction_settings,
    calc_goniometer_rotation_matrix,
    get_rotation_data_from_nexus,
)
from subhkl.convex_hull.peak_integrator import PeakIntegrator
from subhkl.threshold_peak_finder import ThresholdingPeakFinder


DetectorPeaks = namedtuple(
    "DetectorPeaks",
    [
        "R",
        "two_theta",
        "az_phi",
        "wavelength_mins",
        "wavelength_maxes",
        "intensity",
        "sigma",
        "bank",
    ]
)

IntegrationResult = namedtuple(
    "IntegrationResult",
    [
        "h",
        "k",
        "l",
        "intensity",
        "sigma",
        "tt",
        "az",
        "wavelength",
        "bank"
    ]
)


class Peaks:
    def __init__(
        self,
        filename: str,
        instrument: str,
        goniometer_axes: typing.Optional[list[list[float]]] = None,
        goniometer_angles: typing.Optional[list[float]] = None,
        wavelength_min: typing.Optional[float] = None,
        wavelength_max: typing.Optional[float] = None,
    ):
        """
        Find peaks from an image.

        Parameters
        ----------
        filename : str
            Filename of detector image.
        goniometer_axes : list[list[float]]
            Optional axes of the goniometer specified in the same manner as
            Mantid `SetGoniometer`. See also notes in
            `subhkl.config.goniometer.py`. If either this or goniometer_angles
            is not specified, the goniometer rotation will be loaded from the
            file, if possible, and will be set to the identity otherwise.
        goniometer_angles : list[float]
            Optional angles of the goniometer in degrees about the given axes.
        """

        name, ext = os.path.splitext(filename)

        self.instrument = instrument

        if goniometer_axes is not None and goniometer_angles is not None:
            self.goniometer_rotation = calc_goniometer_rotation_matrix(
                goniometer_axes, goniometer_angles
            )
        else:
            # Use identity if goniometer matrix cannot otherwise be loaded
            self.goniometer_rotation = np.eye(3)

        self.wavelength_min = None
        self.wavelength_max = None

        if ext == ".h5":
            self.ims = self.load_nexus(filename)
            # self.wavelength_min, self.wavelength_max = self.get_wavelength_from_nexus(filename)
            self.wavelength_min, self.wavelength_max = (
                self.get_wavelength_from_settings()
            )
            if goniometer_axes is None or goniometer_angles is None:
                self.goniometer_rotation = self.get_goniometer_from_nexus(filename)
        else:
            self.ims = {0: np.array(Image.open(filename)).T}
            self.wavelength_min, self.wavelength_max = (
                self.get_wavelength_from_settings()
            )

        # Override wavelength or define if TIFF
        if wavelength_min:
            self.wavelength_min = wavelength_min
        if wavelength_max:
            self.wavelength_max = wavelength_max

    # TODO: implement for each instrument...
    def get_wavelength_from_nexus(self, filename: str) -> tuple[float, float]:
        print("NOT YET IMPLEMENTED: returning None for wavelength...")
        raise NotImplementedError

    def get_wavelength_from_settings(self) -> tuple[float, float]:
        settings = reduction_settings[self.instrument]
        wavelength_min, wavelength_max = settings.get("Wavelength")
        return wavelength_min, wavelength_max

    def get_goniometer_from_nexus(self, filename: str) -> npt.NDArray:
        """
        Get goniometer rotation matrix from nexus file

        Parameters
        ----------
        filename : str
            Nexus filename

        Returns
        -------
        matrix : 3x3 numpy array
            The goniometer rotation matrix calculated from the angles in the
            nexus file
        """
        axes, angles = get_rotation_data_from_nexus(filename, self.instrument)
        return calc_goniometer_rotation_matrix(axes, angles)

    def load_nexus(self, filename: str) -> dict[int, npt.NDArray]:
        """
        Return images from a Nexus file.

        Parameters
        ----------
        filename: str
            Nexus filename.

        Returns
        -------
        output: dict, npt.NDArray
            Dict of image from Nexus file.

        """

        detectors = beamlines[self.instrument]

        ims = {}

        with File(filename, "r") as f:
            keys = []
            banks = []
            for key in f["/entry/"].keys():
                match = re.match(r"bank(\d+).*", key)
                if match is not None:
                    keys.append(key)
                    banks.append(int(match.groups()[0]))

            for rel_key, bank in zip(keys, banks):
                key = "/entry/" + rel_key + "/event_id"

                array = f[key][()]

                det = detectors.get(str(bank))

                if det is not None:
                    m, n, offset = det["m"], det["n"], det["offset"]

                    bc = np.bincount(array - offset, minlength=m * n)

                    ims[bank] = bc.reshape(m, n)

        return ims

    def harvest_peaks(
        self, bank, max_peaks=200, min_pix=50, min_rel_intensity=0.5, normalize=False
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Locate peak positions in pixel coordinates.

        Parameters
        ----------
        bank : int
            Bank number.
        max_peaks: int
            Maximum number of peaks to limit for output. The default is 200.
        min_pix : int, optional
            Minimum pixel distance between peaks. The default is 50.
        min_rel_intensity: float, optional
            Minimum intensity relative to maximum value. The default is 0.5
        normalize: bool, optional
            Whether to apply adaptive normalization to the image before
            searching for peaks


        Returns
        -------
        i : array, int
            x-pixel coordinates (row).
        j : array, int
            y-pixel coordinates (column).

        """
        im = self.ims[bank]
        if normalize:
            blur = scipy.ndimage.gaussian_filter(im, 4)
            div = scipy.ndimage.gaussian_filter(im, 60)
            processed = blur / div
        else:
            processed = im

        coords = skimage.feature.peak_local_max(
            processed,
            num_peaks=max_peaks,
            min_distance=min_pix,
            threshold_rel=min_rel_intensity,
            exclude_border=min_pix * 3,
        )

        return coords[:, 0], coords[:, 1]

    def harvest_peaks_thresholding(
        self,
        bank: int,
        noise_cutoff_quantile: float = 0.9,
        min_peak_dist_pixels: float = 8.0,
        blur_kernel_sigma: int = 5,
        open_kernel_size_pixels: int = 3,
        mask_file: str | None = None,
        mask_rel_erosion_radius: float = 0.05,
        show_steps: bool = False,
        show_scale: str = "linear"
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Find peaks using a thresholding algorithm.

        Parameters
        ----------
        bank : int
            Bank ID of the image to search for peaks in
        noise_cutoff_quantile : float
            The quantile at which to threshold noise
        min_peak_dist_pixels : int
            Minimum distance in pixels allowed between detected peaks
        blur_kernel_sigma : int
            Typical size of the smaller blurring kernel used in difference-of-
            Gaussians blob detection filter
        open_kernel_size_pixels : int
            Size of the opening kernel; either 3 5, or 7. 3 catches weaker peaks
            but may introduce false positive detections. 7 is mainly useful
            for high resolution images.
        mask_file : str | None
            Optional file containing a mask that indicates (by nonzero pixels)
            which pixels in the image should be IGNORED for peak detection
        mask_rel_erosion_radius : float
            Radius (relative to smaller size of the image) by which to
            expand the ignored region of the mask, if it is given
        show_steps : bool
            Whether to show a plot visualizing the intermediate steps of the
            algorithm. Useful for tuning parameters for a new instrument
        show_scale : str
            Scale of color units in image plots. Either "log" or "linear".
            (currently only supports "linear")
        """
        alg = ThresholdingPeakFinder(
            noise_cutoff_quantile=noise_cutoff_quantile,
            min_peak_dist_pixels=min_peak_dist_pixels,
            blur_kernel_sigma=blur_kernel_sigma,
            open_kernel_size_pixels=open_kernel_size_pixels,
            mask_file=mask_file,
            mask_rel_erosion_radius=mask_rel_erosion_radius,
            show_steps=show_steps,
            show_scale=show_scale
        )

        coords = alg.find_peaks(self.ims[bank])
        return coords[:, 0], coords[:, 1]

    def scale_coordinates(self, bank: int, i: npt.NDArray, j: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Scale from pixel coordinates to real positions.

        Parameters
        ----------
        bank : int
            Bank number.
        i, j : array, int
            Image coordinates.

        Returns
        -------
        x, y : array, float
            Image pixel position.

        """
        width, height = self.detector_width_height(bank)

        m, n = self.ims[bank].shape

        return (i / (m - 1) - 0.5) * width, (j / (n - 1) - 0.5) * height

    def scale_ellipsoid(
        self,
        a: float,
        b: float,
        theta: float,
        scale_x: float,
        scale_y: float,
    ) -> tuple[float, float, float]:
        """
        Scale from pixel coordinates to real units.

        Parameters
        ----------
        a, b : float
            Image coordinates (eigenvalues).
        theta: float
            Orientation angle.
        scale_x, scale_y : float
            Pixel scaling factors.

        Returns
        -------
        x, y : array, float
            Image pixel size.

        """
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        if np.isclose(a, 0) or np.isclose(b, 0):
            return 0., 0., 0.

        S_inv = np.diag([1 / scale_x, 1 / scale_y])

        A = R.T @ np.diag([1 / a**2, 1 / b**2]) @ R

        A_new = S_inv.T @ A @ S_inv

        eigvals, eigvecs = np.linalg.eigh(A_new)

        new_a = 1 / np.sqrt(eigvals[0])
        new_b = 1 / np.sqrt(eigvals[1])

        new_theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        return new_a, new_b, new_theta

    def detector_width_height(self, bank: int) -> tuple[float, float]:
        """
        Return bank's width and height for instrument.

        Parameters
        ----------
        bank : int
            Bank number.

        Returns
        -------
        width, height : array, float
            Width and height of bank.
        """
        detector = beamlines[self.instrument][bank]

        width = detector["width"]
        height = detector["height"]

        return width, height

    def transform_from_detector(
        self, bank: int, i: list[float] | npt.NDArray, j: list[float] | npt.NDArray
    ) -> npt.NDArray:
        """
        Return real-space coordinates from detector using bank and image (i,j).

        Parameters
        ----------
        bank : int
            Bank number.
        i, j : array, int
            Image coordinates.

        Returns
        -------
        x, y, z: array, float
            Real-space coordinates
        """
        bank_id = str(bank)
        detector = beamlines[self.instrument][bank_id]

        m = detector["m"]
        n = detector["n"]

        width = detector["width"]
        height = detector["height"]

        c = np.array(detector["center"])
        vhat = np.array(detector["vhat"])

        u = np.array(i) / (m - 1) * width
        v = np.array(j) / (n - 1) * height

        dv = np.einsum("n,d->nd", v, vhat)

        if detector["panel"] == "flat":
            uhat = np.array(detector["uhat"])

            du = np.einsum("n,d->nd", u, uhat)

        else:
            radius = detector["radius"]
            rhat = np.array(detector["rhat"])

            w = np.cross(vhat, rhat)

            dvr = np.einsum("n,d->nd", radius * np.sin(u / radius), w)
            dr = np.einsum("n,d->nd", radius * (np.cos(u / radius) - 1), rhat)

            du = dr + dvr

        return (c + du + dv).T

    def transform_to_detector(
        self, bank: int, X: float, Y: float, Z: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Return image (i,j) using bank number and real-space coordinates (x, y, z).

        Parameters
        ----------
        bank : int
            Bank number.
        X, Y, Z: array, float
            Real-space coordinates

        Returns
        -------
        i, j : array, int
            Image coordinates.
        """
        p = np.array([X, Y, Z])

        detector = beamlines[self.instrument][str(bank)]

        m = detector["m"]
        n = detector["n"]

        width = detector["width"]
        height = detector["height"]

        c = np.array(detector["center"])
        vhat = np.array(detector["vhat"])

        dw = width / (m - 1)
        dh = height / (n - 1)

        j = np.clip(np.dot(p.T - c, vhat) / dh, 0, n)

        if detector["panel"] == "flat":
            uhat = np.array(detector["uhat"])

            i = np.clip(np.dot(p.T - c, uhat) / dw, 0, m)

        else:
            radius = detector["radius"]
            rhat = np.array(detector["rhat"])

            d = p.T - c - (np.dot(p.T - c, vhat)[:, np.newaxis] * vhat)

            what = np.cross(vhat, rhat)

            dt = 2 * np.arctan(-np.dot(d, rhat) / np.dot(d, what))
            dt = np.mod(dt, 2 * np.pi)

            i = np.clip(dt * (radius / dw), 0, m)

        return i.astype(int), j.astype(int)

    def allow_centering(self, h, k, l, centering="P"):
        if centering == "P":
            mask = np.full(l.shape, True, dtype=bool)
        elif centering == "A":
            mask = (k + l) % 2 == 0
        elif centering == "B":
            mask = (h + l) % 2 == 0
        elif centering == "C":
            mask = (h + k) % 2 == 0
        elif centering == "I":
            mask = (h + k + l) % 2 == 0
        elif centering == "F":
            mask = ((h + k) % 2 == 0) & ((h + l) % 2 == 0) & ((k + l) % 2 == 0)
        elif centering == "R":
            mask = (h + k + l) % 3 == 0
        else:
            raise ValueError("Invalid centering")

        return h[mask], k[mask], l[mask]

    def cartesian_matrix_metric_tensor(self, a, b, c, alpha, beta, gamma):
        G = np.array(
            [
                [a ** 2, a * b * np.cos(gamma), a * c * np.cos(beta)],
                [b * a * np.cos(gamma), b ** 2, b * c * np.cos(alpha)],
                [c * a * np.cos(beta), c * b * np.cos(alpha), c ** 2],
            ]
        )

        Gstar = np.linalg.inv(G)

        B = scipy.linalg.cholesky(Gstar, lower=False)

        return B, Gstar

    def reflections(self, a, b, c, alpha, beta, gamma, centering="P", d_min=2):
        constants = a, b, c, *np.deg2rad([alpha, beta, gamma])
        B, Gstar = self.cartesian_matrix_metric_tensor(*constants)

        astar, bstar, cstar = np.sqrt(np.diag(Gstar))

        h_max = int(np.floor(1 / d_min / astar))
        k_max = int(np.floor(1 / d_min / bstar))
        l_max = int(np.floor(1 / d_min / cstar))

        h, k, l = np.meshgrid(
            np.arange(-h_max, h_max + 1),
            np.arange(-k_max, k_max + 1),
            np.arange(-l_max, l_max + 1),
            indexing="ij",
        )

        hkl = [h.flatten(), k.flatten(), l.flatten()]
        h, k, l = hkl

        d = 1 / np.sqrt(np.einsum("ij,jl,il->l", Gstar, hkl, hkl))

        mask = (d > d_min) & (d < np.inf)

        return self.allow_centering(h[mask], k[mask], l[mask], centering)

    def reflections_mask(self, bank: int, xyz: list[float]) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return mask  using bank number and real-space coordinates (x, y, z).

        Parameters
        ----------
        bank : int
            Bank number.
        xyz: array, float
            Real-space coordinates

        Returns
        -------
        mask: array, int
            Mask for reflections.
        i, j : array, int
            Image coordinates.
        """
        x, y, z = xyz

        detector = beamlines[self.instrument][str(bank)]

        m = detector["m"]
        n = detector["n"]

        c = np.array(detector["center"])
        vhat = np.array(detector["vhat"])

        if detector["panel"] == "flat":
            uhat = np.array(detector["uhat"])
            norm = np.cross(uhat, vhat)

            d = np.einsum("i,in->n", norm, [x, y, z])
            t = np.dot(c, norm) / d

        else:
            radius = detector["radius"]

            d = np.einsum("i,in->n", vhat, [x, y, z])

            norm = np.sqrt(
                (x - d * vhat[0]) ** 2 + (y - d * vhat[1]) ** 2 + (z - d * vhat[2]) ** 2
            )

            t = radius / norm

        X, Y, Z = t * x, t * y, t * z

        i, j = self.transform_to_detector(bank, X, Y, Z)

        mask = (i > 0) & (j > 0) & (i < m - 1) & (j < n - 1) & (t > 0)

        return mask, i, j

    def detector_trajectories(
        self,
        bank: int,
        x: list[float] | npt.NDArray,
        y: list[float] | npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Calculate detector trajectories.

        Parameters
        ----------
        bank : int
            Bank number.
        x, y : array, float
            Pixel position in physical units.

        Returns
        -------
        two_theta : array, float
            Scattering angles in degrees.
        az_phi : array, float
            Azimuthal angles in degrees.

        """

        X, Y, Z = self.transform_from_detector(bank, x, y)

        R = np.sqrt(X**2 + Y**2 + Z**2)
        two_theta = np.rad2deg(np.arccos(Z / R))
        az_phi = np.rad2deg(np.arctan2(Y, X))

        return two_theta, az_phi

    def peak(self, x, y, A, B, mu_x, mu_y, sigma_1, sigma_2, theta):
        """
        Calculate detector trajectories.

        Parameters
        ----------
        x, y : array, float
            Pixel position in physical units.

        A, B: float
            Coefficients in linear function A * shape + B

        mu_x, mu_y: float
            Gaussian center coordinates

        sigma_1, sigma_2: float
            Gaussian standard deviations

        theta: float
            Gaussian rotation angle in radians

        Returns
        -------
        two_theta : array, float
            Scattering angles in degrees.
        az_phi : array, float
            Azimuthal angles in degrees.

        """

        a = 0.5 * (np.cos(theta) ** 2 / sigma_1**2 + np.sin(theta) ** 2 / sigma_2**2)
        b = 0.5 * (np.sin(theta) ** 2 / sigma_1**2 + np.cos(theta) ** 2 / sigma_2**2)
        c = 0.5 * (1 / sigma_1**2 - 1 / sigma_2**2) * np.sin(2 * theta)

        shape = np.exp(
            -(a * (x - mu_x) ** 2 + b * (y - mu_y) ** 2 + c * (x - mu_x) * (y - mu_y))
        )

        return A * shape + B

    def residual(self, params, x, y, z) -> npt.NDArray:
        """
        Calculate residual of detector trajectory.

        Parameters
        ----------
        params : dict
            Parameters for peak calculation.
        x, y, z: float
            Real-space coordinates of peak

        Returns
        -------
        output: array, float
            Residual of the detector trajectory

        """

        return (self.peak(x, y, *params) - z).flatten()

    def transform_ellipsoid(
        self,
        sigma_1: npt.ArrayLike,
        sigma_2: npt.ArrayLike,
        theta: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Transform to ellipsoid using sigma and theta.

        Parameters
        ----------
        sigma_1, sigma_2 : array, float
            Sigma for ellipsoid
        theta: array, float
            Angle theta for ellipsoid

        Returns
        -------
        sigma_x, sigma_y, rho: array, float
            Ellipsoid coordinates transformed

        """

        sigma_x = np.hypot(sigma_1 * np.cos(theta), sigma_2 * np.sin(theta))
        sigma_y = np.hypot(sigma_1 * np.sin(theta), sigma_2 * np.cos(theta))
        rho = (sigma_1**2 - sigma_2**2) * np.sin(2 * theta) / (2 * sigma_x * sigma_y)

        return sigma_x, sigma_y, rho

    def intensity(
        self,
        A: float,
        B: float,
        sigma1: float,
        sigma2: float,
        cov_matrix: npt.ArrayLike,
    ) -> tuple[float, float]:
        """
        Calculated intensity of peak.

        Parameters
        ----------
        A : float
            Input scale factor (A * x + B)
        B : float
            Input for shift factor (A * x + B)
        sigma1: float
            Sigma 1 for intensity calculation.
        sigma2: float
            Sigma 1 for intensity calculation.
        cov_matrix: array, float
            Covariance matrix

        Returns
        -------
        I, sigma: array, float
            Intensity and sigma.

        """

        quantity_I = A * 2 * np.pi * sigma1 * sigma2 - B

        dI = np.array(
            [
                2 * np.pi * sigma1 * sigma2,
                -1,
                2 * np.pi * A * sigma2,
                2 * np.pi * A * sigma1,
            ]
        )

        sigma = np.sqrt(dI @ cov_matrix @ dI.T)

        return quantity_I, sigma

    def fit(self, xp, yp, im, roi_pixels=50):
        """
        Fit x, y, image data to peaks.

        Parameters
        ----------
        xp : array, float
            X data
        yp : array, float
            Y data
        im: array, float
            Image array input
        roi_pixels: float
            Region of interest size in pixels. The default is 50.

        Returns
        -------
        I, sigma: array, float
            Intensity and sigma.

        """

        peak_dict = {}

        X, Y = np.meshgrid(
            np.arange(im.shape[0]), np.arange(im.shape[1]), indexing="ij"
        )

        for ind, (x_val, y_val) in enumerate(zip(xp[:], yp[:])):
            x_min = int(max(x_val - roi_pixels, 0))
            x_max = int(min(x_val + roi_pixels + 1, im.shape[0]))

            y_min = int(max(y_val - roi_pixels, 0))
            y_max = int(min(y_val + roi_pixels + 1, im.shape[1]))

            x = X[x_min:x_max, y_min:y_max].copy()
            y = Y[x_min:x_max, y_min:y_max].copy()

            z = im[x_min:x_max, y_min:y_max].copy()

            x0 = np.array(
                [
                    z.max(),
                    z.min(),
                    x_val,
                    y_val,
                    roi_pixels / 6,
                    roi_pixels / 6,
                    0,
                ]
            )

            xmin = np.array(
                [
                    z.min(),
                    0,
                    x_val - roi_pixels * 0.5,
                    y_val - roi_pixels * 0.5,
                    1,
                    1,
                    -np.pi / 2,
                ]
            )

            xmax = np.array(
                [
                    2 * z.max(),
                    z.mean(),
                    x_val + roi_pixels * 0.5,
                    y_val + roi_pixels * 0.5,
                    roi_pixels / 3,
                    roi_pixels / 3,
                    np.pi / 2,
                ]
            )

            if np.all(x0 > xmin) and np.all(x0 < xmax):
                bounds = np.array([xmin, xmax])

                args = (x, y, z)

                sol = scipy.optimize.least_squares(
                    self.residual, x0=x0, bounds=bounds, args=args, loss="soft_l1"
                )

                J = sol.jac
                inv_cov = J.T.dot(J)

                if np.linalg.det(inv_cov) > 0:
                    A, B, mu_1, mu_2, sigma_1, sigma_2, theta = sol.x

                    inds = [0, 1, 4, 5]

                    cov = np.linalg.inv(inv_cov)[inds][:, inds]

                    quantity_I, sig = self.intensity(A, B, sigma_1, sigma_2, cov)

                    if quantity_I < 10 * sig:
                        mu_1, mu_2 = x_val, y_val
                        sigma_1, sigma_2, theta = 0.0, 0.0, 0.0

                    items = mu_1, mu_2, sigma_1, sigma_2, theta

                    peak_dict[(x_val, y_val)] = items

        return peak_dict

    def get_detector_peaks(
        self,
        harvest_peaks_kwargs: dict,
        integration_params: dict,
        show_progress: bool = False,
        visualize: bool = False,
        file_prefix: str | None = None
    ) -> DetectorPeaks:
        """
        Get peaks in detector space (rotation, angles, and wavelength)
        and integrate using convex hull algorithm.

        Parameters
        ----------
        harvest_peaks_kwargs : dict
            Dictionary containing key "algorithm" which specifies either
            "peak_local_max" or "thresholding" algorithm to use for peak finding.
            Should also contain keyword arguments for `harvest_peaks` (if using
            "peak_local_max" algorithm) or `harvest_peaks_thresholding` (if
            using "thresholding" algorithm)
        integration_params : dict
            Parameters for convex hull peak integration algorithm. Must contain
            keys "region_growth_distance_threshold", "region_growth_minimum_intensity",
            "region_growth_maximum_pixel_radius", "peak_center_box_size",
            "peak_smoothing_window_size", "peak_minimum_pixels",
            "peak_minimum_signal_to_noise", "peak_pixel_outlier_threshold"
        show_progress : bool
            Whether to show progress messages
        visualize : bool
            Whether to generate visualizations while running the detection
            algorithm
        file_prefix : str | None
            If generating visualizations, an optional file prefix to add to
            output files

        Returns
        -------
        detector_peaks : DetectorPeaks
            namedtuple of Rotations, angles, wavelength_mins, wavelength_maxes, 
            intensity and sigma of each peak

        """
        if not self.ims:
            raise Exception("ERROR: Must have images for Peaks first...")

        if visualize:
            import matplotlib.pyplot as plt
        else:
            plt = None

        # Define outputs
        R: list[float] = []
        two_theta: list[float] = []
        az_phi: list[float] = []
        lamda_min: list[float] = []
        lamda_max: list[float] = []
        intensity: list[float] = []
        sigma: list[float] = []
        banks: list[int] = []

        integrator = PeakIntegrator.build_from_dictionary(integration_params)
        finder_algorithm = harvest_peaks_kwargs.pop("algorithm")

        # Calculate angles (two theta and phi), rotation, and wavelength
        for bank in sorted(self.ims.keys()):
            print(f"Processing bank {bank}")

            # Find candidate peaks
            if finder_algorithm == "peak_local_max":
                i, j = self.harvest_peaks(bank, **harvest_peaks_kwargs)
            elif finder_algorithm == "thresholding":
                i, j = self.harvest_peaks_thresholding(bank, **harvest_peaks_kwargs)
            else:
                raise ValueError("Invalid finder algorithm")
            if show_progress:
                print(f"Found {len(i)} candidate peaks")

            if visualize:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(self.ims[bank], norm="log", cmap="binary")
                axes[0].scatter(j, i, marker="1", c="blue")
                axes[0].set_title("Candidate peaks")
            else:
                fig, axes = None, None

            centers = np.stack([i, j], axis=-1)

            # Integrate peaks
            if visualize:
                int_result, hulls = integrator.integrate_peaks(bank, self.ims[bank], centers, return_hulls=True)
            else:
                int_result = integrator.integrate_peaks(bank, self.ims[bank], centers)
                hulls = None

            bank_intensity = np.array([peak_in for _, _, _, peak_in, _, _ in int_result])
            bank_sigma = np.array([peak_sigma for _, _, _, _, _, peak_sigma in int_result])
            keep = [peak_in is not None for peak_in in bank_intensity]

            if visualize:
                plt_im = axes[1].imshow(self.ims[bank], norm="log", cmap="binary")
                if show_progress:
                    for peak_in, peak_sigma in zip(bank_intensity[keep], bank_sigma[keep]):
                        print(f'SNR: {peak_in / peak_sigma}')

                for _, hull, _, _ in hulls:
                    if hull is not None:
                        for simplex in hull.simplices:
                            axes[1].plot(hull.points[simplex, 1], hull.points[simplex, 0], c="red")
                axes[1].set_title("Convex hulls")
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
                fig.colorbar(plt_im, cbar_ax)
                output_file = str(bank) + ".png"
                if file_prefix is not None:
                    output_file = file_prefix + "_" + output_file
                fig.savefig(output_file)
                plt.show()

            # Only add integrated peaks to data
            if sum(keep) > 0:
                i, j = i[keep], j[keep]
                bank_intensity = bank_intensity[keep]
                bank_sigma = bank_sigma[keep]

                # Calculate peak angles
                tt, az = self.detector_trajectories(bank, i, j)

                # Add peak data to output
                two_theta += tt.tolist()
                az_phi += az.tolist()
                R += [self.goniometer_rotation] * len(tt)
                lamda_min += [self.wavelength_min] * len(tt)
                lamda_max += [self.wavelength_max] * len(tt)
                intensity += bank_intensity.tolist()
                sigma += bank_sigma.tolist()
                banks += [bank] * sum(keep)

                print(f"Integrated {len(i)}/{len(centers)} peaks")
            else:
                print("Bank had 0 peaks")

        return DetectorPeaks(R, two_theta, az_phi, lamda_min, lamda_max, intensity, sigma, banks)

    def integrate(
        self,
        peak_dict,
        integration_params,
        create_visualizations=False,
        show_progress=False,
        file_prefix=None
    ):
        integrator = PeakIntegrator.build_from_dictionary(integration_params)

        h, k, l = [], [], []
        intensity, sigma = [], []
        tt, az = [], []
        wavelength = []
        banks = []

        for bank, peaks in peak_dict.items():
            bank_i, bank_j, bank_h, bank_k, bank_l, bank_wl = peaks
            centers = np.stack([bank_i, bank_j], axis=-1)
            bank_tt, bank_az = self.detector_trajectories(bank, bank_i, bank_j)

            int_result, hulls = integrator.integrate_peaks(bank, self.ims[bank], centers, return_hulls=True)

            bank_intensity = np.array([peak_in for _, _, _, peak_in, _, _ in int_result])
            bank_sigma = np.array([peak_sigma for _, _, _, _, _, peak_sigma in int_result])
            keep = [peak_in is not None for peak_in in bank_intensity]
            if show_progress:
                print(f"Integrated {sum(keep)} peaks out of {len(keep)} predicted")
            
            if create_visualizations:
                import matplotlib.pyplot as plt
                plt.rc("font", size=8)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(self.ims[bank], norm="log", cmap="binary")
                axes[0].set_title("Predicted peaks")
                axes[0].scatter(bank_j, bank_i, marker="1", c="blue")
                for p_i, p_j, p_h, p_k, p_l in zip(bank_i, bank_j, bank_h, bank_k, bank_l):
                    axes[0].text(p_j, p_i, f"({p_h}, {p_k}, {p_l})")
            	
                plt_im = axes[1].imshow(self.ims[bank], norm="log", cmap="binary")
                axes[1].set_title("Integrated peaks")
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
                fig.colorbar(plt_im, cbar_ax)
            	
                for _, hull, _, _ in hulls:
                    if hull is not None:
                        for simplex in hull.simplices:
                            axes[1].plot(hull.points[simplex, 1], hull.points[simplex, 0], c="red")

                output_file = str(bank) + "_int.png"
                if file_prefix is not None:
                    output_file = file_prefix + output_file
                fig.savefig(output_file)
                plt.show()

            h.extend(bank_h[keep])
            k.extend(bank_k[keep])
            l.extend(bank_l[keep])
            intensity.extend(bank_intensity[keep])
            sigma.extend(bank_sigma[keep])
            tt.extend(bank_tt[keep])
            az.extend(bank_az[keep])
            wavelength.extend(bank_wl[keep])
            banks.extend([bank] * sum(keep))

        return IntegrationResult(h, k, l, intensity, sigma, tt, az, wavelength, banks)

    def coverage(self, h, k, l, UB, wavelength, tol=1e-3):
        wl_min, wl_max = wavelength

        hkl = np.stack([h, k, l], axis=0)
        print(UB.shape, hkl.shape)

        Qx, Qy, Qz = np.einsum("ij,jk->ik", 2 * np.pi * UB, hkl)
        Q = np.sqrt(Qx ** 2 + Qy ** 2 + Qz ** 2)

        lamda = -4 * np.pi * Qz / Q ** 2
        mask = np.logical_and(lamda > wl_min, lamda < wl_max)

        Qx, Qy, Qz, Q = Qx[mask], Qy[mask], Qz[mask], Q[mask]

        h, k, l, lamda = h[mask], k[mask], l[mask], lamda[mask]

        tt = -2 * np.arcsin(Qz / Q)
        az = np.arctan2(Qy, Qx)

        x = np.sin(tt) * np.cos(az)
        y = np.sin(tt) * np.sin(az)
        z = np.cos(tt)

        coords = np.vstack((x, y, z)).T
        rounded = np.round(coords / tol).astype(int)

        _, ind, mult = np.unique(rounded, axis=0, return_index=True, return_counts=True)

        return [x[ind], y[ind], z[ind]], [h[ind], k[ind], l[ind]], lamda[ind], mult

    def predict_peaks(self, a, b, c, alpha, beta, gamma, centering, d_min, UB):
        h, k, l = self.reflections(a, b, c, alpha, beta, gamma, centering, d_min)
        wavelength = [self.wavelength_min, self.wavelength_max]
        xyz, hkl, wl, mult = self.coverage(h, k, l, UB, wavelength)
        h, k, l = hkl

        peak_dict = {}
        for bank in sorted(self.ims.keys()):
            mask, i, j = self.reflections_mask(bank, xyz)
            peak_dict[bank] = [i[mask], j[mask], h[mask], k[mask], l[mask], wl[mask]]

        return peak_dict

    # Write out the output HDF5 peaks file
    def write_hdf5(
        self,
        output_filename: str,
        rotations: list[float],
        two_theta: list[float],
        az_phi: list[float],
        wavelength_mins: list[float],
        wavelength_maxes: list[float],
        intensity: list[float],
        sigma: list[float],
        bank: list[int]
    ):
        """
        Write output HDF5 file for peaks in detector space.

        Parameters
        ----------
        output_filename: str
            Name of file to write to
        rotations: array, float
            Rotation matrices of peaks.
        two_theta: array, float
            Two theta angles of peaks.
        az_phi: array, float
            Azimuthal phi angles of peaks.
        wavelength_mins: array, float
            Wavelength min of each peak.
        wavelength_maxes: array, float
            Wavelength max of each peak.
        intensity: array, float
            Integrated intensity of each peak
        sigma: array, float
            Uncertainty in integrated intensity of each peak
        bank: array, int
            Detector id for each peak
        """
        # Write HDF5 input file for indexer
        with File(output_filename, "w") as f:
            f["wavelength_mins"] = wavelength_mins
            f["wavelength_maxes"] = wavelength_maxes
            f["rotations"] = rotations
            f["two_theta"] = two_theta
            f["azimuthal"] = az_phi
            f["intensity"] = intensity
            f["sigma"] = sigma
            f["goniometer_rotation"] = self.goniometer_rotation
            f["bank"] = bank
