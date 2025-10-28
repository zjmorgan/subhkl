import os
import re
import typing

import numpy as np
import numpy.typing as npt

from h5py import File
from PIL import Image

import skimage.feature
import scipy.optimize

from subhkl.config import (
    beamlines,
    reduction_settings,
    calc_goniometer_rotation_matrix
)


class Peaks:
    def __init__(
        self,
        filename: str,
        instrument: str,
        wavelength_min: typing.Optional[float] = None,
        wavelength_max: typing.Optional[float] = None,
    ):
        """
        Find peaks from an image.

        Parameters
        ----------
        filename : str
            Filename of detector image.

        """

        name, ext = os.path.splitext(filename)

        self.instrument = instrument
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
            self.goniometer_rotation = self.get_goniometer_from_nexus(filename, instrument)
        else:
            self.ims = {0: np.array(Image.open(filename)).T}
            self.wavelength_min, self.wavelength_max = (
                self.get_wavelength_from_settings()
            )

        # Override wavelength or define if TIFF
        if wavelength_min:
            self.wavelength_min = wavelength_min
        if wavelength_max:
            self.wavelength_min = wavelength_max

    # TODO: implement for each instrument...
    def get_wavelength_from_nexus(self, filename: str) -> float:
        print("NOT YET IMPLEMENTED: returning None for wavelength...")
        wavelength_min = None
        wavelength_max = None
        return wavelength_min, wavelength_max

    def get_wavelength_from_settings(self) -> list[float]:
        settings = reduction_settings[self.instrument]
        wavelength_min, wavelength_max = settings.get("Wavelength")
        return wavelength_min, wavelength_max

    def get_goniometer_from_nexus(self, filename: str, instrument: str) -> npt.NDArray:
        """
        Get goniometer rotation matrix from nexus file

        Parameters
        ----------
        filename : str
            Nexus filename
        instrument : str
            Name of the instrument

        Returns
        -------
        matrix : 3x3 numpy array
            The goniometer rotation matrix calculated from the angles in the
            nexus file
        """
        with File(filename) as f:
            return calc_goniometer_rotation_matrix(f, instrument)

    def load_nexus(self, filename: str) -> dict[npt.NDArray]:
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
            keys = f["/entry/"].keys()
            banks = [key for key in keys if re.search(r"bank\d", key)]

            for bank in banks:
                key = "/entry/" + bank + "/event_id"

                b = int(bank.split("bank")[1].split("_")[0])

                array = f[key][()]

                det = detectors.get(b)

                if det is not None:
                    m, n, offset = det["m"], det["n"], det["offset"]

                    bc = np.bincount(array - offset, minlength=m * n)

                    ims[b] = bc.reshape(m, n)

        return ims

    def harvest_peaks(
        self, bank, max_peaks=200, min_pix=50, min_rel_intens=0.5
    ) -> list[npt.NDArray]:
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
        min_rel_intens: float, optional
            Minimum intensity relative to maximum value. The default is 0.5

        Returns
        -------
        i : array, int
            x-pixel coordinates.
        j : array, int
            y-pixel coordinates.

        """
        coords = skimage.feature.peak_local_max(
            self.ims[bank],
            num_peaks=max_peaks,
            min_distance=min_pix,
            threshold_rel=min_rel_intens,
            exclude_border=min_pix * 3,
        )

        return coords[:, 0], coords[:, 1]

    def scale_coordinates(self, bank: int, i: list[int], j: list[int]) -> npt.NDArray:
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
        a: list[float],
        b: list[float],
        theta: list[float],
        scale_x: float,
        scale_y: float,
    ) -> npt.NDArray:
        """
        Scale from pixel coordinates to real units.

        Parameters
        ----------
        a, b : array
            Image coordinates (eigenvalues).
        theta: array
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
            return 0, 0, 0

        S_inv = np.diag([1 / scale_x, 1 / scale_y])

        A = R.T @ np.diag([1 / a**2, 1 / b**2]) @ R

        A_new = S_inv.T @ A @ S_inv

        eigvals, eigvecs = np.linalg.eigh(A_new)

        new_a = 1 / np.sqrt(eigvals[0])
        new_b = 1 / np.sqrt(eigvals[1])

        new_theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        return new_a, new_b, new_theta

    def detector_width_height(self, bank: int) -> npt.NDArray:
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
        self, bank: int, i: list[int], j: list[int]
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
    ) -> npt.NDArray:
        """
        Return image (i,j) using bank number and real-space coordinates (x, y, z).

        Parameters
        ----------
        bank : int
            Bank number.
        x, y, z: array, float
            Real-space coordinates

        Returns
        -------
        i, j : array, int
            Image coordinates.
        """
        p = np.array([X, Y, Z])

        detector = beamlines[self.instrument][bank]

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

    def reflections_mask(self, bank: int, xyz: list[float]) -> npt.NDArray:
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

        detector = beamlines[self.instrument][bank]

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

    def detector_trajectories(self, bank: int, x: float, y: float) -> npt.NDArray:
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
    ) -> npt.NDArray:
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

    def get_detector_peaks(self, **kwargs: dict):
        """
        Get peaks in detector space (rotation, angles, and wavelength).

        Parameters
        ----------
        kwargs : dict
            Method harvest_peaks key-word args.

        Returns
        -------
        R, two_theta, az_phi, lambda: array, float
            Rotations, angles, and wavelength of each peak

        """
        if not self.ims:
            raise Exception("ERROR: Must have images for Peaks first...")

        # Define outputs
        R: list[float] = []
        two_theta: list[float] = []
        az_phi: list[float] = []
        lamda: list[float] = []

        # Calculate angles (two theta and phi), rotation, and wavelength
        for bank in sorted(self.ims.keys()):
            i, j = self.harvest_peaks(bank, **kwargs)
            tt, az = self.detector_trajectories(bank, i, j)
            two_theta += tt.tolist()
            az_phi += az.tolist()
            R += [np.eye(3)] * len(tt)
            lamda += [self.wavelength_min, self.wavelength_max] * len(tt)

        return R, two_theta, az_phi, lamda

    # Write out the output HDF5 peaks file
    def write_hdf5(
        self,
        output_filename: str,
        rotations: list[float],
        two_theta: list[float],
        phi: list[float],
        wavelengths: list[float],
    ):
        """
        Write output HDF5 file for peaks in detector space.

        Parameters
        ----------

        rotations: array, float
            Rotation matrices of peaks.
        two_theta: array, float
            Two theta angles of peaks.
        az_phi: array, float
            Azimuthal phi angles of peaks.
        wavelengths: array, float
            Wavelength min and max of each peak.
        """
        # Write HDF5 input file for indexer
        with File(output_filename, "w") as f:
            f["wavelengths"] = wavelengths
            f["rotations"] = rotations
            f["two_theta"] = two_theta
            f["azimuthal"] = phi
            f["goniometer_rotation"] = self.goniometer_rotation
