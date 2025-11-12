import os
from functools import partial

import h5py

import numpy as np

import scipy.linalg
import scipy.spatial
import scipy.interpolate

import pyswarms


class VectorizedObjective:
    def __init__(self, B, kf_ki_dir, wavelength, angle, tol=0.15):
        """
        Parameters
        ----------
        B : array (3, 3)
            B matrix
        kf_ki_dir : array (3, M)
            difference between incident and scattering directions for M
            reflections
        wavelength : array (M, 2)
            wavelength lower and upper bounds for M reflections
        angle
        tol
        """
        self.B = B
        self.kf_ki_dir = kf_ki_dir
        self.angle = angle
        self.tol = tol

        wl_min = np.full(kf_ki_dir.shape[1], wavelength[0])  # (M)
        wl_max = np.full(kf_ki_dir.shape[1], wavelength[1])  # (M)

        self.lamda = np.linspace(wl_min, wl_max, 100).T
        # (M, 100)

    def indexer(self, UB):
        """
        Laue indexer for a given collection of :math:`UB` matrices

        Parameters
        ----------
        UB : array, (S, 3, 3)
            S, 3x3 sample oriented lattice matrices.

        Returns
        -------
        err : array, (S)
            Indexing cost for each UB
        num : array (S)
            Number of peaks indexed for each UB
        hkl : array (S, M, 3)
            Miller indices of peaks for each UB
        lamda : array (S, M)
            Resolved wavelength of each peak

        """
        UB_inv = np.linalg.inv(UB)

        hkl_lamda = np.einsum("sij,jm->sim", UB_inv, self.kf_ki_dir)
        # (S, 3, M), M = number of reflections

        hkl = hkl_lamda[:, :, :, None] / self.lamda[None, None, :, :]
        # (S, 3, M, 100)

        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl  # (S, 3, M, 100)

        dist = np.einsum("sij,sjmd->simd", UB, diff_hkl)
        # (S, 3, M, 100)

        dist = np.linalg.norm(dist, axis=1)  # (S, M, 100)
        ind = np.argmin(dist, axis=2, keepdims=True)  # (S, M, 1)

        err = np.take_along_axis(dist, ind, axis=2)[:, :, 0]  # (S, M)
        lamda = np.take_along_axis(self.lamda[None], ind, axis=2)[:, :, 0]
        # (S, M)

        hkl = hkl_lamda / lamda[:, None, :]  # (S, 3, M)

        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl  # (S, 3, M)

        mask = (np.abs(diff_hkl) < self.tol).all(axis=1)  # (S, M)
        num = np.sum(mask, axis=1)  # (S)

        return np.sum(err**2, axis=1), num, int_hkl.transpose((0, 2, 1)), lamda

    def orientation_U(self, param):
        """
        Compute orientation matrices (U) from angles

        Parameters
        ----------
        param : array, (3, S)
            Rotation parameters. S = population size

        Returns
        -------
        U : array (S, 3, 3)
            sample orientation matrices for each set of input parameters.

        """

        u0, u1, u2 = param
        theta = np.arccos(1 - 2 * u0)
        phi = 2 * np.pi * u1

        w = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )

        omega = self.angle(u2)

        U = scipy.spatial.transform.Rotation.from_rotvec((omega * w).T).as_matrix()

        return U

    def __call__(self, x):
        """
        Objective function.

        Parameters
        ----------
        x : array (3, S)
            Refineable parameters. S = population size

        Returns
        -------
        neg_ind : int
            Negative number of peaks indexed.

        """

        U = self.orientation_U(x)
        if len(U.shape) == 2:
            U = U[None]

        UB = np.einsum("sij,jk->sik", U, self.B)
        # (S, 3, 3)

        error, num, hkl, lamda = self.indexer(UB)

        return error


class FindUB:
    """
    Optimizer of crystal orientation from peaks and known lattice parameters.

    Attributes
    ----------
    a, b, c : float
        Lattice constants in ansgroms.
    alpha, beta, gamma : float
        Lattice angles in degrees.

    """

    def __init__(self, filename=None):
        """
        Find :math:`UB` from peaks.

        Parameters
        ----------
        filename : str, optional
            Filename of found peaks. The default is None.

        """

        if filename is not None:
            self.load_peaks(filename)

        t = np.linspace(0, np.pi, 1024)
        cdf = (t - np.sin(t)) / np.pi

        self._angle = scipy.interpolate.interp1d(cdf, t, kind="linear")

    def load_peaks(self, filename):
        """
        Obtain peak information from .h5 file.

        Parameters
        ----------
        filename : str
            HDF5 file of peak information.

        """

        with h5py.File(os.path.abspath(filename), "r") as f:
            self.a = f["sample/a"][()]
            self.b = f["sample/b"][()]
            self.c = f["sample/c"][()]
            self.alpha = f["sample/alpha"][()]
            self.beta = f["sample/beta"][()]
            self.gamma = f["sample/gamma"][()]
            self.wavelength = f["instrument/wavelength"][()]
            self.R = f["goniometer/R"][()]
            self.two_theta = f["peaks/scattering"][()]
            self.az_phi = f["peaks/azimuthal"][()]
            self.centering = f["sample/centering"][()].decode("utf-8")

    def uncertainty_line_segements(self):
        """
        The scattering vector scaled with the (unknown) wavelength.

        Returns
        -------
        kf_ki_dir : list
            Difference between scattering and incident beam directions.

        """

        tt = np.deg2rad(self.two_theta)  # (M)
        az = np.deg2rad(self.az_phi)  # (M)

        kf_ki_dir = np.array(
            [np.sin(tt) * np.cos(az), np.sin(tt) * np.sin(az), np.cos(tt) - 1]
        )  # (3, M)

        return np.einsum("ji,jm->im", self.R, kf_ki_dir)
        # (3, M)

    def metric_G_tensor(self):
        """
        Calculate the metric tensor :math:`G`.

        Returns
        -------
        G : 2d-array
            3x3 matrix of lattice parameter info for Cartesian transforms.

        """

        alpha = np.deg2rad(self.alpha)
        beta = np.deg2rad(self.beta)
        gamma = np.deg2rad(self.gamma)

        g11 = self.a**2
        g22 = self.b**2
        g33 = self.c**2
        g12 = self.a * self.b * np.cos(gamma)
        g13 = self.c * self.a * np.cos(beta)
        g23 = self.b * self.c * np.cos(alpha)

        G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])

        return G

    def metric_G_star_tensor(self):
        """
        Calculate the reciprocal metric tensor :math:`G^*`.

        Returns
        -------
        Gstar : 2d-array
            3x3 matrix of reciprocal lattice info for Cartesian transforms.

        """

        return np.linalg.inv(self.metric_G_tensor())

    def reciprocal_lattice_B(self):
        """
        The reciprocal lattice :math:`B`-matrix.

        Returns
        -------
        B : 2d-array
            3x3 matrix of reciprocal lattice in Cartesian coordinates.

        """

        Gstar = self.metric_G_star_tensor()

        return scipy.linalg.cholesky(Gstar, lower=False)

    def orientation_U(self, u0, u1, u2):
        """
        The sample orientation matrix :math:`U`.

        Parameters
        ----------
        u0, u1, u2 : float
            Rotation parameters.

        Returns
        -------
        U : 2d-array
            3x3 sample orientation matrix.

        """

        theta = np.arccos(1 - 2 * u0)
        phi = 2 * np.pi * u1
        w = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        ax = self._angle(u2)

        return scipy.spatial.transform.Rotation.from_rotvec(ax * w).as_matrix()

    def indexer(self, UB, kf_ki_dir, d_min, d_max, wavelength, tol=0.1):
        """

        Parameters
        ----------
        UB : 2d-array
            3x3 sample oriented lattice matrix.
        kf_ki_dir : list
            Difference between scattering and incident beam directions.
        d_min : list
            Lower limit of :math:`d`-spacing.
        d_max : list
            Upper limit of :math:`d`-spacing.
        wavelength : list
            Bandwidth.

        Returns
        -------
        num : int
            Number of peaks index.
        hkl : list
            Miller indices. Un-indexed are labeled [0,0,0].
        lamda : list
            Resolved wavelength. Unindexed are labeled inf.

        """

        wl_min, wl_max = wavelength

        UB_inv = np.linalg.inv(UB)

        hkl_lamda = np.einsum("ij,jk", UB_inv, kf_ki_dir)

        lamda = np.linspace(wl_min, wl_max, 100)

        hkl = hkl_lamda[:, :, np.newaxis] / lamda

        s = np.einsum("ij,j...->i...", UB, hkl)
        s = np.linalg.norm(s, axis=0)

        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl

        dist = np.einsum("ij,j...->i...", UB, diff_hkl)
        dist = np.linalg.norm(dist, axis=0)

        dist[(s.T > 1 / d_min).T] = np.inf
        dist[(s.T < 1 / d_max).T] = np.inf

        h, k, l = int_hkl  # noqa: E741

        valid = np.full_like(l, True, dtype=bool)

        if self.centering == "A":
            valid = (k + l) % 2 == 0
        elif self.centering == "B":
            valid = (h + l) % 2 == 0
        elif self.centering == "C":
            valid = (h + k) % 2 == 0
        elif self.centering == "I":
            valid = (h + k + l) % 2 == 0
        elif self.centering == "F":
            valid = ((h + k) % 2 == 0) & ((l + h) % 2 == 0) & ((k + l) % 2 == 0)
        elif self.centering == "R_obv":
            valid = (-h + k + l) % 3 == 0
        elif self.centering == "R_obv":
            valid = (h - k + l) % 3 == 0

        dist[~valid] = np.inf

        ind = np.argmin(dist, axis=1)

        hkl = hkl[:, np.arange(hkl_lamda.shape[1]), ind]
        lamda = lamda[ind]

        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl

        mask = (np.abs(diff_hkl) < tol).all(axis=0)
        int_hkl[:, ~mask] = 0

        num = np.sum(mask)

        return num, int_hkl.T, lamda

    def UB_matrix(self, U, B):
        """
        Calculate :math:`UB`-matrix.

        Parameters
        ----------
        U : 2d-array
            3x3 orientation matrix.
        B : 2d-array
            3x3 reciprocal lattice vectors Cartesian matrix.

        Returns
        -------
        UB : 2d-array
            3x3 oriented reciprocal lattice.

        """

        return U @ B

    def objective(self, x):
        """
        Cost function.

        Parameters
        ----------
        x : array
            Refineable parameters.

        Returns
        -------
        neg_ind : int
            Negative number of peaks indexed.

        """

        B = self.reciprocal_lattice_B()

        kf_ki_dir, d_min, d_max = self.uncertainty_line_segements()

        wavelength = self.wavelength

        params = np.array(x)

        Us = [self.orientation_U(*param) for param in params]
        UBs = [self.UB_matrix(U, B) for U in Us]

        return [
            -self.indexer(UB, kf_ki_dir, d_min, d_max, wavelength)[0] / len(d_min) * 100
            for UB in UBs
        ]

    def minimize(self, n_proc=-1):
        """
        Fit the orientation and other parameters.

        Parameters
        ----------
        n_proc : int, optional
            Number of processes to use. The default is -1.

        Returns
        -------
        num : int
            Number of peaks index.
        hkl : list
            Miller indices. Un-indexed are labeled [0,0,0].
        lamda : list
            Resolved wavelength. Un-indexed are labeled inf.

        """

        bounds = ([0, 0, 0], [1, 1, 1])
        options = {"c1": 0.5, "c2": 0.5, "w": 0.5}

        optimizer = pyswarms.single.GlobalBestPSO(
            n_particles=3000, dimensions=3, options=options, bounds=bounds
        )

        n_ind, self.x = optimizer.optimize(
            self.objective, n_processes=n_proc, iters=100
        )

        print(-n_ind)

        # bounds = [slice(0, 1, 1/180),
        #           slice(0, 1, 1/180),
        #           slice(0, 0.5, 1/180)]

        # self.x = scipy.optimize.brute(self.objective,
        #                               ranges=bounds,
        #                               workers=n_proc)

        B = self.reciprocal_lattice_B()
        uls = self.uncertainty_line_segements()

        U = self.orientation_U(*self.x)

        UB = self.UB_matrix(U, B)

        return self.indexer(UB, *uls, self.wavelength)

    def indexer_de(self, UB, kf_ki_dir, wavelength, tol=0.15):
        """
        Differential evolution algorithm Laue indexer for a given
         :math:`UB` matrix.

        Parameters
        ----------
        UB : 2d-array
            3x3 sample oriented lattice matrix.
        kf_ki_dir : array (3, M)
            Difference between scattering and incident beam directions.
        wavelength : list
            Bandwidth of each reflection.
        tol : float, optional
            Indexing tolerance. Default is `0.15`.

        Returns
        -------
        err : float
            Indexing cost.
        num : int
            Number of peaks index.
        hkl : list
            Miller indices. Un-indexed are labeled [0,0,0].
        lamda : list
            Resolved wavelength. Unindexed are labeled inf.

        """
        wl_min = np.full(kf_ki_dir.shape[1], wavelength[0])  # (M)
        wl_max = np.full(kf_ki_dir.shape[1], wavelength[1])  # (M)
        x = np.linspace(0, 1, 100)
        lamda = wl_min[:, None] + (wl_max - wl_min)[:, None] * x[None, :]

        UB_inv = np.linalg.inv(UB)

        hkl_lamda = np.einsum("ij,jk", UB_inv, kf_ki_dir)
        hkl = hkl_lamda[:, :, np.newaxis] / lamda[np.newaxis, :, :]
        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl

        dist = np.einsum("ij,j...->i...", UB, diff_hkl)
        dist = np.linalg.norm(dist, axis=0)

        ind = np.argmin(dist, axis=1)
        err = dist[np.arange(dist.shape[0]), ind]

        lamda = lamda[np.arange(lamda.shape[0]), ind]
        hkl = hkl_lamda / lamda
        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl

        mask = (np.abs(diff_hkl) < tol).all(axis=0)
        num = np.sum(mask)

        return np.sum(err ** 2), num, int_hkl.T, lamda

    def index_de(self):
        kf_ki_dir = self.uncertainty_line_segements()

        B = self.reciprocal_lattice_B()
        U = self.orientation_U(*self.x)

        UB = self.UB_matrix(U, B)

        return self.indexer_de(UB, kf_ki_dir, self.wavelength)[1:]

    def cost_de(self, param, B, kf_ki_dir, wavelength):
        """
        Cost function for indexing given a proposed orientation.

        Parameters
        ----------
        param : tuple, float
            Orientation parameters.
        B : array, float
            Reciprocal lattice B-matrix.
        kf_ki_dir : array, float
            Scattering trajectories.
        wavelength : list, float
            Wavelength band (min, max).

        Returns
        -------
        error : float
            Total indexing cost.

        """

        U = self.orientation_U(*param)

        UB = self.UB_matrix(U, B)

        error, num, hkl, lamda = self.indexer_de(UB, kf_ki_dir, wavelength)

        return error

    def objective_de(self, x):
        """
        Objective function.

        Parameters
        ----------
        x : array
            Refineable parameters.

        Returns
        -------
        neg_ind : int
            Negative number of peaks indexed.

        """

        B = self.reciprocal_lattice_B()

        kf_ki_dir = self.uncertainty_line_segements()

        wavelength = self.wavelength

        params = np.reshape(x, (-1, 3))

        compute_with_bounds = partial(
            self.cost_de, B=B, kf_ki_dir=kf_ki_dir, wavelength=wavelength
        )

        results = [compute_with_bounds(param) for param in params]

        return np.array(results)

    def minimize_de(self, num_procs):
        kf_ki_dir = self.uncertainty_line_segements()

        if num_procs == 1:
            # Vectorize if using only one process
            objective = VectorizedObjective(
                self.reciprocal_lattice_B(),
                kf_ki_dir,
                np.array(self.wavelength),
                self._angle
            )

            self.x = scipy.optimize.differential_evolution(
                objective,
                [(0, 1), (0, 1), (0, 1)],
                popsize=1000,
                updating="deferred",
                vectorized=True,
                disp=True,
                callback=lambda x, convergence: print(x, convergence)
            ).x
        else:
            self.x = scipy.optimize.differential_evolution(
                self.objective_de,
                [(0, 1), (0, 1), (0, 1)],
                popsize=1000,
                updating="deferred",
                workers=num_procs,
                disp=True,
                callback=lambda x, convergence: print(x, convergence)
            ).x

        return self.index_de()
