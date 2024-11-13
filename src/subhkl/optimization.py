import os
os.environ['OMP_NUM_THREADS'] = '1'

import h5py

from functools import partial

import numpy as np

import scipy.linalg
import scipy.spatial
import scipy.interpolate

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
        cdf = (t-np.sin(t))/np.pi

        self._angle = scipy.interpolate.interp1d(cdf, t, kind='linear')

    def load_peaks(self, filename):
        """
        Obtain peak information from .h5 file.

        Parameters
        ----------
        filename : str
            HDF5 file of peak information.

        """

        with h5py.File(os.path.abspath(filename), 'r') as f:

            self.a = f['sample/a'][()]
            self.b = f['sample/b'][()]
            self.c = f['sample/c'][()]
            self.alpha = f['sample/alpha'][()]
            self.beta = f['sample/beta'][()]
            self.gamma = f['sample/gamma'][()]
            self.wavelength = f['instrument/wavelength'][()]
            self.R = f['goniometer/R'][()]
            self.two_theta = f['peaks/scattering'][()]
            self.az_phi = f['peaks/azimuthal'][()]
            self.centering = f['sample/centering'][()].decode('utf-8')

    def uncertainty_line_segements(self):
        """
        The scattering vector scaled with the (unknown) wavelength.

        Returns
        -------
        kf_ki_dir : list
            Difference between scattering and incident beam directions.

        """

        wl_min, wl_max = self.wavelength

        tt = np.deg2rad(self.two_theta)
        az = np.deg2rad(self.az_phi)

        kf_ki_dir = np.array([np.sin(tt)*np.cos(az),
                              np.sin(tt)*np.sin(az),
                              np.cos(tt)-1])

        return kf_ki_dir

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
        g12 = self.a*self.b*np.cos(gamma)
        g13 = self.c*self.a*np.cos(beta)
        g23 = self.b*self.c*np.cos(alpha)

        G = np.array([[g11, g12, g13],
                      [g12, g22, g23],
                      [g13, g23, g33]])

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
            Rotation paramters.

        Returns
        -------
        U : 2d-array
            3x3 sample orientation matrix.

        """

        theta = np.arccos(1-2*u0)
        phi = 2*np.pi*u1

        w = np.array([np.sin(theta)*np.cos(phi),
                      np.sin(theta)*np.sin(phi),
                      np.cos(theta)])

        omega = self._angle(u2)

        U = scipy.spatial.transform.Rotation.from_rotvec(omega*w).as_matrix()

        return U

    def indexer(self, UB, kf_ki_dir, wavelength, tol=0.15):
        """
        Laue indexer for a given :math:`UB` matrix.

        Parameters
        ----------
        UB : 2d-array
            3x3 sample oriented lattice matrix.
        kf_ki_dir : list
            Difference between scattering and incident beam directions.
        wavelength : list, float
            Bandwidth.
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

        wl_min, wl_max = wavelength

        UB_inv = np.linalg.inv(UB)

        hkl_lamda = np.einsum('ij,jk', UB_inv, kf_ki_dir)

        lamda = np.linspace(wl_min, wl_max, 100)

        hkl = hkl_lamda[:,:,np.newaxis]/lamda

        s = np.einsum('ij,j...->i...', UB, hkl)
        s = np.linalg.norm(s, axis=0)

        int_hkl = np.round(hkl)
        diff_hkl = hkl-int_hkl

        dist = np.einsum('ij,j...->i...', UB, diff_hkl)
        dist = np.linalg.norm(dist, axis=0)

        ind = np.argmin(dist, axis=1)
        err = dist[np.arange(dist.shape[0]),ind]

        hkl = hkl[:,np.arange(hkl_lamda.shape[1]),ind]
        lamda = lamda[ind]

        hkl = hkl_lamda/lamda
        int_hkl = np.round(hkl)
        diff_hkl = hkl-int_hkl

        mask = (np.abs(diff_hkl) < tol).all(axis=0)

        num = np.sum(mask)

        return np.sum(err), num, int_hkl.T, lamda

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

    def cost(self, param, B, kf_ki_dir, wavelength):
        """
        Cost function for indexing given a proposed orientation.

        Parameters
        ----------
        param : tuple, float
            Orientation parameters.
        B : array, float
            Reciprocal lattice B-matrix.
        kf_ki_dir : array, float
            DESCRIPTION.
        wavelength : list, float
            Wavelength band (min, max).

        Returns
        -------
        error : float
            Total indexing cost.

        """

        U = self.orientation_U(*param)

        UB = self.UB_matrix(U, B)

        error, num, hkl, lamda = self.indexer(UB, kf_ki_dir, wavelength)

        return error

    def objective(self, x):
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

        compute_with_bounds = partial(self.cost,
                                      B=B, 
                                      kf_ki_dir=kf_ki_dir,
                                      wavelength=wavelength)

        results = [compute_with_bounds(param) for param in params]

        return np.array(results)

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

        self.x = scipy.optimize.differential_evolution(self.objective,
                                                        [(0,1), (0,1), (0,1)], 
                                                        popsize=1000,
                                                        updating='deferred',
                                                        workers=-1).x

        kf_ki_dir = self.uncertainty_line_segements()

        B = self.reciprocal_lattice_B()
        U = self.orientation_U(*self.x)

        UB = self.UB_matrix(U, B)

        return self.indexer(UB, kf_ki_dir, self.wavelength)