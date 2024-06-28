import os
os.environ['OMP_NUM_THREADS'] = '1'

import h5py

import numpy as np

import scipy.linalg
import scipy.spatial
import scipy.interpolate

import pyswarms

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
        d_min : list
            Lower limit of :math:`d`-spacing.
        d_max : list
            Upper limit of :math:`d`-spacing.

        """

        wl_min, wl_max = self.wavelength

        tt = np.deg2rad(self.two_theta)
        az = np.deg2rad(self.az_phi)

        kf_ki_dir = np.array([np.sin(tt)*np.cos(az),
                              np.sin(tt)*np.sin(az),
                              np.cos(tt)-1])

        d_min = 0.5*wl_min/np.sin(0.5*tt)
        d_max = 0.5*wl_max/np.sin(0.5*tt)

        return kf_ki_dir, d_min, d_max

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

        theta = np.arccos(2*u0-1)
        phi = 2*np.pi*u1
        w = np.array([np.sin(theta)*np.cos(phi),
                      np.sin(theta)*np.sin(phi),
                      np.cos(theta)])

        ax = self._angle(u2)

        return scipy.spatial.transform.Rotation.from_rotvec(ax*w).as_matrix()

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

        hkl_lamda = np.einsum('ij,jk', UB_inv, kf_ki_dir)

        lamda = np.linspace(wl_min, wl_max, 100)

        hkl = hkl_lamda[:,:,np.newaxis]/lamda

        s = np.einsum('ij,j...->i...', UB, hkl)
        s = np.linalg.norm(s, axis=0)

        int_hkl = np.round(hkl)
        diff_hkl = hkl-int_hkl

        dist = np.einsum('ij,j...->i...', UB, diff_hkl)
        dist = np.linalg.norm(dist, axis=0)

        dist[(s.T > 1/d_min).T] = np.inf
        dist[(s.T < 1/d_max).T] = np.inf

        h, k, l = int_hkl

        valid = np.full_like(l, True, dtype=bool)

        if self.centering == 'A':
            valid = (k + l) % 2 == 0
        elif self.centering == 'B':
            valid = (h + l) % 2 == 0
        elif self.centering == 'C':
            valid = (h + k) % 2 == 0
        elif self.centering == 'I':
            valid = (h + k + l) % 2 == 0
        elif self.centering == 'F':
            valid = ((h + k) % 2 == 0) \
                  & ((l + h) % 2 == 0) \
                  & ((k + l) % 2 == 0)
        elif self.centering == 'R_obv':
            valid = (-h + k + l) % 3 == 0
        elif self.centering == 'R_obv':
            valid = (h - k + l) % 3 == 0

        dist[~valid] = np.inf

        ind = np.argmin(dist, axis=1)

        hkl = hkl[:,np.arange(hkl_lamda.shape[1]),ind]
        lamda = lamda[ind]

        int_hkl = np.round(hkl)
        diff_hkl = hkl-int_hkl

        mask = (np.abs(diff_hkl) < tol).all(axis=0)
        int_hkl[:,~mask] = 0

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

        return [-self.indexer(UB, kf_ki_dir, d_min, d_max, wavelength)[0]/len(d_min)*100 for UB in UBs]

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

        bounds = ([0,0,0], [1,1,1])

        options = {'c1': 0.5, 'c2': 0.5, 'w': 0.5}

        optimizer = pyswarms.single.GlobalBestPSO(n_particles=30000,
                                                  dimensions=3,
                                                  options=options,
                                                  bounds=bounds)

        n_ind, self.x = optimizer.optimize(self.objective,
                                           n_processes=n_proc,
                                           iters=100)

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