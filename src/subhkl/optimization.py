import os
os.environ['OMP_NUM_THREADS'] = '1'

import h5py

import numpy as np

import scipy.linalg
import scipy.spatial
import scipy.optimize

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
        B ; 2d-array
            3x3 matrix of reciprocal lattice in Cartesian coordinates.

        """

        Gstar = self.metric_G_star_tensor()

        return scipy.linalg.cholesky(Gstar, lower=False)

    def orientation_U(self, phi, theta, omega):
        """
        The sample orientation matrix :math:`U`.

        Parameters
        ----------
        phi : float
            Azimuthal angle in radians.
        theta : float
            Zenith angle in radians.
        omega : float
            Rotation angle in radians.

        Returns
        -------
        U : 2d-array
            3x3 sample orientation matrix.

        """

        u0 = np.cos(phi)*np.sin(theta)
        u1 = np.sin(phi)*np.sin(theta)
        u2 = np.cos(theta)

        w = omega*np.array([u0, u1, u2])

        return scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

    def indexer(self, UB, kf_ki_dir, d_min, d_max, wavelength):
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

        inv_wl_max = 1/wl_min
        inv_wl_min = 1/wl_max

        astar, bstar, cstar = np.sqrt(np.diag(UB.T @ UB))

        s_min = np.min([astar, bstar, cstar])*0.5

        integrality = np.array([1,1,1,0])

        num = 0
        hkl = []
        lamda = []

        for d, kfi in zip(d_min, kf_ki_dir.T):

            s_max = 1/d

            max_h, max_k, max_l = s_max/np.array([astar, bstar, cstar])
            min_h, min_k, min_l = -max_h, -max_k, -max_l

            bounds = scipy.optimize.Bounds([min_h, min_k, min_l, inv_wl_min],
                                           [max_h, max_k, max_l, inv_wl_max])

            A = np.column_stack((UB, -kfi))
            A = np.row_stack((A, [0,0,0,1]))

            lb = [-s_min]*3+[inv_wl_min]
            ub = [+s_min]*3+[inv_wl_max]

            constraints = scipy.optimize.LinearConstraint(A, lb, ub)

            c = np.einsum('i,ij->j', -kfi, UB)
            c = np.concatenate((c,[0]))

            res = scipy.optimize.milp(c=c,
                                      bounds=bounds,
                                      constraints=constraints,
                                      integrality=integrality)

            if res.x is not None:
                num += 1
                hkl.append(res.x[:-1])
                lamda.append(1/res.x[-1])
            else:
                hkl.append([0,0,0])
                lamda.append(np.inf)

        hkl = np.array(hkl)
        lamda = np.array(lamda)

        return num, hkl, lamda

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
        x : list
            Refineable parameters.

        Returns
        -------
        neg_ind : int
            Negative number of peaks indexed.

        """

        B = self.reciprocal_lattice_B()

        kf_ki_dir, d_min, d_max = self.uncertainty_line_segements()

        wavelength = self.wavelength

        U = self.orientation_U(*x)
        UB = self.UB_matrix(U, B)

        num, hkl, lamda = self.indexer(UB, kf_ki_dir, d_min, d_max, wavelength)

        return -num

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
            Resolved wavelength. Unindexed are labeled inf.

        """

        bounds = [(-np.pi, np.pi), (0, np.pi), (-np.pi, np.pi)]        

        self.x = scipy.optimize.shgo(self.objective,
                                     bounds=bounds,
                                     workers=n_proc).x

        B = self.reciprocal_lattice_B()
        uls = self.uncertainty_line_segements()

        U = self.orientation_U(*self.x)

        UB = self.UB_matrix(U, B)

        return self.indexer(UB, *uls, self.wavelength)