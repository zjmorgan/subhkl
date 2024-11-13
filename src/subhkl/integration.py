import numpy as np

from PIL import Image

import skimage.feature
import scipy.optimize
    
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

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

        coords = skimage.feature.peak_local_max(self.im,
                                                min_distance=min_pix,
                                                threshold_rel=min_rel_intens)

        return coords[:,1], coords[:,0]

    def scale_coordinates(self, xp, yp, scale_x, scale_y):
        """
        Scale from pixel coordinates to real positions.

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

        return (xp-nx/2)*scale_x, (yp-ny/2)*scale_y

    def scale_size(self, xp, yp, scale_x, scale_y):
        """
        Scale from pixel coordinates to real width/height

        Parameters
        ----------
        xp, yp : array, int
            Image coordinates.
        scale_x, scale_y : float
            Pixel scaling factors.

        Returns
        -------
        x, y : array, float
            Image pixel size.

        """

        return xp*scale_x, yp*scale_y


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

        X = x*np.cos(gamma)+d*np.sin(gamma)
        Y = y+h
        Z = d*np.cos(gamma)-x*np.sin(gamma)

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

        X = d*np.sin(x/d)
        Y = y+h
        Z = d*np.cos(x/d)

        return X, Y, Z

    def detector_trajectories(self, x, y, d, h, gamma, panel='curved'):
        """
        Calculate detector trajectories.

        Parameters
        ----------
        x, y : array, float
            Pixel position in physical units.
        d : float
            Horizontal detector distance.
        h : float
            Vertical detector height.
        gamma : float, optional
            Image orientation. The default is 0.

        Returns
        -------
        two_theta : array, float
            Scattering angles in degrees.
        az_phi : array, float
            Azimuthal angles in degrees.

        """

        if panel == 'curved':
            X, Y, Z = self.curved_panel(x, y, d, h, gamma)
        else:
            X, Y, Z = self.flat_panel(x, y, d, h, gamma)

        R = np.sqrt(X**2+Y**2+Z**2)
        two_theta = np.rad2deg(np.arccos(Z/R))
        az_phi = np.rad2deg(np.arctan2(Y, X))

        return two_theta, az_phi

    def peak(self, x, y, A, B, mu_x, mu_y, sigma_1, sigma_2, theta):

        a = 0.5*(np.cos(theta)**2/sigma_1**2+np.sin(theta)**2/sigma_2**2)
        b = 0.5*(np.sin(theta)**2/sigma_1**2+np.cos(theta)**2/sigma_2**2)
        c = 0.5*(1/sigma_1**2-1/sigma_2**2)*np.sin(2*theta)

        shape = np.exp(-(a*(x-mu_x)**2+b*(y-mu_y)**2+c*(x-mu_x)*(y-mu_y)))

        return A*shape+B # /(2*np.pi*sigma_1*sigma_2)

    def residual(self, params, x, y, z):

        return (self.peak(x, y, *params)-z).flatten()

    def transform_ellipsoid(self, sigma_1, sigma_2, theta):

        sigma_x = np.hypot(sigma_1*np.cos(theta), sigma_2*np.sin(theta))
        sigma_y = np.hypot(sigma_1*np.sin(theta), sigma_2*np.cos(theta))
        rho = (sigma_1**2-sigma_2**2)*np.sin(2*theta)/(2*sigma_x*sigma_y)

        return sigma_x, sigma_y, rho

    def fit(self, xp, yp, im, roi_pixels=50):

        peak_dict = {}

        Y, X = np.meshgrid(np.arange(im.shape[0]), 
                           np.arange(im.shape[1]),
                           indexing='ij')


        for ind, (x_val, y_val) in enumerate(zip(xp[:], yp[:])):

            y_min = int(max(y_val-roi_pixels, 0))
            y_max = int(min(y_val+roi_pixels+1, im.shape[0]))
            x_min = int(max(x_val-roi_pixels, 0))
            x_max = int(min(x_val+roi_pixels+1, im.shape[1]))

            x = X[y_min:y_max, x_min:x_max].copy()
            y = Y[y_min:y_max, x_min:x_max].copy()

            z = im[y_min:y_max, x_min:x_max].copy()

            x0 = (z.max(), z.min(),
                  x_val, y_val,
                  roi_pixels*0.25, roi_pixels*0.25, 0)

            xmin = (z.min(), 0, 
                    x_val-roi_pixels*0.5, y_val-roi_pixels*0.5,
                    1, 1, -np.pi/2)

            xmax = (2*z.max(), z.mean(), 
                    x_val+roi_pixels*0.5, y_val+roi_pixels*0.5, 
                    roi_pixels, roi_pixels, np.pi/2)

            bounds = np.array([xmin, xmax])

            args = (x, y, z)

            sol = scipy.optimize.least_squares(self.residual,
                                               x0=x0,
                                               bounds=bounds,
                                               args=args)

            J = sol.jac
            inv_cov = J.T.dot(J)

            if np.linalg.det(inv_cov) > 0:

                A, B, mu_x, mu_y, sigma_1, sigma_2, theta = sol.x

                params = self.transform_ellipsoid(sigma_1, sigma_2, theta)

                peak_dict[(x_val, y_val)] = mu_x, mu_y, *params

                # cov = np.linalg.inv(inv_cov)

                # chi2dof = np.sum(sol.fun**2)/(sol.fun.size-sol.x.size)
                # cov *= chi2dof

                # stderr = np.sqrt(np.diag(cov))

                # A, B, mu_x, mu_y, sigma_1, sigma_2, theta = stderr

        return peak_dict

    # def envelope(self, peak_dict):

    #     X, Sx, Sy, R = [], [], [], []

    #     for items in peak_dict.items():
    #         mu_x, mu_y, sigma_x, sigma_y, rho = items
    #         X.append(mu_x, mu_y)
    #         Sx.append(sigma_x)
    #         Sy.append(sigma_y)
    #         R.append(rho)
        
    #     kernel = RBF(length_scale=1.0)
    #     gpx = GaussianProcessRegressor(kernel=kernel).fit(X, Sx)
    #     gpy = GaussianProcessRegressor(kernel=kernel).fit(X, Sy)
    #     gpr = GaussianProcessRegressor(kernel=kernel).fit(X, R)
        
    #     # Predict at a new point
    #     new_point = [[x_new, y_new]]
    #     sx_pred, sy_std = gpx.predict(new_point, return_std=True)
    #     sy_pred, sy_std = gpy.predict(new_point, return_std=True)
    #     rho_pred, rho_std = gpr.predict(new_point, return_std=True)