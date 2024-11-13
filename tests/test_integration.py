import os

import h5py
import numpy as np

import matplotlib
#matplotlib.use('pgf')

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from subhkl.integration import FindPeaks
from subhkl.optimization import FindUB

def test_mesolite():

    directory = '/HFIR/CG4D/shared/images/ndip_data_test/meso_may/'

    im_name = 'meso_2_15min_2-0_4-5_078.tif'

    filename = os.path.join(directory, im_name)

    pks = FindPeaks(filename)
    xp, yp = pks.harvest_peaks(min_pix=30, min_rel_intens=0.05)

    ny, nx = pks.im.shape

    r = 0.2
    p = 2*np.pi*r*180/180
    h = 0.45

    x, y = pks.scale_coordinates(xp, yp, p/nx, h/ny)

    name, ext = os.path.splitext(im_name)

    directory = os.path.dirname(os.path.abspath(__file__))

    fig, ax = plt.subplots(1, 1, figsize=(12.8,6.4), layout='tight')

    extent = [-p/2, p/2, -h/2, h/2]

    ax.imshow(pks.im,
              norm='log',
              cmap='binary',
              origin='lower',
              extent=extent)

    ax.minorticks_on()
    ax.set_aspect(1)

    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')

    fig.savefig(os.path.join(directory, name+'_im.pdf'),
                bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(12.8,6.4), layout='tight')

    ax.imshow(pks.im,
                 norm='log',
                 cmap='binary',
                 origin='lower',
                 extent=extent)

    ax.scatter(x, y, edgecolor='r', facecolor='none')
    ax.minorticks_on()
    ax.set_aspect(1)

    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')

    fig.savefig(os.path.join(directory, name+'_find.pdf'), 
                bbox_inches='tight')
    fig.savefig(os.path.join(directory, name+'_find.pgf'), 
                bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(12.8,6.4), layout='tight')

    two_theta, az_phi = pks.detector_trajectories(x, y, r, 0, 0)

    peaks_file = os.path.join(directory, 'mesolite_imagine.h5')

    wl_min, wl_max = 2, 4.5

    with h5py.File(os.path.join(directory, peaks_file), 'w') as f:

        f['sample/a'] = 18.39
        f['sample/b'] = 56.55
        f['sample/c'] = 6.54
        f['sample/alpha'] = 90
        f['sample/beta'] = 90
        f['sample/gamma'] = 90
        f['sample/centering'] = 'F'
        f['sample/B'] = np.diag([1/18.39, 1/56.55, 1/6.54])
        f['instrument/wavelength'] = [wl_min, wl_max]
        f['goniometer/R'] = np.eye(3)
        f['peaks/scattering'] = two_theta
        f['peaks/azimuthal'] = az_phi

    opt = FindUB(peaks_file)

    error, num, hkl, lamda = opt.minimize()
    assert num/len(lamda) > 0.5

    ax.imshow(pks.im,
              norm='log',
              cmap='binary',
              origin='lower',
              extent=extent)

    ax.plot(x, y, 'r.')
    ax.minorticks_on()
    ax.set_aspect(1)

    for i in range(len(hkl)):
        if np.linalg.norm(hkl[i]) > 0:
            coord = (x[i], y[i])
            label = '{:.0f}{:.0f}{:.0f}'.format(*hkl[i])
            ax.annotate(label, coord)

    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')

    fig.savefig(os.path.join(directory, name+'_index.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(directory, name+'_index.pgf'),
                bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(12.8,6.4), layout='tight')

    B = opt.reciprocal_lattice_B()
    U = opt.orientation_U(*opt.x)
    UB = opt.UB_matrix(U, B)

    Qx, Qy, Qz = np.einsum('ij,kj->ik', 2*np.pi*UB, hkl)
    Q = np.sqrt(Qx**2+Qy**2+Qz**2)

    lamda = -4*np.pi*Qz/Q**2
    mask = np.logical_and(lamda > wl_min, lamda < wl_max)

    Qx, Qy, Qz, Q, lamda = Qx[mask], Qy[mask], Qz[mask], Q[mask], lamda[mask]

    tt = -2*np.arcsin(Qz/Q)
    az = np.arctan2(Qy, Qx)

    xv = np.sin(tt)*np.cos(az)
    yv = np.sin(tt)*np.sin(az)
    zv = np.cos(tt)

    assert np.allclose(xv**2+yv**2+zv**2, 1)

    t = r/np.sqrt(xv**2+zv**2)

    xv *= t
    yv *= t
    zv *= t

    assert np.allclose(xv**2+zv**2, r**2)

    theta = np.arctan2(xv, zv)

    y_ = yv.copy()
    x_ = r*theta

    ax.imshow(pks.im,
              norm='log',
              cmap='binary',
              origin='lower',
              extent=extent)

    ax.scatter(x, y, edgecolor='r', facecolor='none')
    ax.plot(x_, y_, 'w.')
    ax.minorticks_on()
    ax.set_aspect(1)

    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')

    fig.savefig(os.path.join(directory, name+'_predict.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(directory, name+'_predict.pgf'),
                bbox_inches='tight')

    peak_dict = pks.fit(xp, yp, pks.im)

    fig, ax = plt.subplots(1, 1, figsize=(12.8,6.4), layout='tight')

    ax.imshow(pks.im,
              norm='log',
              cmap='binary',
              origin='lower',
              extent=extent)

    ax.minorticks_on()
    ax.set_aspect(1)

    for key in peak_dict.keys():

        mu_x, mu_y, sigma_x, sigma_y, rho = peak_dict[key]
        
        mu_x, mu_y = pks.scale_coordinates(mu_x, mu_y, p/nx, h/ny)
        sigma_x, sigma_y = pks.scale_size(sigma_x, sigma_y, p/nx, h/ny)

        theta = np.rad2deg(0.5*np.arctan2(2*rho*sigma_x*sigma_y,
                                          sigma_x**2-sigma_y**2))

        elli = Ellipse(xy=(mu_x, mu_y),
                       width=6*sigma_x,
                       height=6*sigma_y,
                       angle=theta,
                       linestyle='-',
                       edgecolor='w',
                       facecolor='none',
                       rasterized=False,
                       zorder=100)

        ax.add_patch(elli)

    fig.savefig(os.path.join(directory, name+'_integrate.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(directory, name+'_integrate.pgf'),
                bbox_inches='tight')

test_mesolite()