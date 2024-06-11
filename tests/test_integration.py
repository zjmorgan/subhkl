import os

directory = os.path.dirname(os.path.abspath(__file__))

import h5py
import numpy as np

import matplotlib.pyplot as plt

from subhkl.integration import FindPeaks
from subhkl.optimization import FindUB


def test_mesolite():
    im_name = "meso_2_15min_2-0_4-5_050.tif"
    filename = os.path.join(directory, im_name)

    pks = FindPeaks(filename)
    xp, yp = pks.harvest_peaks(min_pix=30, min_rel_intens=0.05)

    ny, nx = pks.im.shape

    r = 0.2
    p = 2*np.pi*r*180/180
    h = 0.45

    x, y = pks.scale_coordinates(xp, yp, p/nx, h/ny)

    fig, ax = plt.subplots(4,
                           1,
                           figsize=(12.8,19.2),
                           sharex=False,
                           layout='constrained')

    extent = [-p/2, p/2, -h/2, h/2]

    ax[0].imshow(pks.im,
                 norm='log',
                 cmap='binary',
                 origin='lower',
                 extent=extent)

    ax[0].minorticks_on()
    ax[0].set_aspect(1)

    ax[1].imshow(pks.im,
                 norm='log',
                 cmap='binary',
                 origin='lower',
                 extent=extent)

    ax[1].scatter(x, y, edgecolor='r', facecolor='none')
    ax[1].minorticks_on()
    ax[1].set_aspect(1)

    two_theta, az_phi = pks.detector_trajectories(x, y, r, 0, 0)

    peaks_file = os.path.join(directory, 'mesolite_imagine.h5')

    wl_min, wl_max = 2, 4.5

    with h5py.File(os.path.join(directory, peaks_file), 'w') as f:

        f['sample/a'] = 18.405
        f['sample/b'] = 56.655
        f['sample/c'] = 6.544
        f['sample/alpha'] = 90
        f['sample/beta'] = 90
        f['sample/gamma'] = 90
        f['sample/centering'] = 'F'
        f['sample/B'] = np.diag([1/18.405, 1/56.665, 1/6.544])
        f['instrument/wavelength'] = [wl_min, wl_max]
        f['goniometer/R'] = np.eye(3)
        f['peaks/scattering'] = two_theta
        f['peaks/azimuthal'] = az_phi

    opt = FindUB(peaks_file)

    num, hkl, lamda = opt.minimize(48)
    assert num/len(lamda) > 0.5

    ax[2].imshow(pks.im,
                 norm='log',
                 cmap='binary',
                 origin='lower',
                 extent=extent)

    ax[2].plot(x, y, 'r.')
    ax[2].minorticks_on()
    ax[2].set_aspect(1)

    for i in range(len(hkl)):
        if np.linalg.norm(hkl[i]) > 0:
            coord = (x[i], y[i])
            label = '{:.0f}{:.0f}{:.0f}'.format(*hkl[i])
            ax[2].annotate(label, coord)

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

    yp = yv.copy()
    xp = r*theta

    ax[3].imshow(pks.im,
                 norm='log',
                 cmap='binary',
                 origin='lower',
                 extent=extent)

    ax[3].scatter(x, y, edgecolor='r', facecolor='none')
    ax[3].plot(xp, yp, 'w.')
    ax[3].minorticks_on()
    ax[3].set_aspect(1)

    name, ext = os.path.splitext(im_name)

    fig.savefig(name + ".png")


test_mesolite()
