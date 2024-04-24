#/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2.colors)

l2 = 0.4

a = 13
b = 12
gamma = np.deg2rad(110)

lamda_min, lamda_max = 2, 4

G = np.array([[a**2,a*b*np.cos(gamma)],
              [a*b*np.cos(gamma),b**2]])

G_inv = np.linalg.inv(G)

t = np.rad2deg(10)

B = scipy.linalg.cholesky(G_inv)
U = np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])

UB = np.dot(U, B)

h = np.arange(-5,5+1,1)
k = np.arange(-5,5+1,1)

h, k = np.meshgrid(h, k)

h, k = h.flatten(), k.flatten()

Qx, Qz = 2*np.pi*np.einsum('ij,jk->ik', UB, [h,k])

Q = np.sqrt(Qx**2+Qz**2)

lamda = -4*np.pi*Qz/Q**2

mask = (lamda > lamda_min) & (lamda < lamda_max)

Qx, Qz, Q, lamda = Qx[mask], Qz[mask], Q[mask], lamda[mask]

d = 2*np.pi/Q

theta = np.arcsin(0.5*lamda/d)
phi = np.arctan2(0, Qx)

x, z = l2*np.sin(2*theta)*np.cos(phi), l2*np.cos(2*theta)

wl = np.linspace(lamda_min, lamda_max)

x_hat, z_hat = x/l2, z/l2

tt = 2*np.deg2rad(np.linspace(0,360,500))

fig, ax = plt.subplots(1, 2, figsize=(9.6,4.8))
ax[0].plot(2*np.pi/lamda_min*np.sin(tt), 2*np.pi/lamda_min*(np.cos(tt)-1), linewidth=1, linestyle='-', color='k', zorder=10)
ax[0].plot(2*np.pi/lamda_max*np.sin(tt), 2*np.pi/lamda_max*(np.cos(tt)-1), linewidth=1, linestyle='-', color='k', zorder=10)
ax[0].scatter(Qx, Qz, color='C1', zorder=10)
ax[0].set_aspect(1)
ax[0].minorticks_on()
ax[0].set_xlabel(r'$Q_x$ [$\AA^{-1}$]')
ax[0].set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

for _x, _z in zip(x, z):
    ax[0].plot(2*np.pi/wl*_x/l2, 2*np.pi/wl*(_z/l2-1), color='C0')

t = np.deg2rad(np.linspace(-175,175,500)+90)

ax[1].plot(100*l2*np.cos(t), 100*l2*np.sin(t), color='k')
ax[1].scatter(x*100, z*100, color='C1', zorder=10)
ax[1].set_aspect(1)
ax[1].minorticks_on()
ax[1].set_xlabel(r'$x$ [cm]')
ax[1].set_ylabel(r'$z$ [cm]')
ax[1].arrow(0, -l2*150, 0, l2*150, length_includes_head=True, 
            head_width=3, width=1, color='C2', zorder=10)

for _x, _z in zip(x, z):
    ax[1].plot([_x*5,_x*100], [_z*5,_z*100], linestyle='-', linewidth=1, color='C0')
