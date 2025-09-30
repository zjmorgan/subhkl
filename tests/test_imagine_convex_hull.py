import os

import numpy as np
import matplotlib.pyplot as plt

from subhkl.integration import FindPeaks
from subhkl.convex_hull_expansion import RegionGrower, PeakIntegrator

# Whether to add intensity labels to detections
show_intensity = False

# Whether to show candidate peak locations
show_candidates = True

# Whether to zoom in on a particular patch of the plot
zoom = False

# Title of plot and suffix for output file
title = 'convex_hull_norm'

# Configure input image here
folder = "ndip_data_test/meso_feb"
file = os.path.join(folder, "MESO_02-27_2-4-5_30min136.tif")

directory = os.path.dirname(os.path.abspath(__file__))

peak_integrator = PeakIntegrator(
    RegionGrower(distance_threshold=1.5, min_intensity=2000, max_size=17),
    box_size=15,
    smoothing_window_size=5,
    min_peak_pixels=50,
    outlier_threshold=2.0
)
peaks = FindPeaks(file, peak_integrator=peak_integrator)

file = os.path.basename(file)

name, ext = os.path.splitext(file)

fname = os.path.join(directory, name + "_im." + title + ".pdf")

ny, nx = peaks.im.shape

r = 0.2
p = 2 * np.pi * r * 180 / 180
h = 0.45

i, j = peaks.harvest_peaks(min_pix=15, min_rel_intens=0.04, normalize=True)
x, y = peaks.scale_coordinates(i, j, p / nx, h / ny)

extent = (-p / 2, p / 2, -h / 2, h / 2)


fig, ax = plt.subplots(1, 1, figsize=(12.8, 6.4), layout='constrained')

ax.imshow(
    peaks.im,
    norm="log",
    cmap="binary",
    origin="lower",
    extent=extent,
    # interpolation="none",
    rasterized=True,
)

ax.set_aspect(1)
ax.minorticks_on()

ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$y$ [m]")

# ---

peak_dict = peaks.fit_convex_hull(i, j)

for (x, y), (center, peak_hull, peak_intensity, sigma) in peak_dict.items():
    if show_candidates:
        ax.scatter(*peaks.scale_coordinates(x, y, p / nx, h / ny), c='blue', zorder=99, marker='1')

    if peak_hull is None:
        continue

    for simplex in peak_hull.simplices:
        simplex_i = peak_hull.points[simplex, 0]
        simplex_j = peak_hull.points[simplex, 1]

        simplex_x, simplex_y = peaks.scale_coordinates(simplex_i, simplex_j, p / nx, h / ny)

        ax.plot(
            simplex_x,
            simplex_y,
            c='red',
            zorder=100
        )

    if show_intensity:
        cx, cy = peaks.scale_coordinates(*center, p / nx, h / ny)
        ax.text(
            cx,
            cy,
            f'{peak_intensity:.02e} +- {sigma:.02e}',
            c='green'
        )

if zoom:
    ax.set_xlim(-0.58, -0.28)
    ax.set_ylim(0, 0.1)

ax.set_title(title)
fig.savefig(fname, bbox_inches='tight')
