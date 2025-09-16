import os

import numpy as np
import matplotlib.pyplot as plt

from subhkl.integration import Peaks
from subhkl.convex_hull_expansion import RegionGrower, BankPeakIntegrator

# Whether to add intensity labels to detections
show_intensity = False

# Whether to show candidate peak locations
show_candidates = False

# Whether to zoom in on a particular patch of the plot
zoom = False

# Title of plot and suffix for output file
title = 'convex_hull_no_norm'

cell = "Orthorhombic"
centering = "F"

instrument = "IMAGINE"
# Configure input image here
folder = "ndip_data_test/meso_may"
file = os.path.join(folder, "meso_2_15min_2-0_4-5_078.tif")

directory = os.path.dirname(os.path.abspath(__file__))

peak_integrator = BankPeakIntegrator(
    RegionGrower(distance_threshold=1.5, min_intensity=4500, max_size=17),
    box_size=15,
    smoothing_window_size=5,
    min_peak_pixels=30,
    outlier_threshold=2.0
)
peaks = Peaks(file, instrument, peak_integrator=peak_integrator)

file = os.path.basename(file)

name, ext = os.path.splitext(file)

fname = os.path.join(directory, name + "_im." + title + ".pdf")


for bank in sorted(peaks.ims.keys()):
    i, j = peaks.harvest_peaks(bank, min_pix=15, min_rel_intens=0.05, normalize=False)
    x, y = peaks.scale_coordinates(bank, i, j)

    width, height = peaks.detector_width_height(bank)

    im = peaks.ims[bank]

    extent = (-width / 2, width / 2, -height / 2, height / 2)


    fig, ax = plt.subplots(1, 1, figsize=(12.8, 6.4), layout='constrained')

    ax.imshow(
        im.T,
        norm="log",
        cmap="binary",
        origin="lower",
        extent=extent,
        # interpolation="none",
        rasterized=True,
    )

    ax.set_aspect(1)
    ax.minorticks_on()
    ax.set_title(bank)

    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")

    # ---

    peak_dict = peaks.fit_convex_hull(i, j, im)

    nx, ny = im.shape

    for (x, y), (peak_hull, peak_intensity, sigma) in peak_dict.items():
        if show_candidates:
            ax.scatter(*peaks.scale_coordinates(bank, x, y), c='blue', zorder=99, marker='1')

        if peak_hull is None:
            continue

        for simplex in peak_hull.simplices:
            simplex_i = peak_hull.points[simplex, 0]
            simplex_j = peak_hull.points[simplex, 1]

            simplex_x, simplex_y = peaks.scale_coordinates(bank, simplex_i, simplex_j)

            ax.plot(
                simplex_x,
                simplex_y,
                c='red',
                zorder=100
            )

        if show_intensity:
            center = np.mean(peak_hull.points[peak_hull.vertices], axis=0)
            cx, cy = peaks.scale_coordinates(bank, *center)
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
    plt.show()
    fig.savefig(fname, bbox_inches='tight')
