import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Configure input image here
folder = "ndip_data_test/meso_feb"


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def show_image(file, patch_size):
    image_name = os.path.splitext(file)[0]
    im = np.array(Image.open(os.path.join(folder, file)))
    ch_file = str(os.path.join(folder, image_name + '.txt'))
    ch_peaks = np.loadtxt(ch_file, delimiter=',')
    original_file = str(os.path.join(folder, image_name + '_original.txt'))
    original_peaks = np.loadtxt(original_file, delimiter=',')

    n_patches_x = (im.shape[0] + patch_size - 1) // patch_size
    n_patches_y = (im.shape[1] + patch_size - 1) // patch_size

    for patch_x in range(n_patches_x):
        for patch_y in range(n_patches_y):
            fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4), layout='constrained')

            start_x = patch_x * patch_size
            end_x = min(start_x + patch_size, im.shape[0])
            start_y = patch_y * patch_size
            end_y = min(start_y + patch_size, im.shape[1])
            keep_original = (start_y <= original_peaks[:, 0]) & (original_peaks[:, 0] < end_y)
            keep_original &= (start_x <= original_peaks[:, 1]) & (original_peaks[:, 1] < end_x)
            keep_ch = (start_y <= ch_peaks[:, 0]) & (ch_peaks[:, 0] < end_y)
            keep_ch &= (start_x <= ch_peaks[:, 1]) & (ch_peaks[:, 1] < end_x)

            original_rel = original_peaks[keep_original, :]
            ch_rel = ch_peaks[keep_ch, :]

            ax.imshow(
                im[start_x: end_x, start_y: end_y],
                norm="log",
                cmap="binary",
                origin="lower",
                extent=[start_y, end_y, start_x, end_x],
                rasterized=True,
            )
            ax.scatter(original_rel[:, 0], original_rel[:, 1], c='blue', zorder=99, marker='1')
            ax.scatter(ch_rel[:, 0], ch_rel[:, 1], facecolors='none', edgecolors='red')

            move_figure(fig, 50, 100)
            plt.show()


show_image('MESO_02-27_2-4-5_30min172.tif', 500)
