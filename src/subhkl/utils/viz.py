"""Visualization utilities"""

import numpy as np
from skimage.exposure import equalize_adapthist

def plot_detector_data(ax, data, perc_low=1, perc_high=99.9, cmap='viridis'):
    from skimage.exposure import equalize_adapthist

    data_clipped = np.clip(data, np.percentile(data, perc_low), np.percentile(data, perc_high))
    data_norm = (data_clipped - np.min(data_clipped)) / (np.max(data_clipped) - np.min(data_clipped))
    clahe_result = equalize_adapthist(data_norm, clip_limit=0.03)
    ax.imshow(clahe_result, cmap=cmap, origin='lower')
