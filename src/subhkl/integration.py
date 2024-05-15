import numpy as np

from PIL import Image

import skimage.feature

class FindPeaks:

    def __init__(self, filename):
        """
        Find peaks from image

        Parameters
        ----------
        filename : str
            Filename of detector image

        """

        self.im = np.array(Image.open(filename))

    def harvest_peaks(self, min_pix=80):

        coords = skimage.feature.peak_local_max(self.im, min_distance=min_pix)        

        return coords[:,1], coords[:,0]
