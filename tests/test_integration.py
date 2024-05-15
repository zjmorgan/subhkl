import os
import matplotlib.pyplot as plt

from subhkl.integration import FindPeaks

def test_mesolite():

    directory = '/HFIR/CG4D/shared/images/ndip_data_test/meso_may/'    

    im_name = 'meso_2_15min_2-0_4-5_050.tif'

    filename = os.path.join(directory, im_name)

    pks = FindPeaks(filename)
    x, y = pks.harvest_peaks()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].imshow(pks.im, norm='log', cmap='binary_r')
    ax[0].minorticks_on()
    ax[0].set_aspect(1)

    ax[1].imshow(pks.im, norm='log')
    ax[1].scatter(x, y, s=1, color='w')
    ax[1].minorticks_on()
    ax[1].set_aspect(1)

    name, ext = os.path.splitext(im_name)

    fig.savefig(name+'.png')

test_mesolite()