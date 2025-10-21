# import os

# import numpy as np
# import matplotlib.pyplot as plt

# from subhkl.integration import FindPeaks


def test_mesolite():
    pass
    # directory = '/HFIR/CG4D/shared/images/ndip_data_test/meso_may/'
    #
    # files = []
    # for root, _, filenames in os.walk(directory):
    #     for filename in filenames:
    #         if filename.endswith('.tif'):
    #             files.append(os.path.join(directory, filename))
    #
    # directory = os.path.dirname(os.path.abspath(__file__))
    #
    # for file in files:
    #
    #     pks = FindPeaks(file)
    #
    #     file = os.path.basename(file)
    #
    #     name, ext = os.path.splitext(file)
    #
    #     ny, nx = pks.im.shape
    #
    #     r = 0.2
    #     p = 2*np.pi*r*180/180
    #     h = 0.45
    #
    #     fig, ax = plt.subplots(1,
    #                            1,
    #                            figsize=(25.6,19.2),
    #                            layout='constrained')
    #     ax.axvline(x=p/4, color='k', linewidth=1)
    #     ax.axvline(x=-p/4, color='k', linewidth=1)
    #
    #     extent = [-p/2, p/2, -h/2, h/2]
    #
    #     ax.imshow(pks.im, #[::-1,:]
    #               norm='log',
    #               cmap='binary',
    #               origin='lower',
    #               extent=extent)
    #     ax.set_aspect(1)
    #     ax.minorticks_on()
    #     ax.set_title(name)
    #
    #     fname = os.path.join(directory, name+'_im.png')
    #
    #     fig.savefig(fname, bbox_inches='tight')

if __name__ == "__main__":
    test_mesolite()
