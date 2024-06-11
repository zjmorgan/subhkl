import os
directory = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import scipy.spatial

import h5py

from mantid.simpleapi import (LoadEmptyInstrument,
                              LoadCIF,
                              SetUB,
                              SetGoniometer,
                              PredictPeaks,
                              CreatePeaksWorkspace,
                              mtd)

from mantid import config

config['Q.convention'] = 'Crystallography'

refl_cond_dict = {'P': 'Primitive', 
                  'I': 'Body centred',
                  'F': 'All-face centred',
                  'R': 'Primitive',
                  'Robv': 'Rhombohderally centred, obverse',
                  'Rrev': 'Rhombohderally centred, reverse',
                  'A': 'A-face centred',
                  'B': 'B-face centred',
                  'C': 'C-face centred'}   

seed = 13

def generate_peaks(instrument, cif_file, wavelength_band=[2,4], min_d=1):

    wl_min, wl_max = wavelength_band

    LoadEmptyInstrument(InstrumentName='MANDI',
                        OutputWorkspace='instrument')

    LoadCIF(Workspace='instrument', 
            InputFile=os.path.join(directory, cif_file))

    cs = mtd['instrument'].sample().getCrystalStructure()

    uc = cs.getUnitCell()

    CreatePeaksWorkspace(NumberOfPeaks=0,
                         OutputType='LeanElasticPeak',
                         OutputWorkspace='reference')

    SetUB(Workspace='reference', 
          a=uc.a(), 
          b=uc.b(), 
          c=uc.c(), 
          alpha=uc.alpha(),
          beta=uc.beta(),
          gamma=uc.gamma())

    B = mtd['reference'].sample().getOrientedLattice().getB().copy()
    U, R = scipy.spatial.transform.Rotation.random(2, seed).as_matrix()

    UB = U @ B
 
    SetUB(Workspace='instrument', UB=UB)

    SetGoniometer(Workspace='instrument',
                  Axis0='0,0,1,0,1',
                  Axis1='0,0,0,1,1',
                  Axis2='0,0,1,0,1')

    mtd['instrument'].run().getGoniometer().setR(R)

    max_d = np.max([uc.a(),uc.b(),uc.c()])

    symbol = cs.getSpaceGroup().getHMSymbol()[0]

    centering = symbol[0]

    if centering == 'R':
        if symbol[-1] != 'r':
            centering += 'obv'

    PredictPeaks(InputWorkspace='instrument',
                 WavelengthMax=wl_max,
                 WavelengthMin=wl_min,
                 MaxDSpacing=max_d*1.1,
                 MinDSpacing=min_d,
                 ReflectionCondition=refl_cond_dict[centering],
                 CalculateStructureFactors=True,
                 OutputWorkspace='peaks')

    two_theta = []
    az_phi = []
    h = []
    k = []
    l = []
    lamda = []

    for peak in mtd['peaks']:

        two_theta.append(np.rad2deg(peak.getScattering()))
        az_phi.append(np.rad2deg(peak.getAzimuthal()))
        h.append(peak.getH())
        k.append(peak.getK())
        l.append(peak.getL())
        lamda.append(peak.getWavelength())

    peaks_file = cif_file.replace('.cif','_{}.h5'.format(instrument)).lower()

    with h5py.File(os.path.join(directory, peaks_file), 'w') as f:

        f['sample/a'] = uc.a()
        f['sample/b'] = uc.b()
        f['sample/c'] = uc.c()
        f['sample/alpha'] = uc.alpha()
        f['sample/beta'] = uc.beta()
        f['sample/gamma'] = uc.gamma()
        f['sample/B'] = B
        f['sample/U'] = U
        f['sample/centering'] = centering
        f['instrument/wavelength'] = [wl_min, wl_max]
        f['goniometer/R'] = R
        f['peaks/scattering'] = two_theta
        f['peaks/azimuthal'] = az_phi
        f['peaks/h'] = h
        f['peaks/k'] = k
        f['peaks/l'] = l
        f['peaks/lambda'] = lamda

generate_peaks('MANDI', '5vnq.cif', wavelength_band=[2,4], min_d=6)
generate_peaks('MANDI', 'sucrose.cif', wavelength_band=[2,4], min_d=1)