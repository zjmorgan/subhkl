# Data Reduction on MANDI mesolite data

## Setup

- Get the code by cloning `subhkl` from [here](https://github.com/zjmorgan/subhkl),
making sure to get the most recent branch, and install `subhkl` by running
```commandline
git clone https://github.com/zjmorgan/subhkl -b 7-merging-in-subhkl_reduction---peak-finder-prepare-peaks-jacob
```
```commandline
cd subhkl
```
```commandline
pip install -e .
```
(it needs to be an editable installation)
- I will send you the data. You need the following:
  - Nexus file (`.nxs.h5`)
  - Goniometer rotation file (`goniometer.csv`)
  - Crystallographic parameters
    - `a = 18.39`
    - `b = 56.55`
    - `c = 6.54`
    - `alpha = 90`
    - `beta = 90`
    - `gamma = 90`
    - `wavelength_min = 2`
    - `wavelength_max = 4`
    - `sample_centering = "F"`
  - Place the files in the same directory you are going to
  run the following commands in

## Running the workflow

### `finder`

First, we need to find and integrate peaks. Use the
`finder` subcommand for this:
```commandline
python -m subhkl.io.parser finder MANDI_11612.nxs.h5 MANDI
 --output-filename finder_output_11612.h5 
 --min-pixel-distance 2 
 --min-relative-intensity 0.4
 --region-growth-minimum-intensity 30.0
 --region-growth-maximum-pixel-radius 6 
 --peak-center-box-size 9
 --peak-smoothing-window-size 7
 --peak-minimum-pixels 10
```
This will create the output file `finder_output_11612.h5`,
which contains peak scattering angles, intensities and sigmas.

### `indexer`

With the peaks found and integrated, we can run indexing
to find the $h,k,l$ indices for the peaks. Use the `indexer`
subcommand for this:
```commandline
python -m subhkl.io.parser indexer 4 finder_output_11612.h5 goniometer.csv indexed_11612.h5 18.39 56.55 6.54 90 90 90 2 4 F
```
This will create the output file `indexed_11612.h5`. You can also
see where the goniometer rotation and crystallographic parameters
come into play.