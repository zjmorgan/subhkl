# subhkl
Solving crystal orientation from Laue diffraction images

# KB added from here
Download the github branch:https://github.com/zjmorgan/subhkl/edit/forward_kernel/
resolve the dependancies and environment and then install
conda activate env
module load rocm/6.4.1
pip install -e .
run sh garnet.sh  working_dir
---
about garnet.sh:
set -e
WAVEL_MIN=0.5
D_MIN=0.5
WAVEL_MAX=8.5
INTEGRATOR_THRESHOLDS="--region-growth-minimum-sigma 0.5 --region-growth-maximum-pixel-radius 12 --peak-center-box-size 3 --peak-smoothing-window-size 5 --peak-minimum-pixels 40 --peak-minimum-signal-to-noise 0.0 --peak-pixel-outlier-threshold 4"
FINDER_THRESHOLDS="--region-growth-minimum-sigma 1.5 --peak-minimum-pixels 20 --peak-minimum-signal-to-noise 1.0 --thresholding-noise-cutoff-quantile 0.96"
FINDER_PARAMS="11.93 11.93 11.93 90 90 90 ${WAVEL_MIN} ${WAVEL_MAX} I"

# per h5
FIND="python -m subhkl.io.parser finder \$FN MANDI --output-filename \$FN.finder.h5 --finder-algorithm thresholding $FINDER_THRESHOLDS --create-visualizations"

# single call
INDEX=$(cat <<EOF
python -m subhkl.io.parser finder-merger $1/finder_files.txt $1/finder_merged.h5 $FINDER_PARAMS;
python -m subhkl.io.parser indexer $1/finder_merged.h5 $1/stage1.h5 $FINDER_PARAMS --n-runs=50 --popsize=100 --gens=250 --strategy=de --softness=0.1 --refine-lattice --loss-method gaussian  --d-min 0.5 --d-max 10 --batch-size=1;
python -m subhkl.io.parser indexer --bootstrap $1/stage1.h5 $1/finder_merged.h5 $1/indexer.h5 $FINDER_PARAMS --n-runs=50 --popsize=100 --gens=250 --strategy=de --softness=0.01 --refine-lattice --loss-method gaussian --d-min 0.5 --d-max 10 --batch-size=1;
python -m subhkl.io.parser metrics $1/indexer.h5;
EOF
)
# per h5
INTEGRATE=$(cat <<EOF
python -m subhkl.io.parser peak-predictor \$FN MANDI $1/indexer.h5 \$FN.peak_predictor_multi.h5 --d-min ${D_MIN}; #subhkl works well to predict and rest are developing stage.
python -m subhkl.io.parser integrator \$FN MANDI \$FN.peak_predictor_multi.h5 \$FN.integrator_multi.h5 free_fit $INTEGRATOR_THRESHOLDS --create-visualizations --show-progress;
python -m subhkl.io.parser mtz-exporter \$FN.integrator_multi.h5 \$FN.mtz "I a -3 d";
python -c "import reciprocalspaceship as rs; ds = rs.read_mtz('\$FN.mtz'); print(ds.head()); print(f'{len(ds)} reflections')";
EOF
)

for Z in $1/*nxs.h5; do
###    FN=$Z srun -n 1 -N1 -c 1 --gpus-per-task=1 bash -c "$FIND" &
    FN=$Z bash -c "$FIND" &
done
wait

ls $1/*.finder.h5 > $1/finder_files.txt
eval $INDEX

for Z in $1/*nxs.h5; do
#    FN=$Z srun -n 1 -N1 -c 1 --gpus-per-task=1 bash -c "$INTEGRATE" &
    FN=$Z bash -c "$INTEGRATE" &
done
wait

## KB added to here
## Running with docker

Building:

```
docker build -t subhkl .
```

Running:

```
docker run -it --rm --name=subhkl subhkl
```

subhkl will be available for import inside of Python in the container.

## Workflow example (without normalization, for now)

You will need to get the raw mesolite IMAGINE images from GitLab.
Assume that they are stored in the folder `mesolite_202405`.

The script `run_all_imagine.sh` runs the full workflow for a single
image. You can use the following command to apply the script to all the images in
`mesolite_202405`. This will generate a `.mtz` file for each input image.

```bash
for Z in mesolite_202405/*.tif; do run_all_imagine.sh $Z& done
```

To merge the output `.mtz` files, you can use `reciprocalspaceship`. We
will probably add this as a command, but for now the following python code 
works.

```python
import reciprocalspaceship as rs
import os

mtzs = []
for file in os.listdir("mesolite_202405"):
    if os.path.splitext(file)[1] == ".mtz":
        mtzs.append(rs.read_mtz(os.path.join("mesolite_202405", file)))
rs.concat(mtzs).hkl_to_asu().write_mtz("mesolite_202405/meso.mtz")
```

which creates a single `.mtz` file `mesolite_202405/meso.mtz` that contains
all reflections.

## Sparse RBF Peak Finder

The **Sparse RBF (Radial Basis Function)** peak finder is an advanced algorithm designed to resolve overlapping peaks ("necklaces") and dense clusters that standard segmentation methods (like Watershed) often fail to separate. It treats peak finding as a function approximation problem, reconstructing the image as a sparse sum of Gaussian atoms using an iterative pursuit strategy.

### Key Features
* **Adaptive Resolution:** Dynamically adds peaks only where they significantly improve the fit.
* **Joint Relaxation:** Once a peak is found, its position and width are jointly optimized with all other peaks, allowing overlapping spots to "slide" apart naturally.
* **Parallel Acceleration:** Uses JAX to run fully parallelized pursuit on all detector banks simultaneously.

### Parameters

| Parameter | Default | Description | Tuning Advice |
| :--- | :--- | :--- | :--- |
| `--sparse-rbf-alpha` | `0.02` | **Sparsity Penalty.** The minimum relative intensity (0.0-1.0) required for a peak to be kept. | **Critical Knob.** Set to `0.05` to ignore everything below 5% brightness. Set to `0.001` to catch very faint peaks. If you get 0 peaks, lower this value. |
| `--sparse-rbf-min-sigma` | `1.0` | Minimum peak width (pixels). | Set to the physical point-spread function (PSF) size. If peaks are sharp (single pixel), set to `0.5`. |
| `--sparse-rbf-max-sigma` | `10.0` | Maximum peak width (pixels). | Prevent the algorithm from fitting large background gradients as giant peaks. |
| `--sparse-rbf-tile-rows` | `2` | Number of spatial tiles (rows) to split the image into. | `2` (making a 2x2 grid) is a good default. Increase to `4` for large images (2k+) to improve GPU parallelism and speed. |
| `--sparse-rbf-tile-cols` | `2` | Number of spatial tiles (cols) to split the image into. | `2` (making a 2x2 grid) is a good default. Increase to `4` for large images (2k+) to improve GPU parallelism and speed. |
| `--sparse-rbf-max-peaks` | `500` | Maximum number of peaks to find *per bank*. | Increase if you expect very dense diffraction patterns. |

### Usage Example

```bash
# Standard run for dense patterns
python -m subhkl.io.parser finder data.h5 MANDI \
    --finder-algorithm sparse_rbf \
    --sparse-rbf-alpha 0.02 \
    --sparse-rbf-tile-rows 4 \
    --sparse-rbf-tile-cols 4

# Debugging: Visualize the pursuit process
python -m subhkl.io.parser finder data.h5 MANDI \
    --finder-algorithm sparse_rbf \
    --show-steps
