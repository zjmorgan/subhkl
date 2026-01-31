# subhkl
Solving crystal orientation from Laue diffraction images

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
