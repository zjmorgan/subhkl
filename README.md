# subhkl
Solving crystal orientation from Laue diffraction images

## Physics and Conventions

This project uses the **Laue Equation** relating Miller indices $(h, k, l)$ to the scattering vector $Q$ in the laboratory frame:

$$Q_l = 2\pi R \cdot U \cdot B \cdot \mathbf{h}$$

where:
- **$\mathbf{h}$**: Miller indices vector $\begin{pmatrix} h \\ k \\ l \end{pmatrix}$.
- **$B$**: Reciprocal lattice matrix (Cartesian system). Transforms Miller indices to reciprocal space units ($1/\text{\AA}$ if $2\pi$ is not absorbed).
- **$U$**: Orientation matrix (Sample to Cartesian). Transforms reciprocal lattice to the goniometer/sample frame.
- **$R$**: Goniometer rotation matrix (Lab to Sample). Calculated from goniometer axes and angles using Mantid's `SetGoniometer` convention ($R = R_{\text{omega}} R_{\text{chi}} R_{\text{phi}}$).

The scattering vector $Q$ is defined by the change in wavevector:
$$Q = k_f - k_i = \frac{2\pi}{\lambda} (\hat{k}_f - \hat{k}_i)$$

where $\hat{k}_f$ and $\hat{k}_i$ are unit vectors along the scattered and incident beam directions, respectively.

### Coordinate Systems
- **Lab Frame**: $Z$ is along the incident beam, $Y$ is vertically upward.
- **Sample Frame**: Attached to the innermost goniometer axis.
- **Angles**: $2\theta$ is the scattering angle, $\phi$ is the azimuthal angle.

---

## Installation

### Option 1: Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver. If you don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create a virtual environment and install the project:

```bash
uv venv env
source env/bin/activate  # On Windows: env\Scripts\activate
uv pip install -e .
```

### Option 2: Using standard Python venv

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
python -m pip install -e .
```

### Installing test dependencies

```bash
uv pip install -e ".[test]"  # with uv
# or
python -m pip install -e ".[test]"  # with pip
```

### Installing optional JAX dependencies

JAX is an optional dependency used for GPU-accelerated optimization algorithms. The package automatically uses JAX when available, falling back to NumPy otherwise.

**Installation options:**

```bash
# Standard installation (NumPy backend only - no GPU)
pip install subhkl

# CPU-only JAX acceleration (faster, but no GPU)
pip install subhkl[jax]

# NVIDIA GPU support (CUDA 12.x)
pip install subhkl[jax-cuda12]

# NVIDIA GPU support (CUDA 11.x - for older systems)
pip install subhkl[jax-cuda11]

# AMD GPU support (ROCm)
pip install subhkl[jax-rocm]
```

**For development (editable install):**

```bash
# NumPy backend
uv pip install -e .

# JAX CPU
uv pip install -e ".[jax]"

# JAX with CUDA 12
uv pip install -e ".[jax-cuda12]"

# JAX with CUDA 11
uv pip install -e ".[jax-cuda11]"

# JAX with ROCm (AMD)
uv pip install -e ".[jax-rocm]"
```

**Backend detection:**

The optimization backend is automatically selected at import time. You can check which backend is being used:

```python
import subhkl

print(f"Backend: {subhkl.OPTIMIZATION_BACKEND}")  # "jax" or "numpy"
print(f"JAX available: {subhkl.HAS_JAX}")  # True or False

# VectorizedObjective automatically uses the best available backend
objective = subhkl.VectorizedObjective(...)  # JIT-compiled if JAX available
```

**Performance notes:**
- **NumPy backend**: Works everywhere, no GPU required, good for small-scale problems
- **JAX CPU**: ~2-5x faster than NumPy due to JIT compilation, no GPU required
- **JAX GPU (CUDA/ROCm)**: ~10-100x faster for large-scale optimization, requires compatible GPU

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
## Developer Guide

### Running Tests

```bash
pytest -v
```

### Running Linting

```bash
ruff format --check && ruff check
```

To auto-fix formatting issues:

```bash
ruff format
```

### Publishing a Release

The project uses automated publishing to PyPI and GitHub Container Registry when you create a semantic version tag.

**Prerequisites:**

1. **Set up PyPI trusted publishing** (one-time setup):
   - Go to https://pypi.org/manage/account/publishing/
   - Add GitHub as a trusted publisher for your repository
   - Set the workflow filename to `publish.yaml`
   - Set the environment name to `pypi`

2. **Create and push a release tag**:

```bash
# Create a new version tag (e.g., v0.1.0)
git tag v0.1.0

# Push the tag to GitHub
git push origin v0.1.0
```

This will automatically:
- Build and publish the package to PyPI
- Build and push a Docker image to `ghcr.io/zjmorgan/subhkl:v0.1.0` (and `latest`)
