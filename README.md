# subhkl
Solving crystal orientation from Laue diffraction images

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

JAX is an optional dependency used for optimization algorithms. Since it can be difficult to install across platforms, it's not required by default. To use JAX-based optimization features (e.g., `minimize_evosax`, `VectorizedObjectiveJAX`):

```bash
uv pip install -e ".[jax]"  # with uv
# or
python -m pip install -e ".[jax]"  # with pip
```

**Checking for JAX availability in code:**

```python
import subhkl

if subhkl.HAS_JAX:
    # Use JAX-accelerated methods
    opt.minimize_evosax("DE", population_size=1000)
else:
    # Fall back to alternative methods or inform user
    print("JAX not available. Install with: pip install -e '.[jax]'")
```

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