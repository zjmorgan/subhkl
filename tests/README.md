# Tests for subhkl

This contains the tests and data for subhkl

## Test Data

### Large Test Datasets (Zenodo)

The mesolite test dataset (72 files, ~493 MB) is stored on Zenodo and automatically downloaded when running tests:
- **Zenodo DOI**: [10.5281/zenodo.18475332](https://doi.org/10.5281/zenodo.18475332)
- **Location**: `tests/data/MANDI/mesolite/`
- **Auto-download**: Configured in `conftest.py` (runs once per test session)
- **Files**: MANDI_11613.nxs.h5 through MANDI_11684.nxs.h5 (72 total)

The data will be automatically downloaded on first test run. If you need to manually download all files:

```bash
# Download all 72 files from Zenodo (in parallel)
ZENODO_RECORD_ID="18475332"
ZENODO_BASE_URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files"
DATA_DIR="tests/data/MANDI/mesolite"

mkdir -p "${DATA_DIR}"

# Download in parallel for speed
for i in {11613..11684}; do
    filename="MANDI_${i}.nxs.h5"
    wget "${ZENODO_BASE_URL}/${filename}" -O "${DATA_DIR}/${filename}" &
done

wait  # Wait for all downloads to complete
```

**Limit files for faster testing:**
```bash
# Download only first 5 files (faster for testing)
MESOLITE_MAX_FILES=5 uv run pytest tests/test_mandi_mesolite.py

# Download only first 10 files
MESOLITE_MAX_FILES=10 uv run pytest tests/test_mandi_mesolite.py
```

### Generating Test Data

The script `data/generate_test_data.py` in this directory
allows for generating the test data for the tests.

Yet, this script does require `mantid-framework` to be installed.

This is not part of the packages dependences or dev dependencies.


