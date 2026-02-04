# Tests for subhkl

This contains the tests and data for subhkl

## Test Data

### Large Test Datasets (Zenodo)

The large mesolite test dataset (~4GB) is stored on Zenodo and automatically downloaded when running tests:
- **Zenodo DOI**: [10.5281/zenodo.18475332](https://doi.org/10.5281/zenodo.18475332)
- **Location**: `tests/data/MANDI/mesolite/`
- **Auto-download**: Configured in `conftest.py` (runs once per test session)

The data will be automatically downloaded on first test run. If you need to manually download:
```bash
wget https://zenodo.org/records/18475332/files/mesolite.tar.gz
tar -xzf mesolite.tar.gz -C tests/data/MANDI/
```

### Generating Test Data

The script `data/generate_test_data.py` in this directory
allows for generating the test data for the tests.

Yet, this script does require `mantid-framework` to be installed.

This is not part of the packages dependences or dev dependencies.


