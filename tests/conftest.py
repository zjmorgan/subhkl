import os
import urllib.request
from pathlib import Path

import matplotlib
import pytest

# Use non-interactive backend for tests to prevent plot windows from popping up
matplotlib.use("Agg")


# Zenodo dataset configuration
ZENODO_RECORD_ID = "18475332"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"
DATA_DIR = Path(__file__).parent / "data" / "MANDI" / "mesolite"

# List of all files on Zenodo (generated from the actual upload)
ZENODO_FILES = [f"MANDI_{i}.nxs.h5" for i in range(11613, 11685)]


@pytest.fixture(scope="session", autouse=True)
def download_mesolite_data():
    """Download mesolite test data from Zenodo if not already present.

    Set MESOLITE_MAX_FILES environment variable to limit number of files downloaded (for testing).
    Example: MESOLITE_MAX_FILES=5 uv run pytest tests/
    """
    max_files = os.environ.get("MESOLITE_MAX_FILES")
    max_files = int(max_files) if max_files is not None else len(ZENODO_FILES)

    files_to_download = ZENODO_FILES[:max_files]

    # Check which files we are missing
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = {f.name for f in DATA_DIR.glob("*.h5")}
    needed_files = [f for f in files_to_download if f not in existing_files]

    if not needed_files:
        print(f"All {len(files_to_download)} requested mesolite test files already exist at {DATA_DIR}")
        return

    print(
        f"Downloading {len(needed_files)} missing mesolite files from Zenodo (record {ZENODO_RECORD_ID})..."
    )

    # Download each file
    for i, filename in enumerate(needed_files, 1):
        file_url = f"{ZENODO_BASE_URL}/{filename}"
        file_path = DATA_DIR / filename

        try:
            print(f"  [{i}/{len(needed_files)}] Downloading {filename}...")
            urllib.request.urlretrieve(file_url, file_path)
        except Exception as e:
            print(f"  Warning: Failed to download {filename}: {e}")
            continue

    downloaded_count = len(list(DATA_DIR.glob("*.h5")))
    print(f"✓ Downloaded {downloaded_count} files to {DATA_DIR}")


@pytest.fixture(name="test_data_dir")
def fixture__test_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(name="meso_tiff")
def fixture__meso_tiff(test_data_dir) -> str:
    return os.path.join(test_data_dir, "meso_2_15min_2-0_4-5_050.tif")
