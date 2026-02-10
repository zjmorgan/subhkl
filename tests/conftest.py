import os
import pytest
import matplotlib
import urllib.request
from pathlib import Path


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
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"Mesolite test data already exists at {DATA_DIR}")
        return

    max_files = os.environ.get("MESOLITE_MAX_FILES")
    max_files = int(max_files) if max_files else None

    files_to_download = ZENODO_FILES[:max_files] if max_files else ZENODO_FILES

    print(
        f"Downloading {len(files_to_download)} mesolite files from Zenodo (record {ZENODO_RECORD_ID})..."
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download each file
    for i, filename in enumerate(files_to_download, 1):
        file_url = f"{ZENODO_BASE_URL}/{filename}"
        file_path = DATA_DIR / filename

        try:
            print(f"  [{i}/{len(files_to_download)}] Downloading {filename}...")
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
