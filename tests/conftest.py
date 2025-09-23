import os
import pytest

@pytest.fixture(name="test_data_dir")
def fixture__test_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@pytest.fixture(name="meso_tiff")
def fixture__meso_tiff(test_data_dir) -> str:
    return os.path.join(test_data_dir, "meso_2_15min_2-0_4-5_050.tif")
