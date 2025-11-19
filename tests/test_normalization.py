import h5py
import os
import pytest

from subhkl.io.parser import normalize, normalize_intensities
from subhkl.normalization import lorentz_correction

directory = os.path.dirname(os.path.abspath(__file__))


def test_lorentz_correction():
    """
    Test the lorentz correction.
    """

    # Test some known values
    tolerance = 0.001
    assert 0 == lorentz_correction(0, 1)
    assert lorentz_correction(1, 1) == pytest.approx(0.706141463718696, tolerance)
    assert lorentz_correction(2, 1) == pytest.approx(11.298263419499136, tolerance)

    # In-plane angle = 0 is a math error, so return 0
    assert 0 == lorentz_correction(1, 0)


def test_parser():
    """
    Test the parser normalization command.
    """

    # Normalize the h5 file
    infile = os.path.join(directory, "data/5vnq_mandi.h5")
    normalize(infile, "out_test_parser.h5")

    # Check that the output file has normalization data
    with h5py.File("out_test_parser.h5") as f:
        # Each row should have normalization data
        assert len(f["peaks"]["lambda"]) == len(f["peaks"]["normalization"]["lorentz"])

    # Delete test file
    os.unlink("out_test_parser.h5")


def test_parser_intensities():
    """
    Test the parser normalization command.
    """

    # Normalize the h5 file
    raw_file = os.path.join(directory, "data/MANDI_12937.nxs.h5")
    infile = os.path.join(directory, "data/5vnq_mandi.h5")
    normalize_intensities(raw_file, infile, "out_test_parser.h5", "MANDI")

    # Check that the output file has normalization data
    with h5py.File("out_test_parser.h5") as f:
        # Each row should have normalization data
        assert len(f["peaks"]["lambda"]) == len(f["peaks"]["normalization"]["lorentz"])

    # Delete test file
    os.unlink("out_test_parser.h5")
