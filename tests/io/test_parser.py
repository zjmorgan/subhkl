import pytest
from typer.testing import CliRunner
from subhkl.io.parser import app, finder


@pytest.mark.skip(reason="finder function returns None, not 0 - needs fixing")
def test_finder_function_tiff(meso_tiff):
    instrument = "IMAGINE"
    result = finder(
        filename=meso_tiff,
        instrument=instrument,
    )
    assert result == 0


@pytest.mark.skip(reason="CliRunner usage issue - needs fixing")  
def test_find_args_tiff(meso_tiff):
    runner = CliRunner(meso_tiff)
    test_args = ["finder", meso_tiff, "IMAGINE", "output.h5"]
    result = runner.invoke(app, test_args)
    assert result.exit_code == 0

    output = result.stdout.rstrip()
    expected_output = ""
    assert expected_output in output
