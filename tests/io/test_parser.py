from typer.testing import CliRunner
from subhkl.io.parser import app, finder


def test_finder_function(meso_tiff):
    instrument = "IMAGINE"
    finder(
        filename=meso_tiff,
        instrument=instrument,
    )


def test_find_args():
    runner = CliRunner()
    test_args = ["finder", "tests/data/meso_2_15min_2-0_4-5_050.tif", "out.csv"]
    result = runner.invoke(app, test_args)
    print(result)
    assert result.exit_code == 0

    output = result.stdout.rstrip()
    expected_output = ""
    assert expected_output in output
