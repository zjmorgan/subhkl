from typer.testing import CliRunner
from subhkl.io.parser import app


def test_find_args():
    runner = CliRunner()
    test_args = ["find", "my.tiff"]
    result = runner.invoke(app, test_args)
    assert result.exit_code == 0

    output = result.stdout.rstrip()
    expected_output = ""
    assert expected_output in output
    args = parse_optimization_args([])
