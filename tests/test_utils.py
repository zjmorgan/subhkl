import pytest

from subhkl.utils import scale_coordinates


def test_scale_coordinates():
    xp = 878
    yp = 455
    scale_x = 0.00025133
    scale_y = 0.00025
    nx = 5000
    ny = 1800

    x, y = scale_coordinates(xp, yp, scale_x, scale_y, nx, ny)

    tolerance = 0.001
    assert x == pytest.approx(-0.40765, tolerance)
    assert y == pytest.approx(-0.11125, tolerance)


def test_scale_coordinates_no_scaling():
    xp = -1.2
    yp = 2.1
    scale_x = 1.0
    scale_y = 1.0
    nx = 0
    ny = 0

    x, y = scale_coordinates(xp, yp, scale_x, scale_y, nx, ny)

    assert x == xp
    assert y == yp
