def scale_coordinates(xp, yp, scale_x, scale_y, nx, ny):
    """
    Scale from pixel coordinates to real positions

    Parameters
    ----------
    xp, yp : array, int
        Image coordinates.
    scale_x, scale_y : float
        Pixel scaling factors.
    nx, ny : int
       Image shape / resolution

    Returns
    -------
    x, y : array, float
        Image pixel position.

    """
    return (xp - nx / 2) * scale_x, (yp - ny / 2) * scale_y
