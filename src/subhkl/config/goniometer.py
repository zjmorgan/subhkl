"""
Compute goniometer rotation matrix from Euler angles.

Euler specification is loaded `reduction_settings.json`, and
angle values are loaded from .nxs.h5.

---

Notes
- Mantid `SetGoniometer` assumes that angles are *intrinsic* and composed
  in the same order as pushed onto m_motors, i.e., if the axis order is
  Axis0, Axis1, Axis2, with corresponding local rotation matrices R0, R1, R2,
  then the global rotation matrix is R = R0 * R1 * R2.
  (see https://github.com/mantidproject/mantid/blob/main/Framework/Geometry/src/Instrument/Goniometer.cpp#L346,
  which is called by `SetGoniometer` https://github.com/mantidproject/mantid/blob/main/Framework/Crystal/src/SetGoniometer.cpp#L181)
- Mantid `SetGoniometer` expects rotation axes to be specified by the axis
  local x,y,z coordinates (since angles are intrinsic) and an orientation
  o in {-1, 1}, which indicates whether the angle is taken in clockwise (-1)
  or counter-clockwise (1) sense about the axis. Data are packed in an array
  as [x, y, z, o] (see `reduction_settings.json` for examples).
- ??? The order of the axes in `reduction_settings.json` corresponds to the
  order they would be input into Mantid `SetGoniometer` ???
- Based on experimentation, `scipy.spatial.transform.Rotation.from_rotvec`
  constructs a rotation from a rotation vector according to counter-clockwise
  orientation in a right-handed coordinate system (you can achieve clockwise by
  negating the rotation vector and using `from_rotvec`). A rotation vector is
  easily constructed from the axis-angle obtained by combining the Euler angle
  specification [x, y, z, o] and the corresponding angle read from a .nxs.h5
  file.
- Mantid converts the axis-angle representation directly into a quaternion using
  the standard method (see https://github.com/mantidproject/mantid/blob/main/Framework/Kernel/src/Quat.cpp#L114)
  which results in the same sense of the rotation as the interpretation as a
  rotation counter-clockwise about the axis in a right-handed coordinate system
  (that is, it is consistent with constructing the rotation with from_rotvec)
  See here if you are interested: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Rotation_identity
"""
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .config import reduction_settings


def get_rotation_data_from_nexus(filename, instrument):
    """
    Get goniometer axes and rotation angles from Nexus file

    Parameters
    ----------
    filename : str
        Name of nexus file to load angles from

    instrument : str
        Name of instrument used to collect data

    Returns
    -------
    axes : list[length 4 numpy array]
        List of axes in format used by Mantid `SetGoniometer`
    angles : list[float]
        List of angles in degrees about the axes
    names : list[str]
        List of axis names
    """
    settings = reduction_settings[instrument]
    axes, angles, names = [], [], []
    with h5py.File(filename) as f:
        das_logs = f["entry/DASlogs"]

        # We can iterate directly over settings["Goniometer"] as of Python 3.6
        # which guarantees that `json.load` keeps the iteration order of keys
        # the same as it is in the original file.
        # So this should work fine--assuming the order is correct in
        # `reduction_settings.json`, that is!
        for axis_name, axis_spec in settings["Goniometer"].items():
            angle_deg = float(das_logs[axis_name]["average_value"][0])
            axis = np.array(axis_spec, dtype=float)
            angles.append(angle_deg)
            axes.append(axis)
            names.append(axis_name)

    return axes, angles, names


def calc_goniometer_rotation_matrix(axes, angles):
    """
    Calculate the goniometer rotation matrix.

    Parameters
    ----------
    axes : list[list[float]]
        Parallel list of axes corresponding to the angles; each list is packed
        as in Mantid `SetGoniometer`.
    angles : list[float]
        List of the angles in degrees (in the same order as Mantid
        `SetGoniometer`)

    Returns
    -------
    matrix : 3x3 numpy array
        The goniometer rotation matrix
    """
    matrix = np.eye(3)

    for angle_deg, axis_spec in zip(angles, axes):
        # Make rotation vector by combining angle and spec
        sign = axis_spec[3]
        direction = np.array(axis_spec[:3], dtype=float)
        rot_vec = sign * angle_deg * direction

        # Multiply rotation matrix on the right to achieve the ordering
        # used by Mantid `SetGoniometer`
        axis_matrix = Rotation.from_rotvec(rot_vec, degrees=True).as_matrix()
        matrix = matrix @ axis_matrix

    return matrix
