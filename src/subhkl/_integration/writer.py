import h5py
import numpy as np


def write_hdf5(
    peaks,
    output_filename: str,
    rotations: list[float],
    two_theta: list[float],
    az_phi: list[float],
    wavelength_mins: list[float],
    wavelength_maxes: list[float],
    intensity: list[float],
    sigma: list[float],
    radii: list[float],
    xyz: list[list[float]],
    bank: list[int],
    image_index: list[int] = None,
    run_id: list[int] = None,
    gonio_axes: list[list[float]] = None,
    gonio_angles: list[list[float]] = None,
    gonio_names: list[str] = None,
    instrument_wavelength: tuple[float, float] = None,
):
    with h5py.File(output_filename, "w") as f:
        f.attrs["instrument"] = peaks.instrument
        f["wavelength_mins"] = wavelength_mins
        f["wavelength_maxes"] = wavelength_maxes
        f["goniometer/R"] = rotations
        f["peaks/two_theta"] = two_theta
        f["peaks/azimuthal"] = az_phi
        f["peaks/intensity"] = intensity
        f["peaks/sigma"] = sigma
        f["peaks/radius"] = radii
        f["peaks/xyz"] = xyz
        f["bank"] = bank

        if image_index is not None:
            f["peaks/image_index"] = image_index

        if run_id is not None:
            f["peaks/run_index"] = run_id

        if peaks.image.raw_files:
            f["files"] = np.array([s.encode("utf-8") for s in peaks.image.raw_files])
            f["file_offsets"] = peaks.image.file_offsets

        if gonio_axes is not None:
            f["goniometer/axes"] = gonio_axes

        if gonio_angles is not None:
            f["goniometer/angles"] = gonio_angles

        if gonio_names is not None:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("goniometer/names", data=gonio_names, dtype=dt)

        if instrument_wavelength is not None:
            f["instrument/wavelength"] = instrument_wavelength
