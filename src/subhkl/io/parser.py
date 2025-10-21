import typing
import h5py
import numpy as np
import typer
import uuid

from subhkl.integration import Peaks
from subhkl.optimization import FindUB

app = typer.Typer()


def index(num_procs: int, hdf5_peaks_filename: str, output_peaks_filename: str):
    """
    Index the given peak file and save it.

    Params:
        num_procs: Number of pyswarm threads to use in optimization.
        hdf5_peaks_filename: Path to the input hdf5 file to index
        output_peaks_filename: Path to write the output hdf file.
    """

    # Index the peaks
    opt = FindUB(hdf5_peaks_filename)
    num, hkl, lamda = opt.minimize(num_procs)
    h = [i[0] for i in hkl]
    k = [i[1] for i in hkl]
    l_list = [i[2] for i in hkl]

    # Get UB to save to output
    B = opt.reciprocal_lattice_B()
    U = opt.orientation_U(*opt.x)

    with h5py.File(hdf5_peaks_filename) as f:
        intensity = np.array(f["peaks/intensity"])
        sigma = np.array(f["peaks/sigma"])

    # Save output to HDF5 file
    with h5py.File(output_peaks_filename, "w") as f:
        f["sample/B"] = B
        f["sample/U"] = U
        f["peaks/h"] = h
        f["peaks/k"] = k
        f["peaks/l"] = l_list
        f["peaks/lambda"] = lamda
        f["peaks/intensity"] = intensity
        f["peaks/sigma"] = sigma


@app.command()
def finder(
    filename: str,
    instrument: str,
    output_filename: str = "output.h5",
    min_pixel_distance: int = -1,
    min_relative_intensity: float = -1,
    wavelength_min: typing.Optional[float] = None,
    wavelength_max: typing.Optional[float] = None,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_intensity: float = 4500.0,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 30,
    peak_pixel_outlier_threshold: float = 2.0
):
    # Create peak finder from file + instrument
    print(f"Creating peaks from {filename} for instrument {instrument}")

    # Setup optional arguments for wavelength range
    wavelength_kwargs = {}
    if wavelength_min:
        wavelength_kwargs["wavelength_min"] = wavelength_min
    if wavelength_max:
        wavelength_kwargs["wavelength_max"] = wavelength_max

    peaks = Peaks(filename, instrument, **wavelength_kwargs)

    # Setup optional arguments for peak finding
    peak_kwargs = {}
    if min_pixel_distance > 0:
        peak_kwargs["min_pix"] = min_pixel_distance
    if min_relative_intensity > 0:
        peak_kwargs["min_rel_intensity"] = min_relative_intensity

    # Setup parameters for integration with convex hull algorithm
    integration_params = {
        "region_growth_distance_threshold": region_growth_distance_threshold,
        "region_growth_minimum_intensity": region_growth_minimum_intensity,
        "region_growth_maximum_pixel_radius": region_growth_maximum_pixel_radius,
        "peak_center_box_size": peak_center_box_size,
        "peak_smoothing_window_size": peak_smoothing_window_size,
        "peak_minimum_pixels": peak_minimum_pixels,
        "peak_pixel_outlier_threshold": peak_pixel_outlier_threshold
    }

    # Calculate the peaks in detector space
    detector_peaks = peaks.get_detector_peaks(peak_kwargs, integration_params, True)

    # Write out the output HDF5 peaks file
    peaks.write_hdf5(
        output_filename=output_filename,
        rotations=detector_peaks.R,
        two_theta=detector_peaks.two_theta,
        az_phi=detector_peaks.az_phi,
        wavelengths=detector_peaks.wavelengths,
        intensity=detector_peaks.intensity,
        sigma=detector_peaks.sigma
    )


@app.command()
def indexer(
    num_procs: int,
    peaks_h5_filename: str,
    goniometer_csv_filename: str,
    output_peaks_filename: str,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    wavelength_min: float,
    wavelength_max: float,
    sample_centering: str,
) -> None:
    # Load peaks h5 file
    with h5py.File(peaks_h5_filename) as f:
        two_theta = np.array(f["two_theta"])
        az_phi = np.array(f["azimuthal"])
        intensity = np.array(f["intensity"])
        sigma = np.array(f["sigma"])

    # Read in goniometer from CSV filename
    R = np.loadtxt(goniometer_csv_filename, delimiter=",")

    # Write HDF5 input file for indexer
    unique_filename = str(uuid.uuid4()) + ".h5"
    with h5py.File(unique_filename, "w") as f:
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/centering"] = sample_centering
        f["instrument/wavelength"] = [wavelength_min, wavelength_max]
        f["goniometer/R"] = R
        f["peaks/scattering"] = two_theta
        f["peaks/azimuthal"] = az_phi
        f["peaks/intensity"] = intensity
        f["peaks/sigma"] = sigma

    index(num_procs, unique_filename, output_peaks_filename)


@app.command()
def indexer_using_file(
    num_procs: int, hdf5_peaks_filename: str, output_peaks_filename: str
):
    index(num_procs, hdf5_peaks_filename, output_peaks_filename)


if __name__ == "__main__":
    app()
