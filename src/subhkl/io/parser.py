import h5py
import numpy as np
import typer
import uuid

from subhkl.convex_hull_expansion import PeakIntegrator, RegionGrower
from subhkl.detector import Detector, DetectorShape
from subhkl.integration import FindPeaks
from subhkl.optimization import FindUB
from subhkl.utils import scale_coordinates

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

    # Save output to HDF5 file
    with h5py.File(output_peaks_filename, "w") as f:
        f["sample/B"] = B
        f["sample/U"] = U
        f["peaks/h"] = h
        f["peaks/k"] = k
        f["peaks/l"] = l_list
        f["peaks/lambda"] = lamda


@app.command()
def finder(
    tiff_filename: str,
    output_xy_csv_filename: str,
    min_pixel_distance: float = -1,
    min_relative_intensities: float = -1,
    normalize: bool = False
) -> None:
    # Create peak finder from tiff file
    print(f"Creating peaks from {tiff_filename}")
    peaks = FindPeaks(tiff_filename)

    # Setup optional arguments
    kwargs = {"normalize": normalize}
    if min_pixel_distance > 0:
        kwargs["min_pix"] = min_pixel_distance
    if min_relative_intensities > 0:
        kwargs["min_rel_intens"] = min_relative_intensities

    # Calculate the x,y of each peak in pixel space
    xp, yp = peaks.harvest_peaks(**kwargs)

    # Output CSV-style filename
    print(f"Printing {output_xy_csv_filename}")
    np.savetxt(
        output_xy_csv_filename,
        np.column_stack((xp, yp)),
        delimiter=",",
    )


@app.command()
def finder_convex_hull(
    tiff_filename: str,
    output_xy_csv_filename: str,
    min_pixel_distance: float = -1,
    min_relative_intensities: float = -1,
    normalize: bool = False,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_intensity: float = 4500.0,
    region_growth_max_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 30,
    peak_pixel_outlier_threshold: float = 2.0
):
    # Create peak finder from tiff file
    print(f"Creating peaks from {tiff_filename}")

    # Make PeakIntegrator
    peak_integrator = PeakIntegrator(
        RegionGrower(
            distance_threshold=region_growth_distance_threshold,
            min_intensity=region_growth_minimum_intensity,
            max_size=region_growth_max_pixel_radius
        ),
        box_size=peak_center_box_size,
        smoothing_window_size=peak_smoothing_window_size,
        min_peak_pixels=peak_minimum_pixels,
        outlier_threshold=peak_pixel_outlier_threshold
    )

    peaks = FindPeaks(tiff_filename, peak_integrator=peak_integrator)

    # Setup optional arguments
    kwargs = {"normalize": normalize}
    if min_pixel_distance > 0:
        kwargs["min_pix"] = min_pixel_distance
    if min_relative_intensities > 0:
        kwargs["min_rel_intens"] = min_relative_intensities

    # Calculate candidate x,y of each peak in pixel space
    xp, yp = peaks.harvest_peaks(**kwargs)

    # Fit convex hulls and extract corrected peak centers; discard bad peaks
    peak_dict = peaks.fit_convex_hull(xp, yp)
    xy_ch = np.stack([
        center
        for center, _, _, _ in peak_dict.values()
        if center is not None
    ])

    # Convert to integer pixel coordinates to match output from finder
    xy_p = np.round(xy_ch).astype(int)

    # Output CSV-style filename
    print(f"Printing {output_xy_csv_filename}")
    np.savetxt(
        output_xy_csv_filename,
        xy_p,
        delimiter=",",
    )


@app.command()
def preparer(
    xy_csv_filename: str,
    output_peaks_csv_filename: str,
    detector_shape: DetectorShape,
    detector_height: float,
    detector_distance: float,
    image_orientation: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    nx: int = 0,
    ny: int = 0,
) -> None:
    # Read in X,Y in pixel coordinates
    xp, yp = np.loadtxt(xy_csv_filename, delimiter=",", unpack=True)

    # Scale pixel coordinates to real positions
    # Default: no scaling
    x, y = scale_coordinates(xp, yp, scale_x, scale_y, nx, ny)

    # Create detector object based on shape, height, and distance
    detector = Detector(
        x,
        y,
        detector_distance,
        detector_height,
        image_orientation,
        detector_shape,
    )

    # Compute scattering angles (in-plane and out-of-plane) for each peak
    two_theta, az_phi = detector.detector_trajectories()

    print(f"Printing {output_peaks_csv_filename}")
    np.savetxt(
        output_peaks_csv_filename,
        np.column_stack((two_theta, az_phi)),
        delimiter=",",
    )


@app.command()
def indexer(
    num_procs: int,
    peaks_csv_filename: str,
    goniometer_filename: str,
    output_peaks_filename: str,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    wavelength_min: float,
    wavelength_max: float,
    sample_centering,
) -> None:
    # Read in goniometer from CSV filename
    two_theta, az_phi = np.loadtxt(peaks_csv_filename, delimiter=",", unpack=True)
    R = np.loadtxt(goniometer_filename, delimiter=",")

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

    index(num_procs, unique_filename, output_peaks_filename)


@app.command()
def indexer_using_file(
    num_procs: int, hdf5_peaks_filename: str, output_peaks_filename: str
):
    index(num_procs, hdf5_peaks_filename, output_peaks_filename)


if __name__ == "__main__":
    app()
