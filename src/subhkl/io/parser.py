import h5py
import numpy as np
import subhkl.normalization as normalization
import typer
import uuid

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
) -> None:
    # Create peak finder from tiff file
    print(f"Creating peaks from {tiff_filename}")
    peaks = FindPeaks(tiff_filename)

    # Setup optional arguments
    kwargs = {}
    if min_pixel_distance > 0:
        kwargs["min_pix"] = min_pixel_distance
    if min_relative_intensities > 0:
        kwargs["min_rel_intensities"]

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

@app.command()
def normalize(
        hdf5_peaks_filename: str, output_peaks_filename: str
    ):
    '''
    Create a copy of the input file, with each normalization value (from the normalization module) added to each row.
    
    Args:
        hdf5_peaks_filename: String path for the input file to calculate normalizations
        output_peaks_file: String path for the output file location to write to.
    '''
    
    with h5py.File(hdf5_peaks_filename, "r") as original:
        with h5py.File(output_peaks_filename, "w") as output:
            
            # Copy all data to the output file
            for h5obj in original.keys():
                original.copy(h5obj, output)
            
            # Lists of each normalization type's values per row
            lorentz = []
            absorption = []
            detector_efficiency = []
            extinction = []
            
            # Calculate normalization for each row
            for i in range(len(output["peaks"]["lambda"])):
                
                # Get the lambda and theta values for each row
                lambda_value = output["peaks"]["lambda"][i]
                theta = output["peaks"]["scattering"][i] / 2.0
                
                # Add each normalization to the list
                lorentz.append(normalization.lorentz_correction(lambda_value, theta))
                absorption.append(normalization.absorption(lambda_value))
                detector_efficiency.append(normalization.detector_efficiency(theta))
                extinction.append(normalization.extinction(lambda_value))
                
            # Write final data to new normalization group
            output["peaks"].create_group("normalization")
            output["peaks"]["normalization"]["lorentz"] = lorentz
            output["peaks"]["normalization"]["absorption"] = absorption
            output["peaks"]["normalization"]["detecter_efficiency"] = detector_efficiency
            output["peaks"]["normalization"]["extinction"] = extinction
        


if __name__ == "__main__":
    app()
