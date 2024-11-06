import h5py
import numpy as np
import os
import typer
import uuid

from subhkl.detector import Detector, DetectorShape
from subhkl.integration import FindPeaks
from subhkl.optimization import FindUB
from subhkl.utils import scale_coordinates

app = typer.Typer()

def index(
        num_procs: int,
        hdf5_peaks_filename: str,
        output_peaks_filename: str
    ):
    '''
    Index the given peak file and save it.
    
    Params:
        num_procs: Number of pyswarm threads to use in optimization.
        hdf5_peaks_filename: Path to the input hdf5 file to index
        output_peaks_filename: Path to write the output hdf file.
    '''
    
    # Index the peaks
    opt = FindUB(hdf5_peaks_filename)
    num, hkl, lamda = opt.minimize(num_procs)
    h, k, l = hkl
    
    # Get UB to save to output
    B = opt.reciprocal_lattice_B()
    U = opt.orientation_U(*opt.x)
    
    # Save output to HDF5 file
    with h5py.File(output_peaks_filename, 'w') as f:
        f['sample/B'] = B
        f['sample/U'] = U
        f['peaks/h'] = h
        f['peaks/k'] = k
        f['peaks/l'] = l
        f['peaks/lambda'] = lamda

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
        kwargs["min_pix"] = min_pixel_dstance
    if min_relative_intensities > 0:
        kwargs["min_rel_intensities"]

    # Calculate the x,y of each peak in pixel space
    xp, yp = peaks.harvest_peaks(**kwargs)

    # Output CSV-style filename
    print(f"Printing {output_xy_csv_filename}")
    np.savetxt(
        output_xy_csv_filename,
        np.column_stack((xp,yp)),
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
        np.column_stack((two_theta,az_phi)),
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
    with h5py.File(unique_filename, 'w') as f:
        f['sample/a'] = a
        f['sample/b'] = b
        f['sample/c'] = c
        f['sample/alpha'] = alpha
        f['sample/beta'] = beta
        f['sample/gamma'] = gamma
        f['sample/centering'] = sample_centering
        f['instrument/wavelength'] = [wavelength_min, wavelength_max]
        f['goniometer/R'] = R
        f['peaks/scattering'] = two_theta
        f['peaks/azimuthal'] = az_phi

    index(num_procs, unique_filename, output_peaks_filename)


@app.command()
def indexer_using_file(
        num_procs: int,
        hdf5_peaks_filename: str,
        output_peaks_filename: str
    ):
    
    index(num_procs, hdf5_peaks_filename, output_peaks_filename)


if __name__ == "__main__":
    app()
