import typing
import h5py
import numpy as np
import typer
import uuid

from subhkl import normalization
from subhkl.export import (
    FinderConcatenateMerger,
    IndexerConcatenateMerger,
    MTZExporter
)
from subhkl.integration import Peaks
from subhkl.optimization import FindUB

app = typer.Typer()


def index(
    hdf5_peaks_filename: str,
    output_peaks_filename: str,
    strategy_name: str,
    population_size: int,
    gens: int,
    n_runs: int,
    seed: int
):
    """
    Index the given peak file and save it using the evosax optimizer.

    Params:
        hdf5_peaks_filename: Path to the input hdf5 file to index
        output_peaks_filename: Path to write the output hdf file.
        strategy_name: Optimization strategy ('DE' or 'PSO').
        population_size: Population size for each generation.
        gens: Number of generations to run.
        n_runs: Number of optimization runs with different seeds.
        seed: Base seed for the first optimization run.
    """

    # Index the peaks
    opt = FindUB(hdf5_peaks_filename)

    print(f"Starting evosax optimization with strategy: {strategy_name}")
    print(f"Running {n_runs} run(s)...")
    print(f"Settings per run: Population Size={population_size}, Generations={gens}")

    # Call the new evosax minimizer
    num, hkl, lamda = opt.minimize_evosax(
        strategy_name=strategy_name,
        population_size=population_size,
        num_generations=gens,
        n_runs=n_runs,
        seed=seed
    )

    print(f"\nOptimization complete. Best solution indexed {num} peaks.")

    h = [i[0] for i in hkl]
    k = [i[1] for i in hkl]
    l_list = [i[2] for i in hkl]

    # Get UB to save to output
    B = opt.reciprocal_lattice_B()
    U = opt.orientation_U(*opt.x)

    # Copy data from temporary HDF5
    copy_keys = [
        "sample/a",
        "sample/b",
        "sample/c",
        "sample/alpha",
        "sample/beta",
        "sample/gamma",
        "sample/centering",
        "instrument/wavelength",
        "goniometer/R",
        "peaks/intensity",
        "peaks/sigma",
        "peaks/scattering",
        "peaks/azimuthal",
    ]

    copied_data = {}

    with h5py.File(hdf5_peaks_filename) as f:
        for key in copy_keys:
            copied_data[key] = np.array(f[key])

    # Save output to HDF5 file
    print(f"Saving indexed peaks to {output_peaks_filename}...")
    with h5py.File(output_peaks_filename, "w") as f:
        for key, value in copied_data.items():
            f[key] = value

        f["sample/B"] = B
        f["sample/U"] = U
        f["peaks/h"] = h
        f["peaks/k"] = k
        f["peaks/l"] = l_list
        f["peaks/lambda"] = lamda
    print("Done.")


@app.command()
def finder(
    filename: str,
    instrument: str,
    output_filename: str = "output.h5",
    finder_algorithm: str = "peak_local_max",
    show_progress: bool = False,
    create_visualizations: bool = False,
    peak_local_max_min_pixel_distance: int = -1,
    peak_local_max_min_relative_intensity: float = -1,
    peak_local_max_normalization: bool = False,
    thresholding_noise_cutoff_quantile: float = 0.8,
    thresholding_min_peak_dist_pixels: float = 8.0,
    thresholding_mask_file: typing.Optional[str] = None,
    thresholding_mask_rel_erosion_radius: float = 0.05,
    thresholding_blur_kernel_sigma: int = 5,
    thresholding_open_kernel_size_pixels: int = 3,
    wavelength_min: typing.Optional[float] = None,
    wavelength_max: typing.Optional[float] = None,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_intensity: float = 4500.0,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 30,
    peak_minimum_signal_to_noise: float = 1.0,
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
    peak_kwargs = {"algorithm": finder_algorithm}
    if finder_algorithm == "peak_local_max":
        if peak_local_max_min_pixel_distance > 0:
            peak_kwargs["min_pix"] = peak_local_max_min_pixel_distance
        if peak_local_max_min_relative_intensity > 0:
            peak_kwargs["min_rel_intensity"] = peak_local_max_min_relative_intensity
        peak_kwargs['normalize'] = peak_local_max_normalization
    elif finder_algorithm == "thresholding":
        peak_kwargs.update({
            "noise_cutoff_quantile": thresholding_noise_cutoff_quantile,
            "min_peak_dist_pixels": thresholding_min_peak_dist_pixels,
            "mask_file": thresholding_mask_file,
            "mask_rel_erosion_radius": thresholding_mask_rel_erosion_radius,
            "blur_kernel_sigma": thresholding_blur_kernel_sigma,
            "open_kernel_size_pixels": thresholding_open_kernel_size_pixels,
            #"show_steps": True,
            "show_scale": "log"
        })
    else:
        raise ValueError("Invalid finder algorithm; only \"peak_local_max\" "
                         "and \"thresholding\" are allowed")

    # Setup parameters for integration with convex hull algorithm
    integration_params = {
        "region_growth_distance_threshold": region_growth_distance_threshold,
        "region_growth_minimum_intensity": region_growth_minimum_intensity,
        "region_growth_maximum_pixel_radius": region_growth_maximum_pixel_radius,
        "peak_center_box_size": peak_center_box_size,
        "peak_smoothing_window_size": peak_smoothing_window_size,
        "peak_minimum_pixels": peak_minimum_pixels,
        "peak_minimum_signal_to_noise": peak_minimum_signal_to_noise,
        "peak_pixel_outlier_threshold": peak_pixel_outlier_threshold
    }

    # Calculate the peaks in detector space
    detector_peaks = peaks.get_detector_peaks(
        peak_kwargs,
        integration_params,
        visualize=create_visualizations,
        show_progress=show_progress,
        file_prefix=filename
    )

    # Write out the output HDF5 peaks file
    peaks.write_hdf5(
        output_filename=output_filename,
        rotations=detector_peaks.R,
        two_theta=detector_peaks.two_theta,
        az_phi=detector_peaks.az_phi,
        wavelength_mins=detector_peaks.wavelength_mins,
        wavelength_maxes=detector_peaks.wavelength_maxes,
        intensity=detector_peaks.intensity,
        sigma=detector_peaks.sigma,
        bank=detector_peaks.bank
    )


@app.command()
def finder_merger(
    finder_h5_txt_list_filename: str,
    output_pre_index_filename: str,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    wavelength_min: float,
    wavelength_max: float,
    sample_centering: str
):
    with open(finder_h5_txt_list_filename) as f:
        finder_h5_files = f.read().splitlines()

    merging_algorithm = FinderConcatenateMerger(finder_h5_files)
    merging_algorithm.merge(output_pre_index_filename)

    # Write HDF5 input file for indexer
    with h5py.File(output_pre_index_filename, "r+") as f:
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/centering"] = sample_centering
        f["instrument/wavelength"] = [wavelength_min, wavelength_max]
        f["goniometer/R"] = f["rotations"]
        f["peaks/scattering"] = f["two_theta"]
        f["peaks/azimuthal"] = f["azimuthal"]
        f["peaks/intensity"] = f["intensity"]
        f["peaks/sigma"] = f["sigma"]
        del f["rotations"], f["two_theta"], f["azimuthal"], f["intensity"], f["sigma"]


@app.command()
def indexer(
    peaks_h5_filename: str,
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
    goniometer_csv_filename: typing.Optional[str] = None,
    # --- Updated evosax CLI arguments ---
    strategy_name: str = typer.Option(
        "DE", 
        "--strategy", 
        help="Optimization strategy to use (e.g., 'DE' or 'PSO')."
    ),
    n_runs: int = typer.Option(
        1, 
        "--n-runs", "-n", 
        help="Number of optimization runs with different seeds."
    ),
    population_size: int = typer.Option(
        1000, 
        "--population-size", "--popsize", 
        help="Population size for each generation."
    ),
    gens: int = typer.Option(
        100, 
        "--gens", 
        help="Number of generations to run."
    ),
    seed: int = typer.Option(
        0, 
        "--seed", 
        help="Base seed for the first optimization run."
    ),
) -> None:
    """
    Find peaks, prepare, and index them from command-line parameters.
    """
    # Load peaks h5 file
    print(f"Loading peaks from: {peaks_h5_filename}")
    with h5py.File(peaks_h5_filename) as f:
        two_theta = np.array(f["two_theta"])
        az_phi = np.array(f["azimuthal"])
        intensity = np.array(f["intensity"])
        sigma = np.array(f["sigma"])
        rotations = np.array(f["rotations"])

    # Read in goniometer from CSV filename, if given
    if goniometer_csv_filename is not None:
        print(f"Loading goniometer from: {goniometer_csv_filename}")
        R = np.loadtxt(goniometer_csv_filename, delimiter=",")
    else:
        print("Using goniometer rotation from peaks file.")
        R = goniometer_rotation

    # Write HDF5 input file for indexer
    unique_filename = str(uuid.uuid4()) + ".h5"
    print(f"Creating temporary indexer input file: {unique_filename}")
    with h5py.File(unique_filename, "w") as f:
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/centering"] = sample_centering
        f["instrument/wavelength"] = [wavelength_min, wavelength_max]
        f["goniometer/R"] = rotations
        f["peaks/scattering"] = two_theta
        f["peaks/azimuthal"] = az_phi
        f["peaks/intensity"] = intensity
        f["peaks/sigma"] = sigma

    # Call the internal index function with the new parameters
    index(
        hdf5_peaks_filename=unique_filename,
        output_peaks_filename=output_peaks_filename,
        strategy_name=strategy_name,
        population_size=population_size,
        gens=gens,
        n_runs=n_runs,
        seed=seed
    )


@app.command()
def indexer_using_file(
    hdf5_peaks_filename: str, 
    output_peaks_filename: str,
    # --- Updated evosax CLI arguments ---
    strategy_name: str = typer.Option(
        "DE", 
        "--strategy", 
        help="Optimization strategy to use (e.g., 'DE' or 'PSO')."
    ),
    n_runs: int = typer.Option(
        1, 
        "--n-runs", "-n", 
        help="Number of optimization runs with different seeds."
    ),
    population_size: int = typer.Option(
        1000, 
        "--population-size", "--popsize", 
        help="Population size for each generation."
    ),
    gens: int = typer.Option(
        100, 
        "--gens", 
        help="Number of generations to run."
    ),
    seed: int = typer.Option(
        0, 
        "--seed", 
        help="Base seed for the first optimization run."
    ),
):
    """
    Index a pre-prepared HDF5 file that already contains all sample/instrument info.
    """
    # Call the internal index function with the new parameters
    index(
        hdf5_peaks_filename=hdf5_peaks_filename,
        output_peaks_filename=output_peaks_filename,
        strategy_name=strategy_name,
        population_size=population_size,
        gens=gens,
        n_runs=n_runs,
        seed=seed
    )


@app.command()
def peak_predictor(
    filename: str,
    instrument: str,
    indexed_hdf5_filename: str,
    integration_peaks_filename: str,
    d_min: float = 1.0,
    create_visualizations: bool = False
):
    peaks = Peaks(filename, instrument, wavelength_min=1, wavelength_max=4)

    with h5py.File(indexed_hdf5_filename) as f_indexed:
        a = float(np.array(f_indexed["sample/a"]))
        b = float(np.array(f_indexed["sample/b"]))
        c = float(np.array(f_indexed["sample/c"]))
        alpha = float(np.array(f_indexed["sample/alpha"]))
        beta = float(np.array(f_indexed["sample/beta"]))
        gamma = float(np.array(f_indexed["sample/gamma"]))
        centering = np.array(f_indexed["sample/centering"]).item().decode('utf-8')
        wavelength = np.array(f_indexed["instrument/wavelength"])
        R = peaks.goniometer_rotation
        U = np.array(f_indexed["sample/U"])
        B = np.array(f_indexed["sample/B"])

    UB = R @ U @ B

    peak_dict = peaks.predict_peaks(
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        centering,
        d_min,
        UB
    )

    if create_visualizations:
        import matplotlib.pyplot as plt

        for bank, predicted_peaks in peak_dict.items():
            plt.imshow(peaks.ims[bank].T + 1, cmap="binary", norm="log")
            plt.scatter(predicted_peaks[0], predicted_peaks[1], edgecolors='r', facecolors='none')
            plt.title(str(bank))
            plt.savefig(filename + str(bank) + "_pred.png")
            plt.show()

    with h5py.File(integration_peaks_filename, "w") as f:
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/centering"] = centering
        f["instrument/wavelength"] = wavelength
        f["goniometer/R"] = R

        for bank, (i, j, h, k, l, wl) in peak_dict.items():
            f[f"banks/{bank}/i"] = i
            f[f"banks/{bank}/j"] = j
            f[f"banks/{bank}/h"] = h
            f[f"banks/{bank}/k"] = k
            f[f"banks/{bank}/l"] = l
            f[f"banks/{bank}/wavelength"] = wl


@app.command()
def integrator(
    filename: str,
    instrument: str,
    integration_peaks_filename: str,
    output_filename: str,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_intensity: float = 4500.0,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 30,
    peak_minimum_signal_to_noise: float = 1.0,
    peak_pixel_outlier_threshold: float = 2.0,
    create_visualizations: bool = False,
    show_progress: bool = False
):
    peak_dict = {}
    with h5py.File(integration_peaks_filename) as f:
        for bank in f["banks"].keys():
            peak_dict[int(bank)] = [
                np.array(f[f"banks/{bank}/i"]),
                np.array(f[f"banks/{bank}/j"]),
                np.array(f[f"banks/{bank}/h"]),
                np.array(f[f"banks/{bank}/k"]),
                np.array(f[f"banks/{bank}/l"]),
                np.array(f[f"banks/{bank}/wavelength"])
            ]

    # Setup parameters for integration with convex hull algorithm
    integration_params = {
        "region_growth_distance_threshold": region_growth_distance_threshold,
        "region_growth_minimum_intensity": region_growth_minimum_intensity,
        "region_growth_maximum_pixel_radius": region_growth_maximum_pixel_radius,
        "peak_center_box_size": peak_center_box_size,
        "peak_smoothing_window_size": peak_smoothing_window_size,
        "peak_minimum_pixels": peak_minimum_pixels,
        "peak_minimum_signal_to_noise": peak_minimum_signal_to_noise,
        "peak_pixel_outlier_threshold": peak_pixel_outlier_threshold
    }

    peaks = Peaks(filename, instrument)
    result = peaks.integrate(
        peak_dict,
        integration_params,
        create_visualizations=create_visualizations,
        show_progress=show_progress,
        file_prefix=filename
    )

    copy_keys = [
        "sample/a",
        "sample/b",
        "sample/c",
        "sample/alpha",
        "sample/beta",
        "sample/gamma",
        "sample/centering",
        "instrument/wavelength",
        "goniometer/R",
    ]

    with h5py.File(output_filename, "w") as f:
        f["peaks/h"] = result.h
        f["peaks/k"] = result.k
        f["peaks/l"] = result.l
        f["peaks/lambda"] = result.wavelength
        f["peaks/intensity"] = result.intensity
        f["peaks/sigma"] = result.sigma
        f["peaks/scattering"] = result.tt
        f["peaks/azimuthal"] = result.az
        f["peaks/bank"] = result.bank

        with h5py.File(integration_peaks_filename) as f_in:
            for key in copy_keys:
                f_in.copy(f_in[key], f, key)


@app.command()
def normalizer(
    hdf5_peaks_filename: str, output_peaks_filename: str
):
    # Open the input filename
    with h5py.File(hdf5_peaks_filename, "r") as f:
        theta = np.array(f["peaks/scattering"]) / 2.0
        lamda = np.array(f["peaks/lambda"])
        detector_efficiency = normalization.detector_efficiency(lamda)
        absorption = normalization.absorption(lamda)
        extinction = normalization.extinction(lamda)
        lorentz = normalization.lorentz_correction(lamda, theta)
        full = detector_efficiency * extinction * absorption * lorentz

        # Save the result
        with h5py.File(output_peaks_filename, "w") as o:
            for key in f.keys():
                f.copy(f[key], o, key)

            o["peaks/intensity"] = f["peaks/intensity"] / full
            o["peaks/sigma"] = f["peaks/sigma"] / full


@app.command()
def merger(
    indexed_h5_txt_list_filename: str,
    output_filename: str,
    method: str = "concatenate"
):
    with open(indexed_h5_txt_list_filename) as f:
        indexed_h5_files = f.read().splitlines()

    if method.lower() == "concatenate":
        merging_algorithm = IndexerConcatenateMerger(indexed_h5_files)
    else:
        raise ValueError("Invalid merging method")

    merging_algorithm.merge(output_filename)


@app.command()
def mtz_exporter(
    indexed_h5_filename: str,
    output_mtz_filename: str,
    space_group: str
):
    algorithm = MTZExporter(indexed_h5_filename, space_group)
    algorithm.write_mtz(output_mtz_filename)


if __name__ == "__main__":
    app()
