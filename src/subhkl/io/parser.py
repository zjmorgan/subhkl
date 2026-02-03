import typing
import h5py
import numpy as np
import typer
import uuid
import os
import re
import glob

from subhkl.export import FinderConcatenateMerger, MTZExporter
from subhkl.export import FinderConcatenateMerger, MTZExporter, ImageStackMerger
from subhkl.integration import Peaks
from subhkl.optimization import FindUB
from subhkl.config.goniometer import get_rotation_data_from_nexus, calc_goniometer_rotation_matrix
from subhkl.utils import calculate_angular_error

app = typer.Typer()

def index(
    hdf5_peaks_filename: str = None, # Made optional
    output_peaks_filename: str = None,
    strategy_name: str = "DE",
    population_size: int = 1000,
    gens: int = 100,
    n_runs: int = 1,
    seed: int = 0,
    softness: float = 0.01,
    sigma_init: float = None,
    refine_lattice: bool = False,
    lattice_bound_frac: float = 0.05,
    bootstrap_filename: str = None,
    refine_goniometer: bool = False,
    refine_goniometer_axes: list = None,
    goniometer_bound_deg: float = 5.0,
    refine_sample: bool = False,
    sample_bound_meters: float = 0.002,
    refine_beam: bool = False,
    beam_bound_deg: float = 1.0,
    nexus_filename: str = None,
    instrument_name: str = None,
    loss_method: str = 'cosine',
    hkl_search_range: int = 20,
    d_min: float = None,
    d_max: float = None,
    search_window_size: int = 256,
    batch_size: int = None,
    window_batch_size: int = 32,
    chunk_size: int = 256,
    num_iters: int = 20,
    top_k: int = 32,
    B_sharpen: float = None,
    input_data: dict = None, 
):
    """
    Index the given peak file and save it using the evosax optimizer.
    """

    # 1. Initialize Optimizer with Data (No temp file)
    if input_data is not None:
        opt = FindUB(data=input_data)
    else:
        opt = FindUB(filename=hdf5_peaks_filename)

    print(f"Starting evosax optimization with strategy: {strategy_name}")
    print(f"Running {n_runs} run(s)...")
    print(f"Settings per run: Population Size={population_size}, Generations={gens}")
    if refine_lattice:
        print(f"Refining lattice parameters with {lattice_bound_frac*100}% bounds.")
    if refine_sample:
        print(f"Refining sample offset with {1000*sample_bound_meters} mm bounds.")
    if refine_beam:
        print(f"Refining beam tilt with {beam_bound_deg}° bounds.")

    goniometer_names = None
    if refine_goniometer:
        if nexus_filename and instrument_name:
             print(f"Refining goniometer angles with {goniometer_bound_deg} deg bounds.")
             axes, angles, names = get_rotation_data_from_nexus(nexus_filename, instrument_name)
             opt.goniometer_axes = np.array(axes)
             num_peaks = len(opt.two_theta)
             opt.goniometer_angles = np.array(angles)[np.newaxis, :].repeat(num_peaks, axis=0)
             goniometer_names = names
        elif opt.goniometer_axes is not None:
             print(f"Refining goniometer angles from HDF5 file with {goniometer_bound_deg} deg bounds.")
             pass
        else:
            print("WARNING: refine_goniometer requested but goniometer data not found. Skipping goniometer refinement.")
            refine_goniometer = False

    init_params = None
    if bootstrap_filename:
        # Call the reconstruction method
        init_params = opt.get_bootstrap_params(
            bootstrap_filename,
            refine_lattice=refine_lattice,
            lattice_bound_frac=lattice_bound_frac,
            refine_sample=refine_sample,
            sample_bound_meters=sample_bound_meters,
            refine_beam=refine_beam,
            beam_bound_deg=beam_bound_deg,
            refine_goniometer=refine_goniometer,
            goniometer_bound_deg=goniometer_bound_deg,
            refine_goniometer_axes=refine_goniometer_axes
        )

    num, hkl, lamda, U = opt.minimize_evosax(
        strategy_name=strategy_name,
        population_size=population_size,
        num_generations=gens,
        n_runs=n_runs,
        sigma_init=sigma_init,
        seed=seed,
        softness=softness,
        init_params=init_params,
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        refine_goniometer=refine_goniometer,
        refine_goniometer_axes=refine_goniometer_axes,
        goniometer_bound_deg=goniometer_bound_deg,
        goniometer_names=goniometer_names,
        refine_sample=refine_sample,
        sample_bound_meters=sample_bound_meters,
        refine_beam=refine_beam,
        beam_bound_deg=beam_bound_deg,
        loss_method=loss_method,
        d_min=d_min,
        d_max=d_max,
        hkl_search_range=hkl_search_range,
        search_window_size=search_window_size,
        batch_size=batch_size,
        window_batch_size=window_batch_size,
        chunk_size=chunk_size,
        num_iters=num_iters,
        top_k=top_k,
        B_sharpen=B_sharpen,
    )

    print(f"\nOptimization complete. Best solution indexed {num} peaks.")

    B = opt.reciprocal_lattice_B()
    refined_R = opt.R

    copy_keys = [
        "sample/space_group",
        "instrument/wavelength",
        "peaks/intensity",
        "peaks/sigma",
        "peaks/two_theta",
        "peaks/azimuthal",
        "peaks/radius",
        "peaks/xyz", 
        "goniometer/axes", 
        "goniometer/angles",
        "goniometer/names",
        "files",
        "file_offsets",
    ]

    copied_data = {}
    
    if input_data is not None:
        for key in copy_keys:
            if key in input_data:
                copied_data[key] = input_data[key]
    else:
        with h5py.File(hdf5_peaks_filename, 'r') as f:
            for key in copy_keys:
                if key in f:
                    copied_data[key] = np.array(f[key])

    print(f"Saving indexed peaks to {output_peaks_filename}...")
    with h5py.File(output_peaks_filename, "w") as f:
        for key, value in copied_data.items():
            f[key] = value

        f["goniometer/R"] = opt.R
        
        if opt.goniometer_offsets is not None:
            f["optimization/goniometer_offsets"] = opt.goniometer_offsets
            
        if opt.sample_offset is not None:
            f["sample/offset"] = opt.sample_offset
            
        if opt.ki_vec is not None:
            f["beam/ki_vec"] = opt.ki_vec

        f["sample/a"] = opt.a
        f["sample/b"] = opt.b
        f["sample/c"] = opt.c
        f["sample/alpha"] = opt.alpha
        f["sample/beta"] = opt.beta
        f["sample/gamma"] = opt.gamma

        B_mat = opt.reciprocal_lattice_B()
        f["sample/B"] = B_mat
        f["sample/U"] = U
        
        # hkl is (3, N) or (N, 3)? optimize output is (N, 3) usually or we construct lists
        # opt.minimize_evosax returns hkl (3, N).
        f["peaks/h"] = hkl[:,0]
        f["peaks/k"] = hkl[:,1]
        f["peaks/l"] = hkl[:,2]
        f["peaks/lambda"] = lamda
        f["optimization/best_params"] = opt.x
    print("Done.")

@app.command()
def finder(
    filename: str,
    instrument: str,
    output_filename: str = "output.h5",
    finder_algorithm: str = "peak_local_max",
    show_progress: bool = False,
    create_visualizations: bool = False,
    show_steps: bool = False,
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
    region_growth_minimum_sigma: typing.Optional[float] = None,
    region_growth_minimum_intensity: float = 4500.0,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 30,
    peak_minimum_signal_to_noise: float = 1.0,
    peak_pixel_outlier_threshold: float = 2.0,
    sparse_rbf_alpha: float = 0.1,      # Regularization (Higher = fewer, stronger peaks)
    sparse_rbf_gamma: float = 2.0,      # Besov space coefficient (shape prior)
    sparse_rbf_min_sigma: float = 0.5,  # Min spot size (pixels)
    sparse_rbf_max_sigma: float = 10.0, # Max spot size (pixels)
    sparse_rbf_max_peaks: int = 500,    # Max peaks per bank
    sparse_rbf_chunk_size: int = 4096,  # reduce if OOM
    sparse_rbf_tile_rows: int = 2,      # NEW: Number of row divisions for tiling
    sparse_rbf_tile_cols: int = 2,      # NEW: Number of col divisions for tiling
):

    print(f"Creating peaks from {filename} for instrument {instrument}")

    wavelength_kwargs = {}
    if wavelength_min:
        wavelength_kwargs["wavelength_min"] = wavelength_min
    if wavelength_max:
        wavelength_kwargs["wavelength_max"] = wavelength_max

    peaks = Peaks(filename, instrument, **wavelength_kwargs)

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
            "show_steps": show_steps,
            "show_scale": "log"
        })
    elif finder_algorithm == "sparse_rbf":
        peak_kwargs.update({
            "alpha": sparse_rbf_alpha,
            "gamma": sparse_rbf_gamma,
            "min_sigma": sparse_rbf_min_sigma,
            "max_sigma": sparse_rbf_max_sigma,
            "max_peaks": sparse_rbf_max_peaks,
            "chunk_size": sparse_rbf_chunk_size,
            "show_steps": show_steps,
            "show_scale": "linear",
            "tiles": (sparse_rbf_tile_rows, sparse_rbf_tile_cols),
        })
    else:
        raise ValueError("Invalid finder algorithm")

    integration_params = {
        "region_growth_distance_threshold": region_growth_distance_threshold,
        "region_growth_minimum_sigma": region_growth_minimum_sigma,
        "region_growth_minimum_intensity": region_growth_minimum_intensity,
        "region_growth_maximum_pixel_radius": region_growth_maximum_pixel_radius,
        "peak_center_box_size": peak_center_box_size,
        "peak_smoothing_window_size": peak_smoothing_window_size,
        "peak_minimum_pixels": peak_minimum_pixels,
        "peak_minimum_signal_to_noise": peak_minimum_signal_to_noise,
        "peak_pixel_outlier_threshold": peak_pixel_outlier_threshold
    }

    detector_peaks = peaks.get_detector_peaks(
        peak_kwargs,
        integration_params,
        visualize=create_visualizations,
        show_progress=show_progress,
        file_prefix=filename
    )

    peaks.write_hdf5(
        output_filename=output_filename,
        rotations=detector_peaks.R,
        two_theta=detector_peaks.two_theta,
        az_phi=detector_peaks.az_phi,
        wavelength_mins=detector_peaks.wavelength_mins,
        wavelength_maxes=detector_peaks.wavelength_maxes,
        intensity=detector_peaks.intensity,
        sigma=detector_peaks.sigma,
        radii=detector_peaks.radii,
        xyz=detector_peaks.xyz, # Store XYZ
        bank=detector_peaks.bank,
        gonio_axes=detector_peaks.gonio_axes,
        gonio_angles=detector_peaks.gonio_angles,
        gonio_names=detector_peaks.gonio_names,
        instrument_wavelength=[peaks.wavelength_min, peaks.wavelength_max],
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
    space_group: str
):
    with open(finder_h5_txt_list_filename) as f:
        finder_h5_files = f.read().splitlines()

    merging_algorithm = FinderConcatenateMerger(finder_h5_files)
    merging_algorithm.merge(output_pre_index_filename)

    with h5py.File(output_pre_index_filename, "r+") as f:
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/space_group"] = space_group
        f["instrument/wavelength"] = [wavelength_min, wavelength_max]


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
    space_group: str,
    wavelength_min: float = None,
    wavelength_max: float = None,
    goniometer_csv_filename: typing.Optional[str] = None,
    original_nexus_filename: typing.Optional[str] = None,
    instrument_name: typing.Optional[str] = None,
    strategy_name: str = typer.Option(
        "DE", 
        "--strategy", 
        help="Optimization strategy to use (e.g., 'DE' or 'PSO')."
    ),
    sigma_init: float = typer.Option(
        None,
        "--sigma-init",
        help="Parameter exploration range."
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
    softness: float = 0.1,
    refine_lattice: bool = typer.Option(
        False, 
        "--refine-lattice", 
        help="Refine unit cell parameters during optimization."
    ),
    lattice_bound_frac: float = typer.Option(
        0.05,
        "--lattice-bound-frac",
        help="Fractional bound for lattice parameter refinement."
    ),
    refine_goniometer: bool = typer.Option(
        False, 
        "--refine-goniometer", 
        help="Refine goniometer angles during optimization."
    ),
    refine_goniometer_axes: str = typer.Option(
        None,
        "--refine-goniometer-axes",
        help="Comma-separated list of goniometer axis names to refine."
    ),
    goniometer_bound_deg: float = typer.Option(
        5.0,
        "--goniometer-bound-deg",
        help="Bound for goniometer angle refinement in degrees."
    ),
    refine_sample: bool = typer.Option(
        False,
        "--refine-sample",
        help="Refine sample position offset."
    ),
    sample_bound_meters: float = typer.Option(
        2.0,
        "--sample-bound-meters",
        help="Bound for sample offset in meters."
    ),
    refine_beam: bool = typer.Option(
        False,
        "--refine-beam",
        help="Refine beam direction. Default (0,0,1)."
    ),
    beam_bound_deg: float = typer.Option(
        1.0,
        "--beam-bound-deg",
        help="Bound for beam direction in degrees."
    ),
    bootstrap_filename: typing.Optional[str] = typer.Option(None, "--bootstrap", help="Previous HDF5 solution to refine"),
    loss_method: str = typer.Option(
        'cosine',
        "--loss-method",
        help="Loss to use for optimization."
    ),
    d_min: float = typer.Option(None, "--d-min"),
    d_max: float = typer.Option(None, "--d-max"),
    hkl_search_range: int = typer.Option(20, "--hkl-search-range"),
    search_window_size: int = typer.Option(256, "--search-window-size"),
    batch_size: int = typer.Option(None, "--batch-size"),
    window_batch_size: int = typer.Option(32, "--window-batch-size"),
    chunk_size: int = typer.Option(256, "--chunk-size"),
    num_iters: int = typer.Option(20, "--num-iters"),
    top_k: int = typer.Option(32, "--top-k"),
    B_sharpen: float = typer.Option(None, "--b-sharpen", help="Wilson B-factor for peak sharpening (~50 for protein crystals)"),
) -> None:
    # Logic to resolve SG
    sg_to_use = "P 1"
    if space_group:
        sg_to_use = space_group

    print(f"Loading peaks from: {peaks_h5_filename}")
    input_data = {}
    with h5py.File(peaks_h5_filename, "r") as f:
        # Load auto-detected wavelength if not provided
        if wavelength_min is None or wavelength_max is None:
            if "instrument/wavelength" in f:
                wl = f["instrument/wavelength"][()]
                if wavelength_min is None: wavelength_min = float(wl[0])
                if wavelength_max is None: wavelength_max = float(wl[1])
                print(f"Auto-detected wavelength: {wavelength_min:.2f} - {wavelength_max:.2f} A")
            else:
                raise ValueError("Wavelength not provided and not found in input file.")

        # Read standard datasets
        keys_to_load = [
            "peaks/two_theta", "peaks/azimuthal", "peaks/intensity", "peaks/sigma", 
            "peaks/radius", "peaks/xyz",
            "goniometer/R", "goniometer/axes", "goniometer/angles", "goniometer/names",
            "files", "file_offsets",
        ]
        
        for k in keys_to_load:
            if k in f:
                input_data[k] = f[k][()]

    input_data["sample/a"] = a
    input_data["sample/b"] = b
    input_data["sample/c"] = c
    input_data["sample/alpha"] = alpha
    input_data["sample/beta"] = beta
    input_data["sample/gamma"] = gamma
    input_data["sample/space_group"] = sg_to_use
    input_data["instrument/wavelength"] = [wavelength_min, wavelength_max]

    index(
        input_data=input_data,
        output_peaks_filename=output_peaks_filename,
        strategy_name=strategy_name,
        population_size=population_size,
        gens=gens,
        sigma_init=sigma_init,
        n_runs=n_runs,
        seed=seed,
        softness=softness,
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        bootstrap_filename=bootstrap_filename,
        refine_goniometer=refine_goniometer,
        goniometer_bound_deg=goniometer_bound_deg,
        refine_sample=refine_sample,
        sample_bound_meters=sample_bound_meters,
        refine_beam=refine_beam,
        beam_bound_deg=beam_bound_deg,
        nexus_filename=original_nexus_filename,
        instrument_name=instrument_name,
        loss_method=loss_method,
        hkl_search_range=hkl_search_range,
        d_min=d_min,
        d_max=d_max,
        search_window_size=search_window_size,
        batch_size=batch_size,
        window_batch_size=window_batch_size,
        chunk_size=chunk_size,
        num_iters=num_iters,
        top_k=top_k,
        B_sharpen=B_sharpen,
    )


@app.command()
def indexer_using_file(
    hdf5_peaks_filename: str, 
    output_peaks_filename: str,
    original_nexus_filename: typing.Optional[str] = None,
    instrument_name: typing.Optional[str] = None,
    strategy_name: str = typer.Option("DE", "--strategy"),
    n_runs: int = typer.Option(1, "--n-runs"),
    population_size: int = typer.Option(1000, "--population-size"),
    gens: int = typer.Option(100, "--gens"),
    seed: int = typer.Option(0, "--seed"),
    refine_lattice: bool = typer.Option(False, "--refine-lattice"),
    lattice_bound_frac: float = typer.Option(0.05, "--lattice-bound-frac"),
    refine_goniometer: bool = typer.Option(False, "--refine-goniometer"),
    goniometer_bound_deg: float = typer.Option(5.0, "--goniometer-bound-deg"),
    softness: float = 0.1,
):
    index(
        hdf5_peaks_filename=hdf5_peaks_filename,
        output_peaks_filename=output_peaks_filename,
        strategy_name=strategy_name,
        population_size=population_size,
        gens=gens,
        n_runs=n_runs,
        seed=seed,
        softness=softness,
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        refine_goniometer=refine_goniometer,
        goniometer_bound_deg=goniometer_bound_deg,
        nexus_filename=original_nexus_filename,
        instrument_name=instrument_name
    )


@app.command()
def metrics(
    filename: str,
    d_min: float = typer.Option(None, "--d-min", help="Optional minimum d-spacing filter for metrics calculation.")
):
    try:
        with h5py.File(filename, "r") as f:
            if "peaks/xyz" not in f:
                print("METRICS: 9.99 9.99 9.99 9.99 9.99 9.99")
                return

            xyz_det = f["peaks/xyz"][()] 
            h = f["peaks/h"][()]
            k = f["peaks/k"][()]
            l = f["peaks/l"][()]
            lam = f["peaks/lambda"][()]

            ub_helper = FindUB()
            ub_helper.a = f["sample/a"][()]
            ub_helper.b = f["sample/b"][()]
            ub_helper.c = f["sample/c"][()]
            ub_helper.alpha = f["sample/alpha"][()]
            ub_helper.beta = f["sample/beta"][()]
            ub_helper.gamma = f["sample/gamma"][()]

            if "beam/ki_vec" in f:
                ki_vec = f["beam/ki_vec"][()]
            else:
                ki_vec = np.array([0.0, 0.0, 1.0])

            if "goniometer/R" in f:
                R_all = f["goniometer/R"][()]
            else:
                R_all = np.eye(3)[None, ...].repeat(len(h), axis=0)

            if "sample/U" in f:
                U = f["sample/U"][()]
            else:
                U = np.eye(3)

            if "sample/offset" in f:
                sample_offset = f["sample/offset"][()]
            else:
                sample_offset = np.zeros(3)

        mask = (h != 0) | (k != 0) | (l != 0)
        if np.sum(mask) == 0:
            print("METRICS: 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000")
            return

        h, k, l = h[mask], k[mask], l[mask]
        lam = lam[mask]
        xyz_det = xyz_det[mask]
        R_all = R_all[mask]

        B_mat = ub_helper.reciprocal_lattice_B()
        
        # --- NEW: Filter by d_min if provided ---
        if d_min is not None:
            # Calculate d = 1 / |B * hkl|
            # Note: |B*h| is 1/d in subhkl units (standard crystallographic definition)
            # hkl shape (N, 3)
            hkl_vecs = np.stack([h, k, l], axis=1)
            q_cryst = hkl_vecs @ B_mat.T
            q_mag = np.linalg.norm(q_cryst, axis=1)
            
            with np.errstate(divide='ignore'):
                d_vals = 1.0 / q_mag
            
            # Keep peaks where d >= d_min
            d_mask = d_vals >= d_min
            
            if np.sum(d_mask) == 0:
                print(f"METRICS: No peaks found with d >= {d_min} A.")
                return

            h, k, l = h[d_mask], k[d_mask], l[d_mask]
            lam = lam[d_mask]
            xyz_det = xyz_det[d_mask]
            R_all = R_all[d_mask]
            print(f"METRICS: Filtered to {len(h)} peaks with d >= {d_min} A.")

        UB = U @ B_mat
        
        if R_all.ndim == 3:
            RUB = np.matmul(R_all, UB) 
        else:
            RUB = R_all @ UB 

        d_err, ang_err = calculate_angular_error(
            xyz_det, h, k, l, lam, RUB, sample_offset, ki_vec
        )

        print(f"METRICS: {np.median(d_err):.5f} {np.mean(d_err):.5f} {np.max(d_err):.5f} "
              f"{np.median(ang_err):.5f} {np.mean(ang_err):.5f} {np.max(ang_err):.5f}")

    except Exception as e:
        # print(e)
        print("METRICS: 9.99 9.99 9.99 9.99 9.99 9.99")

@app.command()
def peak_predictor(
    filename: str,  # Now expects the MERGED HDF5 (scan_master.h5)
    instrument: str,
    indexed_hdf5_filename: str,
    integration_peaks_filename: str,
    d_min: float = 1.0,
    create_visualizations: bool = False,
    space_group: str = None,
    wavel_min: float = None,
    wavel_max: float = None,
):
    """
    Predicts peaks for a full dataset using the optimized geometry from indexer.
    Input `filename` should be the merged HDF5 used for indexing.
    """
    # 1. Load Optimized Parameters
    with h5py.File(indexed_hdf5_filename, 'r') as f_idx:
        a = float(f_idx["sample/a"][()])
        b = float(f_idx["sample/b"][()])
        c = float(f_idx["sample/c"][()])
        alpha = float(f_idx["sample/alpha"][()])
        beta = float(f_idx["sample/beta"][()])
        gamma = float(f_idx["sample/gamma"][()])
        
        if space_group is None:
            space_group = f_idx["sample/space_group"][()].decode('utf-8')
            
        wavelength = f_idx["instrument/wavelength"][()]
        if wavel_min: wavelength[0] = wavel_min
        if wavel_max: wavelength[1] = wavel_max
        
        U = f_idx["sample/U"][()]
        B = f_idx["sample/B"][()]
        
        # Load Geometry Stacks
        # R is likely (N_images, 3, 3)
        all_R = f_idx["goniometer/R"][()]
        
        if "sample/offset" in f_idx:
            sample_offset = f_idx["sample/offset"][()]
        else:
            sample_offset = np.zeros(3)
            
        if "beam/ki_vec" in f_idx:
            ki_vec = f_idx["beam/ki_vec"][()]
        else:
            ki_vec = np.array([0., 0., 1.])

    # 2. Initialize Data Handler
    # Peaks class auto-detects merged HDF5
    peaks = Peaks(filename, instrument, wavelength_min=wavelength[0], wavelength_max=wavelength[1])
    
    print(f"Predicting peaks for {len(peaks.ims)} images using solution from {indexed_hdf5_filename}")

    # 3. Prediction Loop
    # We must predict per-image because R (and thus RUB) changes per image in a scan.
    results_map = {} # Store results by img_key
    
    # Iterate sorted keys (0, 1, 2... N)
    img_keys = sorted(peaks.ims.keys())
    
    # Handle R shape mismatch (singleton vs stack)
    if all_R.ndim == 2: all_R = all_R[np.newaxis, :, :] # (1,3,3)
    if len(all_R) == 1 and len(img_keys) > 1:
        # constant R
        get_R = lambda i: all_R[0]
    else:
        # stack R
        get_R = lambda i: all_R[i] if i < len(all_R) else all_R[-1]

    for img_key in img_keys:
        # Get Physics for this specific frame
        R_current = get_R(img_key)
        RUB = R_current @ U @ B
        
        # Determine physical bank for this image (needed for detector geometry)
        if hasattr(peaks, 'bank_mapping'):
            phys_bank = peaks.bank_mapping.get(img_key, img_key)
        else:
            phys_bank = img_key
            
        det = peaks.get_detector(phys_bank)
        
        # Predict
        # We manually call the lower-level utility to avoid overhead/complexity
        from subhkl.utils import predict_reflections_on_panel
        h_all, k_all, l_all = peaks.reflections(a, b, c, alpha, beta, gamma, space_group, d_min)
        
        row, col, h_f, k_f, l_f, wl_f = predict_reflections_on_panel(
            detector=det,
            h=h_all, k=k_all, l=l_all,
            RUB=RUB,
            wavelength_min=wavelength[0],
            wavelength_max=wavelength[1],
            sample_offset=sample_offset,
            ki_vec=ki_vec
        )
        
        if len(row) > 0:
            results_map[img_key] = (row, col, h_f, k_f, l_f, wl_f)
            
        if create_visualizations and len(row) > 0:
            import matplotlib.pyplot as plt
            plt.imshow(peaks.ims[img_key] + 1, cmap="binary", norm="log", origin='lower')
            plt.scatter(col, row, edgecolors='r', facecolors='none', s=20)
            plt.title(f"Img {img_key} (Bank {phys_bank})")
            plt.savefig(f"{filename}_pred_{img_key}.png")
            plt.close()

    # 4. Save Predictions
    print(f"Saving predictions to {integration_peaks_filename}")
    with h5py.File(integration_peaks_filename, "w") as f:
        # Save Global Physics
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/space_group"] = space_group
        f["sample/U"] = U
        f["sample/B"] = B
        f["instrument/wavelength"] = wavelength
        f["goniometer/R"] = all_R # Save full stack
        f["sample/offset"] = sample_offset
        f["beam/ki_vec"] = ki_vec

        # Save Peaks
        # Structure: banks/{img_key}/...
        # Using img_key as the identifier ensures 1:1 mapping with the merged input file
        for img_key, (i, j, h, k, l, wl) in results_map.items():
            grp = f.create_group(f"banks/{img_key}")
            grp.create_dataset("i", data=i)
            grp.create_dataset("j", data=j)
            grp.create_dataset("h", data=h)
            grp.create_dataset("k", data=k)
            grp.create_dataset("l", data=l)
            grp.create_dataset("wavelength", data=wl)


@app.command()
def integrator(
    filename: str, # Merged HDF5
    instrument: str,
    integration_peaks_filename: str,
    output_filename: str,
    integration_method: str = "free_fit",
    integration_mask_file: typing.Optional[str] = None,
    integration_mask_rel_erosion_radius: typing.Optional[float] = 0.05,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_intensity: float = 50.0, # Adjusted default
    region_growth_minimum_sigma: typing.Optional[float] = None,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 10,
    peak_minimum_signal_to_noise: float = 1.0,
    peak_pixel_outlier_threshold: float = 2.0,
    create_visualizations: bool = False,
    show_progress: bool = False,
    found_peaks_file: str = None,
):
    """
    Integrates predicted peaks using the merged image stack.
    """
    # 1. Load Predictions
    peak_dict = {}
    with h5py.File(integration_peaks_filename, 'r') as f:
        # Load Physics for context (passed to integrate if needed, mainly R)
        if "sample/U" in f: U = f["sample/U"][()]
        if "sample/B" in f: B = f["sample/B"][()]
        if "goniometer/R" in f: all_R = f["goniometer/R"][()]
        else: all_R = np.eye(3)[np.newaxis, :, :]
        
        if "sample/offset" in f: sample_offset = f["sample/offset"][()]
        else: sample_offset = np.zeros(3)
        if "beam/ki_vec" in f: ki_vec = f["beam/ki_vec"][()]
        else: ki_vec = np.array([0., 0., 1.])

        # Load Per-Image Peaks
        # Structure is banks/{img_key}
        for key in f["banks"].keys():
            # key is string "0", "1", etc.
            img_idx = int(key)
            grp = f[f"banks/{key}"]
            peak_dict[img_idx] = [
                grp["i"][()], grp["j"][()],
                grp["h"][()], grp["k"][()], grp["l"][()],
                grp["wavelength"][()]
            ]

    # 2. Initialize Data
    peaks = Peaks(filename, instrument)
    
    # 3. Setup Parameters
    integration_params = {
        "region_growth_distance_threshold": region_growth_distance_threshold,
        "region_growth_minimum_intensity": region_growth_minimum_intensity,
        "region_growth_minimum_sigma": region_growth_minimum_sigma,
        "region_growth_maximum_pixel_radius": region_growth_maximum_pixel_radius,
        "peak_center_box_size": peak_center_box_size,
        "peak_smoothing_window_size": peak_smoothing_window_size,
        "peak_minimum_pixels": peak_minimum_pixels,
        "peak_minimum_signal_to_noise": peak_minimum_signal_to_noise,
        "peak_pixel_outlier_threshold": peak_pixel_outlier_threshold,
        "integration_mask_file": integration_mask_file,
        "integration_mask_rel_erosion_radius": integration_mask_rel_erosion_radius,
    }

    # 4. Run Integration
    # Note: RUB passed here is symbolic; integrate method will calculate 
    # per-image geometry if we update it, or we pass specific R. 
    # The current 'Peaks.integrate' iterates the dictionary keys.
    # Since we loaded 'merged.h5', peaks.ims keys are 0..N.
    # peak_dict keys are 0..N.
    # They match!
    
    # Construct a composite RUB for the *first* frame just to satisfy the signature,
    # or rely on metric calculation internals.
    # ideally 'integrate' should be aware of variable R, but for now we pass identity 
    # and rely on the fact that prediction is already done.
    RUB_nominal = np.eye(3) 

    result = peaks.integrate(
        peak_dict,
        integration_params,
        RUB=RUB_nominal, 
        sample_offset=sample_offset,
        ki_vec=ki_vec,
        create_visualizations=create_visualizations,
        show_progress=show_progress,
        integration_method=integration_method,
        file_prefix=filename,
        found_peaks_file=found_peaks_file,
    )

    # 5. Save Output
    print(f"Saving integrated peaks to {output_filename}")
    
    # Copy metadata from prediction file
    copy_keys = [
        "sample/a", "sample/b", "sample/c", "sample/alpha", "sample/beta", "sample/gamma",
        "sample/space_group", "sample/U", "sample/B", "sample/offset",
        "beam/ki_vec", "instrument/wavelength", "goniometer/R",
    ]

    with h5py.File(output_filename, "w") as f:
        f["peaks/h"] = result.h
        f["peaks/k"] = result.k
        f["peaks/l"] = result.l
        f["peaks/lambda"] = result.wavelength
        f["peaks/intensity"] = result.intensity
        f["peaks/sigma"] = result.sigma
        f["peaks/two_theta"] = result.tt
        f["peaks/azimuthal"] = result.az
        f["peaks/bank"] = result.bank
        f["peaks/xyz"] = result.xyz

        with h5py.File(integration_peaks_filename, 'r') as f_in:
            for key in copy_keys:
                if key in f_in:
                    f_in.copy(f_in[key], f, key)

@app.command()
def mtz_exporter(
    indexed_h5_filename: str,
    output_mtz_filename: str,
    space_group: str,
):
    algorithm = MTZExporter(indexed_h5_filename, space_group)
    algorithm.write_mtz(output_mtz_filename)


@app.command()
def reduce(
    nexus_filename: str,
    output_filename: str,
    instrument: str,
    wavelength_min: float = typer.Option(None, help="Override min wavelength"),
    wavelength_max: float = typer.Option(None, help="Override max wavelength"),
):
    """
    Reduces a single Nexus event file to a dense image stack HDF5 file.
    Output shape: (N_banks, Height, Width).
    """
    print(f"Reducing {nexus_filename} -> {output_filename}")

    # 1. Load Data using existing Peaks class logic
    # This handles loading the event data into 2D histograms (self.ims)
    # and parsing the goniometer/wavelength from the Nexus file.
    peaks_handler = Peaks(
        nexus_filename,
        instrument,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max
    )

    if not peaks_handler.ims:
        print("Warning: No images found in file.")
        return

    # 2. Stack Data for Batch Processing
    # Ensure consistent ordering of banks
    sorted_banks = sorted(peaks_handler.ims.keys())

    # Stack images: (N_banks, H, W)
    # Note: Assumes all banks have the same shape, which is standard for one instrument.
    image_stack = np.stack([peaks_handler.ims[b] for b in sorted_banks])

    bank_ids = np.array(sorted_banks, dtype=np.int32)
    n_images = len(sorted_banks)

    # 3. Prepare Metadata
    # Repeat angles for each bank so they stay aligned after merging
    if peaks_handler.goniometer_angles_raw is not None:
        # shape (1, 3) -> (N_banks, 3)
        angles_repeated = np.tile(peaks_handler.goniometer_angles_raw, (n_images, 1))
    else:
        angles_repeated = np.zeros((n_images, 3)) # Fallback

    if peaks_handler.goniometer_axes_raw is not None:
        axes = np.array(peaks_handler.goniometer_axes_raw)
    else:
        axes = np.array([0.0, 1.0, 0.0]) # Fallback

    # 4. Save to HDF5
    # We use LZF compression for speed as these are intermediate training files
    with h5py.File(output_filename, "w") as f:
        # Data
        f.create_dataset("images", data=image_stack, compression="lzf")
        f.create_dataset("bank_ids", data=bank_ids)

        # Metadata (Repeated per image)
        f.create_dataset("goniometer/angles", data=angles_repeated)

        # Metadata (Constant)
        f.create_dataset("goniometer/axes", data=axes)

        if peaks_handler.goniometer_names_raw:
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset("goniometer/names", data=peaks_handler.goniometer_names_raw, dtype=dt)

        # Save Wavelength (Min/Max)
        # Using format compatible with downstream indexer
        wl = [peaks_handler.wavelength_min, peaks_handler.wavelength_max]
        f.create_dataset("instrument/wavelength", data=wl)

        # Save Instrument Name
        f.attrs["instrument"] = instrument

    print(f"Saved {n_images} banks to {output_filename}")

@app.command()
def merge_images(
    input_pattern: str = typer.Argument(..., help="Glob pattern for reduced .h5 files (e.g. 'reduced/*.h5')"),
    output_filename: str = typer.Argument(..., help="Output master .h5 file"),
):
    """
    Merges multiple reduced HDF5 image files into a single master dataset.
    """
    # 1. Resolve file list
    # Sort is crucial to ensure scan order (time/angle) is preserved
    h5_files = sorted(glob.glob(input_pattern))
    
    if not h5_files:
        print(f"No files found matching: {input_pattern}")
        raise typer.Exit(code=1)
        
    print(f"Found {len(h5_files)} files. Merging...")

    # 2. Merge
    # Uses the new ImageStackMerger class in export.py
    merger = ImageStackMerger(h5_files)
    merger.merge(output_filename)
    
    print(f"Successfully created {output_filename}")
if __name__ == "__main__":
    app()
