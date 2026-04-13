import glob

import h5py
import numpy as np
import typer

from subhkl.instrument.goniometer import (
    calc_goniometer_rotation_matrix,
    get_rotation_data_from_nexus,
)
from subhkl.io.export import FinderConcatenateMerger, ImageStackMerger, MTZExporter
from subhkl.integration import Peaks
from subhkl.instrument.metrics import compute_metrics
from subhkl.optimization import FindUB

app = typer.Typer()


def index(
    hdf5_peaks_filename: str | None = None,
    output_peaks_filename: str | None = None,
    strategy_name: str = "DE",
    population_size: int = 1000,
    gens: int = 100,
    n_runs: int = 1,
    seed: int = 0,
    tolerance_deg: float = 0.1,
    sigma_init: float | None = None,
    refine_lattice: bool = False,
    lattice_bound_frac: float = 0.05,
    bootstrap_filename: str | None = None,
    refine_goniometer: bool = False,
    refine_goniometer_axes: list | None = None,
    goniometer_bound_deg: float = 5.0,
    refine_sample: bool = False,
    sample_bound_meters: float = 0.002,
    refine_beam: bool = False,
    beam_bound_deg: float = 1.0,
    refine_detector: bool = False,
    refine_detector_banks: list | None = None,
    detector_trans_bound_meters: float = 0.005,
    detector_rot_bound_deg: float = 1.0,
    nexus_filename: str | None = None,
    instrument_name: str | None = None,
    loss_method: str = "cosine",
    hkl_search_range: int = 20,
    d_min: float | None = None,
    d_max: float | None = None,
    search_window_size: int = 512,
    batch_size: int | None = None,
    window_batch_size: int = 32,
    chunk_size: int = 256,
    num_iters: int = 20,
    top_k: int = 32,
    B_sharpen: float | None = None,
    input_data: dict | None = None,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
):
    """
    Index the given peak file and save it using the evosax optimizer.
    """

    if input_data is not None:
        opt = FindUB(data=input_data)
    else:
        opt = FindUB(filename=hdf5_peaks_filename)

    if wavelength_min is not None and wavelength_max is not None:
        opt.wavelength = [wavelength_min, wavelength_max]

    print(f"Starting evosax optimization with strategy: {strategy_name}")
    print(f"Running {n_runs} run(s)...")
    print(f"Settings per run: Population Size={population_size}, Generations={gens}")
    if refine_lattice:
        print(f"Refining lattice parameters with {lattice_bound_frac * 100}% bounds.")
    if refine_sample:
        print(f"Refining sample offset with {1000 * sample_bound_meters} mm bounds.")
    if refine_beam:
        print(f"Refining beam tilt with {beam_bound_deg}° bounds.")

    goniometer_names = None
    if refine_goniometer:
        if nexus_filename and instrument_name:
            print(f"Refining goniometer angles from Nexus with {goniometer_bound_deg} deg bounds.")
            axes, angles, names = get_rotation_data_from_nexus(nexus_filename, instrument_name)
            opt.goniometer_axes = np.array(axes)

            if opt.run_indices is not None:
                num_runs = np.max(opt.run_indices) + 1
                opt.goniometer_angles = np.array(angles)[:, np.newaxis].repeat(num_runs, axis=1)
            else:
                num_peaks = len(opt.two_theta)
                opt.goniometer_angles = np.array(angles)[:, np.newaxis].repeat(num_peaks, axis=1)
            goniometer_names = names
        elif opt.goniometer_axes is not None:
            print(f"Refining goniometer angles from HDF5 file with {goniometer_bound_deg} deg bounds.")
        else:
            print("WARNING: refine_goniometer requested but goniometer data not found. Skipping.")
            refine_goniometer = False

    # --- DETECTOR METROLOGY SETUP ---
    detector_params = None
    peak_pixel_coords = None

    if refine_detector:
        if not nexus_filename or not instrument_name:
            print("WARNING: refine_detector requires nexus_filename and instrument_name to build geometry. Disabling.")
            refine_detector = False
        else:
            print(f"Joint Metrology Refinement active. Bounds: trans={detector_trans_bound_meters*1000}mm, rot={detector_rot_bound_deg}°")
            
            # Use Peaks to instantiate beamlines geometry mapping
            try:
                peaks_obj = Peaks(nexus_filename, instrument_name)
                bank_array = opt.run_indices if "bank" not in input_data else input_data.get("bank", opt.run_indices)
                
                # Default to all uniquely hit banks if no explicit list provided
                unique_hit_banks = np.unique(bank_array).astype(int)
                target_banks = refine_detector_banks if refine_detector_banks else unique_hit_banks

                centers, uhats, vhats, m, n, pw, ph = [], [], [], [], [], [], []
                bank_to_idx = {}
                
                for idx, b_id in enumerate(target_banks):
                    try:
                        det = peaks_obj.get_detector(b_id)
                        centers.append(det.center)
                        uhats.append(det.uhat)
                        vhats.append(det.vhat)
                        m.append(det.m)
                        n.append(det.n)
                        pw.append(det.width / det.m)
                        ph.append(det.height / det.n)
                        bank_to_idx[b_id] = idx
                    except Exception as e:
                        print(f"WARNING: Could not load geometry for bank {b_id}: {e}")

                detector_params = {
                    'centers': centers, 'uhats': uhats, 'vhats': vhats,
                    'm': m, 'n': n, 'pw': pw, 'ph': ph
                }

                # Construct the peak-to-panel mapping for the objective
                if "peaks/xyz" in input_data:
                    xyz = input_data["peaks/xyz"]
                    # We have absolute xyz, we need the local rows/cols
                    # Calculate this outside JAX once to pass statically
                    rows, cols, bank_indices = [], [], []
                    for i_p in range(len(xyz)):
                        b_id = int(bank_array[i_p])
                        if b_id in bank_to_idx:
                            det = peaks_obj.get_detector(b_id)
                            r, c = det.lab_to_pixel(xyz[i_p, 0], xyz[i_p, 1], xyz[i_p, 2], clip=False)
                            rows.append(r)
                            cols.append(c)
                            bank_indices.append(bank_to_idx[b_id])
                        else:
                            # If peak belongs to a bank we aren't refining, point it to 0 but it won't matter
                            rows.append(0.0)
                            cols.append(0.0)
                            bank_indices.append(0)

                    peak_pixel_coords = {
                        'rows': rows,
                        'cols': cols,
                        'bank_indices': bank_indices
                    }
                else:
                    print("WARNING: No peaks/xyz found in file. Disabling refine_detector.")
                    refine_detector = False

            except Exception as e:
                print(f"WARNING: Failed to initialize joint detector refinement geometry: {e}")
                refine_detector = False

    init_params = None
    if bootstrap_filename:
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
            refine_goniometer_axes=refine_goniometer_axes,
        )

    num, hkl, lamda, U = opt.minimize(
        strategy_name=strategy_name,
        population_size=population_size,
        num_generations=gens,
        n_runs=n_runs,
        sigma_init=sigma_init,
        seed=seed,
        tolerance_deg=tolerance_deg,
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
        refine_detector=refine_detector,
        detector_params=detector_params,
        peak_pixel_coords=peak_pixel_coords,
        detector_trans_bound_meters=detector_trans_bound_meters,
        detector_rot_bound_deg=detector_rot_bound_deg,
    )

    print(f"\nOptimization complete. Best solution indexed {num} peaks.")

    opt.reciprocal_lattice_B()

    copy_keys = [
        "sample/space_group", "instrument/wavelength", "peaks/intensity",
        "peaks/sigma", "peaks/two_theta", "peaks/azimuthal", "peaks/radius",
        "peaks/xyz", "goniometer/R", "goniometer/axes", "goniometer/angles",
        "goniometer/names", "files", "file_offsets", "peaks/run_index",
        "bank", "sample/offset", "beam/ki_vec",
    ]

    copied_data = {}

    if input_data is not None:
        for key in copy_keys:
            if key in input_data:
                copied_data[key] = input_data[key]
    else:
        with h5py.File(hdf5_peaks_filename, "r") as f:
            for key in copy_keys:
                if key in f:
                    copied_data[key] = np.array(f[key])

    print(f"Saving indexed peaks to {output_peaks_filename}...")
    with h5py.File(output_peaks_filename, "w") as f:
        if instrument_name:
            f.attrs["instrument"] = instrument_name
        elif "instrument" in input_data:
            f.attrs["instrument"] = input_data["instrument"]
        for key, value in copied_data.items():
            f[key] = value

        def safe_write(grp, name, data):
            if name in grp:
                del grp[name]
            grp[name] = data

        safe_write(f, "goniometer/R", opt.R)

        if opt.goniometer_offsets is not None:
            safe_write(f, "optimization/goniometer_offsets", opt.goniometer_offsets)

        if opt.sample_offset is not None:
            safe_write(f, "sample/offset", opt.sample_offset)

        if opt.ki_vec is not None:
            safe_write(f, "beam/ki_vec", opt.ki_vec)

        safe_write(f, "sample/a", opt.a)
        safe_write(f, "sample/b", opt.b)
        safe_write(f, "sample/c", opt.c)
        safe_write(f, "sample/alpha", opt.alpha)
        safe_write(f, "sample/beta", opt.beta)
        safe_write(f, "sample/gamma", opt.gamma)

        B_mat = opt.reciprocal_lattice_B()
        safe_write(f, "sample/B", B_mat)
        f["sample/U"] = U

        if opt.run_indices is not None:
            safe_write(f, "peaks/run_index", opt.run_indices)

        f["peaks/h"] = hkl[:, 0]
        f["peaks/k"] = hkl[:, 1]
        f["peaks/l"] = hkl[:, 2]
        f["peaks/lambda"] = lamda
        f["optimization/best_params"] = opt.x
        
        # --- Save Refined Detector Metadata ---
        if refine_detector and hasattr(opt, 'calibrated_centers'):
            for b_idx, b_id in enumerate(target_banks):
                grp_name = f"detector_calibration/bank_{b_id}"
                f.create_group(grp_name)
                f[f"{grp_name}/center"] = opt.calibrated_centers[b_idx]
                f[f"{grp_name}/uhat"] = opt.calibrated_uhats[b_idx]
                f[f"{grp_name}/vhat"] = opt.calibrated_vhats[b_idx]

    print("Done.")


@app.command()
def finder(
    filename: str,
    instrument: str,
    output_filename: str = "output.h5",
    finder_algorithm: str = "peak_local_max",
    show_progress: bool = True,
    create_visualizations: bool = False,
    show_steps: bool = False,
    peak_local_max_min_pixel_distance: int = -1,
    peak_local_max_min_relative_intensity: float = -1,
    peak_local_max_normalization: bool = False,
    thresholding_noise_cutoff_quantile: float = 0.8,
    thresholding_min_peak_dist_pixels: float = 8.0,
    thresholding_mask_file: str | None = None,
    thresholding_mask_rel_erosion_radius: float = 0.05,
    thresholding_blur_kernel_sigma: int = 5,
    thresholding_open_kernel_size_pixels: int = 3,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_sigma: float | None = None,
    region_growth_minimum_intensity: float = 4500.0,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 30,
    peak_minimum_signal_to_noise: float = 1.0,
    peak_pixel_outlier_threshold: float = 2.0,
    sparse_rbf_alpha: float = 0.1,  
    sparse_rbf_gamma: float = 2.0,  
    sparse_rbf_min_sigma: float = 0.5,  
    sparse_rbf_max_sigma: float = 10.0, 
    sparse_rbf_max_peaks: int = 500,  
    sparse_rbf_chunk_size: int = 4096,  
    sparse_rbf_tile_rows: int = 2,  
    sparse_rbf_tile_cols: int = 2,  
    max_workers: int = 16,
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
        peak_kwargs["normalize"] = peak_local_max_normalization
    elif finder_algorithm == "thresholding":
        peak_kwargs.update(
            {
                "noise_cutoff_quantile": thresholding_noise_cutoff_quantile,
                "min_peak_dist_pixels": thresholding_min_peak_dist_pixels,
                "mask_file": thresholding_mask_file,
                "mask_rel_erosion_radius": thresholding_mask_rel_erosion_radius,
                "blur_kernel_sigma": thresholding_blur_kernel_sigma,
                "open_kernel_size_pixels": thresholding_open_kernel_size_pixels,
                "show_steps": show_steps,
                "show_scale": "log",
            }
        )
    elif finder_algorithm == "sparse_rbf":
        peak_kwargs.update(
            {
                "alpha": sparse_rbf_alpha,
                "gamma": sparse_rbf_gamma,
                "min_sigma": sparse_rbf_min_sigma,
                "max_sigma": sparse_rbf_max_sigma,
                "max_peaks": sparse_rbf_max_peaks,
                "chunk_size": sparse_rbf_chunk_size,
                "show_steps": show_steps,
                "show_scale": "linear",
                "tiles": (sparse_rbf_tile_rows, sparse_rbf_tile_cols),
            }
        )
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
        "peak_pixel_outlier_threshold": peak_pixel_outlier_threshold,
    }

    detector_peaks = peaks.get_detector_peaks(
        peak_kwargs,
        integration_params,
        visualize=create_visualizations,
        show_progress=show_progress,
        file_prefix=filename,
        max_workers=max_workers,
    )

    peaks.write_hdf5(
        output_filename=output_filename,
        detector_peaks=detector_peaks,
        instrument_wavelength=[peaks.wavelength.min, peaks.wavelength.max],
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
    space_group: str,
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
    peaks_h5_filename: str, output_peaks_filename: str, a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float, space_group: str,
    wavelength_min: float | None = None, wavelength_max: float | None = None,
    goniometer_csv_filename: str | None = None, original_nexus_filename: str | None = None,
    instrument_name: str | None = None, strategy_name: str = typer.Option("DE", "--strategy"),
    sigma_init: float = typer.Option(None, "--sigma-init"), n_runs: int = typer.Option(1, "--n-runs", "-n"),
    population_size: int = typer.Option(1000, "--population-size", "--popsize"),
    gens: int = typer.Option(100, "--gens"), seed: int = typer.Option(0, "--seed"),
    refine_lattice: bool = typer.Option(False, "--refine-lattice"),
    lattice_bound_frac: float = typer.Option(0.05, "--lattice-bound-frac"),
    refine_goniometer: bool = typer.Option(False, "--refine-goniometer"),
    refine_goniometer_axes: str = typer.Option(None, "--refine-goniometer-axes"),
    goniometer_bound_deg: float = typer.Option(5.0, "--goniometer-bound-deg"),
    refine_sample: bool = typer.Option(False, "--refine-sample"),
    sample_bound_meters: float = typer.Option(0.005, "--sample-bound-meters"),
    refine_beam: bool = typer.Option(False, "--refine-beam"),
    beam_bound_deg: float = typer.Option(1.0, "--beam-bound-deg"),
    refine_detector: bool = typer.Option(False, "--refine-detector"),
    refine_detector_banks: str = typer.Option(None, "--refine-detector-banks"),
    detector_trans_bound_meters: float = typer.Option(0.005, "--detector-trans-bound-meters"),
    detector_rot_bound_deg: float = typer.Option(1.0, "--detector-rot-bound-deg"),
    bootstrap_filename: str | None = typer.Option(None, "--bootstrap"),
    batch_size: int = typer.Option(None, "--batch-size"),
) -> None:
    sg_to_use = "P 1"
    if space_group:
        from subhkl.core.spacegroup import get_space_group_object
        try:
            get_space_group_object(space_group)
            sg_to_use = space_group
        except ValueError as e:
            print(f"ERROR: Invalid space group '{space_group}': {e}")
            raise typer.Exit(code=1)

    print(f"Loading peaks from: {peaks_h5_filename}")
    input_data = {}

    def _val(x): return x.default if hasattr(x, "default") else x

    with h5py.File(peaks_h5_filename, "r") as f:
        w_min_val, w_max_val = _val(wavelength_min), _val(wavelength_max)
        if w_min_val is None or w_max_val is None:
            if "instrument/wavelength" in f:
                wl = f["instrument/wavelength"][()]
                if w_min_val is None: wavelength_min = float(wl[0])
                if w_max_val is None: wavelength_max = float(wl[1])
            else:
                raise ValueError("Wavelength not provided and not found in input file.")

        keys_to_load = [
            "peaks/two_theta", "peaks/azimuthal", "peaks/intensity", "peaks/sigma",
            "peaks/radius", "peaks/xyz", "goniometer/R", "goniometer/axes",
            "goniometer/angles", "goniometer/names", "files", "file_offsets",
            "peaks/run_index", "peaks/image_index", "bank", "bank_ids", "sample/offset", "beam/ki_vec",
        ]
        for k in keys_to_load:
            if k in f: input_data[k] = f[k][()]

    if "peaks/image_index" in input_data:
        input_data["peaks/run_index"] = input_data["peaks/image_index"]

    input_data["sample/a"], input_data["sample/b"], input_data["sample/c"] = a, b, c
    input_data["sample/alpha"], input_data["sample/beta"], input_data["sample/gamma"] = alpha, beta, gamma
    input_data["sample/space_group"] = sg_to_use
    input_data["instrument/wavelength"] = [float(_val(wavelength_min)), float(_val(wavelength_max))]

    gonio_axes_list = [x.strip() for x in _val(refine_goniometer_axes).split(",")] if _val(refine_goniometer_axes) else None
    
    det_banks_list = None
    if _val(refine_detector_banks):
        det_banks_list = [int(x.strip()) for x in _val(refine_detector_banks).split(",")]

    index(
        input_data=input_data, output_peaks_filename=output_peaks_filename,
        strategy_name=_val(strategy_name), population_size=_val(population_size),
        gens=_val(gens), sigma_init=_val(sigma_init), n_runs=_val(n_runs), seed=_val(seed),
        refine_lattice=_val(refine_lattice), lattice_bound_frac=_val(lattice_bound_frac), 
        bootstrap_filename=_val(bootstrap_filename), refine_goniometer=_val(refine_goniometer), 
        refine_goniometer_axes=gonio_axes_list, goniometer_bound_deg=_val(goniometer_bound_deg), 
        refine_sample=_val(refine_sample), sample_bound_meters=_val(sample_bound_meters), 
        refine_beam=_val(refine_beam), beam_bound_deg=_val(beam_bound_deg), 
        refine_detector=_val(refine_detector), refine_detector_banks=det_banks_list,
        detector_trans_bound_meters=_val(detector_trans_bound_meters), detector_rot_bound_deg=_val(detector_rot_bound_deg),
        nexus_filename=original_nexus_filename, instrument_name=instrument_name, 
        batch_size=_val(batch_size),
        wavelength_min=input_data["instrument/wavelength"][0], wavelength_max=input_data["instrument/wavelength"][1],
    )

@app.command()
def indexer_using_file(
    hdf5_peaks_filename: str,
    output_peaks_filename: str,
    original_nexus_filename: str | None = None,
    instrument_name: str | None = None,
    strategy_name: str = typer.Option("DE", "--strategy"),
    n_runs: int = typer.Option(1, "--n-runs"),
    population_size: int = typer.Option(1000, "--population-size"),
    gens: int = typer.Option(100, "--gens"),
    seed: int = typer.Option(0, "--seed"),
    refine_lattice: bool = typer.Option(False, "--refine-lattice"),
    lattice_bound_frac: float = typer.Option(0.05, "--lattice-bound-frac"),
    refine_goniometer: bool = typer.Option(False, "--refine-goniometer"),
    goniometer_bound_deg: float = typer.Option(5.0, "--goniometer-bound-deg"),
    refine_detector: bool = typer.Option(False, "--refine-detector"),
    tolerance_deg: float = 0.1,
):
    index(
        hdf5_peaks_filename=hdf5_peaks_filename,
        output_peaks_filename=output_peaks_filename,
        strategy_name=strategy_name,
        population_size=population_size,
        gens=gens,
        n_runs=n_runs,
        seed=seed,
        tolerance_deg=tolerance_deg,
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        refine_goniometer=refine_goniometer,
        goniometer_bound_deg=goniometer_bound_deg,
        refine_detector=refine_detector,
        nexus_filename=original_nexus_filename,
        instrument_name=instrument_name,
    )


@app.command()
def metrics(
    filename: str,
    found_peaks_file: str | None = typer.Option(
        None,
        "--found-peaks",
        help="Optional file with found peaks to compare against (e.g. finder.h5).",
    ),
    instrument: str | None = typer.Option(
        None,
        "--instrument",
        help="Instrument name (required if matching peaks).",
    ),
    d_min: float = typer.Option(
        None,
        "--d-min",
        help="Optional minimum d-spacing filter for metrics calculation.",
    ),
    per_run: bool = typer.Option(
        False,
        "--per-run",
        help="Calculate and display metrics for each run/image.",
    ),
):
    """
    CLI command to compute and display indexing quality metrics.

    Calls compute_metrics from subhkl.instrument.metrics and formats output for display.
    """
    if hasattr(found_peaks_file, "default"):
        found_peaks_file = found_peaks_file.default
    if hasattr(instrument, "default"):
        instrument = instrument.default
    if hasattr(d_min, "default"):
        d_min = d_min.default
    if hasattr(per_run, "default"):
        per_run = per_run.default

    result = compute_metrics(
        filename=filename,
        found_peaks_file=found_peaks_file,
        instrument=instrument,
        d_min=d_min,
        per_run=per_run,
    )

    if "error_message" in result:
        print(result["error_message"])
        if result["error_message"].startswith("Exception"):
            print("METRICS: 9.99 9.99 9.99 9.99 9.99 9.99")
        return

    if "filter_message" in result:
        print(f"METRICS: {result['filter_message']}")

    print(
        f"METRICS: {result['median_d_err']:.5f} {result['mean_d_err']:.5f} {result['max_d_err']:.5f} "
        f"{result['median_ang_err']:.5f} {result['mean_ang_err']:.5f} {result['max_ang_err']:.5f}"
    )

    if per_run and "per_run_errors" in result:
        print("\nPER-RUN MEDIAN ANGULAR ERROR (deg) - Sorted by error:")
        for r, err, count in result["per_run_errors"]:
            status = "BAD" if err > 1.0 else "OK"
            print(f"  Run {r:4d}: {err:6.3f} ({count:4d} peaks) [{status}]")


@app.command()
def peak_predictor(
    filename: str,
    instrument: str,
    indexed_hdf5_filename: str,
    integration_peaks_filename: str,
    d_min: float = 1.0,
    create_visualizations: bool = False,
    space_group: str | None = None,
    wavel_min: float | None = None,
    wavel_max: float | None = None,
    max_workers: int = 16,
):
    with h5py.File(indexed_hdf5_filename, "r") as f_idx:
        a = float(f_idx["sample/a"][()])
        b = float(f_idx["sample/b"][()])
        c = float(f_idx["sample/c"][()])
        alpha = float(f_idx["sample/alpha"][()])
        beta = float(f_idx["sample/beta"][()])
        gamma = float(f_idx["sample/gamma"][()])

        if space_group is None:
            space_group = f_idx["sample/space_group"][()].decode("utf-8")

        wavelength = f_idx["instrument/wavelength"][()]
        if wavel_min:
            wavelength[0] = wavel_min
        if wavel_max:
            wavelength[1] = wavel_max

        U = f_idx["sample/U"][()]
        B = f_idx["sample/B"][()]

        offsets = (
            f_idx["optimization/goniometer_offsets"][()]
            if "optimization/goniometer_offsets" in f_idx
            else None
        )

        if "sample/offset" in f_idx:
            sample_offset = f_idx["sample/offset"][()]
        else:
            sample_offset = np.zeros(3)

        if "beam/ki_vec" in f_idx:
            ki_vec = f_idx["beam/ki_vec"][()]
        else:
            ki_vec = np.array([0.0, 0.0, 1.0])

    peaks = Peaks(
        filename,
        instrument,
        wavelength_min=wavelength[0],
        wavelength_max=wavelength[1],
    )

    print(
        f"Predicting peaks for {len(peaks.image.ims)} images using solution from {indexed_hdf5_filename}"
    )

    all_R = peaks.goniometer.rotation

    if offsets is not None:
        print(f"Applying refined goniometer offsets from indexer: {offsets}")
        if peaks.goniometer.angles_raw is not None and peaks.goniometer.axes_raw is not None:
            angles_refined = peaks.goniometer.angles_raw + offsets[None, :]
            all_R = np.stack([
                calc_goniometer_rotation_matrix(peaks.goniometer.axes_raw, ang)
                for ang in angles_refined
            ])
        else:
            print("WARNING: Cannot apply refined offsets. Using nominal R stack.")
    else:
        print("Using nominal R stack directly from raw images (no offsets applied).")

    UB = U @ B

    if all_R.ndim == 3:
        RUB = np.matmul(all_R, UB)
    else:
        RUB = all_R @ UB

    results_map = peaks.predict_peaks(
        a, b, c, alpha, beta, gamma, d_min,
        RUB=RUB, space_group=space_group, sample_offset=sample_offset,
        ki_vec=ki_vec, max_workers=max_workers, R_all=all_R,
    )

    print(f"Saving predictions to {integration_peaks_filename}")
    with h5py.File(integration_peaks_filename, "w") as f:
        f.attrs["instrument"] = instrument
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        sorted_keys = sorted(peaks.image.ims.keys())
        bank_ids = np.array(
            [peaks.image.bank_mapping.get(k, k) for k in sorted_keys], dtype=np.int32
        )
        f.create_dataset("bank_ids", data=bank_ids)
        f["sample/gamma"] = gamma
        f["sample/space_group"] = space_group
        f["sample/U"] = U
        f["sample/B"] = B
        f["instrument/wavelength"] = wavelength
        f["goniometer/R"] = all_R 

        try:
            goniometer_angles_to_save = angles_refined
        except NameError:
            goniometer_angles_to_save = peaks.goniometer.angles_raw

        f["goniometer/angles"] = goniometer_angles_to_save
        f["goniometer/axes"] = peaks.goniometer.axes_raw
        if peaks.goniometer.names_raw:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(
                "goniometer/names", data=peaks.goniometer.names_raw, dtype=dt
            )

        f["sample/offset"] = sample_offset
        f["beam/ki_vec"] = ki_vec

        for img_key, (i, j, h, k, l, wl) in results_map.items():  # noqa: E741
            grp = f.create_group(f"banks/{img_key}")
            grp.create_dataset("i", data=i)
            grp.create_dataset("j", data=j)
            grp.create_dataset("h", data=h)
            grp.create_dataset("k", data=k)
            grp.create_dataset("l", data=l)
            grp.create_dataset("wavelength", data=wl)


@app.command()
def integrator(
    filename: str, 
    instrument: str,
    integration_peaks_filename: str,
    output_filename: str,
    integration_method: str = "free_fit",
    integration_mask_file: str | None = None,
    integration_mask_rel_erosion_radius: float | None = 0.05,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_intensity: float = 50.0,  
    region_growth_minimum_sigma: float | None = None,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 10,
    peak_minimum_signal_to_noise: float = 1.0,
    peak_pixel_outlier_threshold: float = 2.0,
    create_visualizations: bool = False,
    show_progress: bool = True,
    found_peaks_file: str = None,
    max_workers: int = 16,
):
    peak_dict = {}
    angles_stack = None
    all_R = None
    with h5py.File(integration_peaks_filename, "r") as f:
        if "sample/U" in f:
            U = f["sample/U"][()]
        if "sample/B" in f:
            B = f["sample/B"][()]

        if "goniometer/R" in f:
            all_R = f["goniometer/R"][()]

        if "goniometer/angles" in f:
            angles_stack = f["goniometer/angles"][()]

        if "sample/offset" in f:
            sample_offset = f["sample/offset"][()]
        else:
            sample_offset = np.zeros(3)
        if "beam/ki_vec" in f:
            ki_vec = f["beam/ki_vec"][()]
        else:
            ki_vec = np.array([0.0, 0.0, 1.0])

        for key in f["banks"].keys():
            img_idx = int(key)
            grp = f[f"banks/{key}"]
            peak_dict[img_idx] = [
                grp["i"][()],
                grp["j"][()],
                grp["h"][()],
                grp["k"][()],
                grp["l"][()],
                grp["wavelength"][()],
            ]

    peaks = Peaks(filename, instrument)

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

    if all_R is None:
        print("Warning: Refined R stack not found in prediction file. Using nominal.")
        all_R = peaks.goniometer.rotation

    if angles_stack is None:
        angles_stack = peaks.goniometer.angles_raw

    UB = U @ B
    if all_R.ndim == 3:
        RUB = np.matmul(all_R, UB)
    else:
        RUB = all_R @ UB

    result = peaks.integrate(
        peak_dict,
        integration_params,
        RUB=RUB,
        R_stack=all_R,
        angles_stack=angles_stack,
        sample_offset=sample_offset,
        ki_vec=ki_vec,
        create_visualizations=create_visualizations,
        show_progress=show_progress,
        integration_method=integration_method,
        file_prefix=filename,
        found_peaks_file=found_peaks_file,
        max_workers=max_workers,
    )

    print(f"Saving integrated peaks to {output_filename}")

    copy_keys = [
        "sample/a",
        "sample/b",
        "sample/c",
        "sample/alpha",
        "sample/beta",
        "sample/gamma",
        "sample/space_group",
        "sample/U",
        "sample/B",
        "sample/offset",
        "beam/ki_vec",
        "instrument/wavelength",
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
        f["peaks/run_index"] = result.run_id 
        f["peaks/xyz"] = result.xyz

        if result.R and any(r is not None for r in result.R):
            f["goniometer/R"] = np.array(result.R)
        if result.angles and any(a is not None for a in result.angles):
            f["goniometer/angles"] = np.array(result.angles)

        with h5py.File(integration_peaks_filename, "r") as f_in:
            for key in copy_keys:
                if key in f_in:
                    f_in.copy(f_in[key], f, key)

            for k in ["goniometer/axes", "goniometer/names"]:
                if k in f_in:
                    f_in.copy(f_in[k], f, k)


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
    print(f"Reducing {nexus_filename} -> {output_filename}")

    peaks_handler = Peaks(
        nexus_filename,
        instrument,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )

    if not peaks_handler.image.ims:
        print("Warning: No images found in file.")
        return

    sorted_banks = sorted(peaks_handler.image.ims.keys())

    image_stack = np.stack([peaks_handler.image.ims[b] for b in sorted_banks])

    bank_ids = np.array(sorted_banks, dtype=np.int32)
    n_images = len(sorted_banks)

    if peaks_handler.goniometer.angles_raw is not None:
        angles_repeated = np.tile(peaks_handler.goniometer.angles_raw, (n_images, 1))
    else:
        angles_repeated = np.zeros((n_images, 3)) 

    if peaks_handler.goniometer.axes_raw is not None:
        axes = np.array(peaks_handler.goniometer.axes_raw)
    else:
        axes = np.array([0.0, 1.0, 0.0])  

    with h5py.File(output_filename, "w") as f:
        f.create_dataset("images", data=image_stack, compression="lzf")
        f.create_dataset("bank_ids", data=bank_ids)
        f.create_dataset("goniometer/angles", data=angles_repeated)
        f.create_dataset("goniometer/axes", data=axes)

        if peaks_handler.goniometer.names_raw:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(
                "goniometer/names",
                data=peaks_handler.goniometer.names_raw,
                dtype=dt,
            )

        wl = [peaks_handler.wavelength.min, peaks_handler.wavelength.max]
        f.create_dataset("instrument/wavelength", data=wl)
        f.attrs["instrument"] = instrument

    print(f"Saved {n_images} banks to {output_filename}")


@app.command()
def merge_images(
    input_pattern: str = typer.Argument(
        ..., help="Glob pattern for reduced .h5 files (e.g. 'reduced/*.h5')"
    ),
    output_filename: str = typer.Argument(..., help="Output master .h5 file"),
):
    if " " in input_pattern:
        h5_files = []
        for p in input_pattern.split():
            h5_files.extend(glob.glob(p))
    else:
        h5_files = glob.glob(input_pattern)

    h5_files = sorted(list(set(h5_files)))

    if not h5_files:
        print(f"No files found matching: {input_pattern}")
        raise typer.Exit(code=1)

    print(f"Found {len(h5_files)} files. Merging...")
    merger = ImageStackMerger(h5_files)
    merger.merge(output_filename)

    print(f"Successfully created {output_filename}")


if __name__ == "__main__":
    app()
