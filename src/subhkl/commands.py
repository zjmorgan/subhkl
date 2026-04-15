import glob
import h5py
import numpy as np

# NOTE(Vivek): deprecate and use Goniometer class to handler rotation calc
from subhkl.instrument.goniometer import (
    calc_goniometer_rotation_matrix,
    get_rotation_data_from_nexus,
)
from subhkl.io.export import FinderConcatenateMerger, ImageStackMerger, MTZExporter
from subhkl.integration import Peaks
from subhkl.instrument.metrics import compute_metrics
from subhkl.optimization import FindUB


def run_index(
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
            print(
                f"Refining goniometer angles from Nexus with {goniometer_bound_deg} deg bounds."
            )
            axes, angles, names = get_rotation_data_from_nexus(
                nexus_filename, instrument_name
            )
            opt.goniometer_axes = np.array(axes)

            if opt.run_indices is not None:
                num_runs = np.max(opt.run_indices) + 1
                opt.goniometer_angles = np.array(angles)[:, np.newaxis].repeat(
                    num_runs, axis=1
                )
            else:
                num_peaks = len(opt.two_theta)
                opt.goniometer_angles = np.array(angles)[:, np.newaxis].repeat(
                    num_peaks, axis=1
                )
            goniometer_names = names
        elif opt.goniometer_axes is not None:
            print(
                f"Refining goniometer angles from HDF5 file with {goniometer_bound_deg} deg bounds."
            )
        else:
            print(
                "WARNING: refine_goniometer requested but goniometer data not found. Skipping."
            )
            refine_goniometer = False

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
    )

    print(f"\nOptimization complete. Best solution indexed {num} peaks.")

    opt.reciprocal_lattice_B()

    copy_keys = [
        "sample/space_group",
        "instrument/wavelength",
        "peaks/intensity",
        "peaks/sigma",
        "peaks/two_theta",
        "peaks/azimuthal",
        "peaks/radius",
        "peaks/xyz",
        "goniometer/R",
        "goniometer/axes",
        "goniometer/angles",
        "goniometer/names",
        "files",
        "file_offsets",
        "peaks/run_index",
        "bank",
        "sample/offset",
        "beam/ki_vec",
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
    print("Done.")


def run_finder(
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
    mask_file: str | None = None,
    mask_rel_erosion_radius: float | None = None,
    thresholding_noise_cutoff_quantile: float = 0.8,
    thresholding_min_peak_dist_pixels: float = 8.0,
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
    sparse_rbf_gamma: float = 1.0,
    sparse_rbf_min_sigma: float = 1.5,
    sparse_rbf_max_sigma: float = 10.0,
    sparse_rbf_max_peaks: int = 500,
    sparse_rbf_chunk_size: int = 512,
    sparse_rbf_tile_rows: int = 2,
    sparse_rbf_tile_cols: int = 2,
    sparse_rbf_loss: str = "gaussian",
    sparse_rbf_auto_tune_alpha: bool = False,
    sparse_rbf_candidate_alphas: str = "3.0,5.0,10.0,15.0,20.0,25.0,30",
    max_workers: int = 16,
):
    from subhkl.integration import Peaks  # Ensure imported globally or locally
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
                "blur_kernel_sigma": thresholding_blur_kernel_sigma,
                "open_kernel_size_pixels": thresholding_open_kernel_size_pixels,
                "show_steps": show_steps,
                "show_scale": "log",
            }
        )
    elif finder_algorithm == "sparse_rbf":
        # Because we separated Typer from core logic, this split is 100% safe
        alpha_list = [float(k.strip()) for k in sparse_rbf_candidate_alphas.split(",")]

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
                "loss": sparse_rbf_loss,
                "auto_tune_alpha": sparse_rbf_auto_tune_alpha,
                "candidate_alphas": alpha_list,
            }
        )
    else:
        raise ValueError("Invalid finder algorithm")

    peak_kwargs.update(
        {
            "mask_file": mask_file,
            "mask_rel_erosion_radius": mask_rel_erosion_radius,
        }
    )

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


def run_rbf_integrator(
    filename: str,
    instrument: str,
    integration_peaks_filename: str,
    output_filename: str,
    alpha: float = 1.0,
    gamma: float = 1.0,
    sigmas: str = "1.0,2.0,4.0",
    nominal_sigma: float = 1.0,
    anisotropic: bool = False,
    fit_mosaicity: bool = False,
    max_peaks: int = 500,
    rel_border_width: float = 0.0,
    show_progress: bool = True,
    create_visualizations: bool = False,
    chunk_size: int = 256,
    max_workers: int | None = None,
):
    import h5py
    from subhkl.integration import Peaks
    from subhkl.peakfinder.sparse_rbf import integrate_peaks_rbf_ssn

    sigma_list = [float(k.strip()) for k in sigmas.split(",")]
    print(f"Starting Dense Sparse RBF Integration on {filename}")
    print(f"Parameters: Alpha={alpha}, Gamma={gamma}, Sigma={sigma_list}, Max Peaks Padding={max_peaks}")

    peak_dict = {}

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
            
        for key in f["banks"].keys():
            img_idx = int(key)
            grp = f[f"banks/{key}"]
            peak_dict[img_idx] = [
                grp["i"][()], grp["j"][()], grp["h"][()],
                grp["k"][()], grp["l"][()], grp["wavelength"][()]
            ]

    peaks = Peaks(filename, instrument)

    if all_R is None:
        all_R = peaks.goniometer.rotation
    if angles_stack is None:
        angles_stack = peaks.goniometer.angles_raw

    one_image = next(iter(peaks.image.ims.values()))
    border_width = int(rel_border_width * min(one_image.shape[0], one_image.shape[1]))

    result = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict,
        peaks_obj=peaks,             # Pass the full Peaks object
        alpha=alpha,
        sigmas=sigma_list,
        gamma=gamma,
        nominal_sigma=nominal_sigma,
        max_peaks=max_peaks,
        show_progress=show_progress,
        all_R=all_R,                 # Pass rotation and offset downstream
        sample_offset=sample_offset,
        anisotropic=anisotropic,
        fit_mosaicity=fit_mosaicity,
        border_width=border_width,
        chunk_size=chunk_size,
        create_visualizations=create_visualizations,
        max_workers=max_workers,
    )

    print(f"Saving RBF integrated peaks to {output_filename}")
    with h5py.File(output_filename, "w") as f:
        f["peaks/h"] = result.h
        f["peaks/k"] = result.k
        f["peaks/l"] = result.l
        f["peaks/lambda"] = result.wavelength
        f["peaks/intensity"] = result.intensity
        f["peaks/sigma"] = result.sigma  # SVD-stabilized Fisher Info UQ
        f["peaks/two_theta"] = result.tt
        f["peaks/azimuthal"] = result.az
        f["peaks/bank"] = result.bank
        f["peaks/run_index"] = result.run_id
        
        # Copy full metadata context from predictor output
        copy_keys = [
            "sample/a", "sample/b", "sample/c", 
            "sample/alpha", "sample/beta", "sample/gamma",
            "sample/space_group", "sample/U", "sample/B", 
            "sample/offset", "beam/ki_vec", "instrument/wavelength"
        ]
        
        with h5py.File(integration_peaks_filename, "r") as f_in:
            for key in copy_keys:
                if key in f_in:
                    f_in.copy(f_in[key], f, key)
                    
            for k in ["goniometer/axes", "goniometer/names"]:
                if k in f_in:
                    f_in.copy(f_in[k], f, k)
