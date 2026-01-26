import typing
import h5py
import numpy as np
import typer
import uuid
import os

from subhkl.export import FinderConcatenateMerger, MTZExporter
from subhkl.integration import Peaks
from subhkl.optimization import FindUB
from subhkl.config.goniometer import get_rotation_data_from_nexus, calc_goniometer_rotation_matrix

app = typer.Typer()


def index(
    hdf5_peaks_filename: str,
    output_peaks_filename: str,
    strategy_name: str,
    population_size: int,
    gens: int,
    n_runs: int,
    seed: int,
    softness: float,
    sigma_init: float = None,
    refine_lattice: bool = False,
    lattice_bound_frac: float = 0.05,
    bootstrap_filename: str = None,
    refine_goniometer: bool = False,
    refine_goniometer_axes: list = None,
    goniometer_bound_deg: float = 5.0,
    refine_sample: bool = False, # NEW
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
    B_sharpen: float = None,
):
    """
    Index the given peak file and save it using the evosax optimizer.
    """

    # Index the peaks
    opt = FindUB(hdf5_peaks_filename)

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
        B_sharpen=B_sharpen,
    )

    print(f"\nOptimization complete. Best solution indexed {num} peaks.")

    h = [i[0] for i in hkl]
    k = [i[1] for i in hkl]
    l_list = [i[2] for i in hkl]

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
        "peaks/xyz", # Copy XYZ
        "goniometer/axes", 
        "goniometer/angles",
        "goniometer/names"
    ]

    copied_data = {}

    with h5py.File(hdf5_peaks_filename) as f:
        for key in copy_keys:
            if key in f:
                copied_data[key] = np.array(f[key])

    print(f"Saving indexed peaks to {output_peaks_filename}...")
    with h5py.File(output_peaks_filename, "w") as f:
        for key, value in copied_data.items():
            f[key] = value

        f["goniometer/R"] = refined_R
        
        if opt.goniometer_offsets is not None:
            f["optimization/goniometer_offsets"] = opt.goniometer_offsets
            
        if opt.sample_offset is not None:
            f["sample/offset"] = opt.sample_offset

        f["beam/ki_vec"] = opt.ki_vec
        f["sample/a"] = opt.a
        f["sample/b"] = opt.b
        f["sample/c"] = opt.c
        f["sample/alpha"] = opt.alpha
        f["sample/beta"] = opt.beta
        f["sample/gamma"] = opt.gamma

        f["sample/B"] = B
        f["sample/U"] = U
        f["peaks/h"] = h
        f["peaks/k"] = k
        f["peaks/l"] = l_list
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
    peak_pixel_outlier_threshold: float = 2.0
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
        gonio_names=detector_peaks.gonio_names
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
    wavelength_min: float,
    wavelength_max: float,
    space_group: str,
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
    B_sharpen: float = typer.Option(None, "--b-sharpen", help="Wilson B-factor for peak sharpening (~50 for protein crystals)"),
) -> None:
    # Logic to resolve SG
    sg_to_use = "P 1"
    if space_group:
        sg_to_use = space_group

    print(f"Loading peaks from: {peaks_h5_filename}")
    with h5py.File(peaks_h5_filename) as f:
        two_theta = np.array(f["peaks/two_theta"])
        az_phi = np.array(f["peaks/azimuthal"])
        intensity = np.array(f["peaks/intensity"])
        sigma = np.array(f["peaks/sigma"])
        rotations = np.array(f["goniometer/R"])
        
        if "peaks/radius" in f:
            radii = np.array(f["peaks/radius"])
        else:
            radii = None
            
        if "peaks/xyz" in f:
            peak_xyz = np.array(f["peaks/xyz"])
        else:
            peak_xyz = None
        
        gonio_axes = None
        gonio_angles = None
        gonio_names = None
        if "goniometer/axes" in f:
            gonio_axes = np.array(f["goniometer/axes"])
        if "goniometer/angles" in f:
            gonio_angles = np.array(f["goniometer/angles"])
        if "goniometer/names" in f:
            gonio_names = [n.decode('utf-8') for n in f["goniometer/names"][()]]

    if goniometer_csv_filename is not None:
        print(f"Loading goniometer from: {goniometer_csv_filename}")
        R = np.loadtxt(goniometer_csv_filename, delimiter=",")
        rotations = np.stack([R] * len(two_theta))
    else:
        print("Using goniometer rotation from peaks file.")
        R = rotations

    unique_filename = str(uuid.uuid4()) + ".h5"
    print(f"Creating temporary indexer input file: {unique_filename}")
    with h5py.File(unique_filename, "w") as f:
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/space_group"] = sg_to_use
        f["instrument/wavelength"] = [wavelength_min, wavelength_max]
        f["goniometer/R"] = rotations
        f["peaks/two_theta"] = two_theta
        f["peaks/azimuthal"] = az_phi
        f["peaks/intensity"] = intensity
        f["peaks/sigma"] = sigma
        
        if radii is not None:
            f["peaks/radius"] = radii
        if peak_xyz is not None:
            f["peaks/xyz"] = peak_xyz
        
        if gonio_axes is not None:
            f["goniometer/axes"] = gonio_axes
        if gonio_angles is not None:
            f["goniometer/angles"] = gonio_angles
        if gonio_names is not None:
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset("goniometer/names", data=gonio_names, dtype=dt)

    gonio_axes_list = None
    if refine_goniometer_axes:
        gonio_axes_list = [x.strip() for x in refine_goniometer_axes.split(',')]

    index(
        hdf5_peaks_filename=unique_filename,
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
        refine_goniometer_axes=gonio_axes_list,
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
def metrics(filename: str):
    """
    Calculate and print D-spacing and Angular errors for an indexed solution.
    Output Format: METRICS: median_d mean_d max_d median_ang mean_ang max_ang
    """
    try:
        # 1. LOAD DATA
        with h5py.File(filename, "r") as f:
            if "peaks/xyz" not in f:
                # Fallback or error if XYZ not present
                print("METRICS: 9.99 9.99 9.99 9.99 9.99 9.99")
                return

            xyz_det = f["peaks/xyz"][()] # Lab coordinates (N, 3)
            h = f["peaks/h"][()]
            k = f["peaks/k"][()]
            l = f["peaks/l"][()]
            lam = f["peaks/lambda"][()]

            # Geometry
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

        # Filter Indexed
        mask = (h != 0) | (k != 0) | (l != 0)
        if np.sum(mask) == 0:
            print("METRICS: 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000")
            return

        h, k, l = h[mask], k[mask], l[mask]
        lam = lam[mask]
        xyz_det = xyz_det[mask]
        R_all = R_all[mask]

        # --- PHYSICS CALCULATION ---

        # 1. B Matrix (subhkl units: 1/d)
        B_mat = ub_helper.reciprocal_lattice_B()

        # 2. Geometric Q (Unitless, just directions)
        # Correct for Sample Offset: v = P_raw - S_total
        v = xyz_det - sample_offset
        dist = np.linalg.norm(v, axis=1, keepdims=True)
        kf_dir = v / dist

        # Incident Beam (Normalized)
        ki_dir = ki_vec / np.linalg.norm(ki_vec)

        # Vector Difference: |kf - ki| = 2*sin(theta)
        delta_k = kf_dir - ki_dir[None, :]
        two_sin_theta = np.linalg.norm(delta_k, axis=1)

        # 3. METRIC 1: D-SPACING ERROR
        # d_obs = lambda / 2sin(theta)
        # Avoid divide by zero
        d_obs = np.divide(lam, two_sin_theta, where=two_sin_theta!=0)

        # d_calc = 1 / |B * hkl|
        hkl = np.stack([h, k, l]).T
        q_cryst_simple = hkl @ B_mat.T
        q_cryst_mag = np.linalg.norm(q_cryst_simple, axis=1)
        d_calc = np.divide(1.0, q_cryst_mag, where=q_cryst_mag!=0)

        d_err = np.abs(d_obs - d_calc)

        # 4. METRIC 2: ANGULAR ERROR
        # Q_calc (Lab Frame Direction)
        # q_lab = R * U * B * h
        # Note: Code uses row vector convention: h @ B.T @ U.T
        q_cryst = hkl @ B_mat.T
        q_phi = q_cryst @ U.T

        # Rotate by Goniometer R
        # R_all shape (N, 3, 3), q_phi shape (N, 3)
        # We need (R @ q_phi.T).T -> (N, 3)
        # Einstein sum: n=batch, i=row, j=col. R[n,i,j] * q[n,j] -> out[n,i]
        q_lab = np.einsum('nij,nj->ni', R_all, q_phi)

        q_calc_norm = q_lab / np.linalg.norm(q_lab, axis=1, keepdims=True)

        # Q_obs (Lab Frame Direction)
        # Direction of scattering vector is same as delta_k
        q_obs_norm = delta_k / two_sin_theta[:, None]

        dot = np.sum(q_obs_norm * q_calc_norm, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        ang_err = np.rad2deg(np.arccos(dot))

        # 5. PRINT RESULTS
        print(f"METRICS: {np.median(d_err):.5f} {np.mean(d_err):.5f} {np.max(d_err):.5f} "
              f"{np.median(ang_err):.5f} {np.mean(ang_err):.5f} {np.max(ang_err):.5f}")

    except Exception:
        # Silently fail to standard error format for the pipeline
        print("METRICS: 9.99 9.99 9.99 9.99 9.99 9.99")

@app.command()
def peak_predictor(
    filename: str,
    instrument: str,
    indexed_hdf5_filename: str,
    integration_peaks_filename: str,
    d_min: float = 1.0,
    create_visualizations: bool = False,
    space_group: str = None,
    wavel_min: float = None,
    wavel_max: float = None,
):
    with h5py.File(indexed_hdf5_filename) as f_indexed:
        a = float(np.array(f_indexed["sample/a"]))
        b = float(np.array(f_indexed["sample/b"]))
        c = float(np.array(f_indexed["sample/c"]))
        alpha = float(np.array(f_indexed["sample/alpha"]))
        beta = float(np.array(f_indexed["sample/beta"]))
        gamma = float(np.array(f_indexed["sample/gamma"]))
        if space_group is None:
            space_group = np.array(f_indexed["sample/space_group"]).item().decode('utf-8')
        wavelength = np.array(f_indexed["instrument/wavelength"])
        if wavel_min is not None:
            wavelength[0] = wavel_min
        if wavel_max is not None:
            wavelength[1] = wavel_max
        U = np.array(f_indexed["sample/U"])
        B = np.array(f_indexed["sample/B"])

        refined_offsets = None
        if "optimization/goniometer_offsets" in f_indexed:
            refined_offsets = np.array(f_indexed["optimization/goniometer_offsets"])
            
        sample_offset = None
        if "sample/offset" in f_indexed:
            sample_offset = np.array(f_indexed["sample/offset"])

        ki_vec = None
        if "beam/ki_vec" in f_indexed:
            ki_vec = np.array(f_indexed["beam/ki_vec"])
            print(f"Using refined beam direction {ki_vec} in peak prediction.")

    peaks = Peaks(filename,
                  instrument,
                  wavelength_min=wavelength[0],
                  wavelength_max=wavelength[1])

    if refined_offsets is not None:
        if peaks.goniometer_axes_raw is not None and peaks.goniometer_angles_raw is not None:
            new_angles = np.array(peaks.goniometer_angles_raw) + refined_offsets
            new_R = calc_goniometer_rotation_matrix(peaks.goniometer_axes_raw, new_angles)
            peaks.goniometer_rotation = new_R
            print("Applied refined goniometer offsets to peak prediction.")
        else:
            print("Warning: Refined offsets found but raw goniometer data not available. Using default R.")
    
    R_used = peaks.goniometer_rotation
    UB = R_used @ U @ B

    peak_dict = peaks.predict_peaks(
        a, b, c, alpha, beta, gamma, d_min, UB, space_group=space_group, sample_offset=sample_offset, ki_vec=ki_vec,
    )

    if create_visualizations:
        import matplotlib.pyplot as plt
        for bank, predicted_peaks in peak_dict.items():
            plt.imshow(peaks.ims[bank] + 1, cmap="binary", norm="log")
            plt.scatter(predicted_peaks[1], predicted_peaks[0], edgecolors='r', facecolors='none')
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
        f["sample/space_group"] = space_group
        f["sample/U"] = U
        f["sample/B"] = B
        f["instrument/wavelength"] = wavelength
        f["goniometer/R"] = R_used 

        if sample_offset is not None:
            f["sample/offset"] = sample_offset

        if ki_vec is not None:
            f["beam/ki_vec"] = ki_vec

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
    integration_method: str,
    integration_mask_file: typing.Optional[str] = None,
    integration_mask_rel_erosion_radius: typing.Optional[float] = 0.05,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_intensity: float = 4500.0,
    region_growth_minimum_sigma: typing.Optional[float] = None,
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

    peaks = Peaks(filename, instrument)
    result = peaks.integrate(
        peak_dict,
        integration_params,
        create_visualizations=create_visualizations,
        show_progress=show_progress,
        integration_method=integration_method,
        file_prefix=filename
    )

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
        "goniometer/R",
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

        with h5py.File(integration_peaks_filename) as f_in:
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


if __name__ == "__main__":
    app()
