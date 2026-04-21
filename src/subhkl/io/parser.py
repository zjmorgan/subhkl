# src/subhkl/io/command_line_parser.py
from typing import Annotated
import typer
import h5py
import os

from subhkl.commands import (
    run_index,
    run_rbf_integrator,
    run_finder,
    run_metrics,
    run_peak_predictor,
    run_integrator,
    run_mtz_exporter,
    run_reduce,
    run_merge_images,
    run_zone_axis_search,
)


app = typer.Typer()


def apply_detector_calibration(hdf5_filename: str, instrument: str):
    """
    Reads refined detector metrology from an indexer/prediction file (if present)
    and overrides the in-memory beamlines configuration so downstream
    tasks natively use the calibrated geometry.
    """
    from subhkl.config import beamlines

    if not os.path.exists(hdf5_filename):
        return

    with h5py.File(hdf5_filename, "r") as f:
        if "detector_calibration" in f:
            print(f"Loading calibrated detector geometry from {hdf5_filename}...")
            calib_grp = f["detector_calibration"]
            count = 0
            for bank_key in calib_grp.keys():
                bank_id = bank_key.replace("bank_", "")
                if instrument in beamlines and bank_id in beamlines[instrument]:
                    beamlines[instrument][bank_id]["center"] = calib_grp[bank_key][
                        "center"
                    ][()].tolist()
                    beamlines[instrument][bank_id]["uhat"] = calib_grp[bank_key][
                        "uhat"
                    ][()].tolist()
                    beamlines[instrument][bank_id]["vhat"] = calib_grp[bank_key][
                        "vhat"
                    ][()].tolist()
                    count += 1
            if count > 0:
                print(f"Successfully applied calibration to {count} detector panels.")


app = typer.Typer()


@app.command()
def finder(
    filename: Annotated[str, typer.Argument(help="Input raw/event Nexus file")],
    instrument: Annotated[str, typer.Argument(help="Instrument name")],
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
    sparse_rbf_loss: Annotated[
        str, typer.Option(help="Likelihood for peak finder.")
    ] = "gaussian",
    sparse_rbf_auto_tune_alpha: Annotated[
        bool, typer.Option(help="Auto-tune SNR threshold.")
    ] = False,
    sparse_rbf_candidate_alphas: Annotated[
        str, typer.Option(help="Candidate SNR thresholds alpha for auto-tuning")
    ] = "3.0,5.0,10.0,15.0,20.0,25.0,30.0",
    max_workers: int = 16,
):
    # Pass everything straight into the core logic function
    run_finder(
        filename=filename,
        instrument=instrument,
        output_filename=output_filename,
        finder_algorithm=finder_algorithm,
        show_progress=show_progress,
        create_visualizations=create_visualizations,
        show_steps=show_steps,
        peak_local_max_min_pixel_distance=peak_local_max_min_pixel_distance,
        peak_local_max_min_relative_intensity=peak_local_max_min_relative_intensity,
        peak_local_max_normalization=peak_local_max_normalization,
        mask_file=mask_file,
        mask_rel_erosion_radius=mask_rel_erosion_radius,
        thresholding_noise_cutoff_quantile=thresholding_noise_cutoff_quantile,
        thresholding_min_peak_dist_pixels=thresholding_min_peak_dist_pixels,
        thresholding_blur_kernel_sigma=thresholding_blur_kernel_sigma,
        thresholding_open_kernel_size_pixels=thresholding_open_kernel_size_pixels,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        region_growth_distance_threshold=region_growth_distance_threshold,
        region_growth_minimum_sigma=region_growth_minimum_sigma,
        region_growth_minimum_intensity=region_growth_minimum_intensity,
        region_growth_maximum_pixel_radius=region_growth_maximum_pixel_radius,
        peak_center_box_size=peak_center_box_size,
        peak_smoothing_window_size=peak_smoothing_window_size,
        peak_minimum_pixels=peak_minimum_pixels,
        peak_minimum_signal_to_noise=peak_minimum_signal_to_noise,
        peak_pixel_outlier_threshold=peak_pixel_outlier_threshold,
        sparse_rbf_alpha=sparse_rbf_alpha,
        sparse_rbf_gamma=sparse_rbf_gamma,
        sparse_rbf_min_sigma=sparse_rbf_min_sigma,
        sparse_rbf_max_sigma=sparse_rbf_max_sigma,
        sparse_rbf_max_peaks=sparse_rbf_max_peaks,
        sparse_rbf_chunk_size=sparse_rbf_chunk_size,
        sparse_rbf_loss=sparse_rbf_loss,
        sparse_rbf_auto_tune_alpha=sparse_rbf_auto_tune_alpha,
        sparse_rbf_candidate_alphas=sparse_rbf_candidate_alphas,
        max_workers=max_workers,
    )


@app.command()
def indexer(
    peaks_h5_filename: str,
    output_peaks_filename: str,
    a: Annotated[float | None, typer.Option(help="Unit cell parameter a")] = None,
    b: Annotated[float | None, typer.Option(help="Unit cell parameter b")] = None,
    c: Annotated[float | None, typer.Option(help="Unit cell parameter c")] = None,
    alpha: Annotated[
        float | None, typer.Option(help="Unit cell parameter alpha")
    ] = None,
    beta: Annotated[float | None, typer.Option(help="Unit cell parameter beta")] = None,
    gamma: Annotated[
        float | None, typer.Option(help="Unit cell parameter gamma")
    ] = None,
    space_group: Annotated[
        str | None, typer.Option(help="Space group (e.g. 'P 1')")
    ] = None,
    wavelength_min: Annotated[float | None, typer.Option("--wavelength-min")] = None,
    wavelength_max: Annotated[float | None, typer.Option("--wavelength-max")] = None,
    ki_vec: Annotated[
        str | None,
        typer.Option(
            "--ki-vec", help="Override incident beam vector (e.g., '0,0,1' or '0,0,-1')"
        ),
    ] = None,
    original_nexus_filename: Annotated[
        str | None,
        typer.Option("--nexus", help="Original nexus file for instrument definitions"),
    ] = None,
    instrument_name: Annotated[str | None, typer.Option("--instrument")] = None,
    strategy_name: Annotated[str, typer.Option("--strategy")] = "DE",
    sigma_init: Annotated[float | None, typer.Option("--sigma-init")] = None,
    n_runs: Annotated[int, typer.Option("--n-runs", "-n")] = 1,
    population_size: Annotated[
        int, typer.Option("--population-size", "--popsize")
    ] = 1000,
    gens: Annotated[int, typer.Option("--gens")] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    tolerance_deg: Annotated[float, typer.Option("--tolerance-deg")] = 0.1,
    freeze_orientation: Annotated[
        bool,
        typer.Option(
            "--freeze-orientation", help="Lock the U matrix to its initial state."
        ),
    ] = False,
    refine_lattice: Annotated[bool, typer.Option("--refine-lattice")] = False,
    lattice_bound_frac: Annotated[float, typer.Option("--lattice-bound-frac")] = 0.05,
    refine_goniometer: Annotated[bool, typer.Option("--refine-goniometer")] = False,
    refine_goniometer_axes: Annotated[
        str | None, typer.Option("--refine-goniometer-axes")
    ] = None,
    goniometer_bound_deg: Annotated[
        float, typer.Option("--goniometer-bound-deg")
    ] = 5.0,
    refine_sample: Annotated[bool, typer.Option("--refine-sample")] = False,
    sample_bound_meters: Annotated[
        float, typer.Option("--sample-bound-meters")
    ] = 0.005,
    refine_beam: Annotated[bool, typer.Option("--refine-beam")] = False,
    beam_bound_deg: Annotated[float, typer.Option("--beam-bound-deg")] = 1.0,
    refine_detector: Annotated[bool, typer.Option("--refine-detector")] = False,
    refine_detector_banks: Annotated[
        str | None,
        typer.Option(
            "--refine-detector-banks", help="Comma-separated bank IDs to refine"
        ),
    ] = None,
    detector_modes: Annotated[
        str,
        typer.Option(
            "--detector-modes",
            help="Comma-separated list of refinement modes (e.g. radial,global_rot,independent)",
        ),
    ] = "independent",
    detector_trans_bound_meters: Annotated[
        float, typer.Option("--detector-trans-bound-meters")
    ] = 0.005,
    detector_rot_bound_deg: Annotated[
        float, typer.Option("--detector-rot-bound-deg")
    ] = 1.0,
    detector_global_rot_bound_deg: Annotated[
        float, typer.Option("--detector-global-rot-bound-deg")
    ] = 2.0,
    detector_global_rot_axis: Annotated[
        str,
        typer.Option(
            "--detector-global-rot-axis",
            help="Axis vector for global_rot_axis mode (e.g. 0,1,0)",
        ),
    ] = "0,1,0",
    detector_global_trans_bound_meters: Annotated[
        float, typer.Option("--detector-global-trans-bound-meters")
    ] = 0.01,
    detector_radial_bound_frac: Annotated[
        float, typer.Option("--detector-radial-bound-frac")
    ] = 0.05,
    bootstrap_filename: Annotated[str | None, typer.Option("--bootstrap")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
    num_candidates: Annotated[
        int | None, typer.Option(help="Number of lambda candidates (default: 64)")
    ] = None,
) -> None:
    # 1. Safely Parse Comma-Separated Strings into Python Lists
    ki_vec_parsed = [float(x.strip()) for x in ki_vec.split(",")] if ki_vec else None
    gonio_axes_parsed = (
        [x.strip() for x in refine_goniometer_axes.split(",")]
        if refine_goniometer_axes
        else None
    )
    det_banks_parsed = (
        [int(x.strip()) for x in refine_detector_banks.split(",")]
        if refine_detector_banks
        else None
    )
    det_modes_parsed = (
        [x.strip().lower() for x in detector_modes.split(",")]
        if detector_modes
        else ["independent"]
    )
    global_rot_axis_parsed = (
        [float(x.strip()) for x in detector_global_rot_axis.split(",")]
        if detector_global_rot_axis
        else [0.0, 1.0, 0.0]
    )

    # 2. Hand off to Core Logic
    run_index(
        peaks_h5_filename=peaks_h5_filename,
        output_peaks_filename=output_peaks_filename,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        space_group=space_group,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        ki_vec=ki_vec_parsed,
        original_nexus_filename=original_nexus_filename,
        instrument_name=instrument_name,
        strategy_name=strategy_name,
        sigma_init=sigma_init,
        n_runs=n_runs,
        population_size=population_size,
        gens=gens,
        seed=seed,
        tolerance_deg=tolerance_deg,
        freeze_orientation=freeze_orientation,
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        refine_goniometer=refine_goniometer,
        refine_goniometer_axes=gonio_axes_parsed,
        goniometer_bound_deg=goniometer_bound_deg,
        refine_sample=refine_sample,
        sample_bound_meters=sample_bound_meters,
        refine_beam=refine_beam,
        beam_bound_deg=beam_bound_deg,
        refine_detector=refine_detector,
        refine_detector_banks=det_banks_parsed,
        detector_modes=det_modes_parsed,
        detector_trans_bound_meters=detector_trans_bound_meters,
        detector_rot_bound_deg=detector_rot_bound_deg,
        detector_global_rot_bound_deg=detector_global_rot_bound_deg,
        detector_global_rot_axis=global_rot_axis_parsed,
        detector_global_trans_bound_meters=detector_global_trans_bound_meters,
        detector_radial_bound_frac=detector_radial_bound_frac,
        bootstrap_filename=bootstrap_filename,
        batch_size=batch_size,
        num_candidates=num_candidates,
    )


@app.command()
def rbf_integrator(
    filename: Annotated[str, typer.Argument(help="Merged HDF5 image stack")],
    instrument: Annotated[str, typer.Argument(help="Instrument name")],
    integration_peaks_filename: Annotated[
        str, typer.Argument(help="Predicted peaks HDF5 file")
    ],
    output_filename: Annotated[
        str, typer.Argument(help="Output integrated peaks HDF5 file")
    ],
    alpha: Annotated[
        float, typer.Option("--alpha", help="Peak over background threshold (Z-score)")
    ] = 1.0,
    gamma: Annotated[
        float, typer.Option("--gamma", help="Besov space weight exponent")
    ] = 1.0,
    sigmas: Annotated[str, typer.Option(help="Unstretched peak radii")] = "1.0,2.0,4.0",
    nominal_sigma: Annotated[
        float,
        typer.Option(
            help="The typical peak radius, used as a fallback for weak reflections"
        ),
    ] = 1.0,
    anisotropic: Annotated[
        bool, typer.Option(help="Integrate anisotropic quasi-Laue peaks")
    ] = False,
    fit_mosaicity: Annotated[
        bool,
        typer.Option(
            help="Whether to fit the mosaicity separately from sample dimensions to explain peak shape. Only use in non-spherical detector geometries."
        ),
    ] = False,
    max_peaks: Annotated[
        int,
        typer.Option(
            "--max-peaks", help="Maximum peaks per panel (used for JAX matrix padding)"
        ),
    ] = 500,
    rel_border_width: Annotated[
        float, typer.Option(help="Border width in fraction of image size")
    ] = 0.0,
    show_progress: Annotated[bool, typer.Option("--show-progress")] = True,
    create_visualizations: bool = False,
    chunk_size: int = 256,
    max_workers: Annotated[
        int | None, typer.Option(help="Maximum number of CPU tasks for visualization.")
    ] = None,
):
    """
    Integrates predicted peaks using the Dense Sparse RBF network approach on GPU.
    Calculates intensities and rigorous I/SIGI via Fisher Information matrix SVD.
    """
    run_rbf_integrator(
        filename=filename,
        instrument=instrument,
        integration_peaks_filename=integration_peaks_filename,
        output_filename=output_filename,
        alpha=alpha,
        gamma=gamma,
        sigmas=sigmas,
        nominal_sigma=nominal_sigma,
        anisotropic=anisotropic,
        fit_mosaicity=fit_mosaicity,
        max_peaks=max_peaks,
        rel_border_width=rel_border_width,
        show_progress=show_progress,
        create_visualizations=create_visualizations,
        chunk_size=chunk_size,
        max_workers=max_workers,
    )


@app.command()
def metrics(
    file1: Annotated[
        str, typer.Argument(help="Primary file (e.g., indexer.h5 or predictor.h5)")
    ],
    file2: Annotated[
        str | None,
        typer.Option(
            "--file2",
            help="Optional secondary file to match against (e.g., finder.h5).",
        ),
    ] = None,
    instrument: Annotated[
        str | None,
        typer.Option(
            "--instrument",
            help="Instrument name (required if using file2 or predictor outputs).",
        ),
    ] = None,
    d_min: Annotated[
        float | None,
        typer.Option(
            "--d-min", help="Optional minimum d-spacing filter for metrics calculation."
        ),
    ] = None,
    per_run: Annotated[
        bool,
        typer.Option(
            "--per-run", help="Calculate and display metrics for each run/image."
        ),
    ] = False,
    ki_vec: Annotated[
        str | None,
        typer.Option(
            "--ki-vec", help="Override incident beam vector (e.g., '0,0,1' or '0,0,-1')"
        ),
    ] = None,
):
    """
    CLI command to compute and display indexing quality metrics.
    Compares HKL accuracy internally (file1), or spatial matching between file1 (predicted) and file2 (observed).
    """
    ki_vec_parsed = [float(x.strip()) for x in ki_vec.split(",")] if ki_vec else None

    run_metrics(
        file1=file1,
        file2=file2,
        instrument=instrument,
        d_min=d_min,
        per_run=per_run,
        ki_vec=ki_vec_parsed,
    )


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
    run_peak_predictor(
        filename,
        instrument,
        indexed_hdf5_filename,
        integration_peaks_filename,
        d_min=d_min,
        wavel_min=wavel_min,
        wavel_max=wavel_max,
        space_group=space_group,
        max_workers=max_workers,
        create_visualizations=create_visualizations,
    )


@app.command()
def integrator(
    filename: str,
    instrument: str,
    integration_peaks_filename: str,
    output_filename: str,
    integration_method: str = "free_fit",
    integration_mask_file: str | None = None,
    integration_mask_rel_erosion_radius: float | None = None,
    region_growth_distance_threshold: float = 1.5,
    region_growth_minimum_intensity: float = 50.0,
    region_growth_minimum_sigma: float | None = None,
    region_growth_maximum_pixel_radius: float = 17.0,
    peak_center_box_size: int = 15,
    peak_smoothing_window_size: int = 15,
    peak_minimum_pixels: int = 10,
    peak_minimum_signal_to_noise: float = 1.0,
    peak_pixel_outlier_threshold: float = 2.0,
    ki_vec: str = typer.Option(None, "--ki-vec", help="Override incident beam vector"),
    create_visualizations: bool = False,
    show_progress: bool = True,
    found_peaks_file: str | None = None,
    max_workers: int = 16,
):
    run_integrator(
        filename,
        instrument,
        integration_peaks_filename,
        output_filename,
        integration_method,
        integration_mask_file,
        integration_mask_rel_erosion_radius,
        region_growth_distance_threshold,
        region_growth_minimum_intensity,
        region_growth_minimum_sigma,
        region_growth_maximum_pixel_radius,
        peak_center_box_size,
        peak_smoothing_window_size,
        peak_minimum_pixels,
        peak_minimum_signal_to_noise,
        peak_pixel_outlier_threshold,
        create_visualizations,
        show_progress,
        found_peaks_file,
        max_workers,
    )


@app.command()
def mtz_exporter(
    indexed_h5_filename: str,
    output_mtz_filename: str,
    space_group: str = typer.Option(
        None, help="Optional. Loaded from indexer h5 if missing."
    ),
):
    run_mtz_exporter(indexed_h5_filename, output_mtz_filename, space_group)


@app.command()
def reduce(
    nexus_filename: str,
    output_filename: str,
    instrument: str,
    wavelength_min: Annotated[
        float | None, typer.Option(help="Override min wavelength")
    ] = None,
    wavelength_max: Annotated[
        float | None, typer.Option(help="Override max wavelength")
    ] = None,
):
    run_reduce(
        nexus_filename, output_filename, instrument, wavelength_min, wavelength_max
    )


@app.command()
def merge_images(
    input_pattern: Annotated[
        str,
        typer.Argument(help="Glob pattern for reduced .h5 files (e.g. 'reduced/*.h5')"),
    ],
    output_filename: Annotated[str, typer.Argument(help="Output master .h5 file")],
    a: float = typer.Argument(..., help="Unit cell parameter a"),
    b: float = typer.Argument(..., help="Unit cell parameter b"),
    c: float = typer.Argument(..., help="Unit cell parameter c"),
    alpha: float = typer.Argument(..., help="Unit cell parameter alpha"),
    beta: float = typer.Argument(..., help="Unit cell parameter beta"),
    gamma: float = typer.Argument(..., help="Unit cell parameter gamma"),
    space_group: str = typer.Argument(..., help="Space group (e.g. 'P 1')"),
):
    try:
        run_merge_images(
            input_pattern, output_filename, a, b, c, alpha, beta, gamma, space_group
        )

    except ValueError as e:
        print(str(e))
        raise typer.Exit(code=1)


@app.command()
def zone_axis_search(
    merged_h5_filename: str,
    peaks_h5_filename: str,
    instrument: str,
    output_h5_filename: str,
    d_min: float = 1.0,
    space_group: Annotated[
        str,
        typer.Option(help="(Optional) Space group for zone-axis search"),
    ] = None,
    vector_tolerance: Annotated[
        float,
        typer.Option(
            help="Angular capture radius in degrees for the objective function."
        ),
    ] = 0.15,
    border_frac: Annotated[
        float, typer.Option(help="Fraction of image to crop at the border.")
    ] = 0.1,
    min_intensity: Annotated[
        float, typer.Option(help="Minimum peak amplitude.")
    ] = 50.0,
    hough_grid_resolution: Annotated[
        int, typer.Option(help="Lambert grid resolution.")
    ] = 1024,
    n_hough: Annotated[
        int, typer.Option(help="Maximum number of empirical zone axes.")
    ] = 15,
    davenport_angle_tol: Annotated[
        float, typer.Option(help="Graph search angle tolerance in degrees.")
    ] = 0.5,
    top_k_rays: Annotated[
        int, typer.Option(help="Max rays per image to feed the Hough Transform.")
    ] = 15,
    max_uvw: Annotated[
        int, typer.Option(help="Maximum uvw index for zone axis search")
    ] = 25,
    L_max: Annotated[
        float,
        typer.Option(
            help="Maximum real-space vector length for theoretical zone axes (Angstroms)."
        ),
    ] = 250.0,
    top_k: Annotated[
        int, typer.Option(help="Maximum number of reciprocal grid points to consider.")
    ] = 1000,
    num_runs: Annotated[
        int, typer.Option(help="Number of goniometer runs to use. Set to 0 to use all.")
    ] = 0,
    output_hough: Annotated[
        str | None, typer.Option(help="Diagnostic hough transform image filename.")
    ] = None,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for validation loop")
    ] = 1024,
):
    """
    Global Zone-Axis Search to find the macroscopic crystal orientation (U matrix).
    Outputs an HDF5 file that can be passed directly to 'indexer --bootstrap'.
    """
    run_zone_axis_search(
        merged_h5_filename=merged_h5_filename,
        peaks_h5_filename=peaks_h5_filename,
        instrument=instrument,
        output_h5_filename=output_h5_filename,
        space_group=space_group,
        d_min=d_min,
        vector_tolerance=vector_tolerance,
        border_frac=border_frac,
        min_intensity=min_intensity,
        hough_grid_resolution=hough_grid_resolution,
        n_hough=n_hough,
        davenport_angle_tol=davenport_angle_tol,
        top_k_rays=top_k_rays,
        max_uvw=max_uvw,
        L_max=L_max,
        top_k=top_k,
        num_runs=num_runs,
        output_hough=output_hough,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    app()
