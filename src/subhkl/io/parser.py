# src/subhkl/io/command_line_parser.py
from typing import Annotated
import typer
import h5py

from subhkl.commands import (
    run_index,
    run_finder,
    run_rbf_integrator,
    run_metrics,
    run_peak_predictor,
    run_integrator,
    run_mtz_exporter,
    run_reduce,
    run_merge_images,
)

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
    sparse_rbf_tile_rows: int = 2,
    sparse_rbf_tile_cols: int = 2,
    sparse_rbf_loss: Annotated[str, typer.Option(help="Likelihood for peak finder.")] = "gaussian",
    sparse_rbf_auto_tune_alpha: Annotated[bool, typer.Option(help="Auto-tune SNR threshold.")] = False,
    sparse_rbf_candidate_alphas: Annotated[str, typer.Option(help="Candidate SNR thresholds alpha for auto-tuning")] = "3.0,5.0,10.0,15.0,20.0,25.0,30",
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
        sparse_rbf_tile_rows=sparse_rbf_tile_rows,
        sparse_rbf_tile_cols=sparse_rbf_tile_cols,
        sparse_rbf_loss=sparse_rbf_loss,
        sparse_rbf_auto_tune_alpha=sparse_rbf_auto_tune_alpha,
        sparse_rbf_candidate_alphas=sparse_rbf_candidate_alphas,
        max_workers=max_workers,
    )


@app.command()
def rbf_integrator(
    filename: Annotated[str, typer.Argument(help="Merged HDF5 image stack")],
    instrument: Annotated[str, typer.Argument(help="Instrument name")],
    integration_peaks_filename: Annotated[str, typer.Argument(help="Predicted peaks HDF5 file")],
    output_filename: Annotated[str, typer.Argument(help="Output integrated peaks HDF5 file")],
    alpha: Annotated[float, typer.Option("--alpha", help="Peak over background threshold (Z-score)")] = 1.0,
    gamma: Annotated[float, typer.Option("--gamma", help="Besov space weight exponent")] = 1.0,
    sigmas: Annotated[str, typer.Option(help="Unstretched peak radii")] = "1.0,2.0,4.0",
    nominal_sigma: Annotated[float, typer.Option(help="The typical peak radius, used as a fallback for weak reflections")] = 1.0,
    anisotropic: Annotated[bool, typer.Option(help="Integrate anisotropic quasi-Laue peaks")] = False,
    fit_mosaicity: Annotated[bool, typer.Option(help="Whether to fit the mosaicity separately from sample dimensions to explain peak shape. Only use in non-spherical detector geometries.")] = False,
    max_peaks: Annotated[int, typer.Option("--max-peaks", help="Maximum peaks per panel (used for JAX matrix padding)")] = 500,
    rel_border_width: Annotated[float, typer.Option(help="Border width in fraction of image size")] = 0.0,
    show_progress: Annotated[bool, typer.Option("--show-progress")] = True,
    create_visualizations: bool = False,
    chunk_size: int = 256,
    max_workers: Annotated[int | None, typer.Option(help="Maximum number of CPU tasks for visualization.")] = None,
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

def index(
    hdf5_peaks_filename: str | None = None,  # Made optional
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
        thresholding_noise_cutoff_quantile=thresholding_noise_cutoff_quantile,
        thresholding_min_peak_dist_pixels=thresholding_min_peak_dist_pixels,
        thresholding_mask_file=thresholding_mask_file,
        thresholding_mask_rel_erosion_radius=thresholding_mask_rel_erosion_radius,
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
        sparse_rbf_tile_rows=sparse_rbf_tile_rows,
        sparse_rbf_tile_cols=sparse_rbf_tile_cols,
        max_workers=max_workers,
    )


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
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    goniometer_csv_filename: str | None = None,
    original_nexus_filename: str | None = None,
    instrument_name: str | None = None,
    strategy_name: Annotated[str, typer.Option("--strategy")] = "DE",
    sigma_init: Annotated[float | None, typer.Option("--sigma-init")] = None,
    n_runs: Annotated[int, typer.Option("--n-runs", "-n")] = 1,
    population_size: Annotated[
        int, typer.Option("--population-size", "--popsize")
    ] = 1000,
    gens: Annotated[int, typer.Option("--gens")] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 0,
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
    bootstrap_filename: Annotated[str | None, typer.Option("--bootstrap")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
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

    with h5py.File(peaks_h5_filename, "r") as f:
        if wavelength_min is None or wavelength_max is None:
            if "instrument/wavelength" in f:
                wl = f["instrument/wavelength"][()]
                if wavelength_min is None:
                    wavelength_min = float(wl[0])
                if wavelength_max is None:
                    wavelength_max = float(wl[1])
            else:
                raise ValueError("Wavelength not provided and not found in input file.")

        keys_to_load = [
            "peaks/two_theta",
            "peaks/azimuthal",
            "peaks/intensity",
            "peaks/sigma",
            "peaks/radius",
            "peaks/xyz",
            "goniometer/R",
            "goniometer/axes",
            "goniometer/angles",
            "goniometer/names",
            "files",
            "file_offsets",
            "peaks/run_index",
            "peaks/image_index",
            "bank",
            "bank_ids",
            "sample/offset",
            "beam/ki_vec",
        ]
        for k in keys_to_load:
            if k in f:
                input_data[k] = f[k][()]

    if "peaks/image_index" in input_data:
        input_data["peaks/run_index"] = input_data["peaks/image_index"]

    input_data["sample/a"], input_data["sample/b"], input_data["sample/c"] = a, b, c
    (
        input_data["sample/alpha"],
        input_data["sample/beta"],
        input_data["sample/gamma"],
    ) = alpha, beta, gamma
    input_data["sample/space_group"] = sg_to_use
    input_data["instrument/wavelength"] = [float(wavelength_min), float(wavelength_max)]

    gonio_axes_list = (
        [x.strip() for x in refine_goniometer_axes.split(",")]
        if refine_goniometer_axes
        else None
    )

    run_index(
        input_data=input_data,
        output_peaks_filename=output_peaks_filename,
        strategy_name=strategy_name,
        population_size=population_size,
        gens=gens,
        sigma_init=sigma_init,
        n_runs=n_runs,
        seed=seed,
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
        batch_size=batch_size,
        wavelength_min=input_data["instrument/wavelength"][0],
        wavelength_max=input_data["instrument/wavelength"][1],
    )


@app.command()
def indexer_using_file(
    hdf5_peaks_filename: str,
    output_peaks_filename: str,
    original_nexus_filename: str | None = None,
    instrument_name: str | None = None,
    strategy_name: Annotated[str, typer.Option("--strategy")] = "DE",
    n_runs: Annotated[int, typer.Option("--n-runs")] = 1,
    population_size: Annotated[int, typer.Option("--population-size")] = 1000,
    gens: Annotated[int, typer.Option("--gens")] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    refine_lattice: Annotated[bool, typer.Option("--refine-lattice")] = False,
    lattice_bound_frac: Annotated[float, typer.Option("--lattice-bound-frac")] = 0.05,
    refine_goniometer: Annotated[bool, typer.Option("--refine-goniometer")] = False,
    goniometer_bound_deg: Annotated[
        float, typer.Option("--goniometer-bound-deg")
    ] = 5.0,
    tolerance_deg: float = 0.1,
):
    run_index(
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
        nexus_filename=original_nexus_filename,
        instrument_name=instrument_name,
    )


@app.command()
def metrics(
    filename: str,
    found_peaks_file: Annotated[
        str | None,
        typer.Option(
            "--found-peaks",
            help="Optional file with found peaks to compare against (e.g. finder.h5).",
        ),
    ] = None,
    instrument: Annotated[
        str | None,
        typer.Option(
            "--instrument", help="Instrument name (required if matching peaks)."
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
):
    run_metrics(
        filename=filename,
        found_peaks_file=found_peaks_file,
        instrument=instrument,
        d_min=d_min,
        per_run=per_run,
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
        d_min,
        create_visualizations,
        space_group,
        wavel_min,
        wavel_max,
        max_workers,
    )


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
    space_group: str,
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
):
    try:
        run_merge_images(input_pattern, output_filename)
    except ValueError as e:
        print(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
