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
    run_zone_axis_search,
)

from subhkl.io.export import FinderConcatenateMerger, ImageStackMerger, MTZExporter
from subhkl.integration import Peaks
from subhkl.instrument.metrics import compute_metrics
from subhkl.optimization import FindUB
from subhkl.core.spacegroup import get_space_group_object

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
                    beamlines[instrument][bank_id]["center"] = calib_grp[bank_key]["center"][()].tolist()
                    beamlines[instrument][bank_id]["uhat"] = calib_grp[bank_key]["uhat"][()].tolist()
                    beamlines[instrument][bank_id]["vhat"] = calib_grp[bank_key]["vhat"][()].tolist()
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
def indexer(
    peaks_h5_filename: str, 
    output_peaks_filename: str, 
    a: float = typer.Option(None, help="Unit cell parameter a"), 
    b: float = typer.Option(None, help="Unit cell parameter b"), 
    c: float = typer.Option(None, help="Unit cell parameter c"),
    alpha: float = typer.Option(None, help="Unit cell parameter alpha"), 
    beta: float = typer.Option(None, help="Unit cell parameter beta"), 
    gamma: float = typer.Option(None, help="Unit cell parameter gamma"), 
    space_group: str = typer.Option(None, help="Space group (e.g. 'P 1')"),
    wavelength_min: float | None = typer.Option(None, "--wavelength-min"), 
    wavelength_max: float | None = typer.Option(None, "--wavelength-max"),
    ki_vec: str = typer.Option(None, "--ki-vec", help="Override incident beam vector (e.g., '0,0,1' or '0,0,-1')"),
    original_nexus_filename: str | None = typer.Option(None, "--nexus", help="Original nexus file for instrument definitions"),
    instrument_name: str | None = typer.Option(None, "--instrument"), 
    strategy_name: str = typer.Option("DE", "--strategy"),
    sigma_init: float = typer.Option(None, "--sigma-init"), 
    n_runs: int = typer.Option(1, "--n-runs", "-n"),
    population_size: int = typer.Option(1000, "--population-size", "--popsize"),
    gens: int = typer.Option(100, "--gens"), 
    seed: int = typer.Option(0, "--seed"),
    tolerance_deg: float = typer.Option(0.1, "--tolerance-deg"),
    freeze_orientation: bool = typer.Option(False, "--freeze-orientation", help="Lock the U matrix to its initial state."),
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
    refine_detector_banks: str = typer.Option(None, "--refine-detector-banks", help="Comma-separated bank IDs to refine"),
    detector_modes: str = typer.Option("independent", "--detector-modes", help="Comma-separated list of refinement modes (e.g. radial,global_rot,independent)"),
    detector_trans_bound_meters: float = typer.Option(0.005, "--detector-trans-bound-meters"),
    detector_rot_bound_deg: float = typer.Option(1.0, "--detector-rot-bound-deg"),
    detector_global_rot_bound_deg: float = typer.Option(2.0, "--detector-global-rot-bound-deg"),
    detector_global_rot_axis: str = typer.Option("0,1,0", "--detector-global-rot-axis", help="Axis vector for global_rot_axis mode (e.g. 0,1,0)"),
    detector_global_trans_bound_meters: float = typer.Option(0.01, "--detector-global-trans-bound-meters"),
    detector_radial_bound_frac: float = typer.Option(0.05, "--detector-radial-bound-frac"),
    bootstrap_filename: str | None = typer.Option(None, "--bootstrap"),
    batch_size: int = typer.Option(None, "--batch-size"),
    loss_method: str = typer.Option("cosine", "--loss-method"),
    d_min: float | None = typer.Option(None, "--d-min"),
    d_max: float | None = typer.Option(None, "--d-max"),
) -> None:
    input_data = {}
    def _val(x): return x.default if hasattr(x, "default") else x

    a_val, b_val, c_val = _val(a), _val(b), _val(c)
    alpha_val, beta_val, gamma_val = _val(alpha), _val(beta), _val(gamma)
    sg_val = _val(space_group)
    w_min_val, w_max_val = _val(wavelength_min), _val(wavelength_max)
    strat_val = _val(strategy_name)
    pop_val, gens_val, runs_val, seed_val = _val(population_size), _val(gens), _val(n_runs), _val(seed)
    tol_val = _val(tolerance_deg)
    sigma_val = _val(sigma_init)
    freeze_val = _val(freeze_orientation)
    
    gonio_axes_list = [x.strip() for x in _val(refine_goniometer_axes).split(",")] if _val(refine_goniometer_axes) else None
    det_banks_list = [int(x.strip()) for x in _val(refine_detector_banks).split(",")] if _val(refine_detector_banks) else None
    det_modes_list = [x.strip().lower() for x in _val(detector_modes).split(",")] if _val(detector_modes) else ["independent"]
    det_rot_axis = np.array([float(x.strip()) for x in _val(detector_global_rot_axis).split(",")])

    # --- INJECT BOOTSTRAP PHYSICS DIRECTLY ---
    if _val(bootstrap_filename):
        apply_detector_calibration(_val(bootstrap_filename), _val(instrument_name))
        with h5py.File(_val(bootstrap_filename), 'r') as b_f:
            if "sample/a" in b_f: a_val = b_f["sample/a"][()]
            if "sample/b" in b_f: b_val = b_f["sample/b"][()]
            if "sample/c" in b_f: c_val = b_f["sample/c"][()]
            if "sample/alpha" in b_f: alpha_val = b_f["sample/alpha"][()]
            if "sample/beta" in b_f: beta_val = b_f["sample/beta"][()]
            if "sample/gamma" in b_f: gamma_val = b_f["sample/gamma"][()]

    print(f"Loading peaks from: {peaks_h5_filename}")
    with h5py.File(peaks_h5_filename, "r") as f:
        if a_val is None: a_val = f["sample/a"][()] if "sample/a" in f else None
        if b_val is None: b_val = f["sample/b"][()] if "sample/b" in f else None
        if c_val is None: c_val = f["sample/c"][()] if "sample/c" in f else None
        if alpha_val is None: alpha_val = f["sample/alpha"][()] if "sample/alpha" in f else None
        if beta_val is None: beta_val = f["sample/beta"][()] if "sample/beta" in f else None
        if gamma_val is None: gamma_val = f["sample/gamma"][()] if "sample/gamma" in f else None
        
        if sg_val is None:
            file_sg = f["sample/space_group"][()] if "sample/space_group" in f else None
            sg_val = file_sg.decode('utf-8') if isinstance(file_sg, bytes) else file_sg
        
        if None in (a_val, b_val, c_val, alpha_val, beta_val, gamma_val, sg_val):
            raise ValueError("Unit cell parameters (a,b,c,alpha,beta,gamma) and Space Group must be provided via CLI or exist in the input file.")

        try:
            get_space_group_object(sg_val)
        except ValueError as e:
            print(f"ERROR: Invalid space group '{sg_val}': {e}")
            raise typer.Exit(code=1)

        if w_min_val is None or w_max_val is None:
            if "instrument/wavelength" in f:
                wl = f["instrument/wavelength"][()]
                if w_min_val is None: w_min_val = float(wl[0])
                if w_max_val is None: w_max_val = float(wl[1])
            else:
                raise ValueError("Wavelength min/max not provided and not found in input file.")

        keys_to_load = [
            "peaks/intensity", "peaks/sigma", "peaks/radius",
            "goniometer/R", "goniometer/axes", "goniometer/angles", "goniometer/names", 
            "files", "file_offsets", "peaks/run_index", "peaks/image_index", 
            "bank", "bank_ids", "sample/offset", "beam/ki_vec",
            "peaks/pixel_r", "peaks/pixel_c" 
        ]
        for k in keys_to_load:
            if k in f: input_data[k] = f[k][()]

        if _val(ki_vec) is not None:
            ki_vec_val = np.array([float(x.strip()) for x in _val(ki_vec).split(",")])
        else:
            ki_vec_val = f["beam/ki_vec"][()] if "beam/ki_vec" in f else np.array([0.0, 0.0, 1.0])

        detector_params = None
        peak_pixel_coords = None
        refine_det_flag = _val(refine_detector)
        target_banks = None

        if "peaks/pixel_r" in f and "peaks/pixel_c" in f:
            print("Reconstructing physical geometry from pixels for physics optimization...")
            if not _val(instrument_name) or not _val(original_nexus_filename): 
                raise ValueError("ERROR: Finder file contains pixels. You must provide --instrument and --nexus to rebuild geometry.")
                
            pixel_r = f["peaks/pixel_r"][()]
            pixel_c = f["peaks/pixel_c"][()]
            
            bank_array = None
            if "bank" in f: bank_array = f["bank"][()]
            elif "peaks/bank" in f: bank_array = f["peaks/bank"][()]
            elif "bank_ids" in f and "peaks/image_index" in f:
                b_ids = f["bank_ids"][()]
                img_idx = f["peaks/image_index"][()]
                bank_array = np.array([b_ids[int(idx)] for idx in img_idx])
            else:
                bank_array = f["peaks/image_index"][()] 
                
            peaks_obj = Peaks(_val(original_nexus_filename), _val(instrument_name))
            from subhkl.config import beamlines
            from subhkl.instrument.detector import Detector
            
            if refine_det_flag:
                all_physical_banks = [int(k) for k in beamlines[_val(instrument_name)].keys()]
                target_banks = det_banks_list if det_banks_list else sorted(all_physical_banks)

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
                    'm': m, 'n': n, 'pw': pw, 'ph': ph,
                    'modes': det_modes_list,
                    'radial_bound': _val(detector_radial_bound_frac),
                    'global_rot_bound_deg': _val(detector_global_rot_bound_deg),
                    'global_rot_axis': det_rot_axis,
                    'global_trans_bound_meters': _val(detector_global_trans_bound_meters)
                }
            
            xyz_out = np.zeros((len(pixel_r), 3))
            tt_out = np.zeros(len(pixel_r))
            az_out = np.zeros(len(pixel_r))
            
            u_offsets = np.zeros(len(pixel_r))
            v_offsets = np.zeros(len(pixel_r))
            bank_indices = np.zeros(len(pixel_r), dtype=np.int32)
            
            for phys_bank in np.unique(bank_array):
                mask = bank_array == phys_bank
                if not np.any(mask): continue
                    
                try:
                    det_config = beamlines[_val(instrument_name)][str(int(phys_bank))]
                    det = Detector(det_config)
                    
                    xyz_p = det.pixel_to_lab(pixel_r[mask], pixel_c[mask])
                    xyz_out[mask] = xyz_p
                    
                    tt_out[mask], az_out[mask] = det.pixel_to_angles(
                        pixel_r[mask], pixel_c[mask], ki_vec=ki_vec_val
                    )
                    
                    if refine_det_flag and int(phys_bank) in bank_to_idx:
                        bank_indices[mask] = bank_to_idx[int(phys_bank)]
                        u_offsets[mask] = np.dot(xyz_p - det.center, det.uhat)
                        v_offsets[mask] = np.dot(xyz_p - det.center, det.vhat)
                        
                except KeyError as e:
                    print(f"Warning: Could not rebuild geometry for bank {phys_bank}: {e}")
                    
            input_data["peaks/xyz"] = xyz_out
            input_data["peaks/two_theta"] = tt_out
            input_data["peaks/azimuthal"] = az_out
            
            if refine_det_flag:
                peak_pixel_coords = {
                    'u_offsets': u_offsets.tolist(),
                    'v_offsets': v_offsets.tolist(),
                    'bank_indices': bank_indices.tolist()
                }
        else:
            raise ValueError("ERROR: Input file does not contain peaks/pixel_r and peaks/pixel_c. Cannot perform physically sound indexing.")

    if "peaks/image_index" in input_data:
        input_data["peaks/run_index"] = input_data["peaks/image_index"]

    # --- INJECT SECOND PHASE OF BOOTSTRAP PHYSICS ---
    if _val(bootstrap_filename):
        with h5py.File(_val(bootstrap_filename), 'r') as b_f:
            if "sample/offset" in b_f: input_data["sample/offset"] = b_f["sample/offset"][()]
            if "beam/ki_vec" in b_f: ki_vec_val = b_f["beam/ki_vec"][()]

    input_data["sample/a"], input_data["sample/b"], input_data["sample/c"] = a_val, b_val, c_val
    input_data["sample/alpha"], input_data["sample/beta"], input_data["sample/gamma"] = alpha_val, beta_val, gamma_val
    input_data["sample/space_group"] = sg_val
    input_data["instrument/wavelength"] = [float(w_min_val), float(w_max_val)]
    input_data["beam/ki_vec"] = ki_vec_val

    opt = FindUB(data=input_data)
    opt.wavelength = [float(w_min_val), float(w_max_val)]

    if _val(bootstrap_filename):
        with h5py.File(_val(bootstrap_filename), 'r') as b_f:
            if "optimization/goniometer_offsets" in b_f: 
                opt.goniometer_offsets = b_f["optimization/goniometer_offsets"][()]

    print(f"Starting evosax optimization with strategy: {strat_val}")
    print(f"Running {runs_val} run(s)...")
    print(f"Settings per run: Population Size={pop_val}, Generations={gens_val}")
    if freeze_val: print("ORIENTATION LOCKED: U Matrix will not be refined.")
    if _val(refine_lattice): print(f"Refining lattice parameters with {_val(lattice_bound_frac) * 100}% bounds.")
    if _val(refine_sample): print(f"Refining sample offset with {1000 * _val(sample_bound_meters)} mm bounds.")
    if _val(refine_beam): print(f"Refining beam tilt with {_val(beam_bound_deg)}° bounds.")

    goniometer_names = None
    refine_gonio_flag = _val(refine_goniometer)
    if refine_gonio_flag:
        if _val(original_nexus_filename) and _val(instrument_name):
            print(f"Refining goniometer angles from geometry file with {_val(goniometer_bound_deg)} deg bounds.")
            
            is_merged = False
            with h5py.File(_val(original_nexus_filename), 'r') as f_check:
                if "images" in f_check and "goniometer/axes" in f_check:
                    is_merged = True
                    axes = f_check["goniometer/axes"][()]
                    angles = f_check["goniometer/angles"][()]
                    names = [n.decode('utf-8') for n in f_check["goniometer/names"][()]] if "goniometer/names" in f_check else None
            
            if not is_merged:
                axes, angles, names = get_rotation_data_from_nexus(_val(original_nexus_filename), _val(instrument_name))
                
            if len(axes) == 0:
                raise ValueError("ERROR: Could not extract goniometer axes from the provided nexus file.")
                
            opt.goniometer_axes = np.array(axes)

            # --- FIX: Secure Mapping of Angles to Peaks ---
            if opt.run_indices is not None:
                max_run_id = int(np.max(opt.run_indices))
                
                # Check if the file provided exactly enough angles for the max run ID
                if angles.ndim == 2 and angles.shape[0] > max_run_id:
                    # The file has a 1:1 mapping of image_index -> angle. Just transpose it for JAX.
                    opt.goniometer_angles = angles.T
                elif angles.ndim == 2 and angles.shape[0] == 1:
                    # The file only has 1 angle (e.g. a single snapshot). Broadcast it.
                    opt.goniometer_angles = np.tile(angles.T, (1, max_run_id + 1))
                else:
                    # If we hit here, something is critically mismatched in the file format
                    raise ValueError(f"CRITICAL: Angle shape {angles.shape} cannot map to {max_run_id + 1} runs.")
            else:
                num_peaks = len(opt.two_theta) if opt.two_theta is not None else 1
                if angles.ndim == 2 and angles.shape[0] == 1:
                    opt.goniometer_angles = np.tile(angles.T, (1, num_peaks))
                elif angles.ndim == 2 and angles.shape[0] == num_peaks:
                    opt.goniometer_angles = angles.T
                else:
                    raise ValueError(f"CRITICAL: Angle shape {angles.shape} cannot map to {num_peaks} peaks.")
                
            goniometer_names = names
            
        elif opt.goniometer_axes is not None:
            print(f"Refining goniometer angles from HDF5 file with {_val(goniometer_bound_deg)} deg bounds.")
            goniometer_names = opt.goniometer_names
        else:
            print("WARNING: refine_goniometer requested but goniometer data not found. Skipping.")
            refine_gonio_flag = False

    init_params = None
    if _val(bootstrap_filename):
        init_params = opt.get_bootstrap_params(
            _val(bootstrap_filename),
            freeze_orientation=freeze_val
        )

    num, hkl, lamda, U = opt.minimize(
        strategy_name=strat_val,
        population_size=pop_val,
        num_generations=gens_val,
        n_runs=runs_val,
        sigma_init=sigma_val,
        seed=seed_val,
        tolerance_deg=tol_val,
        init_params=init_params,
        refine_lattice=_val(refine_lattice),
        lattice_bound_frac=_val(lattice_bound_frac),
        refine_goniometer=refine_gonio_flag,
        refine_goniometer_axes=gonio_axes_list,
        goniometer_bound_deg=_val(goniometer_bound_deg),
        goniometer_names=goniometer_names,
        refine_sample=_val(refine_sample),
        sample_bound_meters=_val(sample_bound_meters),
        refine_beam=_val(refine_beam),
        beam_bound_deg=_val(beam_bound_deg),
        loss_method=_val(loss_method),
        d_min=_val(d_min),
        d_max=_val(d_max),
        batch_size=_val(batch_size),
        refine_detector=refine_det_flag,
        detector_params=detector_params,
        peak_pixel_coords=peak_pixel_coords,
        detector_trans_bound_meters=_val(detector_trans_bound_meters),
        detector_rot_bound_deg=_val(detector_rot_bound_deg),
        freeze_orientation=freeze_val,
    )

    print(f"\nOptimization complete. Best solution indexed {num} peaks.")
    opt.reciprocal_lattice_B()

    copy_keys = [
        "sample/space_group", "instrument/wavelength", "peaks/intensity",
        "peaks/sigma", "peaks/radius", "goniometer/R", "goniometer/axes", 
        "goniometer/angles", "goniometer/names", "files", "file_offsets", 
        "peaks/run_index", "peaks/image_index", "bank", "sample/offset", 
        "beam/ki_vec", "peaks/pixel_r", "peaks/pixel_c"
    ]

    copied_data = {}
    for key in copy_keys:
        if key in input_data:
            copied_data[key] = input_data[key]

    print(f"Saving indexed peaks to {output_peaks_filename}...")
    with h5py.File(output_peaks_filename, "w") as f:
        if _val(instrument_name):
            f.attrs["instrument"] = _val(instrument_name)
        elif "instrument" in input_data:
            f.attrs["instrument"] = input_data["instrument"]
            
        for key, value in copied_data.items():
            f[key] = value

        def safe_write(grp, name, data):
            if name in grp: del grp[name]
            grp[name] = data

        safe_write(f, "goniometer/R", opt.R)
        if opt.goniometer_offsets is not None: safe_write(f, "optimization/goniometer_offsets", opt.goniometer_offsets)
        if opt.sample_offset is not None: safe_write(f, "sample/offset", opt.sample_offset)
        if opt.ki_vec is not None: safe_write(f, "beam/ki_vec", opt.ki_vec)

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
        
        if opt.x is not None and opt.x.size > 0:
            f["optimization/best_params"] = opt.x
            
        import json
        flags = {
            "refine_lattice": _val(refine_lattice), "refine_goniometer": refine_gonio_flag,
            "refine_sample": _val(refine_sample), "refine_beam": _val(refine_beam),
            "refine_detector": refine_det_flag, "freeze_orientation": freeze_val,
        }
        f.create_dataset("optimization/flags", data=json.dumps(flags).encode('utf-8'))
        
        if refine_det_flag and hasattr(opt, 'calibrated_centers'):
            for b_idx, b_id in enumerate(target_banks):
                grp_name = f"detector_calibration/bank_{b_id}"
                f.create_group(grp_name)
                f[f"{grp_name}/center"] = opt.calibrated_centers[b_idx]
                f[f"{grp_name}/uhat"] = opt.calibrated_uhats[b_idx]
                f[f"{grp_name}/vhat"] = opt.calibrated_vhats[b_idx]
    print("Done.")



@app.command()
def metrics(
    file1: str = typer.Argument(..., help="Primary file (e.g., indexer.h5 or predictor.h5)"),
    file2: str | None = typer.Option(
        None,
        "--file2",
        help="Optional secondary file to match against (e.g., finder.h5).",
    ),
    instrument: str | None = typer.Option(
        None,
        "--instrument",
        help="Instrument name (required if using file2 or predictor outputs).",
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
    ki_vec: str = typer.Option(None, "--ki-vec", help="Override incident beam vector (e.g. '0,0,-1')"),
):
    """
    CLI command to compute and display indexing quality metrics.
    Compares HKL accuracy internally (file1), or spatial matching between file1 (predicted) and file2 (observed).
    """
    # dynamically shifts coordinates using the detector_calibration group.
    result = run_metrics(
        file1=file1,
        file2=file2,
        instrument=instrument,
        d_min=d_min,
        per_run=per_run,
        ki_vec_override=ki_vec_arr
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
    ki_vec: str = typer.Option(None, "--ki-vec", help="Override incident beam vector"),
    max_workers: int = 16,
):
    run_peak_predictor(
        filename,
        instrument,
        wavelength_min=wavelength[0],
        wavelength_max=wavelength[1],
    )

    print(
        f"Predicting peaks for {len(peaks.image.ims)} images using solution from {indexed_hdf5_filename}"
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
    space_group: str = typer.Option(None, help="Optional. Loaded from indexer h5 if missing."),
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

@app.command()
def zone_axis_search(
    merged_h5_filename: str,
    peaks_h5_filename: str,
    instrument: str,
    output_h5_filename: str,
    d_min: float = 1.0,
    sigma: Annotated[
        float, typer.Option(help="(Legacy) Replaced by vector_tolerance.")
    ] = 2.0,
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
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        space_group=space_group,
        d_min=d_min,
        sigma=sigma,
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

@app.command()
def index_images(
    merged_h5_filename: str,
    instrument: str,
    output_h5_filename: str,
    a: float = typer.Option(None, help="Override unit cell parameter a"), 
    b: float = typer.Option(None, help="Override unit cell parameter b"), 
    c: float = typer.Option(None, help="Override unit cell parameter c"),
    alpha: float = typer.Option(None, help="Override unit cell parameter alpha"), 
    beta: float = typer.Option(None, help="Override unit cell parameter beta"), 
    gamma: float = typer.Option(None, help="Override unit cell parameter gamma"), 
    space_group: str = typer.Option(None, help="Override Space group (e.g. 'P 1')"),
    ki_vec: str = typer.Option("0,0,1", "--ki-vec", help="Incident beam vector"),
    bootstrap: str = typer.Option(None, help="Seed with initial U matrix."),
    d_min: float = 1.0,
    sigma: float = 5.0,
    min_intensity: float = typer.Option(50.0, help="Minimum peak amplitude (photon counts)."),
    population_size: int = 1000,
    gens: int = 400,
    n_runs: int = 1,
    freeze_orientation: bool = typer.Option(False, "--freeze-orientation", help="Do not refine the U matrix over SO(3)."),
    batch_size: int = typer.Option(None, help="Number of runs to execute in parallel on GPU."),
    seed: int = 0,
    create_visualizations: bool = typer.Option(False, "--create-visualizations", help="Output PNG overlays of predicted vs extracted peaks."),
    border_frac: float = typer.Option(0.1, help="Fraction of image to crop at the border."),
):
    from subhkl.config import beamlines, reduction_settings
    ki_vec_arr = np.array([float(x.strip()) for x in ki_vec.split(",")])

    with h5py.File(merged_h5_filename, 'r') as f_in:
        U_initial = f_in["sample/U"][()] if "sample/U" in f_in else None

        if U_initial is None and gens == 0:
            U_initial = np.eye(3)

        file_bank_ids = f_in["bank_ids"][()]
        ax = f_in["goniometer/axes"][()]
        goniometer_angles = np.array(f_in["goniometer/angles"][()])

        from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix
        R_stack = np.stack([calc_goniometer_rotation_matrix(ax, ang) for ang in goniometer_angles])

        file_offsets = f_in["file_offsets"][()]
        file_names_in = list(f_in["files"].asstr())

        file_names = []
        if file_offsets[0] != 0:
            raise ValueError
        offsets_excl = np.concatenate([file_offsets[1:], [len(file_bank_ids)]])
        old_offs = 0
        for offs, f in zip(offsets_excl, file_names_in):
            file_names += [f] * (offs - old_offs)
            old_offs = offs
            
        file_a = f_in["sample/a"][()] if "sample/a" in f_in else None
        a_val = a if a is not None else file_a
        
        file_b = f_in["sample/b"][()] if "sample/b" in f_in else None
        b_val = b if b is not None else file_b
        
        file_c = f_in["sample/c"][()] if "sample/c" in f_in else None
        c_val = c if c is not None else file_c
        
        file_alpha = f_in["sample/alpha"][()] if "sample/alpha" in f_in else None
        alpha_val = alpha if alpha is not None else file_alpha
        
        file_beta = f_in["sample/beta"][()] if "sample/beta" in f_in else None
        beta_val = beta if beta is not None else file_beta
        
        file_gamma = f_in["sample/gamma"][()] if "sample/gamma" in f_in else None
        gamma_val = gamma if gamma is not None else file_gamma
        
        file_sg = f_in["sample/space_group"][()] if "sample/space_group" in f_in else None
        if file_sg is not None and isinstance(file_sg, bytes): file_sg = file_sg.decode('utf-8')
        sg_val = space_group if space_group is not None else file_sg
        
        if None in (a_val, b_val, c_val, alpha_val, beta_val, gamma_val, sg_val):
            raise ValueError("Unit cell parameters and Space Group must be present in the merged.h5 file or provided via CLI.")

        settings = reduction_settings[instrument]
        wavelength_min, wavelength_max = settings.get("Wavelength")

        images_raw = np.stack(f_in["images"][()])

    if bootstrap is not None:
        with h5py.File(bootstrap, 'r') as f_in:
            U_initial = f_in["sample/U"][()]

    from subhkl.optimization import FindUB

    medians = np.median(images_raw, axis=(1, 2), keepdims=True)
    images_bg = np.maximum(images_raw - medians, 0)

    import scipy.ndimage
    images_max = scipy.ndimage.maximum_filter(images_bg, size=3)

    images_landscape = np.zeros_like(images_bg, dtype=np.float32)
    for i in range(len(images_bg)):
        smoothed = scipy.ndimage.gaussian_filter(images_bg[i], sigma=1.0)
        local_max = scipy.ndimage.maximum_filter(smoothed, size=3) == smoothed
        mask = local_max & (smoothed > (min_intensity * 0.5))
        if not np.any(mask):
            continue
        dist = scipy.ndimage.distance_transform_edt(~mask)
        images_landscape[i] = np.exp(-dist / sigma)

    print("Generating theoretical HKL pool...")
    from subhkl.core.crystallography import generate_reflections
    h, k_idx, l = generate_reflections(a_val, b_val, c_val, alpha_val, beta_val, gamma_val, sg_val, d_min)
    hkl_pool = np.vstack([h, k_idx, l])

    ub_helper = FindUB()
    ub_helper.a, ub_helper.b, ub_helper.c = a_val, b_val, c_val
    ub_helper.alpha, ub_helper.beta, ub_helper.gamma = alpha_val, beta_val, gamma_val
    B_mat = ub_helper.reciprocal_lattice_B()

    det_centers, uhats, vhats = [], [], []
    widths, heights, ms, ns = [], [], [], []

    for i, phys_bank in enumerate(file_bank_ids):
        from subhkl.instrument.detector import Detector
        det_config = beamlines[instrument][str(phys_bank)]
        det = Detector(det_config)

        det_centers.append(det.center)
        if det.panel_type.value == "flat":
            uhats.append(det.uhat)
            vhats.append(det.vhat)
        else:
            raise NotImplementedError("Curved panels not yet supported in JAX sparse_laue.")

        widths.append(det.width)
        heights.append(det.height)
        ms.append(det.m)
        ns.append(det.n)

    data_dict = {
        'images_landscape': images_landscape,
        'hkl_pool': hkl_pool, 'B_mat': B_mat, 'R_stack': np.array(R_stack),
        'wl_min': wavelength_min, 'wl_max': wavelength_max,
        'det_centers': np.array(det_centers), 'uhats': np.array(uhats),
        'vhats': np.array(vhats), 'widths': np.array(widths), 'heights': np.array(heights),
        'ms': np.array(ms), 'ns': np.array(ns),
        'ki_vec': ki_vec_arr, 'sample_offset': np.zeros(3),
        'border_frac': border_frac,
    }

    from subhkl.optimization import ImageBasedFindUB
    indexer = ImageBasedFindUB(data_dict)

    injected_rots = None
    if U_initial is not None:
        print(f"Starting from provided U matrix...")

        from scipy.spatial.transform import Rotation as R
        u_rot = R.from_matrix(U_initial)
        rodrigues_vec = u_rot.as_rotvec()
        injected_rots = np.array([rodrigues_vec])

    if gens > 0 and not freeze_orientation:
        print(f"Starting Unified Sparse Laue Optimization over SO(3)...")
        print(f"  Images: {len(images_bg)} | Target HKLs: {hkl_pool.shape[1]}")
        opt_U, opt_params = indexer.minimize_evosax(
            "DE", population_size=population_size, num_generations=gens,
            seed=seed, batch_size=batch_size, n_runs=n_runs,
            injected_rotations=injected_rots
        )
    else:
        if freeze_orientation:
            print(f"Skipping SO(3) search (--freeze-orientation active). Integrating using provided U matrix...")
        else:
            print(f"Skipping SO(3) search (gens=0). Integrating using provided U matrix...")
        opt_U = U_initial
        opt_params = np.zeros(3)

    print("Extracting physical intensities from optimal orientation...")
    c_stars, rows, cols, lams, valids = indexer.get_reflections(np.array(opt_U), images_max)
    c_stars, valids = np.array(c_stars), np.array(valids)

    mask = (c_stars >= min_intensity) & valids
    batch_idx, hkl_idx = np.where(mask)

    final_volumes = c_stars[batch_idx, hkl_idx]
    final_rows = np.array(rows)[batch_idx, hkl_idx]
    final_cols = np.array(cols)[batch_idx, hkl_idx]
    final_lams = np.array(lams)[batch_idx, hkl_idx]
    final_hkls = hkl_pool[:, hkl_idx]
    final_banks = [file_bank_ids[b] for b in batch_idx]
    final_filenames = [file_names[b] for b in batch_idx]

    if create_visualizations:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        print("Generating diagnostic visualizations...")
        for f in range(len(images_raw)):
            valid_mask = valids[f]
            if not np.any(valid_mask):
                continue

            pred_r = np.array(rows)[f, valid_mask]
            pred_c = np.array(cols)[f, valid_mask]

            ext_mask = (c_stars[f] >= min_intensity) & valid_mask
            ext_r = np.array(rows)[f, ext_mask]
            ext_c = np.array(cols)[f, ext_mask]

            phys_bank = file_bank_ids[f]
            fn_in = file_names[f]

            import os
            fn_in = os.path.basename(fn_in)

            img_plot = np.maximum(images_bg[f], 1.0)
            fig, ax = plt.subplots(figsize=(10, 10))
            vmax = max(10.0, img_plot.max())
            ax.imshow(img_plot, norm=mcolors.LogNorm(vmin=1.0, vmax=vmax), cmap='binary', origin='lower')

            ax.scatter(pred_c, pred_r, marker='x', color='blue', s=30, alpha=0.5, label='Predicted HKLs')
            ax.scatter(ext_c, ext_r, facecolors='none', edgecolors='red', marker='o', s=150, linewidths=1.5, label=f'Extracted (>{min_intensity} counts)')

            ax.set_title(f"Sparse Laue - Run {fn_in}, Bank {phys_bank}")
            ax.legend(loc='upper right')

            fname = f"sparse_laue_run_{fn_in}_bank_{phys_bank}.png"
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)

    xyz_out = []
    for f, (row_idx, col_idx) in enumerate(zip(rows, cols)):
        from subhkl.instrument.detector import Detector
        phys_bank = file_bank_ids[f]
        det_config = beamlines[instrument][str(phys_bank)]
        det = Detector(det_config)
        valid_mask = valids[f]
        if not np.any(valid_mask):
            continue

        xyz_out.append(det.pixel_to_lab(np.array(row_idx)[valid_mask],
                                        np.array(col_idx)[valid_mask]))

    xyz_det = np.concatenate(xyz_out)

    print(f"Integration complete. Extracted {len(final_volumes)} valid reflections.")

    print(f"Saving to {output_h5_filename}...")
    with h5py.File(output_h5_filename, "w") as f:
        f["sample/U"] = np.array(opt_U)
        f["sample/B"] = B_mat

        f["sample/a"], f["sample/b"], f["sample/c"] = a_val, b_val, c_val
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = alpha_val, beta_val, gamma_val
        f["sample/space_group"] = sg_val.encode('utf-8')
        f["instrument/wavelength"] = [wavelength_min, wavelength_max]

        f["goniometer/R"] = R_stack
        f["beam/ki_vec"] = ki_vec_arr
        f["sample/offset"] = np.zeros(3)

        f["optimization/best_params"] = np.array(opt_params)

        if len(final_volumes) > 0:
            f["peaks/h"], f["peaks/k"], f["peaks/l"] = final_hkls[0], final_hkls[1], final_hkls[2]
            f["peaks/lambda"] = final_lams
            f["peaks/intensity"] = final_volumes
            f["peaks/sigma"] = np.full_like(final_volumes, sigma)
            f["peaks/xyz"] = xyz_det
            f["bank"] = final_banks
            f["filename"] = final_filenames
            f["peaks/pixel_r"] = final_rows
            f["peaks/pixel_c"] = final_cols

    print("Done.")

if __name__ == "__main__":
    app()
