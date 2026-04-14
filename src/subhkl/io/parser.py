import glob
import os

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
                    # Temporarily override the nominal config in memory
                    beamlines[instrument][bank_id]["center"] = calib_grp[bank_key]["center"][()].tolist()
                    beamlines[instrument][bank_id]["uhat"] = calib_grp[bank_key]["uhat"][()].tolist()
                    beamlines[instrument][bank_id]["vhat"] = calib_grp[bank_key]["vhat"][()].tolist()
                    count += 1
            if count > 0:
                print(f"Successfully applied calibration to {count} detector panels.")

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
            "peaks/pixel_r", "peaks/pixel_c" # <--- Must load pixels to rebuild XYZ dynamically
        ]
        for k in keys_to_load:
            if k in f: input_data[k] = f[k][()]

        # --- DYNAMICALLY RECONSTRUCT XYZ FROM PIXELS FOR THE OPTIMIZER ---
        if "peaks/pixel_r" in f and "peaks/pixel_c" in f:
            print("Reconstructing physical XYZ coordinates from pixels for physics optimization...")
            if not _val(instrument_name): raise ValueError("ERROR: Finder file contains pixels. You must provide --instrument to rebuild geometry.")
            if not _val(original_nexus_filename): raise ValueError("ERROR: Finder file contains pixels. You must provide --nexus to rebuild geometry.")
                
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
            
            calibration_dict = {}
            if "detector_calibration" in f:
                calib_grp = f["detector_calibration"]
                for b_key in calib_grp.keys():
                    calibration_dict[b_key] = {
                        "center": calib_grp[b_key]["center"][()],
                        "uhat": calib_grp[b_key]["uhat"][()],
                        "vhat": calib_grp[b_key]["vhat"][()]
                    }
                    
            xyz_out = np.zeros((len(pixel_r), 3))
            from subhkl.config import beamlines
            from subhkl.instrument.detector import Detector
            
            for phys_bank in np.unique(bank_array):
                mask = bank_array == phys_bank
                if not np.any(mask): continue
                    
                try:
                    det_config = beamlines[_val(instrument_name)][str(int(phys_bank))]
                    det = Detector(det_config)
                    
                    bank_str = f"bank_{int(phys_bank)}"
                    if bank_str in calibration_dict:
                        det.center = calibration_dict[bank_str]["center"]
                        det.uhat = calibration_dict[bank_str]["uhat"]
                        det.vhat = calibration_dict[bank_str]["vhat"]
                        
                    xyz_out[mask] = det.pixel_to_lab(pixel_r[mask], pixel_c[mask])
                except KeyError as e:
                    print(f"Warning: Could not rebuild XYZ for bank {phys_bank}: {e}")
                    
            # Inject xyz into input_data purely so FindUB can use it internally for the math.
            # We will NOT copy it to the output file.
            input_data["peaks/xyz"] = xyz_out
        else:
            raise ValueError("ERROR: Input file does not contain peaks/pixel_r and peaks/pixel_c. Cannot perform physically sound indexing.")

    if "peaks/image_index" in input_data:
        input_data["peaks/run_index"] = input_data["peaks/image_index"]

    input_data["sample/a"], input_data["sample/b"], input_data["sample/c"] = a_val, b_val, c_val
    input_data["sample/alpha"], input_data["sample/beta"], input_data["sample/gamma"] = alpha_val, beta_val, gamma_val
    input_data["sample/space_group"] = sg_val
    input_data["instrument/wavelength"] = [float(w_min_val), float(w_max_val)]

    if _val(ki_vec) is not None:
        input_data["beam/ki_vec"] = np.array([float(x.strip()) for x in _val(ki_vec).split(",")])

    opt = FindUB(data=input_data)
    opt.wavelength = [float(w_min_val), float(w_max_val)]

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
            print(f"Refining goniometer angles from Nexus with {_val(goniometer_bound_deg)} deg bounds.")
            axes, angles, names = get_rotation_data_from_nexus(_val(original_nexus_filename), _val(instrument_name))
            opt.goniometer_axes = np.array(axes)

            if opt.run_indices is not None:
                num_runs = np.max(opt.run_indices) + 1
                opt.goniometer_angles = np.array(angles)[:, np.newaxis].repeat(num_runs, axis=1)
            else:
                num_peaks = len(opt.two_theta)
                opt.goniometer_angles = np.array(angles)[:, np.newaxis].repeat(num_peaks, axis=1)
            goniometer_names = names
        elif opt.goniometer_axes is not None:
            print(f"Refining goniometer angles from HDF5 file with {_val(goniometer_bound_deg)} deg bounds.")
        else:
            print("WARNING: refine_goniometer requested but goniometer data not found. Skipping.")
            refine_gonio_flag = False

    detector_params = None
    peak_pixel_coords = None
    refine_det_flag = _val(refine_detector)
    target_banks = None

    if refine_det_flag:
        if not _val(original_nexus_filename) or not _val(instrument_name):
            print("WARNING: refine_detector requires nexus_filename and instrument_name to build geometry. Disabling.")
            refine_det_flag = False
        else:
            print(f"Joint Metrology Refinement active. Pipeline: {' -> '.join(det_modes_list)}")
            try:
                peaks_obj = Peaks(_val(original_nexus_filename), _val(instrument_name))
                
                from subhkl.config import beamlines
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
                
                if "peaks/xyz" in input_data:
                    xyz = input_data["peaks/xyz"]
                    rows, cols, bank_indices = [], [], []
                    
                    if "bank" in input_data: bank_array = input_data["bank"]
                    elif "peaks/bank" in input_data: bank_array = input_data["peaks/bank"]
                    else: bank_array = opt.run_indices
                        
                    for i_p in range(len(xyz)):
                        b_id = int(bank_array[i_p])
                        if b_id in bank_to_idx:
                            det = peaks_obj.get_detector(b_id)
                            r, c = det.lab_to_pixel(xyz[i_p, 0], xyz[i_p, 1], xyz[i_p, 2], clip=False)
                            rows.append(r)
                            cols.append(c)
                            bank_indices.append(bank_to_idx[b_id])
                        else:
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
                    refine_det_flag = False

            except Exception as e:
                print(f"WARNING: Failed to initialize joint detector refinement geometry: {e}")
                refine_det_flag = False

    init_params = None
    if _val(bootstrap_filename):
        init_params = opt.get_bootstrap_params(
            _val(bootstrap_filename),
            refine_lattice=_val(refine_lattice),
            lattice_bound_frac=_val(lattice_bound_frac),
            refine_sample=_val(refine_sample),
            sample_bound_meters=_val(sample_bound_meters),
            refine_beam=_val(refine_beam),
            beam_bound_deg=_val(beam_bound_deg),
            refine_goniometer=refine_gonio_flag,
            goniometer_bound_deg=_val(goniometer_bound_deg),
            refine_goniometer_axes=gonio_axes_list,
            freeze_orientation=freeze_val,
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

    # Strip physical coordinates from the copy operation
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
        
        if opt.x is not None and opt.x.size > 0:
            f["optimization/best_params"] = opt.x
        
        if refine_det_flag and hasattr(opt, 'calibrated_centers'):
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
    show_candidates: bool = False,
    peak_local_max_min_pixel_distance: int = -1,
    peak_local_max_min_relative_intensity: float = -1,
    peak_local_max_normalization: bool = False,
    mask_file: str | None = None,
    mask_rel_erosion_radius: float = None,
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
    sparse_rbf_loss: str = typer.Option("gaussian", help="Likelihood for peak finder."),
    sparse_rbf_auto_tune_alpha: bool = typer.Option(False, help="Auto-tune SNR threshold."),
    sparse_rbf_candidate_alphas: str = typer.Option("3.0,5.0,10.0,15.0,20.0,25.0,30", help="Candidate SNR thresholds alpha for auto-tuning"),
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
                "blur_kernel_sigma": thresholding_blur_kernel_sigma,
                "open_kernel_size_pixels": thresholding_open_kernel_size_pixels,
                "show_steps": show_steps,
                "show_scale": "log",
            }
        )
    elif finder_algorithm == "sparse_rbf":
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
            "show_candidates": show_candidates
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

    try:
        copy_keys = [
            "sample/a", "sample/b", "sample/c", 
            "sample/alpha", "sample/beta", "sample/gamma", 
            "sample/space_group"
        ]
        with h5py.File(output_filename, "a") as f_out:
            with h5py.File(filename, "r") as f_in:
                for key in copy_keys:
                    if key in f_in:
                        if key in f_out:
                            del f_out[key]
                        f_in.copy(f_in[key], f_out, key)
    except Exception as e:
        print(f"Warning: Could not forward embedded unit cell metadata: {e}")


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
    if hasattr(file2, "default"): file2 = file2.default
    if hasattr(instrument, "default"): instrument = instrument.default
    if hasattr(d_min, "default"): d_min = d_min.default
    if hasattr(per_run, "default"): per_run = per_run.default
    if hasattr(ki_vec, "default"): ki_vec = ki_vec.default

    ki_vec_arr = None
    if ki_vec is not None:
        ki_vec_arr = np.array([float(x.strip()) for x in ki_vec.split(",")])

    # No need to call apply_detector_calibration here because metrics.py
    # dynamically shifts coordinates using the detector_calibration group.
    result = compute_metrics(
        file1=file1,
        file2=file2,
        instrument=instrument,
        d_min=d_min,
        per_run=per_run,
        ki_vec_override=ki_vec_arr
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
    ki_vec: str = typer.Option(None, "--ki-vec", help="Override incident beam vector"),
    max_workers: int = 16,
):
    apply_detector_calibration(indexed_hdf5_filename, instrument)

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

        if ki_vec is not None:
            ki_vec_val = np.array([float(x.strip()) for x in ki_vec.split(",")])
        elif "beam/ki_vec" in f_idx:
            ki_vec_val = f_idx["beam/ki_vec"][()]
        else:
            ki_vec_val = np.array([0.0, 0.0, 1.0])

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
        ki_vec=ki_vec_val, max_workers=max_workers, R_all=all_R,
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
        f["beam/ki_vec"] = ki_vec_val

        for img_key, (i, j, h, k, l, wl) in results_map.items():  # noqa: E741
            grp = f.create_group(f"banks/{img_key}")
            grp.create_dataset("i", data=i)
            grp.create_dataset("j", data=j)
            grp.create_dataset("h", data=h)
            grp.create_dataset("k", data=k)
            grp.create_dataset("l", data=l)
            grp.create_dataset("wavelength", data=wl)

        # Forward the calibration group to the prediction file
        with h5py.File(indexed_hdf5_filename, "r") as f_in:
            if "detector_calibration" in f_in:
                f_in.copy("detector_calibration", f)


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
    found_peaks_file: str = None,
    max_workers: int = 16,
):
    apply_detector_calibration(integration_peaks_filename, instrument)

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

        if ki_vec is not None:
            ki_vec_val = np.array([float(x.strip()) for x in ki_vec.split(",")])
        elif "beam/ki_vec" in f:
            ki_vec_val = f["beam/ki_vec"][()]
        else:
            ki_vec_val = np.array([0.0, 0.0, 1.0])

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
        ki_vec=ki_vec_val,
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

            for k in ["goniometer/axes", "goniometer/names", "detector_calibration"]:
                if k in f_in:
                    f_in.copy(f_in[k], f, k)


@app.command()
def mtz_exporter(
    indexed_h5_filename: str,
    output_mtz_filename: str,
    space_group: str = typer.Option(None, help="Optional. Loaded from indexer h5 if missing."),
):
    sg = space_group
    if sg is None:
        with h5py.File(indexed_h5_filename, 'r') as f:
            if "sample/space_group" in f:
                raw_sg = f["sample/space_group"][()]
                sg = raw_sg.decode('utf-8') if isinstance(raw_sg, bytes) else str(raw_sg)
            else:
                raise ValueError("space_group must be provided as it is missing from the HDF5 file.")
                
    algorithm = MTZExporter(indexed_h5_filename, sg)
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
    a: float = typer.Argument(..., help="Unit cell parameter a"), 
    b: float = typer.Argument(..., help="Unit cell parameter b"), 
    c: float = typer.Argument(..., help="Unit cell parameter c"),
    alpha: float = typer.Argument(..., help="Unit cell parameter alpha"), 
    beta: float = typer.Argument(..., help="Unit cell parameter beta"), 
    gamma: float = typer.Argument(..., help="Unit cell parameter gamma"), 
    space_group: str = typer.Argument(..., help="Space group (e.g. 'P 1')"),
):
    
    try:
        get_space_group_object(space_group)
    except ValueError as e:
        print(f"ERROR: Invalid space group '{space_group}': {e}")
        raise typer.Exit(code=1)

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
    
    with h5py.File(output_filename, "a") as f:
        f["sample/a"] = a
        f["sample/b"] = b
        f["sample/c"] = c
        f["sample/alpha"] = alpha
        f["sample/beta"] = beta
        f["sample/gamma"] = gamma
        f["sample/space_group"] = space_group.encode('utf-8')

    print(f"Successfully created {output_filename} with cell constraints embedded.")


@app.command()
def zone_axis_search(
    merged_h5_filename: str,
    peaks_h5_filename: str,
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
    d_min: float = 1.0,
    sigma: float = typer.Option(2.0, help="(Legacy) Replaced by vector_tolerance."),
    vector_tolerance: float = typer.Option(0.15, help="Angular capture radius in degrees for the objective function."),
    border_frac: float = typer.Option(0.1, help="Fraction of image to crop at the border."),
    min_intensity: float = typer.Option(50.0, help="Minimum peak amplitude."),
    hough_grid_resolution: int = typer.Option(1024, help="Lambert grid resolution."),
    n_hough: int = typer.Option(15, help="Maximum number of empirical zone axes."),
    davenport_angle_tol: float = typer.Option(0.5, help="Graph search angle tolerance in degrees."),
    top_k_rays: int = typer.Option(15, help="Max rays per image to feed the Hough Transform."),
    max_uvw: int = typer.Option(25, help="Maximum uvw index for zone axis search"),
    L_max: float = typer.Option(250.0, help="Maximum real-space vector length for theoretical zone axes (Angstroms)."),
    top_k: int = typer.Option(1000, help="Maximum number of reciprocal grid points to consider."),
    num_runs: int = typer.Option(0, help="Number of goniometer runs to use. Set to 0 to use all."),
    output_hough: str = typer.Option(None, help="Diagnostic hough transform image filename."),
    batch_size: int = typer.Option(1024, help="Batch size for validation loop"),
):
    """
    Global Zone-Axis Search to find the macroscopic crystal orientation (U matrix).
    Outputs an HDF5 file that can be passed directly to 'indexer --bootstrap'.
    """
    import jax.numpy as jnp
    from subhkl.config import reduction_settings
    from subhkl.optimization import VectorizedObjective
    from subhkl.search.prior import HoughPrior

    print(f"Loading data from {merged_h5_filename}...")
    with h5py.File(merged_h5_filename, 'r') as f_in:
        file_bank_ids = list(int(bid) for bid in f_in["bank_ids"])
        ax = f_in["goniometer/axes"][()]
        goniometer_angles = np.array(f_in["goniometer/angles"][()])

        R_stack = np.stack([calc_goniometer_rotation_matrix(ax, ang) for ang in goniometer_angles])
        file_offsets = f_in["file_offsets"][()]
        
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

    if num_runs > 0:
        if len(file_offsets) > num_runs:
            end_idx = file_offsets[num_runs]
        else:
            end_idx = len(file_bank_ids)
            num_runs = len(file_offsets)

        print(f"Limiting search to the first {num_runs} run(s) (Images 0 to {end_idx-1})...")
        file_bank_ids = file_bank_ids[:end_idx]
        R_stack = R_stack[:end_idx]
    else:
        end_idx = len(file_bank_ids)
        print(f"Using all {len(file_offsets)} available runs for the search...")

    settings = reduction_settings[instrument]
    wavelength_min, wavelength_max = settings.get("Wavelength")

    ub_helper = FindUB()
    ub_helper.a, ub_helper.b, ub_helper.c = a_val, b_val, c_val
    ub_helper.alpha, ub_helper.beta, ub_helper.gamma = alpha_val, beta_val, gamma_val
    B_mat = ub_helper.reciprocal_lattice_B()

    with h5py.File(peaks_h5_filename, 'r') as f_peaks:
        ki_vec_override = None if ki_vec == "0,0,1" else np.array([float(x.strip()) for x in ki_vec.split(",")])
        if ki_vec_override is not None:
            ki_vec_val = ki_vec_override
        elif "beam/ki_vec" in f_peaks:
            ki_vec_val = f_peaks["beam/ki_vec"][()]
        else:
            ki_vec_val = np.array([0.0, 0.0, 1.0])

    print("\n--- HOUGH PRIOR GENERATION ---")
    prior_engine = HoughPrior(B_mat, np.array(R_stack), ki_vec=ki_vec_val)

    print(f"Loading empirical rays from {peaks_h5_filename}...")
    
    with h5py.File(peaks_h5_filename, 'r') as f_peaks:
        peaks_xyz = f_peaks["peaks/xyz"][()]
        peaks_intensity = f_peaks["peaks/intensity"][()]

        if "peaks/image_index" in f_peaks:
            group_indices = f_peaks["peaks/image_index"][()]
        else:
            group_indices = f_peaks["peaks/run_index"][()]

        R_peaks_override = f_peaks.get("goniometer/R")
        if R_peaks_override is not None:
            R_peaks_override = R_peaks_override[()]

    q_hat_list, q_lab_list, peaks_xyz_list, intensities_list, mapped_bank_indices = [], [], [], [], []

    unique_groups = np.unique(group_indices)
    for g_idx in unique_groups:
        if g_idx >= end_idx:
            continue

        mask = group_indices == g_idx
        grp_xyz = peaks_xyz[mask]
        grp_intensity = peaks_intensity[mask]

        if R_peaks_override is not None:
            if R_peaks_override.ndim == 3 and R_peaks_override.shape[0] == len(peaks_xyz):
                R_gonio = R_peaks_override[mask][0]
            elif R_peaks_override.ndim == 3 and R_peaks_override.shape[0] > g_idx:
                R_gonio = R_peaks_override[g_idx]
            else:
                R_gonio = R_peaks_override
        else:
            R_gonio = R_stack[g_idx] if g_idx < len(R_stack) else np.eye(3)

        intensity_mask = grp_intensity >= min_intensity
        if not np.any(intensity_mask): 
            continue
            
        grp_xyz = grp_xyz[intensity_mask]
        grp_intensity = grp_intensity[intensity_mask]

        top_k_idx = np.argsort(grp_intensity)[::-1][:min(top_k_rays, len(grp_intensity))]
        grp_xyz_top = grp_xyz[top_k_idx]
        grp_intensity_top = grp_intensity[top_k_idx]

        kf = grp_xyz_top / np.linalg.norm(grp_xyz_top, axis=1, keepdims=True)
        q_lab = kf - ki_vec_val[None, :]
        q_sample = np.dot(q_lab, R_gonio)

        q_norms = np.linalg.norm(q_sample, axis=1, keepdims=True)
        q_hat_grp = q_sample / q_norms
        
        q_hat_list.append(q_hat_grp)
        q_lab_list.append(q_lab)
        peaks_xyz_list.append(grp_xyz_top)
        intensities_list.append(grp_intensity_top)
        
        mapped_bank_indices.append(np.full(len(grp_xyz_top), g_idx))

    if not q_hat_list:
        print("Failed to extract any valid rays from the peaks file. Check your --min-intensity threshold.")
        return

    q_hat = np.vstack(q_hat_list)
    q_lab_all = np.vstack(q_lab_list).T  
    peaks_xyz_all = np.vstack(peaks_xyz_list).T 
    intensities_all = np.concatenate(intensities_list)
    
    bank_indices_all = np.concatenate(mapped_bank_indices)

    median_intensity = np.median(intensities_all)
    weights_all = intensities_all / (median_intensity + 1e-6)
    weights_all = np.clip(weights_all, 0.0, 10.0)

    print(f"Extracted {len(q_hat)} physical rays. Running 3D Combinatorial Hough...")
    n_obs, weights_obs = prior_engine.compute_hough_accumulator(
        q_hat, grid_resolution=hough_grid_resolution, n_hough=n_hough,
        plot_filename=output_hough, border_frac=border_frac
    )

    if len(n_obs) == 0:
        return

    n_calc = prior_engine.generate_theoretical_zones(L_max=L_max, top_k=top_k, max_uvw=max_uvw)
    print(f"Extracted {len(n_obs)} Empirical Zones against {len(n_calc)} Theoretical Zones.")

    quats, _ = prior_engine.solve_permutations(
        jnp.array(n_obs), jnp.array(weights_obs), n_calc, q_hat,
        space_group=sg_val,
        angle_tol_deg=davenport_angle_tol,
        scoring_tol_deg=vector_tolerance,
        d_min=d_min
    )

    if quats is None or len(quats) == 0:
        return

    print(f"Filtering Prior through Exact Physics Forward-Model...")

    ray_objective = VectorizedObjective(
        B=B_mat,
        kf_ki_dir=q_lab_all,
        peak_xyz_lab=peaks_xyz_all,
        wavelength=[wavelength_min, wavelength_max],
        cell_params=[a_val, b_val, c_val, alpha_val, beta_val, gamma_val],
        static_R=R_stack,
        peak_run_indices=bank_indices_all,
        beam_nominal=ki_vec_val
    )

    prior_rots = prior_engine.physics_filter(quats, ray_objective, batch_size=batch_size, z_score_threshold=3.0)

    if prior_rots is None or len(prior_rots) == 0:
        print("Exact physical model rejected all seeds. Exiting.")
        return

    print(f"Success! Saving optimal seed to {output_h5_filename}...")
    with h5py.File(output_h5_filename, "w") as f:
        best_rot = np.array(prior_rots[0])
        f.create_dataset("optimization/best_params", data=best_rot)

        from subhkl.optimization import rotation_matrix_from_rodrigues_jax
        U_matrix = np.array(rotation_matrix_from_rodrigues_jax(best_rot))
        f.create_dataset("sample/U", data=U_matrix)
        f.create_dataset("sample/B", data=B_mat)

        f.create_dataset("sample/a", data=a_val)
        f.create_dataset("sample/b", data=b_val)
        f.create_dataset("sample/c", data=c_val)
        f.create_dataset("sample/alpha", data=alpha_val)
        f.create_dataset("sample/beta", data=beta_val)
        f.create_dataset("sample/gamma", data=gamma_val)

        f.create_dataset("sample/offset", data=np.zeros(3))
        f.create_dataset("beam/ki_vec", data=ki_vec_val)
        f.create_dataset("optimization/goniometer_offsets", data=np.zeros(len(ax)))
        f.create_dataset("sample/space_group", data=sg_val.encode('utf-8'))
        f.create_dataset("instrument/wavelength", data=[wavelength_min, wavelength_max])

    print(f"Done. You can now run:\n subhkl indexer {merged_h5_filename} <output.h5> --bootstrap {output_h5_filename} ...")

@app.command()
def rbf_integrator(
    filename: str = typer.Argument(..., help="Merged HDF5 image stack"),
    instrument: str = typer.Argument(..., help="Instrument name"),
    integration_peaks_filename: str = typer.Argument(..., help="Predicted peaks HDF5 file"),
    output_filename: str = typer.Argument(..., help="Output integrated peaks HDF5 file"),
    alpha: float = typer.Option(1.0, "--alpha", help="Peak over background threshold (Z-score)"),
    gamma: float = typer.Option(1.0, "--gamma", help="Besov space weight exponent"),
    sigmas: str = typer.Option("1.0,2.0,4.0", help="Unstretched peak radii"),
    nominal_sigma: float = typer.Option(1.0, help="The typical peak radius, used as a fallback for weak reflections"),
    anisotropic: bool = typer.Option(False, help="Integrate anisotropic quasi-Laue peaks"),
    fit_mosaicity: bool = typer.Option(False, help="Whether to fit the mosaicity separately from sample dimensions to explain peak shape. Only use in non-spherical detector geometries."),
    max_peaks: int = typer.Option(500, "--max-peaks", help="Maximum peaks per panel (used for JAX matrix padding)"),
    rel_border_width: float = typer.Option(0, help="Border width in fraction of image size"),
    ki_vec: str = typer.Option(None, "--ki-vec", help="Override incident beam vector"),
    show_progress: bool = typer.Option(True, "--show-progress"),
    create_visualizations: bool = False,
    chunk_size: int = 256,
    max_workers: int = typer.Option(None, help="Maximum number of CPU tasks for visualization."),
):
    import h5py
    from subhkl.integration import Peaks
    from subhkl.search.sparse_rbf import integrate_peaks_rbf_ssn

    apply_detector_calibration(integration_peaks_filename, instrument)

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
            
        if ki_vec is not None:
            ki_vec_val = np.array([float(x.strip()) for x in ki_vec.split(",")])
        elif "beam/ki_vec" in f:
            ki_vec_val = f["beam/ki_vec"][()]
        else:
            ki_vec_val = np.array([0.0, 0.0, 1.0])

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
        peaks_obj=peaks,             
        alpha=alpha,
        sigmas=sigma_list,
        gamma=gamma,
        nominal_sigma=nominal_sigma,
        max_peaks=max_peaks,
        show_progress=show_progress,
        all_R=all_R,                 
        sample_offset=sample_offset,
        ki_vec=ki_vec_val,
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
        f["peaks/sigma"] = result.sigma 
        f["peaks/two_theta"] = result.tt
        f["peaks/azimuthal"] = result.az
        f["peaks/bank"] = result.bank
        f["peaks/run_index"] = result.run_id
        
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
                    
            for k in ["goniometer/axes", "goniometer/names", "detector_calibration"]:
                if k in f_in:
                    f_in.copy(f_in[k], f, k)

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
