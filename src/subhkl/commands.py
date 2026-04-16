import h5py
import numpy as np

# NOTE(Vivek): deprecate and use Goniometer class to handler rotation calc
from subhkl.instrument.goniometer import (
    get_rotation_data_from_nexus,
)
from subhkl.integration import Peaks
from subhkl.optimization import FindUB
from subhkl.io.export import ImageStackMerger, MTZExporter

from typing import List


def apply_detector_calibration(hdf5_filename: str, instrument: str):
    """
    Reads refined detector metrology from an indexer/prediction file (if present)
    and overrides the in-memory beamlines configuration so downstream
    tasks natively use the calibrated geometry.
    """
    from subhkl.config import beamlines
    import os

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


def run_index(
    peaks_h5_filename: str,
    output_peaks_filename: str,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    space_group: str | None = None,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    ki_vec: list[float] | np.ndarray | None = None,
    original_nexus_filename: str | None = None,
    instrument_name: str | None = None,
    strategy_name: str = "DE",
    sigma_init: float | None = None,
    n_runs: int = 1,
    population_size: int = 1000,
    gens: int = 100,
    seed: int = 0,
    tolerance_deg: float = 0.1,
    freeze_orientation: bool = False,
    refine_lattice: bool = False,
    lattice_bound_frac: float = 0.05,
    refine_goniometer: bool = False,
    refine_goniometer_axes: list[str] | None = None,
    goniometer_bound_deg: float = 5.0,
    refine_sample: bool = False,
    sample_bound_meters: float = 0.005,
    refine_beam: bool = False,
    beam_bound_deg: float = 1.0,
    refine_detector: bool = False,
    refine_detector_banks: list[int] | None = None,
    detector_modes: list[str] | None = None,
    detector_trans_bound_meters: float = 0.005,
    detector_rot_bound_deg: float = 1.0,
    detector_global_rot_bound_deg: float = 2.0,
    detector_global_rot_axis: list[float] | np.ndarray | None = None,
    detector_global_trans_bound_meters: float = 0.01,
    detector_radial_bound_frac: float = 0.05,
    bootstrap_filename: str | None = None,
    batch_size: int | None = None,
    loss_method: str = "cosine",
    d_min: float | None = None,
    d_max: float | None = None,
    input_data: dict | None = None,
):
    input_data = input_data or {}

    if detector_modes is None:
        detector_modes = ["independent"]
    if detector_global_rot_axis is None:
        detector_global_rot_axis = [0.0, 1.0, 0.0]

    # --- INJECT BOOTSTRAP PHYSICS DIRECTLY ---
    if bootstrap_filename:
        apply_detector_calibration(bootstrap_filename, instrument_name)
        with h5py.File(bootstrap_filename, "r") as b_f:
            if "sample/a" in b_f:
                a = b_f["sample/a"][()]
            if "sample/b" in b_f:
                b = b_f["sample/b"][()]
            if "sample/c" in b_f:
                c = b_f["sample/c"][()]
            if "sample/alpha" in b_f:
                alpha = b_f["sample/alpha"][()]
            if "sample/beta" in b_f:
                beta = b_f["sample/beta"][()]
            if "sample/gamma" in b_f:
                gamma = b_f["sample/gamma"][()]

    print(f"Loading peaks from: {peaks_h5_filename}")
    with h5py.File(peaks_h5_filename, "r") as f:
        if a is None:
            a = f["sample/a"][()] if "sample/a" in f else None
        if b is None:
            b = f["sample/b"][()] if "sample/b" in f else None
        if c is None:
            c = f["sample/c"][()] if "sample/c" in f else None
        if alpha is None:
            alpha = f["sample/alpha"][()] if "sample/alpha" in f else None
        if beta is None:
            beta = f["sample/beta"][()] if "sample/beta" in f else None
        if gamma is None:
            gamma = f["sample/gamma"][()] if "sample/gamma" in f else None

        if space_group is None:
            file_sg = f["sample/space_group"][()] if "sample/space_group" in f else None
            space_group = (
                file_sg.decode("utf-8") if isinstance(file_sg, bytes) else file_sg
            )

        if None in (a, b, c, alpha, beta, gamma, space_group):
            raise ValueError(
                "Unit cell parameters (a,b,c,alpha,beta,gamma) and Space Group must be provided via CLI or exist in the input file."
            )

        from subhkl.core.spacegroup import get_space_group_object

        try:
            get_space_group_object(space_group)
        except ValueError as e:
            raise ValueError(f"Invalid space group '{space_group}': {e}")

        if wavelength_min is None or wavelength_max is None:
            if "instrument/wavelength" in f:
                wl = f["instrument/wavelength"][()]
                if wavelength_min is None:
                    wavelength_min = float(wl[0])
                if wavelength_max is None:
                    wavelength_max = float(wl[1])
            else:
                raise ValueError(
                    "Wavelength min/max not provided and not found in input file."
                )

        keys_to_load = [
            "peaks/intensity",
            "peaks/sigma",
            "peaks/radius",
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
            "peaks/pixel_r",
            "peaks/pixel_c",
        ]
        for k in keys_to_load:
            if k in f:
                input_data[k] = f[k][()]

        if ki_vec is not None:
            ki_vec_val = np.array(ki_vec)
        else:
            ki_vec_val = (
                f["beam/ki_vec"][()]
                if "beam/ki_vec" in f
                else np.array([0.0, 0.0, 1.0])
            )

        detector_params = None
        peak_pixel_coords = None
        target_banks = None

        if "peaks/pixel_r" in f and "peaks/pixel_c" in f:
            print("Reconstructing physical geometry from pixels for optimization...")
            if not instrument_name or not original_nexus_filename:
                raise ValueError(
                    "ERROR: Finder file contains pixels. You must provide --instrument and --nexus to rebuild geometry."
                )

            pixel_r = f["peaks/pixel_r"][()]
            pixel_c = f["peaks/pixel_c"][()]

            bank_array = None
            if "bank" in f:
                bank_array = f["bank"][()]
            elif "peaks/bank" in f:
                bank_array = f["peaks/bank"][()]
            elif "bank_ids" in f and "peaks/image_index" in f:
                b_ids = f["bank_ids"][()]
                img_idx = f["peaks/image_index"][()]
                bank_array = np.array([b_ids[int(idx)] for idx in img_idx])
            else:
                bank_array = f["peaks/image_index"][()]

            peaks_obj = Peaks(original_nexus_filename, instrument_name)
            from subhkl.config import beamlines
            from subhkl.instrument.detector import Detector

            if refine_detector:
                all_physical_banks = [int(k) for k in beamlines[instrument_name].keys()]
                target_banks = (
                    refine_detector_banks
                    if refine_detector_banks
                    else sorted(all_physical_banks)
                )

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
                    "centers": centers,
                    "uhats": uhats,
                    "vhats": vhats,
                    "m": m,
                    "n": n,
                    "pw": pw,
                    "ph": ph,
                    "modes": detector_modes,
                    "radial_bound": detector_radial_bound_frac,
                    "global_rot_bound_deg": detector_global_rot_bound_deg,
                    "global_rot_axis": np.array(detector_global_rot_axis),
                    "global_trans_bound_meters": detector_global_trans_bound_meters,
                }

            xyz_out = np.zeros((len(pixel_r), 3))
            tt_out = np.zeros(len(pixel_r))
            az_out = np.zeros(len(pixel_r))

            u_offsets = np.zeros(len(pixel_r))
            v_offsets = np.zeros(len(pixel_r))
            bank_indices = np.zeros(len(pixel_r), dtype=np.int32)

            for phys_bank in np.unique(bank_array):
                mask = bank_array == phys_bank
                if not np.any(mask):
                    continue

                try:
                    det_config = beamlines[instrument_name][str(int(phys_bank))]
                    det = Detector(det_config)

                    xyz_p = det.pixel_to_lab(pixel_r[mask], pixel_c[mask])
                    xyz_out[mask] = xyz_p

                    tt_out[mask], az_out[mask] = det.pixel_to_angles(
                        pixel_r[mask], pixel_c[mask], ki_vec=ki_vec_val
                    )

                    if refine_detector and int(phys_bank) in bank_to_idx:
                        bank_indices[mask] = bank_to_idx[int(phys_bank)]
                        u_offsets[mask] = np.dot(xyz_p - det.center, det.uhat)
                        v_offsets[mask] = np.dot(xyz_p - det.center, det.vhat)

                except KeyError as e:
                    print(
                        f"Warning: Could not rebuild geometry for bank {phys_bank}: {e}"
                    )

            input_data["peaks/xyz"] = xyz_out
            input_data["peaks/two_theta"] = tt_out
            input_data["peaks/azimuthal"] = az_out

            if refine_detector:
                peak_pixel_coords = {
                    "u_offsets": u_offsets.tolist(),
                    "v_offsets": v_offsets.tolist(),
                    "bank_indices": bank_indices.tolist(),
                }
        else:
            raise ValueError(
                "ERROR: Input file does not contain peaks/pixel_r and peaks/pixel_c. Cannot perform physically sound indexing."
            )

    if "peaks/image_index" in input_data:
        input_data["peaks/run_index"] = input_data["peaks/image_index"]

    # --- INJECT SECOND PHASE OF BOOTSTRAP PHYSICS ---
    if bootstrap_filename:
        with h5py.File(bootstrap_filename, "r") as b_f:
            if "sample/offset" in b_f:
                input_data["sample/offset"] = b_f["sample/offset"][()]
            if "beam/ki_vec" in b_f:
                ki_vec_val = b_f["beam/ki_vec"][()]

    input_data["sample/a"], input_data["sample/b"], input_data["sample/c"] = a, b, c
    (
        input_data["sample/alpha"],
        input_data["sample/beta"],
        input_data["sample/gamma"],
    ) = alpha, beta, gamma
    input_data["sample/space_group"] = space_group
    input_data["instrument/wavelength"] = [float(wavelength_min), float(wavelength_max)]
    input_data["beam/ki_vec"] = ki_vec_val

    opt = FindUB(data=input_data)
    opt.wavelength = [float(wavelength_min), float(wavelength_max)]

    if bootstrap_filename:
        with h5py.File(bootstrap_filename, "r") as b_f:
            if "optimization/goniometer_offsets" in b_f:
                opt.goniometer_offsets = b_f["optimization/goniometer_offsets"][()]

    print(f"Starting evosax optimization with strategy: {strategy_name}")
    print(f"Running {n_runs} run(s)...")
    print(f"Settings per run: Population Size={population_size}, Generations={gens}")
    if freeze_orientation:
        print("ORIENTATION LOCKED: U Matrix will not be refined.")
    if refine_lattice:
        print(f"Refining lattice parameters with {lattice_bound_frac * 100}% bounds.")
    if refine_sample:
        print(f"Refining sample offset with {1000 * sample_bound_meters} mm bounds.")
    if refine_beam:
        print(f"Refining beam tilt with {beam_bound_deg}° bounds.")

    goniometer_names = None
    if refine_goniometer:
        if original_nexus_filename and instrument_name:
            print(
                f"Refining goniometer angles from geometry file with {goniometer_bound_deg} deg bounds."
            )

            is_merged = False
            with h5py.File(original_nexus_filename, "r") as f_check:
                if "images" in f_check and "goniometer/axes" in f_check:
                    is_merged = True
                    axes = f_check["goniometer/axes"][()]
                    angles = f_check["goniometer/angles"][()]
                    names = (
                        [n.decode("utf-8") for n in f_check["goniometer/names"][()]]
                        if "goniometer/names" in f_check
                        else None
                    )

            if not is_merged:
                axes, angles, names = get_rotation_data_from_nexus(
                    original_nexus_filename, instrument_name
                )

            if len(axes) == 0:
                raise ValueError(
                    "ERROR: Could not extract goniometer axes from the provided nexus file."
                )

            opt.goniometer_axes = np.array(axes)

            if opt.run_indices is not None:
                max_run_id = int(np.max(opt.run_indices))
                num_peaks = len(opt.run_indices)
                num_axes = len(opt.goniometer_axes)

                # 1. Force the angles matrix to be (num_axes, num_runs/peaks)
                if angles.ndim == 2:
                    if angles.shape[0] == num_axes:
                        pass  # Already correct
                    elif angles.shape[1] == num_axes:
                        angles = angles.T
                    else:
                        # Ambiguous fallback
                        if angles.shape[0] == max_run_id + 1 or angles.shape[0] == num_peaks:
                            angles = angles.T

                num_angles_provided = angles.shape[1] if angles.ndim == 2 else len(angles)

                # 2. Auto-expand run_indices if we have exactly 1 angle per peak but flat indices
                if num_angles_provided == num_peaks and max_run_id == 0 and num_peaks > 1:
                    opt.run_indices = np.arange(num_peaks, dtype=np.int32)
                    max_run_id = num_peaks - 1

                # 3. Assign the mapped angles
                if num_angles_provided > max_run_id:
                    opt.goniometer_angles = angles
                elif num_angles_provided == 1:
                    opt.goniometer_angles = np.tile(angles, (1, max_run_id + 1))
                else:
                    raise ValueError(f"CRITICAL: Angle shape {angles.shape} cannot map to {max_run_id + 1} runs.")
            else:
                num_peaks = len(opt.two_theta) if opt.two_theta is not None else 1
                num_axes = len(opt.goniometer_axes)
                
                if angles.ndim == 2 and angles.shape[1] == num_axes:
                    angles = angles.T
                    
                num_angles_provided = angles.shape[1] if angles.ndim == 2 else len(angles)
                
                if num_angles_provided == num_peaks:
                    opt.goniometer_angles = angles
                elif num_angles_provided == 1:
                    opt.goniometer_angles = np.tile(angles, (1, num_peaks))
                else:
                    raise ValueError(f"CRITICAL: Angle shape {angles.shape} cannot map to {num_peaks} peaks.")

            goniometer_names = names

        elif opt.goniometer_axes is not None:
            print(
                f"Refining goniometer angles from HDF5 file with {goniometer_bound_deg} deg bounds."
            )
            goniometer_names = opt.goniometer_names
        else:
            print(
                "WARNING: refine_goniometer requested but goniometer data not found. Skipping."
            )
            refine_goniometer = False

    init_params = None
    if bootstrap_filename:
        init_params = opt.get_bootstrap_params(
            refine_goniometer_axes=refine_goniometer_axes,
            bootstrap_filename=bootstrap_filename,
            freeze_orientation=freeze_orientation,
        )

    num, hkl, lamda, U = opt.minimize(
        strategy_name=strategy_name,
        population_size=population_size,
        num_generations=gens,
        n_runs=n_runs,
        sigma_init=sigma_init,
        seed=seed,
        init_params=init_params,
        goniometer_bound_deg=goniometer_bound_deg,
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        refine_goniometer=refine_goniometer,
        refine_goniometer_axes=refine_goniometer_axes,
        goniometer_names=goniometer_names,
        refine_sample=refine_sample,
        sample_bound_meters=sample_bound_meters,
        refine_beam=refine_beam,
        beam_bound_deg=beam_bound_deg,
        d_min=d_min,
        d_max=d_max,
        batch_size=batch_size,
        refine_detector=refine_detector,
        detector_params=detector_params,
        peak_pixel_coords=peak_pixel_coords,
        detector_trans_bound_meters=detector_trans_bound_meters,
        detector_rot_bound_deg=detector_rot_bound_deg,
        freeze_orientation=freeze_orientation,
    )

    print(f"\nOptimization complete. Best solution indexed {num} peaks.")
    opt.reciprocal_lattice_B()

    copy_keys = [
        "sample/space_group",
        "instrument/wavelength",
        "peaks/intensity",
        "peaks/sigma",
        "peaks/radius",
        "goniometer/R",
        "goniometer/axes",
        "goniometer/angles",
        "goniometer/names",
        "files",
        "file_offsets",
        "peaks/run_index",
        "peaks/image_index",
        "bank",
        "sample/offset",
        "beam/ki_vec",
        "peaks/pixel_r",
        "peaks/pixel_c",
    ]

    copied_data = {}
    for key in copy_keys:
        if key in input_data:
            copied_data[key] = input_data[key]

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

        if opt.x is not None and opt.x.size > 0:
            f["optimization/best_params"] = opt.x

        import json

        flags = {
            "refine_lattice": refine_lattice,
            "refine_goniometer": refine_goniometer,
            "refine_sample": refine_sample,
            "refine_beam": refine_beam,
            "refine_detector": refine_detector,
            "freeze_orientation": freeze_orientation,
        }
        f.create_dataset("optimization/flags", data=json.dumps(flags).encode("utf-8"))

        if refine_detector and hasattr(opt, "calibrated_centers"):
            for b_idx, b_id in enumerate(target_banks):
                grp_name = f"detector_calibration/bank_{b_id}"
                f.create_group(grp_name)
                f[f"{grp_name}/center"] = opt.calibrated_centers[b_idx]
                f[f"{grp_name}/uhat"] = opt.calibrated_uhats[b_idx]
                f[f"{grp_name}/vhat"] = opt.calibrated_vhats[b_idx]
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


def run_metrics(
    file1: str,
    file2: str | None = None,
    instrument: str | None = None,
    d_min: float | None = None,
    per_run: bool = False,
    ki_vec: List[float] | np.ndarray = None,
):
    from subhkl.instrument.metrics import compute_metrics

    # No need to call apply_detector_calibration here because metrics.py
    # dynamically shifts coordinates using the detector_calibration group.
    result = compute_metrics(
        file1=file1,
        file2=file2,
        instrument=instrument,
        d_min=d_min,
        per_run=per_run,
        ki_vec_override=ki_vec,
    )

    if "error_message" in result:
        print(result["error_message"])
        if result["error_message"].startswith("Exception"):
            print("METRICS: 9.99 9.99 9.99 9.99 9.99 9.99")
        return

    if "filter_message" in result:
        print(f"METRICS: {result['filter_message']}")

    # Print main metrics
    print(
        f"METRICS: {result['median_d_err']:.5f} {result['mean_d_err']:.5f} {result['max_d_err']:.5f} "
        f"{result['median_ang_err']:.5f} {result['mean_ang_err']:.5f} {result['max_ang_err']:.5f}"
    )

    # Print per-run metrics if requested
    if per_run and "per_run_errors" in result:
        print("\nPER-RUN MEDIAN ANGULAR ERROR (deg) - Sorted by error:")
        for r, err, count in result["per_run_errors"]:
            status = "BAD" if err > 1.0 else "OK"
            print(f"  Run {r:4d}: {err:6.3f} ({count:4d} peaks) [{status}]")


def run_peak_predictor(
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
        sample_offset = (
            f_idx["sample/offset"][()] if "sample/offset" in f_idx else np.zeros(3)
        )
        ki_vec = (
            f_idx["beam/ki_vec"][()]
            if "beam/ki_vec" in f_idx
            else np.array([0.0, 0.0, 1.0])
        )

    peaks = Peaks(
        filename, instrument, wavelength_min=wavelength[0], wavelength_max=wavelength[1]
    )
    print(
        f"Predicting peaks for {len(peaks.image.ims)} images using solution from {indexed_hdf5_filename}"
    )

    all_R = peaks.goniometer.rotation

    if offsets is not None:
        from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix

        print(f"Applying refined goniometer offsets from indexer: {offsets}")
        if (
            peaks.goniometer.angles_raw is not None
            and peaks.goniometer.axes_raw is not None
        ):
            angles_refined = peaks.goniometer.angles_raw + offsets[None, :]
            all_R = np.stack(
                [
                    calc_goniometer_rotation_matrix(peaks.goniometer.axes_raw, ang)
                    for ang in angles_refined
                ]
            )
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
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        d_min,
        RUB=RUB,
        space_group=space_group,
        sample_offset=sample_offset,
        ki_vec=ki_vec,
        max_workers=max_workers,
        R_all=all_R,
    )

    print(f"Saving predictions to {integration_peaks_filename}")
    with h5py.File(integration_peaks_filename, "w") as f:
        f.attrs["instrument"] = instrument
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = alpha, beta, gamma

        sorted_keys = sorted(peaks.image.ims.keys())
        bank_ids = np.array(
            [peaks.image.bank_mapping.get(k, k) for k in sorted_keys], dtype=np.int32
        )
        f.create_dataset("bank_ids", data=bank_ids)

        f["sample/space_group"] = space_group
        f["sample/U"], f["sample/B"] = U, B
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

        for img_key, (i, j, h, k, l, wl) in results_map.items():
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
    apply_detector_calibration(integration_peaks_filename, instrument)

    import h5py
    from subhkl.peakfinder.sparse_rbf import integrate_peaks_rbf_ssn

    sigma_list = [float(k.strip()) for k in sigmas.split(",")]
    print(f"Starting Dense Sparse RBF Integration on {filename}")
    print(
        f"Parameters: Alpha={alpha}, Gamma={gamma}, Sigma={sigma_list}, Max Peaks Padding={max_peaks}"
    )

    peak_dict = {}

    with h5py.File(integration_peaks_filename, "r") as f:
        if "sample/U" in f:
            f["sample/U"][()]
        if "sample/B" in f:
            f["sample/B"][()]
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
                grp["i"][()],
                grp["j"][()],
                grp["h"][()],
                grp["k"][()],
                grp["l"][()],
                grp["wavelength"][()],
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
        peaks_obj=peaks,  # Pass the full Peaks object
        alpha=alpha,
        sigmas=sigma_list,
        gamma=gamma,
        nominal_sigma=nominal_sigma,
        max_peaks=max_peaks,
        show_progress=show_progress,
        all_R=all_R,  # Pass rotation and offset downstream
        sample_offset=sample_offset,
        anisotropic=anisotropic,
        fit_mosaicity=fit_mosaicity,
        border_width=border_width,
        chunk_size=chunk_size,
        create_visualizations=create_visualizations,
        file_prefix=filename,
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

        with h5py.File(integration_peaks_filename, "r") as f_in:
            for key in copy_keys:
                if key in f_in:
                    f_in.copy(f_in[key], f, key)

            for k in ["goniometer/axes", "goniometer/names"]:
                if k in f_in:
                    f_in.copy(f_in[k], f, k)


def run_integrator(
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
    apply_detector_calibration(integration_peaks_filename, instrument)

    peak_dict = {}
    angles_stack = None
    all_R = None
    with h5py.File(integration_peaks_filename, "r") as f:
        U = f["sample/U"][()] if "sample/U" in f else None
        B = f["sample/B"][()] if "sample/B" in f else None
        all_R = f["goniometer/R"][()] if "goniometer/R" in f else None
        angles_stack = f["goniometer/angles"][()] if "goniometer/angles" in f else None
        sample_offset = f["sample/offset"][()] if "sample/offset" in f else np.zeros(3)
        ki_vec = (
            f["beam/ki_vec"][()] if "beam/ki_vec" in f else np.array([0.0, 0.0, 1.0])
        )

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

    UB = U @ B if U is not None and B is not None else None
    RUB = None
    if UB is not None:
        RUB = np.matmul(all_R, UB) if all_R.ndim == 3 else all_R @ UB

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
        f["peaks/h"], f["peaks/k"], f["peaks/l"] = result.h, result.k, result.l
        f["peaks/lambda"] = result.wavelength
        f["peaks/intensity"], f["peaks/sigma"] = result.intensity, result.sigma
        f["peaks/two_theta"], f["peaks/azimuthal"] = result.tt, result.az
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


def run_mtz_exporter(
    indexed_h5_filename: str, output_mtz_filename: str, space_group: str
):
    algorithm = MTZExporter(indexed_h5_filename, space_group)
    algorithm.write_mtz(output_mtz_filename)


def run_reduce(
    nexus_filename: str,
    output_filename: str,
    instrument: str,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
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

    axes = (
        np.array(peaks_handler.goniometer.axes_raw)
        if peaks_handler.goniometer.axes_raw is not None
        else np.array([0.0, 1.0, 0.0])
    )

    with h5py.File(output_filename, "w") as f:
        f.create_dataset("images", data=image_stack, compression="lzf")
        f.create_dataset("bank_ids", data=bank_ids)
        f.create_dataset("goniometer/angles", data=angles_repeated)
        f.create_dataset("goniometer/axes", data=axes)

        if peaks_handler.goniometer.names_raw:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(
                "goniometer/names", data=peaks_handler.goniometer.names_raw, dtype=dt
            )

        f.create_dataset(
            "instrument/wavelength",
            data=[peaks_handler.wavelength.min, peaks_handler.wavelength.max],
        )
        f.attrs["instrument"] = instrument

    print(f"Saved {n_images} banks to {output_filename}")


def run_merge_images(
    input_pattern: str,
    output_filename: str,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    space_group: str,
):
    from subhkl.core.spacegroup import get_space_group_object
    import glob

    try:
        get_space_group_object(space_group)
    except ValueError as e:
        raise ValueError(f"ERROR: Invalid space group '{space_group}': {e}")

    if " " in input_pattern:
        h5_files = []
        for p in input_pattern.split():
            h5_files.extend(glob.glob(p))
    else:
        h5_files = glob.glob(input_pattern)

    h5_files = sorted(list(set(h5_files)))

    if not h5_files:
        raise ValueError(f"No files found matching: {input_pattern}")

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
        f["sample/space_group"] = space_group.encode("utf-8")

    print(f"Successfully created {output_filename} with unit cell info embedded.")


def run_zone_axis_search(
    merged_h5_filename: str,
    peaks_h5_filename: str,
    instrument: str,
    output_h5_filename: str,
    space_group: str = None,
    d_min: float = 1.0,
    vector_tolerance: float = 0.15,
    border_frac: float = 0.1,
    min_intensity: float = 50.0,
    hough_grid_resolution: int = 1024,
    n_hough: int = 15,
    davenport_angle_tol: float = 0.5,
    top_k_rays: int = 15,
    max_uvw: int = 25,
    L_max: float = 250.0,
    top_k: int = 1000,
    num_runs: int = 0,
    output_hough: str | None = None,
    batch_size: int = 1024,
):
    """
    Global Zone-Axis Search to find the macroscopic crystal orientation (U matrix).
    Outputs an HDF5 file that can be passed directly to 'indexer --bootstrap'.
    """
    import h5py
    import numpy as np
    import jax.numpy as jnp
    from subhkl.config import reduction_settings
    from subhkl.optimization import FindUB, VectorizedObjective
    from subhkl.search.prior import HoughPrior

    print(f"Loading data from {merged_h5_filename}...")
    with h5py.File(merged_h5_filename, "r") as f_in:
        file_bank_ids = list(int(bid) for bid in f_in["bank_ids"])
        ax = f_in["goniometer/axes"][()]
        goniometer_angles = np.array(f_in["goniometer/angles"][()])

        from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix

        R_stack = np.stack(
            [calc_goniometer_rotation_matrix(ax, ang) for ang in goniometer_angles]
        )
        file_offsets = f_in["file_offsets"][()]

        a = float(f_in["sample/a"][()])
        b = float(f_in["sample/b"][()])
        c = float(f_in["sample/c"][()])
        alpha = float(f_in["sample/alpha"][()])
        beta = float(f_in["sample/beta"][()])
        gamma = float(f_in["sample/gamma"][()])

        if space_group is None:
            space_group = f_in["sample/space_group"][()].decode("utf-8")

    # Dynamically slice the arrays based on the requested number of runs
    if num_runs > 0:
        if len(file_offsets) > num_runs:
            end_idx = file_offsets[num_runs]
        else:
            end_idx = len(file_bank_ids)
            num_runs = len(file_offsets)

        print(
            f"Limiting search to the first {num_runs} run(s) (Images 0 to {end_idx - 1})..."
        )
        file_bank_ids = file_bank_ids[:end_idx]
        R_stack = R_stack[:end_idx]
    else:
        end_idx = len(file_bank_ids)
        print(f"Using all {len(file_offsets)} available runs for the search...")

    settings = reduction_settings[instrument]
    wavelength_min, wavelength_max = settings.get("Wavelength")

    ub_helper = FindUB()
    ub_helper.a, ub_helper.b, ub_helper.c = a, b, c
    ub_helper.alpha, ub_helper.beta, ub_helper.gamma = alpha, beta, gamma
    B_mat = ub_helper.reciprocal_lattice_B()

    print("\n--- HOUGH PRIOR GENERATION ---")
    prior_engine = HoughPrior(
        B_mat, np.array(R_stack), ki_vec=np.array([0.0, 0.0, 1.0])
    )

    print(f"Loading empirical rays from {peaks_h5_filename}...")

    with h5py.File(peaks_h5_filename, "r") as f_peaks:
        peaks_xyz = f_peaks["peaks/xyz"][()]
        peaks_intensity = f_peaks["peaks/intensity"][()]

        # CRITICAL: image_index maps 1:1 to the N_banks dimension of R_stack in merged.h5
        if "peaks/image_index" in f_peaks:
            group_indices = f_peaks["peaks/image_index"][()]
        else:
            group_indices = f_peaks["peaks/run_index"][()]

        if "beam/ki_vec" in f_peaks:
            ki_vec = f_peaks["beam/ki_vec"][()]
        else:
            ki_vec = np.array([0.0, 0.0, 1.0])

        # If Peaks file overrides the goniometer entirely, use it. Otherwise rely on Peaks API/merged.h5
        R_peaks_override = f_peaks.get("goniometer/R")
        if R_peaks_override is not None:
            R_peaks_override = R_peaks_override[()]

    q_hat_list, q_lab_list, peaks_xyz_list, intensities_list, mapped_bank_indices = (
        [],
        [],
        [],
        [],
        [],
    )

    unique_groups = np.unique(group_indices)
    for g_idx in unique_groups:
        if g_idx >= end_idx:
            continue

        mask = group_indices == g_idx
        grp_xyz = peaks_xyz[mask]
        grp_intensity = peaks_intensity[mask]

        # 2. Safely grab the rotation matrix using the flat bank index (g_idx)
        if R_peaks_override is not None:
            if R_peaks_override.ndim == 3 and R_peaks_override.shape[0] == len(
                peaks_xyz
            ):
                R_gonio = R_peaks_override[mask][0]
            elif R_peaks_override.ndim == 3 and R_peaks_override.shape[0] > g_idx:
                R_gonio = R_peaks_override[g_idx]
            else:
                R_gonio = R_peaks_override
        else:
            # Let the flat R_stack (N_banks, 3, 3) map directly to the image/bank index
            R_gonio = R_stack[g_idx] if g_idx < len(R_stack) else np.eye(3)

        intensity_mask = grp_intensity >= min_intensity
        if not np.any(intensity_mask):
            continue

        grp_xyz = grp_xyz[intensity_mask]
        grp_intensity = grp_intensity[intensity_mask]

        top_k_idx = np.argsort(grp_intensity)[::-1][
            : min(top_k_rays, len(grp_intensity))
        ]
        grp_xyz_top = grp_xyz[top_k_idx]
        grp_intensity_top = grp_intensity[top_k_idx]

        kf = grp_xyz_top / np.linalg.norm(grp_xyz_top, axis=1, keepdims=True)
        q_lab = kf - ki_vec[None, :]
        q_sample = np.dot(q_lab, R_gonio)

        q_norms = np.linalg.norm(q_sample, axis=1, keepdims=True)
        q_hat_grp = q_sample / q_norms

        q_hat_list.append(q_hat_grp)
        q_lab_list.append(q_lab)
        peaks_xyz_list.append(grp_xyz_top)
        intensities_list.append(grp_intensity_top)

        # 3. Map the VectorizedObjective strictly to the flat bank index
        mapped_bank_indices.append(np.full(len(grp_xyz_top), g_idx))

    if not q_hat_list:
        print(
            "Failed to extract any valid rays from the peaks file. Check your --min-intensity threshold."
        )
        return

    q_hat = np.vstack(q_hat_list)
    q_lab_all = np.vstack(q_lab_list).T
    peaks_xyz_all = np.vstack(peaks_xyz_list).T
    intensities_all = np.concatenate(intensities_list)

    # This array now contains the exact bank index (0 to N_banks-1) for every single ray
    bank_indices_all = np.concatenate(mapped_bank_indices)

    median_intensity = np.median(intensities_all)
    weights_all = intensities_all / (median_intensity + 1e-6)
    weights_all = np.clip(weights_all, 0.0, 10.0)

    print(f"Extracted {len(q_hat)} physical rays. Running 3D Combinatorial Hough...")
    n_obs, weights_obs = prior_engine.compute_hough_accumulator(
        q_hat,
        grid_resolution=hough_grid_resolution,
        n_hough=n_hough,
        plot_filename=output_hough,
        border_frac=border_frac,
    )

    if len(n_obs) == 0:
        return

    n_calc = prior_engine.generate_theoretical_zones(
        L_max=L_max, top_k=top_k, max_uvw=max_uvw
    )
    print(
        f"Extracted {len(n_obs)} Empirical Zones against {len(n_calc)} Theoretical Zones."
    )

    quats, _ = prior_engine.solve_permutations(
        jnp.array(n_obs),
        jnp.array(weights_obs),
        n_calc,
        q_hat,
        space_group=space_group,
        angle_tol_deg=davenport_angle_tol,
        scoring_tol_deg=vector_tolerance,
        d_min=d_min,
    )

    if quats is None or len(quats) == 0:
        return

    print("Filtering Prior through Exact Physics Forward-Model...")

    ray_objective = VectorizedObjective(
        B=B_mat,
        kf_ki_dir=q_lab_all,
        peak_xyz_lab=peaks_xyz_all,
        wavelength=[wavelength_min, wavelength_max],
        cell_params=[a, b, c, alpha, beta, gamma],
        # 4. The Magic Link:
        # static_R has length N_banks. peak_run_indices contains values from 0 to N_banks-1.
        # VectorizedObjective will now perfectly map every single ray to its exact physical bank geometry.
        static_R=R_stack,
        peak_run_indices=bank_indices_all,
    )

    prior_rots = prior_engine.physics_filter(
        quats, ray_objective, batch_size=batch_size, z_score_threshold=3.0
    )

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

        f.create_dataset("sample/a", data=a)
        f.create_dataset("sample/b", data=b)
        f.create_dataset("sample/c", data=c)
        f.create_dataset("sample/alpha", data=alpha)
        f.create_dataset("sample/beta", data=beta)
        f.create_dataset("sample/gamma", data=gamma)

        f.create_dataset("sample/offset", data=np.zeros(3))
        f.create_dataset("beam/ki_vec", data=np.array([0.0, 0.0, 1.0]))
        f.create_dataset("optimization/goniometer_offsets", data=np.zeros(len(ax)))
        f.create_dataset("sample/space_group", data=space_group.encode("utf-8"))
        f.create_dataset("instrument/wavelength", data=[wavelength_min, wavelength_max])

    print(
        f"Done. You can now run:\n subhkl indexer {merged_h5_filename} <output.h5> --bootstrap {output_h5_filename} ..."
    )
