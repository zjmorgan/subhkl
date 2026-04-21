import h5py
import numpy as np
import jax.numpy as jnp
from subhkl.config import reduction_settings
from subhkl.core.crystallography import Lattice
from subhkl.optimization import VectorizedObjective
from subhkl.search.prior import HoughPrior

def run(
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

    B_mat = Lattice(a, b, c, alpha, beta, gamma).get_b_matrix()

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

    # Retrieve names to build the map for the objective function
    axis_names = None
    with h5py.File(merged_h5_filename, "r") as f_in:
        if "goniometer/names" in f_in:
            axis_names = [n.decode("utf-8") for n in f_in["goniometer/names"][()]]

    motor_map = None
    if axis_names is not None:
        unique_motors = []
        motor_map = []
        for name in axis_names:
            if name not in unique_motors:
                unique_motors.append(name)
            motor_map.append(unique_motors.index(name))

    ray_objective = VectorizedObjective(
        B=B_mat,
        kf_ki_dir=q_lab_all,
        peak_xyz_lab=peaks_xyz_all,
        wavelength=[wavelength_min, wavelength_max],
        cell_params=[a, b, c, alpha, beta, gamma],
        motor_map=motor_map,
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
