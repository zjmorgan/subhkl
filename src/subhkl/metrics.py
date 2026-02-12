"""
Metrics computation module for indexing quality assessment.

Provides functions to compute and return angular/distance errors from indexed peaks.
"""

import h5py
import numpy as np
import scipy.spatial

from subhkl.config import beamlines
from subhkl.detector import Detector
from subhkl.optimization import FindUB
from subhkl.utils import calculate_angular_error


def _get_safe_R_stack(R_file_in, run_indices_in, target_len):
    """
    Helper function to robustly construct a rotation matrix stack.

    Args:
        R_file_in: Input rotation data (None, 1 matrix, or stack of matrices)
        run_indices_in: Run indices for lookup
        target_len: Target length of output

    Returns:
        List of rotation matrices
    """
    if R_file_in is None:
        return [np.eye(3)] * target_len

    if R_file_in.ndim == 3 and len(R_file_in) == target_len:
        return R_file_in

    # Robust lookup with fallback
    def safe_get_single_R(r_idx):
        ridx = int(r_idx)
        if R_file_in.ndim == 3 and len(R_file_in) > ridx:
            return R_file_in[ridx]
        return R_file_in[0]

    if R_file_in.ndim == 3:
        # Check if R_file_in is per-peak or per-run
        # target_len is number of matched peaks.
        # If stack matches target_len, it is likely already per-peak.
        if len(R_file_in) == target_len:
            return R_file_in
        return [safe_get_single_R(r) for r in run_indices_in]
    return [R_file_in] * target_len


def compute_metrics(
    filename: str,
    found_peaks_file: str | None = None,
    instrument: str | None = None,
    d_min: float | None = None,
    per_run: bool = False,
) -> dict:
    """
    Compute indexing quality metrics from indexed peaks.

    Args:
        filename: HDF5 file with indexed peaks and sample/beam parameters
        found_peaks_file: Optional HDF5 file with observed peaks for comparison
        instrument: Instrument name (required if matching peaks)
        d_min: Optional minimum d-spacing filter
        per_run: If True, include per-run error statistics

    Returns:
        Dictionary with keys:
            - 'median_d_err': Median distance error
            - 'mean_d_err': Mean distance error
            - 'max_d_err': Maximum distance error
            - 'median_ang_err': Median angular error (degrees)
            - 'mean_ang_err': Mean angular error (degrees)
            - 'max_ang_err': Maximum angular error (degrees)
            - 'num_peaks': Number of peaks used
            - 'per_run_errors': List of (run_id, median_error, count) tuples (if per_run=True)
            - 'error_message': Error message if computation failed
    """
    try:
        # Load Global Physics from filename
        with h5py.File(filename, "r") as f:
            ub_helper = FindUB()
            ub_helper.a = f["sample/a"][()]
            ub_helper.b = f["sample/b"][()]
            ub_helper.c = f["sample/c"][()]
            ub_helper.alpha = f["sample/alpha"][()]
            ub_helper.beta = f["sample/beta"][()]
            ub_helper.gamma = f["sample/gamma"][()]
            B_mat = ub_helper.reciprocal_lattice_B()
            U = f["sample/U"][()] if "sample/U" in f else np.eye(3)
            sample_offset = (
                f["sample/offset"][()] if "sample/offset" in f else np.zeros(3)
            )
            ki_vec = (
                f["beam/ki_vec"][()]
                if "beam/ki_vec" in f
                else np.array([0.0, 0.0, 1.0])
            )
            R_file = f["goniometer/R"][()] if "goniometer/R" in f else None

            if instrument is None:
                instrument = f.attrs.get("instrument")

        (
            matched_h,
            matched_k,
            matched_l,
            matched_lam,
            matched_xyz,
            matched_R,
            matched_run,
        ) = ([], [], [], [], [], [], [])

        # Robust Run Index Resolution for Metrics
        def resolve_indices(f_handle):
            idx_run = (
                f_handle["peaks/run_index"][()]
                if "peaks/run_index" in f_handle
                else None
            )
            idx_img = (
                f_handle["peaks/image_index"][()]
                if "peaks/image_index" in f_handle
                else None
            )
            idx_bank = f_handle["bank"][()] if "bank" in f_handle else None

            if R_file is not None and R_file.ndim == 3:
                num_rot = R_file.shape[0]
                if idx_run is not None and int(np.max(idx_run)) + 1 == num_rot:
                    return idx_run
                if idx_img is not None and int(np.max(idx_img)) + 1 == num_rot:
                    return idx_img
                if idx_bank is not None and int(np.max(idx_bank)) + 1 == num_rot:
                    return idx_bank
            return (
                idx_run
                if idx_run is not None
                else (idx_img if idx_img is not None else idx_bank)
            )

        if found_peaks_file is not None:
            if instrument is None:
                return {
                    "error_message": "ERROR: --instrument required for matching when not found in file attributes."
                }

            # Load Found Peaks
            with h5py.File(found_peaks_file, "r") as f_obs:
                xyz_obs = f_obs["peaks/xyz"][()]
                run_obs = resolve_indices(f_obs)
                if run_obs is None:
                    run_obs = np.zeros(len(xyz_obs))

                # Try to get physical bank mapping
                bank_obs = f_obs["bank"][()] if "bank" in f_obs else None
                run_to_bank = {}
                if bank_obs is not None and run_obs is not None:
                    for r in np.unique(run_obs):
                        run_to_bank[int(r)] = int(bank_obs[run_obs == r][0])

            # Load Predicted Peaks from filename
            with h5py.File(filename, "r") as f_pred:
                if "banks" in f_pred:
                    # Predictor format
                    bank_ids = f_pred["bank_ids"][()] if "bank_ids" in f_pred else None

                    for img_key_str in f_pred["banks"].keys():
                        img_idx = int(img_key_str)
                        grp = f_pred[f"banks/{img_key_str}"]
                        h_p = grp["h"][()]
                        k_p = grp["k"][()]
                        l_p = grp["l"][()]
                        lam_p = grp["wavelength"][()]
                        i_p = grp["i"][()]
                        j_p = grp["j"][()]

                        # Get matching observed peaks
                        mask_obs = run_obs == img_idx
                        if not np.any(mask_obs):
                            continue

                        xyz_obs_run = xyz_obs[mask_obs]

                        if bank_ids is not None:
                            phys_bank = bank_ids[img_idx]
                        else:
                            phys_bank = img_idx

                        if run_to_bank:
                            phys_bank = run_to_bank.get(img_idx, phys_bank)

                        try:
                            det_config = beamlines[instrument][str(phys_bank)]
                        except KeyError:
                            continue

                        det = Detector(det_config)
                        xyz_pred_run = det.pixel_to_lab(i_p, j_p)
                        if xyz_pred_run.ndim == 1:
                            xyz_pred_run = xyz_pred_run[np.newaxis, :]

                        # KDTree match
                        tree = scipy.spatial.KDTree(xyz_pred_run)
                        dists, idxs = tree.query(xyz_obs_run)

                        valid = dists < 0.01
                        if np.any(valid):
                            num_valid = np.sum(valid)
                            matched_h.extend(h_p[idxs[valid]])
                            matched_k.extend(k_p[idxs[valid]])
                            matched_l.extend(l_p[idxs[valid]])
                            matched_lam.extend(lam_p[idxs[valid]])
                            matched_xyz.extend(xyz_obs_run[valid])
                            matched_run.extend([img_idx] * num_valid)

                            # Use helper for R assignment
                            matched_R.extend(
                                _get_safe_R_stack(
                                    R_file, [img_idx] * num_valid, num_valid
                                )
                            )
                else:
                    # Non-predictor format (integrator/indexer) but matching requested
                    # Use peaks/xyz from file as predicted positions
                    xyz_pred = f_pred["peaks/xyz"][()]
                    h_p = f_pred["peaks/h"][()]
                    k_p = f_pred["peaks/k"][()]
                    l_p = f_pred["peaks/l"][()]
                    lam_p = f_pred["peaks/lambda"][()]
                    run_pred = resolve_indices(f_pred)
                    if run_pred is None:
                        run_pred = np.zeros(len(h_p))

                    unique_runs = np.unique(run_pred)
                    for r in unique_runs:
                        mask_p = run_pred == r
                        mask_o = run_obs == r
                        if not np.any(mask_p) or not np.any(mask_o):
                            continue

                        xyz_p_run = xyz_pred[mask_p]
                        xyz_o_run = xyz_obs[mask_o]

                        tree = scipy.spatial.KDTree(xyz_p_run)
                        dists, idxs = tree.query(xyz_o_run)
                        valid = dists < 0.01
                        if np.any(valid):
                            num_valid = np.sum(valid)
                            matched_h.extend(h_p[mask_p][idxs[valid]])
                            matched_k.extend(k_p[mask_p][idxs[valid]])
                            matched_l.extend(l_p[mask_p][idxs[valid]])
                            matched_lam.extend(lam_p[mask_p][idxs[valid]])
                            matched_xyz.extend(xyz_o_run[valid])
                            matched_run.extend([r] * num_valid)

                            # Use helper for R assignment
                            matched_R.extend(
                                _get_safe_R_stack(R_file, [r] * num_valid, num_valid)
                            )
        else:
            # Standard case: load from filename
            with h5py.File(filename, "r") as f:
                if "peaks/h" not in f:
                    return {"error_message": "No peaks/h dataset found in file"}
                matched_h = f["peaks/h"][()]
                matched_k = f["peaks/k"][()]
                matched_l = f["peaks/l"][()]
                matched_lam = f["peaks/lambda"][()]
                if "peaks/xyz" in f:
                    matched_xyz = f["peaks/xyz"][()]
                else:
                    # Fallback to reconstructing kf_dir from angles
                    # tt, az are in degrees
                    tt = np.deg2rad(f["peaks/two_theta"][()])
                    az = np.deg2rad(f["peaks/azimuthal"][()])
                    kx = np.sin(tt) * np.cos(az)
                    ky = np.sin(tt) * np.sin(az)
                    kz = np.cos(tt)
                    matched_xyz = np.stack([kx, ky, kz], axis=1)

                matched_run = resolve_indices(f)
                if matched_run is None:
                    matched_run = np.zeros(len(matched_h))

                matched_R = _get_safe_R_stack(R_file, matched_run, len(matched_h))

        # Convert to numpy arrays
        h = np.array(matched_h)
        k = np.array(matched_k)
        l = np.array(matched_l)
        lam = np.array(matched_lam)
        xyz_det = np.array(matched_xyz)
        R_all = np.array(matched_R)
        run_index = np.array(matched_run)

        mask = (h != 0) | (k != 0) | (l != 0)
        if np.sum(mask) == 0:
            return {
                "median_d_err": 0.0,
                "mean_d_err": 0.0,
                "max_d_err": 0.0,
                "median_ang_err": 0.0,
                "mean_ang_err": 0.0,
                "max_ang_err": 0.0,
                "num_peaks": 0,
            }

        h, k, l = h[mask], k[mask], l[mask]
        lam = lam[mask]
        xyz_det = xyz_det[mask]
        R_all = R_all[mask]
        run_index = run_index[mask]

        # --- Filter by d_min if provided ---
        d_filter_message = None
        if d_min is not None:
            hkl_vecs = np.stack([h, k, l], axis=1)
            q_cryst = hkl_vecs @ B_mat.T
            q_mag = np.linalg.norm(q_cryst, axis=1)
            with np.errstate(divide="ignore"):
                d_vals = 1.0 / q_mag
            d_mask = d_vals >= d_min
            if np.sum(d_mask) == 0:
                return {"error_message": f"No peaks found with d >= {d_min} A."}
            h, k, l = h[d_mask], k[d_mask], l[d_mask]
            lam = lam[d_mask]
            xyz_det = xyz_det[d_mask]
            R_all = R_all[d_mask]
            run_index = run_index[d_mask]
            d_filter_message = f"Filtered to {len(h)} peaks with d >= {d_min} A."

        # --- Calculate RUB stack AFTER all filtering ---
        UB = U @ B_mat
        if R_all.ndim == 3:
            RUB = np.matmul(R_all, UB)
        else:
            RUB = R_all @ UB

        d_err, ang_err = calculate_angular_error(
            xyz_det, h, k, l, lam, RUB, sample_offset, ki_vec, R_all
        )

        result = {
            "median_d_err": float(np.median(d_err)),
            "mean_d_err": float(np.mean(d_err)),
            "max_d_err": float(np.max(d_err)),
            "median_ang_err": float(np.median(ang_err)),
            "mean_ang_err": float(np.mean(ang_err)),
            "max_ang_err": float(np.max(ang_err)),
            "num_peaks": len(h),
        }

        if d_filter_message:
            result["filter_message"] = d_filter_message

        if per_run:
            unique_runs = sorted(np.unique(run_index))
            run_errors = []
            for r in unique_runs:
                r_mask = run_index == r
                if np.sum(r_mask) > 0:
                    run_errors.append(
                        (
                            int(r),
                            float(np.median(ang_err[r_mask])),
                            int(np.sum(r_mask)),
                        )
                    )
            run_errors.sort(key=lambda x: x[1], reverse=True)
            result["per_run_errors"] = run_errors

        return result

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error_message": f"Exception during metrics computation: {e!s}"}
