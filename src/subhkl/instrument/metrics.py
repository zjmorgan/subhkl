"""
Metrics computation module for indexing quality assessment.
Provides functions to compute and return angular/distance errors from indexed peaks.
"""

import h5py
import numpy as np
import scipy.spatial

from subhkl.config import beamlines
from subhkl.instrument.detector import Detector
from subhkl.instrument.physics import calculate_angular_error
from subhkl.optimization import FindUB


def _get_safe_R_stack(R_file_in, run_indices_in, target_len):
    if R_file_in is None:
        return [np.eye(3)] * target_len

    if R_file_in.ndim == 3 and len(R_file_in) == target_len:
        return R_file_in

    def safe_get_single_R(r_idx):
        ridx = int(r_idx)
        if R_file_in.ndim == 3 and len(R_file_in) > ridx:
            return R_file_in[ridx]
        return R_file_in[0]

    if R_file_in.ndim == 3:
        if len(R_file_in) == target_len:
            return R_file_in
        return [safe_get_single_R(r) for r in run_indices_in]
    return [R_file_in] * target_len


def resolve_indices(f_handle):
    if "peaks/run_index" in f_handle:
        return f_handle["peaks/run_index"][()]
    if "peaks/image_index" in f_handle:
        return f_handle["peaks/image_index"][()]
    if "bank" in f_handle:
        return f_handle["bank"][()]
    return None


def extract_xyz_from_file(file_path, instrument=None):
    """Safely extracts physical XYZ lab coordinates from a file, applying calibrations if present."""
    with h5py.File(file_path, "r") as f:
        run_idx = resolve_indices(f)
        
        bank_array = None
        if "bank" in f: bank_array = f["bank"][()]
        elif "peaks/bank" in f: bank_array = f["peaks/bank"][()]
        elif "bank_ids" in f and run_idx is not None:
            b_ids = f["bank_ids"][()]
            bank_array = np.array([b_ids[int(r)] for r in run_idx])
        else:
            bank_array = run_idx
            
        calibration_dict = {}
        if "detector_calibration" in f:
            calib_grp = f["detector_calibration"]
            for b_key in calib_grp.keys():
                calibration_dict[b_key] = {
                    "center": calib_grp[b_key]["center"][()],
                    "uhat": calib_grp[b_key]["uhat"][()],
                    "vhat": calib_grp[b_key]["vhat"][()]
                }

        # 1. Coordinate array reconstruction (Finder or Integrator)
        if "peaks/xyz" in f or ("peaks/pixel_r" in f and "peaks/pixel_c" in f):
            has_pixels = "peaks/pixel_r" in f and "peaks/pixel_c" in f
            has_xyz = "peaks/xyz" in f
            
            if has_xyz:
                xyz = f["peaks/xyz"][()]
            else:
                xyz = np.zeros((len(f["peaks/pixel_r"][()]), 3))
                
            if run_idx is None: run_idx = np.zeros(len(xyz))
            if bank_array is None: bank_array = run_idx
            
            if instrument is not None and has_pixels:
                new_xyz = np.copy(xyz)
                pixel_r, pixel_c = f["peaks/pixel_r"][()], f["peaks/pixel_c"][()]
                
                for phys_bank in np.unique(bank_array):
                    bank_str = f"bank_{int(phys_bank)}"
                    mask = bank_array == phys_bank
                    if not np.any(mask): continue
                        
                    try:
                        det_config = beamlines[instrument][str(int(phys_bank))]
                        det = Detector(det_config)
                        
                        if bank_str in calibration_dict:
                            det.center = calibration_dict[bank_str]["center"]
                            det.uhat = calibration_dict[bank_str]["uhat"]
                            det.vhat = calibration_dict[bank_str]["vhat"]
                            
                        new_xyz[mask] = det.pixel_to_lab(pixel_r[mask], pixel_c[mask])
                    except KeyError:
                        pass
                return new_xyz, run_idx
            return xyz, run_idx
            
        # 2. Bank/Pixel format (Predictor output)
        if "banks" in f:
            xyz_list, run_list = [], []
            bank_ids = f["bank_ids"][()] if "bank_ids" in f else None
            
            for img_key_str in f["banks"].keys():
                img_idx = int(img_key_str)
                grp = f[f"banks/{img_key_str}"]
                i_p, j_p = grp["i"][()], grp["j"][()]
                
                phys_bank = bank_ids[img_idx] if bank_ids is not None else img_idx
                try:
                    det_config = beamlines[instrument][str(phys_bank)]
                    det = Detector(det_config)
                    
                    bank_str = f"bank_{phys_bank}"
                    if bank_str in calibration_dict:
                        det.center = calibration_dict[bank_str]["center"]
                        det.uhat = calibration_dict[bank_str]["uhat"]
                        det.vhat = calibration_dict[bank_str]["vhat"]
                    
                    xyz = det.pixel_to_lab(i_p, j_p)
                    if xyz.ndim == 1: xyz = xyz[np.newaxis, :]
                    
                    xyz_list.append(xyz)
                    run_list.extend([img_idx] * len(xyz))
                except KeyError:
                    continue
                    
            if xyz_list: return np.vstack(xyz_list), np.array(run_list)
            
    return None, None

def compute_metrics(
    file1: str,
    file2: str | None = None,
    instrument: str | None = None,
    d_min: float | None = None,
    per_run: bool = False,
    ki_vec_override: np.ndarray | None = None,
) -> dict:
    try:
        with h5py.File(file1, "r") as f:
            ub_helper = FindUB()
            ub_helper.a = f["sample/a"][()]
            ub_helper.b = f["sample/b"][()]
            ub_helper.c = f["sample/c"][()]
            ub_helper.alpha = f["sample/alpha"][()]
            ub_helper.beta = f["sample/beta"][()]
            ub_helper.gamma = f["sample/gamma"][()]
            B_mat = ub_helper.reciprocal_lattice_B()
            U = f["sample/U"][()] if "sample/U" in f else np.eye(3)
            sample_offset = f["sample/offset"][()] if "sample/offset" in f else np.zeros(3)
            
            if ki_vec_override is not None:
                ki_vec = ki_vec_override
            else:
                ki_vec = f["beam/ki_vec"][()] if "beam/ki_vec" in f else np.array([0.0, 0.0, 1.0])
                
            R_file = f["goniometer/R"][()] if "goniometer/R" in f else None

            if instrument is None:
                instrument = f.attrs.get("instrument")

        matched_h, matched_k, matched_l, matched_lam, matched_xyz, matched_R, matched_run = [], [], [], [], [], [], []

        # ==========================================
        # TWO FILE COMPARISON
        # ==========================================
        if file2 is not None:
            if instrument is None:
                return {"error_message": "ERROR: --instrument required for matching when not found in file attributes."}

            xyz_1, run_1 = extract_xyz_from_file(file1, instrument)
            xyz_2, run_2 = extract_xyz_from_file(file2, instrument)
            
            if xyz_1 is None or xyz_2 is None:
                return {"error_message": "ERROR: Could not extract physical XYZ coordinates from one or both files."}

            with h5py.File(file1, "r") as f1:
                if "banks" in f1:
                    for img_key_str in f1["banks"].keys():
                        img_idx = int(img_key_str)
                        grp = f1[f"banks/{img_key_str}"]
                        h_p, k_p, l_p, lam_p = grp["h"][()], grp["k"][()], grp["l"][()], grp["wavelength"][()]
                        
                        mask_1 = run_1 == img_idx
                        mask_2 = run_2 == img_idx
                        if not np.any(mask_1) or not np.any(mask_2): continue
                        
                        xyz_1_run = xyz_1[mask_1]
                        xyz_2_run = xyz_2[mask_2]
                        
                        tree = scipy.spatial.KDTree(xyz_1_run)
                        dists, idxs = tree.query(xyz_2_run)
                        valid = dists < 0.01
                        
                        if np.any(valid):
                            num_valid = np.sum(valid)
                            matched_h.extend(h_p[idxs[valid]])
                            matched_k.extend(k_p[idxs[valid]])
                            matched_l.extend(l_p[idxs[valid]])
                            matched_lam.extend(lam_p[idxs[valid]])
                            matched_xyz.extend(xyz_2_run[valid])
                            matched_run.extend([img_idx] * num_valid)
                            matched_R.extend(_get_safe_R_stack(R_file, [img_idx] * num_valid, num_valid))
                else:
                    h_p = f1["peaks/h"][()]
                    k_p = f1["peaks/k"][()]
                    l_p = f1["peaks/l"][()]
                    lam_p = f1["peaks/lambda"][()]
                    
                    unique_runs = np.unique(run_1)
                    for r in unique_runs:
                        mask_1 = run_1 == r
                        mask_2 = run_2 == r
                        if not np.any(mask_1) or not np.any(mask_2): continue

                        xyz_1_run = xyz_1[mask_1]
                        xyz_2_run = xyz_2[mask_2]

                        tree = scipy.spatial.KDTree(xyz_1_run)
                        dists, idxs = tree.query(xyz_2_run)
                        valid = dists < 0.01
                        
                        if np.any(valid):
                            num_valid = np.sum(valid)
                            matched_h.extend(h_p[mask_1][idxs[valid]])
                            matched_k.extend(k_p[mask_1][idxs[valid]])
                            matched_l.extend(l_p[mask_1][idxs[valid]])
                            matched_lam.extend(lam_p[mask_1][idxs[valid]])
                            matched_xyz.extend(xyz_2_run[valid])
                            matched_run.extend([r] * num_valid)
                            matched_R.extend(_get_safe_R_stack(R_file, [r] * num_valid, num_valid))

        # ==========================================
        # SINGLE FILE METRICS
        # ==========================================
        else:
            with h5py.File(file1, "r") as f:
                if "peaks/h" not in f:
                    return {"error_message": "No peaks/h dataset found in file"}
                matched_h = f["peaks/h"][()]
                matched_k = f["peaks/k"][()]
                matched_l = f["peaks/l"][()]
                matched_lam = f["peaks/lambda"][()]
                
                xyz, r_idx = extract_xyz_from_file(file1, instrument)
                if xyz is None:
                    return {"error_message": "Could not extract XYZ coordinates."}
                    
                matched_xyz = xyz
                matched_run = r_idx
                matched_R = _get_safe_R_stack(R_file, matched_run, len(matched_h))

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
                "median_d_err": 0.0, "mean_d_err": 0.0, "max_d_err": 0.0,
                "median_ang_err": 0.0, "mean_ang_err": 0.0, "max_ang_err": 0.0,
                "num_peaks": 0,
            }

        h, k, l = h[mask], k[mask], l[mask]
        lam = lam[mask]
        xyz_det = xyz_det[mask]
        R_all = R_all[mask]
        run_index = run_index[mask]

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
                        (int(r), float(np.median(ang_err[r_mask])), int(np.sum(r_mask)))
                    )
            run_errors.sort(key=lambda x: x[1], reverse=True)
            result["per_run_errors"] = run_errors

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error_message": f"Exception during metrics computation: {e!s}"}
