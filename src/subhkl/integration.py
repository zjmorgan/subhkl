import bisect
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.spatial
import skimage.feature
from h5py import File
from PIL import Image

import _integration.loader

# Ensure we have tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(x, **kwargs):
        return x


from subhkl.config import (
    beamlines,
    reduction_settings,
)

from subhkl.config.goniometer import Goniometer
from subhkl.convex_hull.peak_integrator import PeakIntegrator
from subhkl.detector import Detector
from subhkl.sparse_rbf_peak_finder import SparseRBFPeakFinder
from subhkl.threshold_peak_finder import ThresholdingPeakFinder
from subhkl.utils import (
    calculate_angular_error,
    generate_reflections,
    predict_reflections_on_panel,
)

from dataclasses import dataclass, astuple
from typing import List, Any, Optional


@dataclass(frozen=True)
class DetectorPeaks:
    R: List[Any]
    two_theta: List[float]
    az_phi: List[float]
    wavelength_mins: List[float]
    wavelength_maxes: List[float]
    intensity: List[float]
    sigma: List[float]
    radii: List[float]
    xyz: List[List[float]]
    bank: List[int]
    image_index: List[int]
    run_id: List[int]
    gonio_axes: Optional[List[List[float]]]
    gonio_angles: List[List[float]]
    gonio_names: Optional[List[str]]

    def __iter__(self):
        """Allows tuple unpacking"""
        return iter(astuple(self))

    def __getitem__(self, index):
        """Allows index access"""
        return astuple(self)[index]


@dataclass(frozen=True)
class IntegrationResult:
    h: List[float]
    k: List[float]
    l: List[float]
    intensity: List[float]
    sigma: List[float]
    tt: List[float]
    az: List[float]
    wavelength: List[float]
    bank: List[int]
    run_id: List[int]
    xyz: List[List[float]]
    R: List[Any]
    angles: List[List[float]]

    def __iter__(self):
        """Allows tuple unpacking"""
        return iter(astuple(self))

    def __getitem__(self, index):
        """Allows index access"""
        return astuple(self)[index]


@dataclass
class Wavelength:
    min: float = None
    max: float = None

    def __iter__(self):
        return iter((self.min, self.max))


# ==============================================================================
# WORKER FUNCTIONS (Multiprocessing)
# ==============================================================================


def _run_harvest_local_max(
    im,
    max_peaks=200,
    min_pix=50,
    min_rel_intensity=0.5,
    normalize=False,
    **kwargs,
):
    """
    Worker for finding peak candidates using local maxima search.
    """
    if normalize:
        blur = scipy.ndimage.gaussian_filter(im, 4)
        div = scipy.ndimage.gaussian_filter(im, 60)
        processed = blur / div
    else:
        processed = im

    coords = skimage.feature.peak_local_max(
        processed,
        num_peaks=max_peaks,
        min_distance=min_pix,
        threshold_rel=min_rel_intensity,
        exclude_border=min_pix * 3,
    )
    return coords[:, 0], coords[:, 1]


def _run_harvest_thresholding(im, **kwargs):
    """
    Worker for finding peak candidates using adaptive thresholding.
    """
    valid_keys = [
        "noise_cutoff_quantile",
        "min_peak_dist_pixels",
        "blur_kernel_sigma",
        "open_kernel_size_pixels",
        "mask_file",
        "mask_rel_erosion_radius",
        "show_steps",
        "show_scale",
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    alg = ThresholdingPeakFinder(**filtered_kwargs)
    coords = alg.find_peaks(im)
    return coords[:, 0], coords[:, 1]


def _process_single_image(
    img_key,
    img_label,
    physical_bank,
    image,
    det_config,
    finder_info,
    integration_params,
    mask_info,
    geometry_info,
    viz_info,
):
    """
    Worker function to process a single image in a separate process.
    Performs peak finding, integration, and optional visualization.
    """
    # Unpack tuple arguments
    algo, harvest_kwargs, pre_coords = finder_info
    mask_file, erosion = mask_info
    gonio_R, gonio_angles, wl_min, wl_max = geometry_info
    do_viz, viz_prefix = viz_info

    det = Detector(det_config)

    # 1. Find Peaks
    if algo == "sparse_rbf":
        i, j = pre_coords
    elif algo == "peak_local_max":
        i, j = _run_harvest_local_max(image, **harvest_kwargs)
    elif algo == "thresholding":
        i, j = _run_harvest_thresholding(image, **harvest_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    centers = np.stack([i, j], axis=-1)

    # 2. Setup Mask
    if mask_file is not None:
        mask_im = np.array(Image.open(mask_file))
        if erosion:
            radius = max(1, int(min(mask_im.shape) * erosion))
            kernel = np.ones((radius, radius), dtype=np.uint8)
            mask = cv2.erode(mask_im, kernel).astype(bool)
        else:
            mask = mask_im.astype(bool)
    else:
        mask = np.full(image.shape, True)

    # 3. Integration Setup (Sigma Override)
    # Rebuild integrator from params to avoid sharing state
    integrator = PeakIntegrator.build_from_dictionary(integration_params.copy())

    if integration_params.get("region_growth_minimum_sigma") is not None:
        mean = np.mean(image[mask])
        std = np.std(image[mask])
        n_sigma = integration_params["region_growth_minimum_sigma"]
        integrator.region_grower.min_intensity = mean + n_sigma * std

    # 4. Integrate
    int_result, hulls, refined_centers = integrator.integrate_peaks(
        physical_bank, image, centers, return_hulls=True
    )

    bank_intensity = np.array([res[3] for res in int_result])
    bank_sigma = np.array([res[5] for res in int_result])

    # Strict Keeping (Hull Required)
    keep = []
    for idx, res in enumerate(int_result):
        has_hull = hulls[idx][1] is not None
        is_valid = res[3] is not None
        keep.append(is_valid and has_hull)

    # 5. Refine centers (DEPRECATED: Keep predicted centers for finder)
    # i, j = refined_centers[keep, 0], refined_centers[keep, 1]
    i, j = i[keep], j[keep]

    # 6. Visualization
    if do_viz:
        import matplotlib.pyplot as plt

        # Force Agg backend to avoid thread issues
        current_backend = plt.get_backend()
        if current_backend.lower() != "agg":
            plt.switch_backend("Agg")
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(1 + image, norm="log", cmap="binary", origin="lower")
            axes[0].scatter(j, i, marker="1", c="blue")
            axes[0].set_title(f"Candidates ({img_label}, Bank {physical_bank})")
            axes[1].imshow(1 + image, norm="log", cmap="binary", origin="lower")
            forbidden = ~mask
            if np.any(forbidden):
                overlay = np.zeros((*forbidden.shape, 4))
                overlay[forbidden] = [0, 1, 1, 0.3]
                axes[1].imshow(overlay, origin="lower")
            for _, hull, _, _ in hulls:
                if hull is not None:
                    for simplex in hull.simplices:
                        axes[1].plot(
                            hull.points[simplex, 1],
                            hull.points[simplex, 0],
                            c="red",
                        )
            axes[1].set_title("Integrated Hulls")
            fname = f"{img_label}_bank{physical_bank}.png"
            if viz_prefix is not None:
                fname = f"{viz_prefix}_{fname}"
            fig.savefig(fname)
            plt.close(fig)
        except Exception as e:
            print(f"Visualization failed for {img_label}: {e}")

    # 6. Gather Results
    if sum(keep) > 0:
        intensities = bank_intensity[keep]
        sigmas = bank_sigma[keep]
        tt, az = det.pixel_to_angles(i, j)
        lab_coords = det.pixel_to_lab(i, j)
        if lab_coords.ndim == 1:
            lab_coords = lab_coords[np.newaxis, :]
        num = len(tt)

        # Radii Calculation
        radii = []
        kept_indices = np.where(keep)[0]
        tt_rad, az_rad = np.deg2rad(tt), np.deg2rad(az)
        v_centers = np.stack(
            [
                np.sin(tt_rad) * np.cos(az_rad),
                np.sin(tt_rad) * np.sin(az_rad),
                np.cos(tt_rad),
            ],
            axis=1,
        )
        for k_idx, orig_idx in enumerate(kept_indices):
            _, hull, _, _ = hulls[orig_idx]
            if hull is None:
                radii.append(0.0)
                continue
            verts = hull.points[hull.vertices]
            v_i, v_j = verts[:, 0], verts[:, 1]
            v_tt, v_az = det.pixel_to_angles(v_i, v_j, sample_offset=None)
            v_tt_r, v_az_r = np.deg2rad(v_tt), np.deg2rad(v_az)
            v_vecs = np.stack(
                [
                    np.sin(v_tt_r) * np.cos(v_az_r),
                    np.sin(v_tt_r) * np.sin(v_az_r),
                    np.cos(v_tt_r),
                ],
                axis=1,
            )
            dots = np.clip(v_vecs @ v_centers[k_idx], -1.0, 1.0)
            radii.append(np.max(np.arccos(dots)))

        res = {
            "two_theta": tt.tolist(),
            "az_phi": az.tolist(),
            "R": [gonio_R] * num,
            "lamda_min": [wl_min] * num,
            "lamda_max": [wl_max] * num,
            "intensity": intensities.tolist(),
            "sigma": sigmas.tolist(),
            "radii": radii,
            "xyz": lab_coords.tolist(),
            "banks": [physical_bank] * num,
            "image_indices": [img_key] * num,
            "gonio_angles": [gonio_angles] * num if gonio_angles is not None else [],
            "count": num,
        }
        log_msg = (
            f"Integrated {len(i)}/{len(centers)} peaks for {img_label} "
            f"(Bank {physical_bank})"
        )
    else:
        res = None
        log_msg = f"{img_label} (Bank {physical_bank}) had 0 valid peaks"
    return res, log_msg


def _predict_single_bank(
    bank_id,
    det_config,
    unit_cell_params,
    RUB,
    wavelength_min,
    wavelength_max,
    sample_offset,
    ki_vec,
    R_all=None,
):
    """
    Worker function for predicting peaks on a single detector bank.
    Generates HKLs locally (lazy generation) to reduce IPC overhead.
    """
    # 1. Generate Reflections locally
    a, b, c, alpha, beta, gamma, space_group, d_min = unit_cell_params
    h, k, l = generate_reflections(  # noqa: E741
        a, b, c, alpha, beta, gamma, space_group, d_min
    )

    det = Detector(det_config)
    row, col, h_f, k_f, l_f, wl_f = predict_reflections_on_panel(
        detector=det,
        h=h,
        k=k,
        l=l,
        RUB=RUB,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        sample_offset=sample_offset,
        ki_vec=ki_vec,
        R_all=R_all,
    )
    if len(row) > 0:
        return bank_id, [row, col, h_f, k_f, l_f, wl_f]
    return bank_id, None


def _integrate_single_bank(
    bank_id,
    physical_bank,
    image,
    peaks,
    det_config,
    integration_params,
    integration_method,
    viz_info,
    metrics_info,
):
    # Unpack predicted peaks (i, j, h, k, l, wl)
    bank_i, bank_j, bank_h, bank_k, bank_l, bank_wl = peaks
    centers = np.stack([bank_i, bank_j], axis=-1)

    det = Detector(det_config)

    # --- METRICS: Comparison with found peaks ---
    metrics_str = ""
    (
        found_peaks_xyz,
        found_peaks_bank,
        found_peaks_run,
        run_id,
        RUB,
        current_angles_val,
        current_R_val,
        sample_offset,
        ki_vec,
    ) = metrics_info

    # CORRECTED: Account for Sample-frame offset rotation
    s_lab = (
        current_R_val @ sample_offset if current_R_val is not None else sample_offset
    )
    bank_tt, bank_az = det.pixel_to_angles(bank_i, bank_j, sample_offset=s_lab)

    # Correctly handle lab coordinate shape (N, 3)
    lab_coords = det.pixel_to_lab(bank_i, bank_j)
    if lab_coords.ndim == 1:
        lab_coords = lab_coords[np.newaxis, :]  # (1, 3) -> (N=1, 3)

    if found_peaks_xyz is not None and len(centers) > 0:
        f_xyz_valid = np.array([])

        # Priority 1: Filter by run index (for merged multi-run files)
        if found_peaks_run is not None:
            mask_run = found_peaks_run == run_id
            f_xyz_valid = found_peaks_xyz[mask_run]
        # Priority 2: Filter by physical bank ID
        elif found_peaks_bank is not None:
            mask_bank = found_peaks_bank == physical_bank
            f_xyz_valid = found_peaks_xyz[mask_bank]
        # Priority 3: Spatial proximity (for single files)
        else:
            s_off = sample_offset if sample_offset is not None else np.zeros(3)
            det_vec = det.center - s_off
            f_vecs = found_peaks_xyz - s_off
            dots = np.dot(f_vecs, det_vec)
            f_xyz_front = found_peaks_xyz[dots > 0]

            if len(f_xyz_front) > 0:
                f_row, f_col = det.lab_to_pixel(
                    f_xyz_front[:, 0],
                    f_xyz_front[:, 1],
                    f_xyz_front[:, 2],
                    clip=False,
                )
                on_sensor = (
                    (f_row >= 0) & (f_row < det.n) & (f_col >= 0) & (f_col < det.m)
                )
                f_xyz_valid = f_xyz_front[on_sensor]

        if len(f_xyz_valid) > 0:
            f_row_valid, f_col_valid = det.lab_to_pixel(
                f_xyz_valid[:, 0],
                f_xyz_valid[:, 1],
                f_xyz_valid[:, 2],
                clip=False,
            )
            on_panel_found = (
                (f_row_valid >= 0)
                & (f_row_valid < det.n)
                & (f_col_valid >= 0)
                & (f_col_valid < det.m)
            )

            if np.sum(on_panel_found) > 0:
                f_row_valid = f_row_valid[on_panel_found]
                f_col_valid = f_col_valid[on_panel_found]
                f_xyz_valid = f_xyz_valid[on_panel_found]

                f_pixels = np.stack([f_row_valid, f_col_valid], axis=1)
                p_pixels = np.stack([bank_i, bank_j], axis=1)

                tree = scipy.spatial.KDTree(p_pixels)
                dists_pix, idxs = tree.query(f_pixels)
                valid_matches = dists_pix < 20.0

                if np.sum(valid_matches) > 0:
                    matched_idxs = idxs[valid_matches]
                    f_xyz_matched = f_xyz_valid[valid_matches]
                    d_err, ang_err = calculate_angular_error(
                        f_xyz_matched,
                        bank_h[matched_idxs],
                        bank_k[matched_idxs],
                        bank_l[matched_idxs],
                        bank_wl[matched_idxs],
                        RUB,
                        sample_offset,
                        ki_vec,
                        current_R_val,
                    )
                    metrics_str = (
                        f" | Med Error: $\\Delta\\theta$={np.median(ang_err):.2f}$^\\circ$, "
                        f"$\\Delta d$={np.median(d_err):.3f}$\\AA$"
                    )

    # --- INTEGRATION ---
    mask_file, mask_erosion = (
        integration_params.get("integration_mask_file"),
        integration_params.get("integration_mask_rel_erosion_radius", 0.05),
    )
    if mask_file is not None:
        mask_im = np.array(Image.open(mask_file))
        radius = max(1, int(min(mask_im.shape) * mask_erosion))
        kernel = np.ones((radius, radius), dtype=np.uint8)
        mask = cv2.erode(mask_im, kernel).astype(bool)
    else:
        mask = np.full(image.shape, True)

    integrator = PeakIntegrator.build_from_dictionary(integration_params.copy())
    if integration_params.get("region_growth_minimum_sigma") is not None:
        mean = np.mean(image[mask])
        std = np.std(image[mask])
        n_sigma = integration_params["region_growth_minimum_sigma"]
        integrator.region_grower.min_intensity = mean + n_sigma * std

    valid_indices = []
    for idx, (r, c) in enumerate(centers):
        r_int, c_int = int(r), int(c)
        if (
            0 <= r_int < mask.shape[0]
            and 0 <= c_int < mask.shape[1]
            and mask[r_int, c_int]
        ):
            valid_indices.append(idx)

    centers = centers[valid_indices]
    bank_tt = bank_tt[valid_indices]
    bank_az = bank_az[valid_indices]
    bank_h = bank_h[valid_indices]
    bank_k = bank_k[valid_indices]
    bank_l = bank_l[valid_indices]
    bank_wl = bank_wl[valid_indices]
    lab_coords = lab_coords[valid_indices]

    int_result, hulls, refined_centers = integrator.integrate_peaks(
        bank_id,
        image,
        centers,
        integration_method=integration_method,
        return_hulls=True,
    )

    bank_intensity = np.array([res[3] for res in int_result])
    bank_sigma = np.array([res[5] for res in int_result])

    keep = []
    for idx, res in enumerate(int_result):
        has_hull = hulls[idx][1] is not None
        is_valid = res[3] is not None
        keep.append(is_valid and has_hull)

    # Re-calculate angles and lab coordinates using refined centers
    # Only for kept peaks
    if not np.any(keep):
        return None

    kept_centers = refined_centers[keep]
    # Re-calculate angles and lab coordinates
    bank_tt, bank_az = det.pixel_to_angles(
        kept_centers[:, 0], kept_centers[:, 1], sample_offset=s_lab
    )
    lab_coords = det.pixel_to_lab(kept_centers[:, 0], kept_centers[:, 1])
    if lab_coords.ndim == 1:
        lab_coords = lab_coords[np.newaxis, :]

    # --- VISUALIZATION ---
    # UPDATED: Use viz_label for filename
    do_viz, viz_prefix, viz_label = viz_info
    if do_viz:
        import matplotlib.pyplot as plt

        if plt.get_backend().lower() != "agg":
            plt.switch_backend("Agg")
        plt.rc("font", size=8)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(1 + image, norm="log", cmap="binary", origin="lower")
        axes[0].set_title(f"{viz_label}")

        label_pred = f"Predicted{metrics_str}"
        if len(centers) > 0:
            axes[0].scatter(
                centers[:, 1],
                centers[:, 0],
                marker="1",
                c="blue",
                label=label_pred,
                s=40,
            )

        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=False,
            ncol=1,
        )

        for p_i, p_j, p_h, p_k, p_l in zip(
            centers[:, 0], centers[:, 1], bank_h, bank_k, bank_l, strict=False
        ):
            is_zone = (p_h == 0) or (p_k == 0) or (p_l == 0)
            is_nodal = (abs(p_h) + abs(p_k) + abs(p_l)) < 8
            if is_zone or is_nodal:
                c = "red" if is_nodal else "black"
                w = "bold" if is_nodal else "normal"
                axes[0].text(
                    p_j,
                    p_i,
                    f"({p_h},{p_k},{p_l})",
                    color=c,
                    fontsize=6,
                    fontweight=w,
                    clip_on=True,
                )

        axes[1].imshow(1 + image, norm="log", cmap="binary", origin="lower")
        forbidden = ~mask
        overlay = np.zeros((*forbidden.shape, 4))
        overlay[forbidden] = [0, 1, 1, 0.3]
        axes[1].imshow(overlay, origin="lower")
        axes[1].set_title("Integrated peaks")

        for _, hull, _, _ in hulls:
            if hull is not None:
                for simplex in hull.simplices:
                    axes[1].plot(
                        hull.points[simplex, 1],
                        hull.points[simplex, 0],
                        c="red",
                    )

        # New Naming: {prefix}_{label}_int.png
        # viz_label already includes _bank{physical_bank} from Peaks.integrate
        out_name = f"{viz_label}_int.png"
        if viz_prefix:
            out_name = f"{viz_prefix}_{out_name}"
        fig.savefig(out_name, bbox_inches="tight")
        plt.close(fig)

    return {
        "h": bank_h[keep],
        "k": bank_k[keep],
        "l": bank_l[keep],
        "intensity": bank_intensity[keep],
        "sigma": bank_sigma[keep],
        "tt": bank_tt,
        "az": bank_az,
        "wavelength": bank_wl[keep],
        "xyz": lab_coords.tolist(),
        "bank": [bank_id] * sum(keep),
        "run_id": [run_id] * sum(keep),
        "R": [current_R_val] * sum(keep) if current_R_val is not None else [],
        "angles": [current_angles_val] * sum(keep)
        if current_angles_val is not None
        else [],
    }


# NOTE(Vivek): currently user provided values are overriden (matches original logic), but i'm pretty sure it should be the other way around. Looking at wavelength, user input is prioritized over files.
def init_goniometer(filename, ext, instrument, is_merged, axes=None, angles=None):
    """
    Build goniometer with the following priority
    - merged hdf5
    - nexus
    - manual override
    - default (assume beam aligned)
    """

    if ext == ".h5" and is_merged:
        with h5py.File(filename, "r") as f:
            if axes is None or angles is None:
                if "goniometer/axes" in f and "goniometer/angles" in f:
                    return Goniometer.from_h5_file(f)

    if ext == ".h5" and not is_merged:
        try:
            return Goniometer.from_nexus(filename, instrument)
        except Exception as e:
            print(f"Warning: Failed to extract goniometer from nexus file: {e}")

    if axes is not None and angles is not None:
        rot = Goniometer.get_rotation(axes, angles)
        return Goniometer(
            axes_raw=axes, angles_raw=angles, names_raw=None, rotation=rot
        )

    return Goniometer()


def init_wavelength(filename, ext, instrument, is_merged, min=None, max=None):
    settings = reduction_settings[instrument]
    wmin, wmax = settings.get("Wavelength")

    if ext == ".h5" and is_merged:
        with h5py.File(filename, "r") as f:
            if "instrument/wavelength" in f:
                wl = f["instrument/wavelength"][()]
                wmin, wmax = float(wl[0]), float(wl[1])

    min = min if min is not None else wmin
    max = max if max is not None else wmax

    return Wavelength(min, max)


def init_ims(filename, ext, is_merged):
    if ext == ".h5":
        if is_merged:
            ims = _integration.loader.load_merged_h5(filename)
        else:
            ims = _integration.loader.load_nexus(filename)
    else:
        ims = {0: np.array(Image.open(filename))}

    return ims


def _check_if_merged(filename, ext) -> bool:
    if ext != ".h5":
        return False
    try:
        with h5py.File(filename, "r") as f:
            return "images" in f
    except OSError:
        raise OSError


class Peaks:
    def __init__(
        self,
        filename: str,
        instrument: str,
        goniometer_axes: list[list[float]] | None = None,
        goniometer_angles: list[float] | None = None,
        wavelength_min: float | None = None,
        wavelength_max: float | None = None,
    ):
        name, ext = os.path.splitext(filename)
        self.filename = filename
        self.instrument = instrument

        self.bank_mapping = {}

        is_merged = _check_if_merged(filename, ext)
        if is_merged:
            print(f"Detected Merged HDF5 file format: {filename}")

        self.goniometer = init_goniometer(
            filename, ext, instrument, is_merged, goniometer_axes, goniometer_angles
        )
        self.wavelength = init_wavelength(
            filename, ext, instrument, is_merged, min=wavelength_min, max=wavelength_max
        )
        self.ims = init_ims(filename, ext, is_merged)

    def get_detector(self, bank: int) -> Detector:
        if bank in self.bank_mapping:
            physical_bank = self.bank_mapping[bank]
        else:
            physical_bank = bank
        bank_id = str(physical_bank)
        det_config = beamlines[self.instrument][bank_id]
        return Detector(det_config)

    def get_run_id(self, img_key: int) -> int:
        """Helper to resolve the run ID for an image key."""
        if hasattr(self, "file_offsets") and self.file_offsets is not None:
            return int(np.searchsorted(self.file_offsets, img_key, side="right") - 1)
        return 0

    def get_image_label(self, img_key):
        """Helper to resolve a readable label for an image key."""
        if (
            hasattr(self, "image_files_raw")
            and self.image_files_raw
            and hasattr(self, "file_offsets")
        ):
            file_idx = bisect.bisect_right(self.file_offsets, img_key) - 1
            if 0 <= file_idx < len(self.image_files_raw):
                orig_name = os.path.basename(self.image_files_raw[file_idx])
                clean_name = os.path.splitext(orig_name)[0]
                clean_name = clean_name.replace(".nxs.h5", "").replace(".h5", "")
                return clean_name
        return f"img{img_key}"

    def get_detector_peaks(
        self,
        harvest_peaks_kwargs: dict,
        integration_params: dict,
        show_progress: bool = False,
        visualize: bool = False,
        file_prefix: str | None = None,
        max_workers: int = None,
    ) -> DetectorPeaks:
        if not self.ims:
            raise Exception("ERROR: Must have images for Peaks first...")

        # --- Define outputs ---
        R: list[float] = []
        two_theta: list[float] = []
        az_phi: list[float] = []
        lamda_min: list[float] = []
        lamda_max: list[float] = []
        intensity: list[float] = []
        sigma: list[float] = []
        radii: list[float] = []
        xyz_out: list[list[float]] = []
        banks: list[int] = []
        image_indices: list[int] = []
        run_ids: list[int] = []
        gonio_angles_out: list[list[float]] = []

        finder_algorithm = harvest_peaks_kwargs.pop("algorithm")

        # --- BATCH PRE-PROCESSING (SparseRBF) ---
        precomputed_peaks = {}
        if finder_algorithm == "sparse_rbf":
            img_keys = sorted(self.ims.keys())
            images_list = [self.ims[k] for k in img_keys]
            img_stack = np.stack(images_list)

            alg = SparseRBFPeakFinder(
                alpha=harvest_peaks_kwargs.get("alpha", 0.1),
                gamma=harvest_peaks_kwargs.get("gamma", 2.0),
                min_sigma=harvest_peaks_kwargs.get("min_sigma", 1.0),
                max_sigma=harvest_peaks_kwargs.get("max_sigma", 10.0),
                max_peaks=harvest_peaks_kwargs.get("max_peaks", 500),
                chunk_size=harvest_peaks_kwargs.get("chunk_size", 1024),
                show_steps=harvest_peaks_kwargs.get("show_steps", False),
            )
            batch_coords = alg.find_peaks_batch(img_stack)
            precomputed_peaks = {
                k: c for k, c in zip(img_keys, batch_coords, strict=False)
            }

        # --- PREPARE PARALLEL TASKS ---
        tasks = []
        for img_key in sorted(self.ims.keys()):
            if hasattr(self, "bank_mapping") and img_key in self.bank_mapping:
                physical_bank = self.bank_mapping[img_key]
            else:
                physical_bank = img_key

            img_label = self.get_image_label(img_key)

            # FIX: Skip banks that are not in beamlines config
            if str(physical_bank) not in beamlines[self.instrument]:
                print(
                    f"WARNING: Bank {physical_bank} not found in beamlines config "
                    f"for {self.instrument}. Skipping..."
                )
                continue

            det_config = beamlines[self.instrument][str(physical_bank)]

            if self.goniometer.rotation.ndim == 3:
                current_R = (
                    self.goniometer.rotation[img_key]
                    if img_key < len(self.goniometer.rotation)
                    else self.goniometer.rotation[-1]
                )
            else:
                current_R = self.goniometer.rotation

            current_angles = None
            if self.goniometer.angles_raw is not None:
                if self.goniometer.angles_raw.ndim == 2:
                    current_angles = (
                        self.goniometer.angles_raw[img_key]
                        if img_key < len(self.goniometer.angles_raw)
                        else self.goniometer.angles_raw[-1]
                    )
                else:
                    current_angles = self.goniometer.angles_raw

            pre_coords = None
            if finder_algorithm == "sparse_rbf":
                coords = precomputed_peaks[img_key]
                pre_coords = (coords[:, 0], coords[:, 1])

            finder_info = (finder_algorithm, harvest_peaks_kwargs, pre_coords)
            mask_info = (
                harvest_peaks_kwargs.get("mask_file"),
                harvest_peaks_kwargs.get("mask_rel_erosion_radius"),
            )
            geo_info = (
                current_R,
                current_angles,
                self.wavelength.min,
                self.wavelength.max,
            )
            viz_info = (visualize, file_prefix)

            self.get_run_id(img_key)

            tasks.append(
                (
                    img_key,
                    img_label,
                    physical_bank,
                    self.ims[img_key],
                    det_config,
                    finder_info,
                    integration_params,
                    mask_info,
                    geo_info,
                    viz_info,
                )
            )

        print(f"Starting parallel integration of {len(tasks)} images...")

        # Use 'spawn' to be safe with JAX threading
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
            # FIX: Map futures to img_key to preserve order
            future_to_key = {
                executor.submit(_process_single_image, *t): t[0] for t in tasks
            }

            results_by_key = {}
            for future in tqdm(
                as_completed(future_to_key),
                total=len(future_to_key),
                desc="Integrating",
                disable=not show_progress,
            ):
                img_key = future_to_key[future]
                try:
                    res, msg = future.result()
                    if show_progress:
                        tqdm.write(msg)
                    if res:
                        results_by_key[img_key] = res
                except Exception as e:
                    print(f"Worker failed for image {img_key}: {e}")

            # Assemble results in DETERMINISTIC (sorted) order
            for img_key in sorted(self.ims.keys()):
                res = results_by_key.get(img_key)
                if res:
                    two_theta.extend(res["two_theta"])
                    az_phi.extend(res["az_phi"])
                    R.extend(res["R"])
                    lamda_min.extend(res["lamda_min"])
                    lamda_max.extend(res["lamda_max"])
                    intensity.extend(res["intensity"])
                    sigma.extend(res["sigma"])
                    radii.extend(res["radii"])
                    xyz_out.extend(res["xyz"])
                    banks.extend(res["banks"])
                    actual_img_key = res["image_indices"][0]
                    image_indices.extend(res["image_indices"])
                    run_ids.extend([self.get_run_id(actual_img_key)] * res["count"])
                    if res["gonio_angles"]:
                        gonio_angles_out.extend(res["gonio_angles"])

        return DetectorPeaks(
            R,
            two_theta,
            az_phi,
            lamda_min,
            lamda_max,
            intensity,
            sigma,
            radii,
            xyz_out,
            banks,
            image_indices,
            run_ids,
            self.goniometer.axes_raw,
            gonio_angles_out,
            self.goniometer.names_raw,
        )

    def predict_peaks(
        self,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        d_min,
        RUB,
        space_group="P 1",
        sample_offset=None,
        ki_vec=None,
        R_all=None,
        max_workers: int = None,
    ):
        """
        Predicts peak positions using parallel processing.
        Handles RUB as either a single (3,3) matrix OR a stack (N,3,3)
        for rotation scans.
        Generates HKLs locally (lazy generation) to reduce IPC overhead.
        """

        peak_dict = {}
        tasks = []

        # Package scalar params to send to workers
        unit_cell_params = (a, b, c, alpha, beta, gamma, space_group, d_min)

        sorted_keys = sorted(self.ims.keys())
        use_stack = RUB.ndim == 3 and RUB.shape[0] > 1

        print(f"Predicting peaks for {len(self.ims)} banks...")

        for _i, bank in enumerate(sorted_keys):
            det_config = beamlines[self.instrument][
                str(self.bank_mapping.get(bank, bank))
            ]
            run_id = self.get_run_id(bank)

            if use_stack:
                # Use image index if stack matches image count, otherwise use run index
                idx = bank if RUB.shape[0] == len(self.ims) else run_id
                if idx >= RUB.shape[0]:
                    idx = -1
                rub_val = RUB[idx]
            else:
                rub_val = RUB if RUB.ndim == 2 else RUB[0]

            # Resolve current R for this bank (used for sample offset rotation)
            current_R_val = None
            if R_all is not None:
                if R_all.ndim == 3:
                    idx_r = bank if R_all.shape[0] == len(self.ims) else run_id
                    if idx_r < R_all.shape[0]:
                        current_R_val = R_all[idx_r]
                    else:
                        current_R_val = R_all[0]
                else:
                    current_R_val = R_all

            tasks.append(
                (
                    bank,
                    det_config,
                    unit_cell_params,  # Pass tuple instead of arrays
                    rub_val,
                    self.wavelength.min,
                    self.wavelength.max,
                    sample_offset,
                    ki_vec,
                    current_R_val,
                )
            )

        # Use 'spawn' to be safe with JAX threading
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
            futures = [executor.submit(_predict_single_bank, *t) for t in tasks]

            # Map results to a temporary list to allow sorting
            results_list = []
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Predicting"
            ):
                try:
                    bank_id, res = future.result()
                    if res:
                        results_list.append((bank_id, res))
                except Exception as e:
                    print(f"Prediction failed for a bank: {e}")

            # Insert into dict in sorted order
            for bank_id, res in sorted(results_list, key=lambda x: x[0]):
                peak_dict[bank_id] = res

        return peak_dict

    # --- UPDATED INTEGRATE ---
    def integrate(
        self,
        peak_dict,
        integration_params,
        RUB,
        R_stack=None,
        angles_stack=None,
        sample_offset=None,
        ki_vec=None,
        integration_method="free_fit",
        create_visualizations=False,
        show_progress=False,
        file_prefix=None,
        found_peaks_file=None,
        max_workers=None,
    ):
        h, k, l = [], [], []  # noqa: E741
        intensity, sigma = [], []
        tt, az = [], []
        wavelength = []
        banks = []
        run_ids = []
        xyz = []
        R_out = []
        angles_out = []

        found_peaks_xyz = None
        found_peaks_bank = None
        found_peaks_run = None
        if found_peaks_file is not None:
            try:
                import h5py

                print(f"Loading found peaks from: {found_peaks_file}")
                with h5py.File(found_peaks_file, "r") as f:
                    if "files" in f and "file_offsets" in f and "peaks/xyz" in f:
                        files_db = f["files"][()]
                        offsets = f["file_offsets"][()]
                        target_name = os.path.basename(self.filename)
                        match_idxs = []
                        # 1. Direct match
                        for i, fname_bytes in enumerate(files_db):
                            fname_str = (
                                fname_bytes.decode("utf-8")
                                if isinstance(fname_bytes, bytes)
                                else str(fname_bytes)
                            )
                            if target_name in fname_str:
                                match_idxs.append(i)

                        # 2. Match via source files (if self is a merged master)
                        if (
                            not match_idxs
                            and hasattr(self, "image_files_raw")
                            and self.image_files_raw
                        ):
                            for src_file in self.image_files_raw:
                                src_name = os.path.basename(src_file)
                                for i, fname_bytes in enumerate(files_db):
                                    fname_str = (
                                        fname_bytes.decode("utf-8")
                                        if isinstance(fname_bytes, bytes)
                                        else str(fname_bytes)
                                    )
                                    if src_name == os.path.basename(fname_str):
                                        if i not in match_idxs:
                                            match_idxs.append(i)

                        if match_idxs:
                            # Load and concatenate from all matched indices
                            xyz_list = []
                            bank_list = []
                            run_list = []
                            for idx in match_idxs:
                                start = int(offsets[idx])
                                end = (
                                    int(offsets[idx + 1])
                                    if idx < len(files_db) - 1
                                    else f["peaks/xyz"].shape[0]
                                )
                                xyz_list.append(f["peaks/xyz"][start:end])
                                if "bank" in f:
                                    bank_list.append(f["bank"][start:end])
                                elif "peaks/bank" in f:
                                    bank_list.append(f["peaks/bank"][start:end])

                                if "peaks/run_index" in f:
                                    run_list.append(f["peaks/run_index"][start:end])

                            found_peaks_xyz = (
                                np.concatenate(xyz_list, axis=0) if xyz_list else None
                            )
                            found_peaks_bank = (
                                np.concatenate(bank_list, axis=0) if bank_list else None
                            )
                            found_peaks_run = (
                                np.concatenate(run_list, axis=0) if run_list else None
                            )
                    elif "peaks/xyz" in f:
                        found_peaks_xyz = f["peaks/xyz"][()]
                        if "bank" in f:
                            found_peaks_bank = f["bank"][()]
                        elif "peaks/bank" in f:
                            found_peaks_bank = f["peaks/bank"][()]
                        if "peaks/run_index" in f:
                            found_peaks_run = f["peaks/run_index"][()]
            except Exception as e:
                print(f"Failed to load found peaks: {e}")

        tasks = []
        os.path.basename(self.filename)

        for bank, peaks in peak_dict.items():
            physical_bank = self.bank_mapping.get(bank, bank)
            det_config = beamlines[self.instrument][str(physical_bank)]
            run_id = self.get_run_id(bank)

            # UPDATED: Generate nice labels for visualization
            img_label = self.get_image_label(bank)
            viz_label = f"{img_label}_bank{physical_bank}"

            # Handle RUB being a stack (N, 3, 3) or a single matrix (3, 3)
            if RUB.ndim == 3 and RUB.shape[0] > 1:
                # Use image index if stack matches image count, otherwise use run index
                idx = bank if RUB.shape[0] == len(self.ims) else run_id
                if idx >= RUB.shape[0]:
                    idx = -1
                current_rub = RUB[idx]
            else:
                current_rub = RUB if RUB.ndim == 2 else RUB[0]

            # Resolve R and angles for this image
            current_R_val = None
            if R_stack is not None:
                idx_r = bank if R_stack.shape[0] == len(self.ims) else run_id
                if idx_r < R_stack.shape[0]:
                    current_R_val = R_stack[idx_r]
                else:
                    current_R_val = R_stack[0]

            current_angles_val = None
            if angles_stack is not None:
                idx_a = bank if angles_stack.shape[0] == len(self.ims) else run_id
                if idx_a < angles_stack.shape[0]:
                    current_angles_val = angles_stack[idx_a]
                else:
                    current_angles_val = angles_stack[0]

            metrics_info = (
                found_peaks_xyz,
                found_peaks_bank,
                found_peaks_run,
                run_id,
                current_rub,
                current_angles_val,
                current_R_val,
                sample_offset,
                ki_vec,
            )
            # Pass viz_label instead of fname_clean
            viz_info = (create_visualizations, file_prefix, viz_label)

            tasks.append(
                (
                    bank,
                    physical_bank,
                    self.ims[bank],
                    peaks,
                    det_config,
                    integration_params,
                    integration_method,
                    viz_info,
                    metrics_info,
                )
            )

        print(f"Integrating {len(tasks)} banks in parallel...")

        # Use 'spawn' to be safe with JAX threading
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
            # FIX: Map futures to bank ID to preserve order
            future_to_bank = {
                executor.submit(_integrate_single_bank, *t): t[0] for t in tasks
            }

            results_by_bank = {}
            for future in tqdm(
                as_completed(future_to_bank),
                total=len(future_to_bank),
                desc="Integrating",
                disable=not show_progress,
            ):
                bank_id = future_to_bank[future]
                try:
                    res = future.result()
                    if res:
                        results_by_bank[bank_id] = res
                except Exception as e:
                    print(f"Integration worker failed for bank {bank_id}: {e}")

            # Assemble results in DETERMINISTIC (sorted) order
            for bank_id in sorted(peak_dict.keys()):
                res = results_by_bank.get(bank_id)
                if res:
                    h.extend(res["h"])
                    k.extend(res["k"])
                    l.extend(res["l"])
                    intensity.extend(res["intensity"])
                    sigma.extend(res["sigma"])
                    tt.extend(res["tt"])
                    az.extend(res["az"])
                    wavelength.extend(res["wavelength"])
                    xyz.extend(res["xyz"])
                    banks.extend(res["bank"])
                    run_ids.extend(res["run_id"])
                    R_out.extend(res["R"])
                    angles_out.extend(res["angles"])

        return IntegrationResult(
            h,
            k,
            l,
            intensity,
            sigma,
            tt,
            az,
            wavelength,
            banks,
            run_ids,
            xyz,
            R_out,
            angles_out,
        )

    def write_hdf5(
        self,
        output_filename: str,
        rotations: list[float],
        two_theta: list[float],
        az_phi: list[float],
        wavelength_mins: list[float],
        wavelength_maxes: list[float],
        intensity: list[float],
        sigma: list[float],
        radii: list[float],
        xyz: list[list[float]],
        bank: list[int],
        image_index: list[int] = None,
        run_id: list[int] = None,
        gonio_axes: list[list[float]] = None,
        gonio_angles: list[list[float]] = None,
        gonio_names: list[str] = None,
        instrument_wavelength: tuple[float, float] = None,
    ):
        with File(output_filename, "w") as f:
            f.attrs["instrument"] = self.instrument
            f["wavelength_mins"] = wavelength_mins
            f["wavelength_maxes"] = wavelength_maxes
            f["goniometer/R"] = rotations
            f["peaks/two_theta"] = two_theta
            f["peaks/azimuthal"] = az_phi
            f["peaks/intensity"] = intensity
            f["peaks/sigma"] = sigma
            f["peaks/radius"] = radii
            f["peaks/xyz"] = xyz
            f["bank"] = bank

            if image_index is not None:
                f["peaks/image_index"] = image_index

            if run_id is not None:
                f["peaks/run_index"] = run_id

            if hasattr(self, "image_files_raw") and self.image_files_raw:
                f["files"] = np.array([s.encode("utf-8") for s in self.image_files_raw])
                f["file_offsets"] = self.file_offsets

            if gonio_axes is not None:
                f["goniometer/axes"] = gonio_axes

            if gonio_angles is not None:
                f["goniometer/angles"] = gonio_angles

            if gonio_names is not None:
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset("goniometer/names", data=gonio_names, dtype=dt)

            if instrument_wavelength is not None:
                f["instrument/wavelength"] = instrument_wavelength
