import os
from dataclasses import dataclass, astuple
import numpy as np
from typing import List, Any, Optional, Dict, Tuple

from .image_data import ImageData
from subhkl.config import beamlines
from subhkl.instrument.goniometer import Goniometer
from subhkl.search.sparse_rbf import SparseRBFPeakFinder

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

@dataclass(frozen=True)
class Wavelength:
    min: float = None
    max: float = None

    def __iter__(self):
        return iter((self.min, self.max))


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
    peak_rows: Optional[List[int]]
    peak_cols: Optional[List[int]]

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


def prepare_harvest_tasks(
    image_data: ImageData,
    instrument: str,
    goniometer: Goniometer,
    wavelength: Wavelength,
    harvest_peaks_kwargs: Dict[str, Any],
    integration_params: Dict[str, Any],
    visualize: bool,
    file_prefix: str,
) -> List[Tuple[Any, ...]]:
    ims = image_data.ims
    bank_mapping = image_data.bank_mapping

    finder_algorithm = harvest_peaks_kwargs.pop("algorithm")

    # --- BATCH PRE-PROCESSING (SparseRBF) ---
    precomputed_peaks = {}
    if finder_algorithm == "sparse_rbf":
        img_keys = sorted(ims.keys())
        images_list = [ims[k] for k in img_keys]
        img_stack = np.stack(images_list)

        border_width = harvest_peaks_kwargs.get("mask_rel_erosion_radius", 0)
        if border_width is None:
            border_width = 0.0
        border_width *= min(img_stack.shape[1], img_stack.shape[2])

        alg = SparseRBFPeakFinder(
            alpha=harvest_peaks_kwargs.get("alpha", 0.1),
            gamma=harvest_peaks_kwargs.get("gamma", 2.0),
            loss=harvest_peaks_kwargs.get("loss", "gaussian"),
            min_sigma=harvest_peaks_kwargs.get("min_sigma", 1.0),
            max_sigma=harvest_peaks_kwargs.get("max_sigma", 10.0),
            max_peaks=harvest_peaks_kwargs.get("max_peaks", 500),
            border_width=int(border_width),
            chunk_size=harvest_peaks_kwargs.get("chunk_size", 128),
            show_steps=harvest_peaks_kwargs.get("show_steps", False),
            auto_tune_alpha=harvest_peaks_kwargs.get("auto_tune_alpha", False),
            candidate_alphas=harvest_peaks_kwargs.get("candidate_alphas", None)
        )
        batch_coords = alg.find_peaks_batch(img_stack)
        precomputed_peaks = {k: c for k, c in zip(img_keys, batch_coords, strict=False)}

    tasks = []
    for img_key in sorted(ims.keys()):
        physical_bank = bank_mapping.get(img_key, img_key)
        img_label = image_data.get_label(img_key)

        # FIX: Skip banks that are not in beamlines config
        if str(physical_bank) not in beamlines[instrument]:
            print(
                f"WARNING: Bank {physical_bank} not found in beamlines config "
                f"for {instrument}. Skipping..."
            )
            continue

        det_config = beamlines[instrument][str(physical_bank)]

        if goniometer.rotation.ndim == 3:
            current_R = (
                goniometer.rotation[img_key]
                if img_key < len(goniometer.rotation)
                else goniometer.rotation[-1]
            )
        else:
            current_R = goniometer.rotation

        current_angles = None
        if goniometer.angles_raw is not None:
            if goniometer.angles_raw.ndim == 2:
                current_angles = (
                    goniometer.angles_raw[img_key]
                    if img_key < len(goniometer.angles_raw)
                    else goniometer.angles_raw[-1]
                )
            else:
                current_angles = goniometer.angles_raw

        pre_coords = None
        if finder_algorithm == "sparse_rbf":
            coords = precomputed_peaks[img_key]
            # coords shape is [intensity, r, c, sigma]
            if len(coords) > 0:
                pre_coords = (coords[:, 1], coords[:, 2])
            else:
                pre_coords = (np.array([]), np.array([]))

        finder_info = (finder_algorithm, harvest_peaks_kwargs, pre_coords)
        mask_info = (
            harvest_peaks_kwargs.get("mask_file"),
            harvest_peaks_kwargs.get("mask_rel_erosion_radius"),
        )
        geo_info = (
            current_R,
            current_angles,
            wavelength.min,
            wavelength.max,
        )
        viz_info = (visualize, file_prefix)

        image_data.get_run_id(img_key)

        tasks.append(
            (
                img_key,
                img_label,
                physical_bank,
                ims[img_key],
                det_config,
                finder_info,
                integration_params,
                mask_info,
                geo_info,
                viz_info,
            )
        )
    return tasks, precomputed_peaks


def prepare_predict_tasks(
    image_data: ImageData,
    instrument: str,
    wavelength_min: float,
    wavelength_max: float,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    d_min: float,
    RUB: np.ndarray,
    space_group: str = "P 1",
    sample_offset: Optional[np.ndarray] = None,
    ki_vec: Optional[np.ndarray] = None,
    R_all: Optional[np.ndarray] = None,
) -> List[Tuple[Any, ...]]:
    """Packages data for predict_single_bank workers."""
    ims = image_data.ims
    bank_mapping = image_data.bank_mapping
    tasks = []

    unit_cell_params = (a, b, c, alpha, beta, gamma, space_group, d_min)

    sorted_keys = sorted(image_data.ims.keys())
    if not sorted_keys:
        return []

    # Calculate exact physical bounds
    num_runs = max([image_data.get_run_id(k) for k in sorted_keys]) + 1
    max_img_key = max(sorted_keys)

    def _resolve(stack, img_key, r_id, name):
        if stack is None: 
            return None
            
        # Distinguish between single matrices/vectors and batched stacks
        is_batch = (stack.ndim == 3) or (stack.ndim == 2 and name == "angles_stack")
        if not is_batch:
            return stack
        
        n_items = stack.shape[0]
        if n_items == 1:
            return stack[0]
            
        # ToF Stack Validation: Must be large enough to be indexed by the highest image key
        if n_items > num_runs and n_items > max_img_key:
            return stack[img_key]
            
        # Run Stack Validation: Must match the exact number of physical runs
        if n_items == num_runs:
            return stack[r_id]
            
        raise ValueError(
            f"Geometry dimension mismatch for {name}: Stack length is {n_items}. "
            f"Expected exactly {num_runs} (for per-run) or >= {max_img_key + 1} (for per-image ToF)."
        )

    print(f"Predicting peaks for {len(ims)} banks...")

    for _i, bank in enumerate(sorted_keys):
        det_config = beamlines[instrument][str(bank_mapping.get(bank, bank))]
        run_id = image_data.get_run_id(bank)

        current_rub = _resolve(RUB, bank, run_id, "RUB")
        current_R_val = _resolve(R_all, bank, run_id, "R_all")

        tasks.append(
            (
                bank,
                det_config,
                unit_cell_params,
                current_rub,
                wavelength_min,
                wavelength_max,
                sample_offset,
                ki_vec,
                current_R_val,
            )
        )
    return tasks

def prepare_integrate_tasks(
    image: ImageData,
    filename: str,
    instrument: str,
    peak_dict: Dict[str, List[Any]],
    integration_params: Dict[str, Any],
    RUB: np.ndarray,
    R_stack: Optional[np.ndarray] = None,
    angles_stack: Optional[np.ndarray] = None,
    sample_offset: Optional[np.ndarray] = None,
    ki_vec: Optional[np.ndarray] = None,
    integration_method: str = "free_fit",
    create_visualizations: bool = False,
    show_progress: bool = False,
    file_prefix: Optional[str] = None,
    found_peaks_file: Optional[str] = None,
) -> List[Tuple[Any, ...]]:

    # ... [Keep all your existing found_peaks loading logic exactly as is] ...

    # [Code resumes below the found_peaks loading block]
    tasks = []
    os.path.basename(filename)

    sorted_keys = sorted(peak_dict.keys())
    if not sorted_keys:
        return []

    # Calculate exact physical bounds
    num_runs = max([image.get_run_id(k) for k in sorted_keys]) + 1
    max_img_key = max(sorted_keys)

    def _resolve(stack, img_key, r_id, name):
        if stack is None:
            return None

        is_batch = (stack.ndim == 3) or (stack.ndim == 2 and name == "angles_stack")
        if not is_batch:
            return stack

        n_items = stack.shape[0]
        if n_items == 1:
            return stack[0]

        if n_items > num_runs and n_items > max_img_key:
            return stack[img_key]

        if n_items == num_runs:
            return stack[r_id]

        raise ValueError(
            f"Geometry dimension mismatch for {name}: Stack length is {n_items}. "
            f"Expected exactly {num_runs} (for per-run) or >= {max_img_key + 1} (for per-image ToF)."
        )

    for bank, peaks in peak_dict.items():
        physical_bank = image.bank_mapping.get(bank, bank)
        det_config = beamlines[instrument][str(physical_bank)]
        run_id = image.get_run_id(bank)

        img_label = image.get_label(bank)
        viz_label = f"{img_label}_bank{physical_bank}"

        current_rub = _resolve(RUB, bank, run_id, "RUB")
        current_R_val = _resolve(R_stack, bank, run_id, "R_stack")
        current_angles_val = _resolve(angles_stack, bank, run_id, "angles_stack")

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
        viz_info = (create_visualizations, file_prefix, viz_label)

        tasks.append(
            (
                bank,
                physical_bank,
                image.ims[bank],
                peaks,
                det_config,
                integration_params,
                integration_method,
                viz_info,
                metrics_info,
            )
        )
    return tasks

