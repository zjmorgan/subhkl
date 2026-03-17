import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, astuple
import multiprocessing
import os
from typing import List, Any, Optional

import h5py
import PIL.Image
import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


from subhkl._integration.loader import ImageData
from subhkl._integration import writer, worker
from subhkl.config import beamlines, reduction_settings
from subhkl.config.goniometer import Goniometer
from subhkl.detector import Detector
from subhkl.sparse_rbf_peak_finder import SparseRBFPeakFinder


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


@dataclass(frozen=True)
class Wavelength:
    min: float = None
    max: float = None

    def __iter__(self):
        return iter((self.min, self.max))


# ==============================================================================
# WORKER FUNCTIONS (Multiprocessing)
# ==============================================================================


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


def init_ims(filename, ext, instrument, is_merged):
    if ext == ".h5":
        if is_merged:
            image_data = ImageData.load_merged_h5(filename)
        else:
            image_data = ImageData.load_nexus(filename, instrument)
        return image_data
    else:
        ims = {0: np.array(PIL.Image.open(filename))}

    return ImageData(ims)


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

        is_merged = _check_if_merged(filename, ext)
        if is_merged:
            print(f"Detected Merged HDF5 file format: {filename}")

        self.goniometer = init_goniometer(
            filename, ext, instrument, is_merged, goniometer_axes, goniometer_angles
        )
        self.wavelength = init_wavelength(
            filename, ext, instrument, is_merged, wavelength_min, wavelength_max
        )
        self.image = init_ims(filename, ext, instrument, is_merged)

    def get_detector(self, bank: int) -> Detector:
        if bank in self.image.bank_mapping:
            physical_bank = self.image.bank_mapping[bank]
        else:
            physical_bank = bank
        bank_id = str(physical_bank)
        det_config = beamlines[self.instrument][bank_id]
        return Detector(det_config)

    def get_run_id(self, img_key: int) -> int:
        """Helper to resolve the run ID for an image key."""
        if self.image.file_offsets is not None:
            return int(
                np.searchsorted(self.image.file_offsets, img_key, side="right") - 1
            )
        return 0

    def get_image_label(self, img_key):
        """Helper to resolve a readable label for an image key."""
        files = self.image.raw_files
        offsets = self.image.file_offsets

        if files and offsets is not None:
            file_idx = bisect.bisect_right(self.image.file_offsets, img_key) - 1
            if 0 <= file_idx < len(self.image.raw_files):
                orig_name = os.path.basename(self.image.raw_files[file_idx])
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
        if not self.image.ims:
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
            img_keys = sorted(self.image.ims.keys())
            images_list = [self.image.ims[k] for k in img_keys]
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
        for img_key in sorted(self.image.ims.keys()):
            physical_bank = self.image.bank_mapping.get(img_key, img_key)
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
                    self.image.ims[img_key],
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
                executor.submit(worker.process_single_image, *t): t[0] for t in tasks
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
            for img_key in sorted(self.image.ims.keys()):
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

        ims = self.image.ims
        bank_mapping = self.image.bank_mapping
        peak_dict = {}
        tasks = []

        # Package scalar params to send to workers
        unit_cell_params = (a, b, c, alpha, beta, gamma, space_group, d_min)

        sorted_keys = sorted(self.image.ims.keys())
        use_stack = RUB.ndim == 3 and RUB.shape[0] > 1

        print(f"Predicting peaks for {len(ims)} banks...")

        for _i, bank in enumerate(sorted_keys):
            det_config = beamlines[self.instrument][str(bank_mapping.get(bank, bank))]
            run_id = self.get_run_id(bank)

            if use_stack:
                # Use image index if stack matches image count, otherwise use run index
                idx = bank if RUB.shape[0] == len(ims) else run_id
                if idx >= RUB.shape[0]:
                    idx = -1
                rub_val = RUB[idx]
            else:
                rub_val = RUB if RUB.ndim == 2 else RUB[0]

            # Resolve current R for this bank (used for sample offset rotation)
            current_R_val = None
            if R_all is not None:
                if R_all.ndim == 3:
                    idx_r = bank if R_all.shape[0] == len(ims) else run_id
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
            futures = [executor.submit(worker.predict_single_bank, *t) for t in tasks]

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
                        if not match_idxs and self.image.raw_files:
                            for src_file in self.image.raw_files:
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
            physical_bank = self.image.bank_mapping.get(bank, bank)
            det_config = beamlines[self.instrument][str(physical_bank)]
            run_id = self.get_run_id(bank)

            # UPDATED: Generate nice labels for visualization
            img_label = self.get_image_label(bank)
            viz_label = f"{img_label}_bank{physical_bank}"

            # Handle RUB being a stack (N, 3, 3) or a single matrix (3, 3)
            if RUB.ndim == 3 and RUB.shape[0] > 1:
                # Use image index if stack matches image count, otherwise use run index
                idx = bank if RUB.shape[0] == len(self.image.ims) else run_id
                if idx >= RUB.shape[0]:
                    idx = -1
                current_rub = RUB[idx]
            else:
                current_rub = RUB if RUB.ndim == 2 else RUB[0]

            # Resolve R and angles for this image
            current_R_val = None
            if R_stack is not None:
                idx_r = bank if R_stack.shape[0] == len(self.image.ims) else run_id
                if idx_r < R_stack.shape[0]:
                    current_R_val = R_stack[idx_r]
                else:
                    current_R_val = R_stack[0]

            current_angles_val = None
            if angles_stack is not None:
                idx_a = bank if angles_stack.shape[0] == len(self.image.ims) else run_id
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
                    self.image.ims[bank],
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
                executor.submit(worker.integrate_single_bank, *t): t[0] for t in tasks
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

    def write_hdf5(self, output_filename, detector_peaks, instrument_wavelength):
        writer.write_hdf5(
            self,
            output_filename,
            detector_peaks.R,
            detector_peaks.two_theta,
            detector_peaks.az_phi,
            detector_peaks.wavelength_mins,
            detector_peaks.wavelength_maxes,
            detector_peaks.intensity,
            detector_peaks.sigma,
            detector_peaks.radii,
            detector_peaks.xyz,
            detector_peaks.bank,
            detector_peaks.image_index,
            detector_peaks.run_id,
            detector_peaks.gonio_axes,
            detector_peaks.gonio_angles,
            detector_peaks.gonio_names,
            instrument_wavelength,
        )
