from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

import h5py
import PIL.Image
import numpy as np
from tqdm import tqdm

from subhkl.integration.image_data import ImageData
from subhkl.integration.orchestrator import DetectorPeaks, IntegrationResult, Wavelength
from subhkl.integration import writer, worker, orchestrator
from subhkl.config import beamlines, reduction_settings
from subhkl.config.goniometer import Goniometer
from subhkl.detector import Detector
from subhkl.io.loader import ImageLoader


# NOTE(Vivek): currently user provided values are overriden (matches original logic), but i'm pretty sure it should be the other way around. Looking at wavelength, user input is prioritized over files.
def _init_goniometer(
    filename: str, ext: str, instrument: str, is_merged: bool, axes=None, angles=None
) -> Goniometer:
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


def _init_wavelength(
    filename: str, ext: str, instrument: str, is_merged: bool, min=None, max=None
) -> Wavelength:
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


def _init_ims(filename: str, ext: str, instrument: str, is_merged: bool) -> ImageData:
    if ext == ".h5":
        if is_merged:
            image_data = ImageLoader.load_merged_h5(filename)
        else:
            image_data = ImageLoader.load_nexus(filename, instrument)
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

        self.goniometer = _init_goniometer(
            filename, ext, instrument, is_merged, goniometer_axes, goniometer_angles
        )
        self.wavelength = _init_wavelength(
            filename, ext, instrument, is_merged, wavelength_min, wavelength_max
        )
        self.image = _init_ims(filename, ext, instrument, is_merged)

    def get_detector(self, bank: int) -> Detector:
        if bank in self.image.bank_mapping:
            physical_bank = self.image.bank_mapping[bank]
        else:
            physical_bank = bank
        bank_id = str(physical_bank)
        det_config = beamlines[self.instrument][bank_id]
        return Detector(det_config)

    def get_run_id(self, img_key: int) -> int:
        return self.image.get_run_id(img_key)

    def get_image_label(self, img_key):
        return self.image.get_label(img_key)

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

        tasks = orchestrator.prepare_harvest_tasks(
            self.image,
            self.instrument,
            self.goniometer,
            self.wavelength,
            harvest_peaks_kwargs,
            integration_params,
            visualize,
            file_prefix,
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

        return self._assemble_detector_peaks(results_by_key)

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
        show_progress: bool = False,
    ) -> dict:
        """
        Predicts peak positions using parallel processing.
        Handles RUB as either a single (3,3) matrix OR a stack (N,3,3)
        for rotation scans.
        Generates HKLs locally (lazy generation) to reduce IPC overhead.
        """

        peak_dict = {}
        tasks = orchestrator.prepare_predict_tasks(
            image_data=self.image,
            instrument=self.instrument,
            wavelength_min=self.wavelength.min,
            wavelength_max=self.wavelength.max,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            d_min=d_min,
            RUB=RUB,
            space_group=space_group,
            sample_offset=sample_offset,
            ki_vec=ki_vec,
            R_all=R_all,
        )

        # Use 'spawn' to be safe with JAX threading
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
            futures_to_bank = {
                executor.submit(worker.predict_single_bank, *t): t[0] for t in tasks
            }

            # Map results to a temporary list to allow sorting
            results_by_bank = {}
            for future in tqdm(
                as_completed(futures_to_bank),
                total=len(futures_to_bank),
                desc="Predicting",
                disable=not show_progress,
            ):
                bank_id = futures_to_bank[future]
                try:
                    _, res = future.result()
                    if res:
                        results_by_bank[bank_id] = res
                except Exception as e:
                    print(f"Prediction failed for bank {bank_id}: {e}")

        # Insert into dict in sorted order
        for bank_id in sorted(results_by_bank.keys()):
            peak_dict[bank_id] = results_by_bank[bank_id]

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
        tasks = orchestrator.prepare_integrate_tasks(
            self.image,
            self.filename,
            self.instrument,
            peak_dict,
            integration_params,
            RUB,
            R_stack,
            angles_stack,
            sample_offset,
            ki_vec,
            integration_method,
            create_visualizations,
            show_progress,
            file_prefix,
            found_peaks_file,
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

        return self._assemble_integration_result(peak_dict, results_by_bank)

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

    def _assemble_detector_peaks(self, results_by_key):
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

    def _assemble_integration_result(self, peak_dict, results_by_bank):
        h, k, l = [], [], []  # noqa: E741
        intensity, sigma = [], []
        tt, az = [], []
        wavelength = []
        banks = []
        run_ids = []
        xyz = []
        R_out = []
        angles_out = []

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
