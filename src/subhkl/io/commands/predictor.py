import h5py
import numpy as np

from subhkl.integration import Peaks
from subhkl.instrument.detector import calibrate_from_file

def run(
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
    calibrate_from_file(indexed_hdf5_filename, instrument)

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

        offsets = None
        if "optimization/goniometer_offsets" in f_idx:
            off_data = f_idx["optimization/goniometer_offsets"]
            if isinstance(off_data, h5py.Group):
                offsets = {k: off_data[k][()] for k in off_data.keys()}
            else:
                offsets = off_data[()]

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
            # --- SAFE NAMED MAPPING ---
            if isinstance(offsets, dict) and peaks.goniometer.names_raw is not None:
                mapped_offsets = np.array(
                    [offsets.get(name, 0.0) for name in peaks.goniometer.names_raw]
                )
            else:
                # Legacy array fallback
                motor_map = []
                if peaks.goniometer.names_raw is not None:
                    unique_motors = []
                    for name in peaks.goniometer.names_raw:
                        if name not in unique_motors:
                            unique_motors.append(name)
                        motor_map.append(unique_motors.index(name))
                else:
                    motor_map = list(range(len(peaks.goniometer.axes_raw)))
                mapped_offsets = np.array(
                    [offsets[motor_map[i]] for i in range(len(motor_map))]
                )

            angles_refined = peaks.goniometer.angles_raw + mapped_offsets[None, :]

            all_R = np.stack(
                [
                    calc_goniometer_rotation_matrix(peaks.goniometer.axes_raw, ang)
                    for ang in angles_refined
                ]
            )

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
