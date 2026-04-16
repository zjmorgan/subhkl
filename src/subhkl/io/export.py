import sys

import gemmi
import h5py
import numpy as np


class BaseConcatenateMerger:
    def __init__(self, h5_files, copy_keys, merge_keys, per_file_keys=None):
        """
        Merges datasets by concatenation

        Parameters
        ----------
        h5_files : list[str]
            List of .h5 file paths
        copy_keys : list[str]
            List of keys in .h5 files to copy once
        merge_keys : list[str]
            List of keys in .h5 files to merge by concatenation
        per_file_keys : list[str]
            List of keys that are per-run/per-file (shape [N_runs, ...])
        """
        self.h5_files = sorted(list(set(h5_files)))
        self.copy_keys = copy_keys
        self.merge_keys = merge_keys
        self.per_file_keys = per_file_keys if per_file_keys is not None else []

    def merge(self, output_filename):
        """
        Merges the indexed datasets into a single dataset

        Parameters
        ----------
        output_filename : str
            Name of the output .h5 file to write the merged dataset to
        """

        total_peaks = 0
        total_runs = 0
        # Determine total peaks AND find a valid template file
        typical_file_path = self.h5_files[0]
        found_valid_template = False

        for file in self.h5_files:
            with h5py.File(file, "r") as f_in:
                num = len(f_in[self.merge_keys[0]])
                total_peaks += num

                num_runs_in_file = 0
                if self.per_file_keys:
                    # Assume each file is ONE run for now, OR check length of first per_file_key
                    if self.per_file_keys[0] in f_in:
                        num_runs_in_file = len(f_in[self.per_file_keys[0]])

                if num_runs_in_file == 0:
                    if "peaks/run_index" in f_in and len(f_in["peaks/run_index"]) > 0:
                        num_runs_in_file = int(np.max(f_in["peaks/run_index"]) + 1)
                    else:
                        num_runs_in_file = 1

                total_runs += num_runs_in_file

                # Use the first file with data as the template to ensure
                # multidimensional shapes (like 3x3 matrices) are preserved.
                if not found_valid_template and num > 0:
                    typical_file_path = file
                    found_valid_template = True

        with h5py.File(output_filename, "w") as f_out:
            # Open the valid template file we found (or the first one if all are empty)
            with h5py.File(typical_file_path, "r") as f_typical:
                for key in self.copy_keys:
                    if key in f_typical:
                        f_out[key] = np.array(f_typical[key])

                for merge_key in self.merge_keys:
                    # Use the shape from the typical file
                    shape = (total_peaks,) + f_typical[merge_key].shape[1:]
                    dtype = f_typical[merge_key].dtype
                    f_out.create_dataset(merge_key, shape, dtype)

                for per_file_key in self.per_file_keys:
                    if per_file_key in f_typical:
                        shape = (total_runs,) + f_typical[per_file_key].shape[1:]
                        dtype = f_typical[per_file_key].dtype
                        f_out.create_dataset(per_file_key, shape, dtype)

            offset = 0
            run_offset = 0
            f_out["files"] = np.array(
                list(map(lambda s: s.encode("utf-8"), self.h5_files))
            )
            f_out.create_dataset("file_offsets", (len(self.h5_files),), dtype=np.int64)

            for i_file, indexed_file in enumerate(self.h5_files):
                with h5py.File(indexed_file, "r") as f_in:
                    num_items = len(f_in[self.merge_keys[0]])
                    peak_range = slice(offset, offset + num_items)
                    f_out["file_offsets"][i_file] = offset

                    # 1. Merge per-peak keys
                    for merge_key in self.merge_keys:
                        if merge_key in f_in:
                            # Only copy if there is data to avoid shape mismatch on empty files
                            if num_items > 0:
                                data = np.array(f_in[merge_key])
                                # Increment run_index or image_index if it's per-file local
                                if (
                                    merge_key == "peaks/run_index"
                                    or merge_key == "peaks/image_index"
                                ):
                                    # We assign a global run index based on the running offset.
                                    # This ensures indices from multiple files do not collide.
                                    data += run_offset

                                f_out[merge_key][peak_range] = data
                        elif num_items > 0 and merge_key == "peaks/run_index":
                            # Fallback: if no run_index exists in input, we assign global run offset
                            f_out["peaks/run_index"][peak_range] = run_offset

                    # 2. Merge per-file/run keys
                    num_runs_in_file = 0
                    if self.per_file_keys:
                        for per_file_key in self.per_file_keys:
                            if per_file_key in f_in:
                                data = np.array(f_in[per_file_key])
                                n_r = len(data)
                                num_runs_in_file = max(num_runs_in_file, n_r)
                                run_range = slice(run_offset, run_offset + n_r)
                                f_out[per_file_key][run_range] = data

                    if num_runs_in_file == 0 and num_items > 0:
                        if "peaks/run_index" in f_in:
                            num_runs_in_file = int(np.max(f_in["peaks/run_index"]) + 1)
                        else:
                            num_runs_in_file = 1

                    offset += num_items
                    run_offset += num_runs_in_file


class FinderConcatenateMerger(BaseConcatenateMerger):
    def __init__(self, h5_files):
        merge_keys = [
            "wavelength_mins",
            "wavelength_maxes",
            "peaks/two_theta",
            "peaks/azimuthal",
            "peaks/intensity",
            "peaks/sigma",
            "peaks/radius",  # Added radius to merge keys
            "peaks/xyz",
            "bank",
            "peaks/image_index",
            "peaks/run_index",
            "goniometer/R",
            "goniometer/angles",
        ]
        per_file_keys = []
        copy_keys = ["goniometer/axes", "goniometer/names"]
        super().__init__(h5_files, copy_keys, merge_keys, per_file_keys=per_file_keys)


class MTZExporter:
    def __init__(self, peaks_file, space_group=None):
        with h5py.File(peaks_file) as f:
            self.a = float(np.array(f["sample/a"]))
            self.b = float(np.array(f["sample/b"]))
            self.c = float(np.array(f["sample/c"]))
            self.alpha = float(np.array(f["sample/alpha"]))
            self.beta = float(np.array(f["sample/beta"]))
            self.gamma = float(np.array(f["sample/gamma"]))

            if space_group is None:
                sg = f["sample/space_group"]
                space_group = sg.decode("utf-8") if isinstance(sg, bytes) else str(sg)

            self.h = np.array(f["peaks/h"])
            self.k = np.array(f["peaks/k"])
            self.l = np.array(f["peaks/l"])
            self.lamda = np.array(f["peaks/lambda"])
            self.theta = np.array(f["peaks/two_theta"]) / 2
            self.phi = np.array(f["peaks/azimuthal"])
            self.intensity = np.array(f["peaks/intensity"])
            self.sigma = np.array(f["peaks/sigma"])
            if "structure_factors" in f["peaks"].keys():
                self.f = np.array(f["peaks/structure_factors"])
                self.f_sigma = np.array(f["peaks/structure_factors_sigma"])
            else:
                self.f = None
                self.f_sigma = None

            if "run_index" in f["peaks"].keys():
                self.runs = np.array(f["peaks/run_index"])
            else:
                self.runs = np.zeros_like(self.h, dtype=np.int32)

            self.runs = 1000 * self.runs + f["peaks/bank"]

        self.space_group = space_group

    def write_mtz(self, filename):
        mtz = gemmi.Mtz(with_base=True)
        mtz.set_logging(sys.stdout)

        sg = gemmi.find_spacegroup_by_name(self.space_group)
        if sg is None:
            raise ValueError(f"Could not find space group: {self.space_group}")
        mtz.spacegroup = sg

        unit_cell = gemmi.UnitCell(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        )
        mtz.set_cell_for_all(unit_cell)

        mtz.add_column("I", "J")
        mtz.add_column("SIGI", "Q")
        if self.f is not None:
            mtz.add_column("FP", "F")
            mtz.add_column("SIGFP", "Q")
        mtz.add_column("WAVEL", "W")
        mtz.add_column("THETA", "W")
        mtz.add_column("PHI", "W")
        mtz.add_column("BATCH", "B")

        # Column order: h, k, l, I, sigI, [FP, sigFP,] wavel, theta, phi, batch
        n_base_cols = 9  # h, k, l, I, sigI, wavel, theta, phi, batch
        n_structure_factor_cols = 2  # FP, sigFP
        n_cols = n_base_cols + (n_structure_factor_cols if self.f is not None else 0)

        data = []

        for i in range(len(self.intensity)):
            h, k, l = self.h[i], self.k[i], self.l[i]  # noqa: E741

            # Drop invalid peaks
            if h == 0 and k == 0 and l == 0:
                continue

            intensity, sigma = self.intensity[i], self.sigma[i]
            wl = self.lamda[i]
            theta = self.theta[i]
            phi = self.phi[i]

            if self.runs is not None:
                run = self.runs[i]
            else:
                run = 0

            if self.f is not None:
                f, f_sigma = self.f[i], self.f_sigma[i]
                row = [
                    h,
                    k,
                    l,
                    intensity,
                    sigma,
                    f,
                    f_sigma,
                    wl,
                    theta,
                    phi,
                    run,
                ]
            else:
                row = [h, k, l, intensity, sigma, wl, theta, phi, run]

            data.append(row)

        if len(data) == 0:
            # Empty case: create a 2D array with correct number of columns
            data = np.empty((0, n_cols), dtype=np.float32)
        else:
            data = np.ascontiguousarray(np.array(data, dtype=np.float32))

        mtz.set_data(data)
        mtz.write_to_file(filename)


class ImageStackMerger(BaseConcatenateMerger):
    def __init__(self, h5_files):
        """
        Merges reduced image HDF5 files into a single stack for batch processing.
        """
        merge_keys = [
            "images",  # The stack of 2D images
            "goniometer/angles",  # Per-image angles
            "bank_ids",  # Per-image detector ID
        ]

        # Keys that should be identical across all files (metadata)
        copy_keys = [
            "goniometer/axes",
            "goniometer/names",
            "instrument/wavelength",
            "instrument/name",
        ]
        super().__init__(h5_files, copy_keys, merge_keys)
