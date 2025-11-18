import sys
import numpy as np
import h5py

import gemmi


class BaseConcatenateMerger:
    def __init__(self, h5_files, copy_keys, merge_keys):
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
        """
        self.h5_files = h5_files
        self.copy_keys = copy_keys
        self.merge_keys = merge_keys

    def merge(self, output_filename):
        """
        Merges the indexed datasets into a single dataset

        Parameters
        ----------
        output_filename : str
            Name of the output .h5 file to write the merged dataset to
        """

        total_peaks = 0
        for file in self.h5_files:
            with h5py.File(file, "r") as f_in:
                total_peaks += len(f_in[self.merge_keys[0]])

        with h5py.File(output_filename, "w") as f_out:
            with h5py.File(self.h5_files[0], "r") as f_typical:
                for key in self.copy_keys:
                    f_out[key] = np.array(f_typical[key])

                for merge_key in self.merge_keys:
                    shape = (total_peaks,) + f_typical[merge_key].shape[1:]
                    dtype = f_typical[merge_key].dtype
                    f_out.create_dataset(merge_key, shape, dtype)

            offset = 0
            f_out["files"] = np.array(list(map(lambda s: s.encode('utf-8'), self.h5_files)))
            f_out.create_dataset("file_offsets", (len(self.h5_files),), dtype=np.int64)
            for i_file, indexed_file in enumerate(self.h5_files):
                with h5py.File(indexed_file, "r") as f_in:
                    num_items = len(f_in[self.merge_keys[0]])
                    peak_range = slice(offset, offset + num_items)
                    f_out["file_offsets"][i_file] = offset
                    for merge_key in self.merge_keys:
                        f_out[merge_key][peak_range] = np.array(f_in[merge_key])

                    offset += num_items


class FinderConcatenateMerger(BaseConcatenateMerger):
    def __init__(self, h5_files):
        merge_keys = [
            "wavelength_mins",
            "wavelength_maxes",
            "goniometer/R",
            "peaks/two_theta",
            "peaks/azimuthal",
            "peaks/intensity",
            "peaks/sigma"
        ]
        super().__init__(h5_files, [], merge_keys)


class IndexerConcatenateMerger(BaseConcatenateMerger):
    def __init__(self, indexed_h5_files):
        """
        Parameters
        ----------
        indexed_h5_files : list[str]
            List of .h5 files from indexer to merge
        """
        copy_keys = [
            "sample/a",
            "sample/b",
            "sample/c",
            "sample/alpha",
            "sample/beta",
            "sample/gamma",
            "sample/centering",
            "instrument/wavelength",
        ]

        merge_keys = [
            "peaks/intensity",
            "peaks/sigma",
            "peaks/structure_factors",
            "peaks/structure_factors_sigma",
            "peaks/h",
            "peaks/k",
            "peaks/l",
            "peaks/lambda",
            "peaks/two_theta",
            "peaks/azimuthal",
        ]

        super().__init__(indexed_h5_files, copy_keys, merge_keys)


class MTZExporter:
    def __init__(self, peaks_file, space_group="P 1"):
        with h5py.File(peaks_file) as f:
            self.a = float(np.array(f["sample/a"]))
            self.b = float(np.array(f["sample/b"]))
            self.c = float(np.array(f["sample/c"]))
            self.alpha = float(np.array(f["sample/alpha"]))
            self.beta = float(np.array(f["sample/beta"]))
            self.gamma = float(np.array(f["sample/gamma"]))

            self.h = np.array(f["peaks/h"])
            self.k = np.array(f["peaks/k"])
            self.l = np.array(f["peaks/l"])
            self.lamda = np.array(f["peaks/lambda"])
            self.theta = np.array(f["peaks/two_theta"])/2
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

        mtz.spacegroup = gemmi.find_spacegroup_by_name(self.space_group)

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

        data = []

        for i in range(len(self.intensity)):
            h, k, l = self.h[i], self.k[i], self.l[i]

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
                row = [h, k, l, intensity, sigma, f, f_sigma, wl, theta, phi, run]
            else:
                row = [h, k, l, intensity, sigma, wl, theta, phi, run]

            data.append(row)

        data = np.array(data)

        mtz.set_data(data)
        mtz.write_to_file(filename)
