import sys
import numpy as np
import h5py

import gemmi


class ConcatenateMerger:
    def __init__(self, indexed_h5_files):
        """
        Merges indexed peak datasets by concatenation

        Parameters
        ----------
        indexed_h5_files: list[str]
            List of file paths for indexed peak dataset .h5 files
        """
        self.indexed_h5_files = indexed_h5_files

    def merge(self, output_filename):
        """
        Merges the indexed datasets into a single dataset

        Parameters
        ----------
        output_filename : str
            Name of the output .h5 file to write the merged dataset to
        """

        total_peaks = 0
        for indexed_file in self.indexed_h5_files:
            with h5py.File(indexed_file, "r") as f_in:
                total_peaks += len(f_in["peaks/intensity"])

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
            "peaks/scattering",
            "peaks/azimuthal",
        ]

        with h5py.File(output_filename, "w") as f_out:
            with h5py.File(self.indexed_h5_files[0], "r") as f_typical:
                for key in copy_keys:
                    f_out[key] = np.array(f_typical[key])

            for merge_key in merge_keys:
                f_out.create_dataset(merge_key, total_peaks)
            f_out.create_dataset("peaks/run_index", total_peaks, dtype=np.int32)

            offset = 0
            for i_file, indexed_file in enumerate(self.indexed_h5_files):
                with h5py.File(indexed_file, "r") as f_in:
                    num_peaks = len(f_in["peaks/intensity"])
                    peak_range = slice(offset, offset + num_peaks)
                    f_out["peaks/run_index"][peak_range] = i_file
                    for merge_key in merge_keys:
                        f_out[merge_key][peak_range] = np.array(f_in[merge_key])

                    offset += num_peaks



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
            self.theta = np.array(f["peaks/scattering"])/2
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
