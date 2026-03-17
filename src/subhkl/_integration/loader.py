import re
import h5py
import numpy as np
from subhkl.config import (
        beamlines,
        reduction_settings
    )

def load_nexus(peaks, filename: str) -> dict[int, np.ndarray]:
    detectors = beamlines[peaks.instrument]
    settings = reduction_settings[peaks.instrument]
    ims = {}
    with h5py.File(filename, "r") as f:
        keys = []
        banks = []
        for key in f["/entry/"].keys():
            match = re.match(r"bank(\d+).*", key)
            if match is not None:
                keys.append(key)
                banks.append(int(match.groups()[0]))
        for rel_key, bank in zip(keys, banks, strict=False):
            key = "/entry/" + rel_key + "/event_id"
            array = f[key][()]
            det = detectors.get(str(bank))
            if det is not None:
                m, n, offset = det["m"], det["n"], det["offset"]
                bc = np.bincount(array - offset, minlength=m * n)
                if np.sum(bc) > 0:
                    if settings.get("YAxisIsFastVaryingIndex"):
                        ims[bank] = bc.reshape(m, n).T
                    else:
                        ims[bank] = bc.reshape(n, m)

    return ims

def load_merged_h5(peaks, filename: str) -> dict[int, np.ndarray]:
    ims = {}
    with h5py.File(filename, "r") as f:
        images = f["images"]
        N = images.shape[0]
        if "bank_ids" in f:
            bank_ids = f["bank_ids"][()]
        else:
            bank_ids = np.zeros(N, dtype=int)
        if "files" in f and "file_offsets" in f:
            peaks.image_files_raw = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in f["files"][()]
            ]
            peaks.file_offsets = f["file_offsets"][()]
        else:
            peaks.image_files_raw = None
            peaks.file_offsets = None
        data = images[()]
        for i in range(N):
            ims[i] = data[i]
            peaks.bank_mapping[i] = int(bank_ids[i])
    return ims

