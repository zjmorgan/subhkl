import os
import bisect
from dataclasses import dataclass, field
import re
import h5py
import numpy as np
from typing import Dict, List, Optional

from subhkl.config import beamlines, reduction_settings


@dataclass(frozen=True)
class ImageData:
    ims: Dict[int, np.ndarray]
    file_offsets: Optional[np.ndarray] = None
    raw_files: Optional[List[str]] = None
    bank_mapping: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def load_nexus(cls, filename: str, instrument) -> dict[int, np.ndarray]:
        detectors = beamlines[instrument]
        settings = reduction_settings[instrument]
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

        return cls(ims=ims)

    @classmethod
    def load_merged_h5(cls, filename: str) -> dict[int, np.ndarray]:
        ims = {}
        bank_mapping = {}
        with h5py.File(filename, "r") as f:
            images = f["images"]
            N = images.shape[0]
            if "bank_ids" in f:
                bank_ids = f["bank_ids"][()]
            else:
                bank_ids = np.zeros(N, dtype=int)
            if "files" in f and "file_offsets" in f:
                image_files_raw = [
                    n.decode("utf-8") if isinstance(n, bytes) else str(n)
                    for n in f["files"][()]
                ]
                file_offsets = f["file_offsets"][()]
            else:
                image_files_raw = None
                file_offsets = None
            data = images[()]
            for i in range(N):
                ims[i] = data[i]
                bank_mapping[i] = int(bank_ids[i])

        return cls(
            ims=ims,
            raw_files=image_files_raw,
            file_offsets=file_offsets,
            bank_mapping=bank_mapping,
        )

    def get_label(self, img_key: int) -> str:
        files = self.raw_files
        offsets = self.file_offsets
        if files and offsets is not None:
            file_idx = bisect.bisect_right(self.file_offsets, img_key) - 1
            if 0 <= file_idx < len(self.raw_files):
                orig_name = os.path.basename(self.raw_files[file_idx])
                clean_name = os.path.splitext(orig_name)[0]
                clean_name = clean_name.replace(".nxs.h5", "").replace(".h5", "")
                return clean_name
        return f"img{img_key}"

    def get_run_id(self, img_key: int) -> int:
        if self.file_offsets is None:
            return 0
        return int(np.searchsorted(self.file_offsets, img_key, side="right") - 1)
