import h5py
import numpy as np
from pathlib import Path


class SAM3Model:
    def __init__(self, dim=512,
                 h5_path="/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/descriptor_mask.h5"):
        self.dim = dim
        self.h5_path = h5_path
        self.h5_file = None
        self.conf = {"name": f"sam3_{dim}"}
        self._load_h5()

    def _load_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

    def process(self, name):
        stem = Path(name).stem
        try:
            self._load_h5()

            if stem not in self.h5_file:
                return np.zeros(self.dim, dtype=np.float32)

            group = self.h5_file[stem]
            if 'globaldesc' not in group:
                return np.zeros(self.dim, dtype=np.float32)

            global_desc = np.array(group['globaldesc'])

            if global_desc.ndim > 1:
                global_desc = global_desc.mean(axis=0).squeeze()

            if len(global_desc) > self.dim:
                global_desc = global_desc[:self.dim]
            elif len(global_desc) < self.dim:
                global_desc = np.pad(global_desc, (0, self.dim - len(global_desc)))

            return global_desc.astype(np.float32)

        except Exception:
            # Silently return zero vector for missing entries (queries, etc.)
            return np.zeros(self.dim, dtype=np.float32)

    def __del__(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass