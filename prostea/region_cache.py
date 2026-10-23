"""Loader for the full-patch-grid embedding cache written by
scripts/cache_regions.py.

Unlike ../prototypeProstate/cache/pnf_plus_embeddings (which keeps only
needle-region patches, discarding everything else at extraction time), this
cache keeps the *entire* patch grid plus both masks, so region membership
(weak / unlabeled / negative) can be recomputed however a genome needs at
search time without re-running the frozen encoder. That's what makes NSGA-II
search tractable on a single GPU -- the expensive forward pass happens once,
during caching, not once per genome per generation.

Layout, at <cache_dir>:
    metadata.csv        core_id, patient_id, center, grade_group, involvement,
                         pri_mus, age, psa, n_prostate_patches, n_needle_patches
    npz/<core_id>.npz
        feats            (Gh, Gw, D) float16   frozen PNF+ patch embeddings
        needle_mask      (Gh, Gw)    uint8 {0,1}
        prostate_mask    (Gh, Gw)    uint8 {0,1} (all-ones if unavailable for
                                      this core -- see scripts/cache_regions.py)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class CoreEntry:
    core_id: str
    patient_id: str
    center: str
    grade_group: int
    involvement: float
    age: float
    psa: float


class RegionCache:
    """Index over a cache_regions.py output directory.

    A genome search calls .load()/.load_many() for the same cores over and
    over -- every epoch of every fold of every genome NSGA-II evaluates. At
    ~1118 cores x ~4.2MB (float32, decompressed) this whole cohort is a few
    GB, small enough to keep resident once touched: by default (`in_memory=
    True`) each core's decompressed tensors are cached in a dict on first
    load, turning O(epochs x folds x genomes) disk reads into O(1) per core
    for the process's lifetime. This is what actually makes a multi-
    generation NSGA-II search tractable in wall-clock time, not just in
    principle -- without it, the disk I/O dominates every genome evaluation.
    Pass `in_memory=False` to fall back to always re-reading from disk (e.g.
    if a future, much larger cohort doesn't fit in memory).
    """

    def __init__(self, cache_dir: str, in_memory: bool = True):
        self.cache_dir = cache_dir
        self.npz_dir = os.path.join(cache_dir, "npz")
        self.meta = pd.read_csv(os.path.join(cache_dir, "metadata.csv"))
        self.meta = self.meta.reset_index(drop=True)
        missing = [c for c in self.meta.core_id if not os.path.isfile(self._npz_path(c))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} core(s) in metadata.csv have no cached .npz, "
                f"e.g. {missing[:3]}. Re-run scripts/cache_regions.py."
            )
        self.in_memory = in_memory
        self._mem: dict[int, dict] = {}

    def _npz_path(self, core_id: str) -> str:
        return os.path.join(self.npz_dir, f"{core_id}.npz")

    def __len__(self) -> int:
        return len(self.meta)

    def entry(self, row_idx: int) -> CoreEntry:
        row = self.meta.iloc[row_idx]
        return CoreEntry(
            core_id=row.core_id,
            patient_id=row.patient_id,
            center=row.center,
            grade_group=int(row.grade_group),
            involvement=float(row.involvement),
            age=float(row.age),
            psa=float(row.psa),
        )

    def _load_from_disk(self, row_idx: int) -> dict:
        entry = self.entry(row_idx)
        npz = np.load(self._npz_path(entry.core_id))
        return dict(
            core_id=entry.core_id,
            patient_id=entry.patient_id,
            center=entry.center,
            grade_group=entry.grade_group,
            involvement=entry.involvement,
            age=entry.age,
            psa=entry.psa,
            feats=torch.from_numpy(npz["feats"].astype(np.float32)),
            needle_mask=torch.from_numpy(npz["needle_mask"] > 0),
            prostate_mask=torch.from_numpy(npz["prostate_mask"] > 0),
        )

    def load(self, row_idx: int, device: str = "cpu") -> dict:
        """Returns feats (Gh, Gw, D) float32, needle_mask/prostate_mask
        (Gh, Gw) bool, plus the metadata row's scalar fields -- all on
        `device`. Cached tensors are kept on CPU (see class docstring) and
        moved to `device` on every call -- a no-op if device="cpu", a real
        (but small, ~4MB/core) transfer if device="cuda".
        """
        if self.in_memory:
            cpu_item = self._mem.get(row_idx)
            if cpu_item is None:
                cpu_item = self._load_from_disk(row_idx)
                self._mem[row_idx] = cpu_item
        else:
            cpu_item = self._load_from_disk(row_idx)

        item = dict(cpu_item)
        item["feats"] = item["feats"].to(device)
        item["needle_mask"] = item["needle_mask"].to(device)
        item["prostate_mask"] = item["prostate_mask"].to(device)
        return item

    def load_many(self, row_idxs, device: str = "cpu") -> list[dict]:
        return [self.load(i, device=device) for i in row_idxs]
