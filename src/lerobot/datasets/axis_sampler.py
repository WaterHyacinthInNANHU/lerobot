"""AXIS row-restricted sampler: uniform draw over an explicit row set.

Even the plain-BC arm samples a restricted set (the corpus's non-idle rows), so this sampler
replaces EpisodeAwareSampler for every AXIS arm. Draw order is a pure function of
(seed, epoch); resume is a skip-offset, mirroring EpisodeAwareSampler's state contract.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler


class AxisRowSampler(Sampler[int]):
    def __init__(self, rows: np.ndarray, seed: int, shuffle: bool = True) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or len(rows) == 0:
            raise ValueError(f"rows must be a non-empty 1-D int64 array, got shape {rows.shape}")
        if rows.min() < 0:
            raise ValueError(f"rows contains negative indices (min={rows.min()})")
        self.rows = rows
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0
        self._start = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "start_index": self._start}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state["epoch"])
        self._start = int(state["start_index"])

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        # Advance epoch state eagerly, before returning the generator.
        epoch, start = self.epoch, self._start
        self.epoch += 1
        self._start = 0
        return self._iter_from_epoch(epoch, start)

    def _iter_from_epoch(self, epoch: int, start: int):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed * 100_003 + epoch)
            order = torch.randperm(len(self.rows), generator=g).numpy()
        else:
            order = np.arange(len(self.rows))
        for i in order[start:]:
            yield int(self.rows[i])
