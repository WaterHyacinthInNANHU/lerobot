"""AXIS row-restricted sampler: uniform draw over an explicit row set.

Even the plain-BC arm samples a restricted set (the corpus's non-idle rows), so this sampler
replaces EpisodeAwareSampler for every AXIS arm. Draw order is a pure function of
(seed, epoch); resume is a skip-offset, mirroring EpisodeAwareSampler's state contract.
"""

from __future__ import annotations

import json

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


class AxisScheduleSampler(Sampler[int]):
    """Replay a precomputed index schedule (Drop/Anneal arms) in exact file order.

    The schedule IS the arm: every draw was decided offline by axis.dataset.build_index_schedule
    and audited via the artifact's own meta block. Nothing here may reorder, dedupe, resume, or
    reshape it -- this mirrors openpi's real `openpi.training.schedule_sampler.ScheduleSampler`
    (NOT the file the task brief named, which is a different, unrelated row planner) and the
    on-disk schema `axis.dataset.build_index_schedule.write_schedule` actually writes:

    - `rows` is 2-D `(total_steps, batch)`, integer dtype (any integer width is accepted and
      safely upcast to int64; a float array is refused rather than silently truncated -- a
      truncated row is a WRONG row, indistinguishable from a correct one once training starts).
    - `meta` is a 0-d array holding a JSON string; `meta["n_rows"]` is REQUIRED (an artifact
      missing it cannot be told apart from one built against a different corpus) and must equal
      `expected_frames` EXACTLY -- the schedule's indices are positions in a specific concatenated
      corpus, so even a same-direction size drift (the corpus merely grew) would leave every
      index in bounds while naming a different frame.
    - Negative rows are rejected: torch indexes a negative int as "from the end", so a negative
      row would silently draw from the dataset tail instead of raising.

    Resume: unlike openpi (which checks `config.resume` at the data-loader call site, before ever
    constructing `ScheduleSampler`), this fork's `lerobot_train.make_dataloaders` calls
    `sampler.load_state_dict(...)` uniformly for every sampler type when `cfg.resume and step > 0`
    (never on a fresh run -- verified at that call site). So `load_state_dict` is where THIS class
    refuses: a resumed replay would restart the schedule from row 0 while the optimizer continues
    from step k, silently changing what the arm saw. Same reasoning as openpi's
    `_check_schedule_resume`, adapted to where this fork's resume path actually calls in.
    """

    def __init__(self, schedule_path: str, expected_frames: int) -> None:
        self.path = schedule_path
        with np.load(schedule_path, allow_pickle=False) as z:
            rows = z["rows"]
            meta = json.loads(str(z["meta"]))
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError(
                f"schedule {schedule_path} stores rows as dtype {rows.dtype}, not an integer "
                "dtype. Casting would silently truncate a float draw into a wrong-but-valid- "
                "looking row; regenerate the artifact with an integer rows array instead."
            )
        rows = rows.astype(np.int64)
        if rows.ndim != 2:
            raise ValueError(
                f"schedule {schedule_path} has shape {rows.shape}, expected 2-D "
                "(total_steps, batch) -- a 1-D array cannot be told apart from a corrupt or "
                "pre-schedule artifact"
            )
        n_rows = meta.get("n_rows")
        if n_rows is None:
            raise ValueError(
                f"schedule {schedule_path} has no meta['n_rows'], so there is nothing to bind "
                "it to the corpus it was built against -- an artifact predating that field "
                "cannot be told apart from one built on a different corpus. Rebuild it with "
                "axis.dataset.build_index_schedule."
            )
        if int(n_rows) != int(expected_frames):
            raise ValueError(
                f"schedule {schedule_path} was drawn against n_rows={n_rows} but the dataset "
                f"has {expected_frames} frames: the schedule's indices are positions in a "
                "different concatenated corpus and would train on the wrong episodes -- rebuild "
                "the schedule against this dataset, or point the run at the corpus it was built "
                "from."
            )
        if rows.size:
            lo = int(rows.min())
            if lo < 0:
                raise ValueError(
                    f"schedule {schedule_path} contains a negative row index ({lo}). Torch "
                    "would wrap that to the dataset tail rather than raise; the artifact is "
                    "corrupt or was built against the wrong indexing convention -- rebuild it."
                )
            hi = int(rows.max())
            if hi >= expected_frames:
                raise ValueError(
                    f"schedule {schedule_path} draws row {hi}, out of range for "
                    f"expected_frames={expected_frames}: refusing to train"
                )
        self.rows = rows
        self.meta = meta

    @property
    def total_steps(self) -> int:
        """Rows in the schedule's own step dimension -- mirrors openpi ScheduleSampler.total_steps."""
        return int(self.rows.shape[0])

    @property
    def batch(self) -> int:
        """The schedule's own batch dimension -- mirrors openpi ScheduleSampler.batch.

        Consumed at the wire-in site (`_check_axis_schedule_budget` in lerobot_train.py) to
        refuse a `cfg.batch_size` that disagrees with it: torch's DataLoader cuts this sampler's
        flat index stream into batches of `batch_size`, so the artifact's row-major
        (total_steps, batch) block only reproduces the artifact's own batches one-for-one when
        `batch_size == self.batch`.
        """
        return int(self.rows.shape[1])

    def __len__(self) -> int:
        return int(self.rows.size)

    def set_epoch(self, epoch: int) -> None:  # the schedule IS the epoch structure
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        raise RuntimeError(
            f"schedule {self.path} cannot resume: this fork checkpoints no schedule-replay "
            "position, so a resume would replay the schedule from row 0 while the optimizer "
            "continues from step k -- silently changing what the arm saw (mirrors openpi's "
            "ScheduleSampler / _check_schedule_resume refusal). Restart the run clean instead "
            "of resuming."
        )

    def __iter__(self):
        # Row-major, no generator, no epoch counter: the order is the artifact's, verbatim.
        return iter(self.rows.reshape(-1).tolist())
