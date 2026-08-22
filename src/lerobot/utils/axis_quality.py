"""AXIS CFG quality-tag batch hook: read a precomputed tag artifact and put the tag in the task.

ONLINE TIER (this fork). Mirrors openpi's `openpi.training.quality_conditioning.QualityTags` /
`QualityTaggedDataset` (see WaterHyacinthInNANHU/openpi, quality_conditioning.py), adapted to
lerobot's batch shape: openpi injects the tag per-sample via `QualityTaggedDataset.__getitem__`'s
own index argument -- the only unambiguously concat-global index available there, since
`data["index"]` under a `ConcatDataset` is measured to be the SUB-DATASET-local row. This fork's
`LeRobotDataset` has no such wrapper layer: `batch["index"]` IS the dataset's own global frame
index (confirmed against the shipped artifact -- see test_axis_quality.py and
p2-task-3-report.md), so the join is a direct `tag[batch["index"]]`, no position map needed.

Byte-exact prompt format parity with openpi's `slb_cfg.apply_metadata(prompt, q, None)`: that
call joins `[prompt, f"Quality: {int(q)}"]` with `"\\n"`, i.e. `prompt + "\\nQuality: " + str(q)`.
`PROMPT_MARKER` here is exactly openpi's `PROMPT_TAG_MARKER` (`"\\nQuality: "`).
"""

from __future__ import annotations

import json

import numpy as np
import torch

# Exact byte-for-byte match to openpi's `quality_conditioning.PROMPT_TAG_MARKER`
# (built there from `slb_cfg.QUALITY_KEY = "Quality"`): `f"\n{QUALITY_KEY}: "`.
PROMPT_MARKER = "\nQuality: "

# Mirrors openpi's `quality_conditioning.NO_TAG` / `NOT_TRAINABLE`: 0 = untagged/unconditional
# (the bare prompt, i.e. the unconditional CFG branch); 255 = the offline builder's sentinel for
# a row that exists in the tag array but must never be sampled (excluded from every row plan).
NO_TAG = 0
NOT_TRAINABLE = 255

# Mirrors openpi's `quality_conditioning.N_BINS` (quality_conditioning.py:75-93) and its exact
# rationale, duplicated here rather than imported for the same reason openpi's own copy is
# duplicated from `axis.dataset.quality_labels`: this fork is a further tier removed (nothing
# under lerobot imports `axis` OR `openpi`).
#
# A self-consistency check against the artifact's OWN `bin_row_counts` would catch a GROWN
# offline bin count (extra tag values with no declared bin) but MISS a SHRUNK one entirely: every
# tag of a 3-bin build is still a legal 5-bin tag, so no self-consistency check -- and no
# downstream range guard -- would ever fire, and the arm would silently train 3-bin conditioning
# while the config and the write-up say 5. N_BINS as a fixed cross-tier constant is the only
# place that catches the shrink, which is why it must NOT be derived from the file being checked.
N_BINS = 5


def is_tagged(task: object) -> bool:
    """True if this task string already carries a quality tag. Mirrors openpi's `is_tagged`."""
    return PROMPT_MARKER in str(task)


class AxisQualityTags:
    """The artifact, plus the checks that bind it to this run.

    Mirrors `openpi.training.quality_conditioning.QualityTags.__init__` line-for-line (1-D uint8
    no-cast, non-empty prompts, reward_id present, n_rows == len(tag), bin-count check against
    the cross-tier N_BINS constant, at-least-one-NO_TAG-row guard), PLUS an `expected_frames`
    check that openpi's twin does not need (there the wrapper's own `check_dataset_rows` performs
    the equivalent join at construction time -- here that binding is done in `__init__` directly,
    since this fork has no separate wrapper object). The parity contract is that this fork
    refuses everything openpi refuses, so every openpi guard is a LITERAL port, not a
    self-consistency approximation of one.
    """

    def __init__(self, path: str, expected_frames: int) -> None:
        self.path = path
        # NOT memory-mapped, for the same reason as openpi's QualityTags: np.load's mmap_mode is
        # silently ignored for a .npz, and the array is small enough (~2 MB) that it does not
        # matter.
        with np.load(path, allow_pickle=False) as z:
            self.tag = np.asarray(z["tag"])
            self.prompts = [str(s) for s in z["prompts"]]
            self.meta = json.loads(str(z["meta"]))

        if self.tag.ndim != 1 or self.tag.dtype != np.uint8:
            # Not cast: a float or int64 array reads through int() without complaint, and a
            # truncated tag is a WRONG tag indistinguishable from a right one. Same reasoning as
            # AxisScheduleSampler's refusal to cast a float rows array.
            raise ValueError(
                f"quality artifact {path} has tag shape {self.tag.shape} dtype {self.tag.dtype}; "
                "expected 1-D uint8 dense over the corpus row space (openpi build_quality_labels "
                "format). Rebuild it with axis.dataset.build_quality_labels."
            )
        if not self.prompts:
            raise ValueError(
                f"quality artifact {path} carries no prompts; the token-budget guard would have "
                "nothing to check and would pass vacuously."
            )
        if self.meta.get("reward_id") is None:
            raise ValueError(
                f"quality artifact {path} has no 'reward_id' in its meta, so its filename cannot "
                "be checked against its contents and the CFG arms -- which share one config name "
                "-- become indistinguishable. Rebuild it with axis.dataset.build_quality_labels."
            )
        declared_n_rows = self.meta.get("n_rows")
        if declared_n_rows is None:
            raise ValueError(
                f"quality artifact {path} has no meta['n_rows'], so the tag array cannot be "
                "checked against the corpus size the builder recorded. Rebuild it with "
                "axis.dataset.build_quality_labels."
            )
        if int(declared_n_rows) != len(self.tag):
            # The builder refuses to write this (mirrors openpi's build_quality_labels.quality_meta
            # guard): the file was truncated or hand-edited after the fact.
            raise ValueError(
                f"quality artifact {path} carries {len(self.tag)} tags but its meta reports "
                f"n_rows={int(declared_n_rows)}; the file disagrees with itself about the index "
                "space it covers. Rebuild it with axis.dataset.build_quality_labels."
            )
        if int(declared_n_rows) != int(expected_frames):
            # The equivalent of openpi's QualityTaggedDataset.__init__ calling
            # tags.check_dataset_rows(len(dataset)) -- done here directly since this fork has no
            # separate wrapper object. A GROWN artifact is the dangerous direction: every index
            # the sampler can draw would still be in range, and the whole run would tag off a
            # longer index space -- i.e. the right-looking tag on the wrong frame, for every frame.
            raise ValueError(
                f"quality artifact {path} covers n_rows={int(declared_n_rows)} but "
                f"expected_frames={int(expected_frames)}: the artifact indexes a different corpus "
                "than the dataset this run is training on. Rebuild the artifact against this "
                "corpus, or point the run at the corpus it was built from."
            )
        self._check_bin_count()
        if not bool((self.tag == NO_TAG).any()):
            # Mirrors openpi's QualityTags.__init__ (quality_conditioning.py:176-186) exactly.
            # DROP_WHOLE=0 (or a dropout that misfired) leaves every trainable row tagged, so the
            # model never sees the bare prompt, never learns p(a | no tag), and guidance at
            # inference -- which subtracts an unconditional forward pass -- is undefined. Nothing
            # online would notice: training loss, throughput and every prompt look normal.
            raise ValueError(
                f"quality artifact {path} has no NO_TAG ({NO_TAG}) rows, so CFG has no "
                "unconditional branch to guide away from and the arm degenerates to plain "
                "conditional BC. Rebuild it with a non-zero dropout "
                "(axis.dataset.quality_labels.DROP_WHOLE)."
            )

    def _check_bin_count(self) -> None:
        """Bind the offline tier's N_BINS to this file's copy of it.

        Mirrors openpi's `_check_bin_count` (quality_conditioning.py:188-213) exactly: the
        downstream range guard (in the transform that applies the tag to the prompt) sees a
        GROWN offline bin count (tags outside [1, N_BINS]) and misses a SHRUNK one entirely,
        since every tag of a 3-bin build is a legal 5-bin tag. The artifact carries the answer:
        `bin_row_counts` is keyed "1".."N_BINS" by the builder, so its largest key must equal
        this tier's N_BINS exactly -- not merely bound the tags actually present.
        """
        counts = self.meta.get("bin_row_counts")
        if not counts:
            raise ValueError(
                f"quality artifact {self.path} has no 'bin_row_counts' in its meta, so the bin "
                f"count it was BUILT with cannot be compared against this tier's N_BINS="
                f"{N_BINS}. A build with fewer bins would pass every range check and train a "
                "coarser arm than the config reports. Rebuild it with "
                "axis.dataset.build_quality_labels."
            )
        offline_bins = max(int(k) for k in counts)
        if offline_bins != N_BINS:
            raise ValueError(
                f"bin-count mismatch: quality artifact {self.path} was built with "
                f"{offline_bins} bins (its meta['bin_row_counts'] keys) but this tier's N_BINS is "
                f"{N_BINS}. A smaller offline count is invisible to the downstream range guard "
                f"-- every tag would be in [1, {N_BINS}] and the arm would train "
                f"{offline_bins}-bin conditioning while the config and the write-up say {N_BINS}. "
                "Rebuild the artifact, or bring the two tiers' N_BINS back into agreement."
            )


def apply_quality_tags(batch: dict, tags: AxisQualityTags) -> dict:
    """Append the quality tag to `batch["task"]`, keyed on `batch["index"]` (global frame index).

    Edits `batch["task"]` (list[str]) in place; `q == NO_TAG` (0) leaves the task untouched --
    that IS the unconditional CFG branch. Returns `{"cfg_tagged_frac": float}` for logging.

    Mirrors `openpi.training.quality_conditioning.AxisQualityConditioning.__call__`'s guards,
    reordered to match the STAGING of the real openpi pipeline: there, `QualityTaggedDataset`'s
    `tag_for_row` (which raises on NOT_TRAINABLE) runs at `__getitem__`, strictly before the
    sample ever reaches the transform's `is_tagged` check -- so the NOT_TRAINABLE guard here
    likewise precedes the double-tag guard.
    """
    index = batch["index"]
    idx_np = index.detach().cpu().numpy() if isinstance(index, torch.Tensor) else np.asarray(index)
    if idx_np.ndim != 1:
        raise ValueError(f"batch['index'] must be 1-D, got shape {idx_np.shape}")
    task = batch["task"]
    if len(task) != len(idx_np):
        raise ValueError(
            f"batch['task'] has {len(task)} entries but batch['index'] has {len(idx_np)}: "
            "cannot join them row-for-row."
        )
    if idx_np.size:
        lo = int(idx_np.min())
        if lo < 0:
            # numpy reads a negative index from the END, returning some OTHER row's tag rather
            # than raising -- the same silent-wrap class AxisScheduleSampler also refuses.
            raise IndexError(
                f"negative row index {lo} into quality artifact {tags.path}; numpy would read it "
                "from the tail and return some other row's tag."
            )
        hi = int(idx_np.max())
        if hi >= len(tags.tag):
            raise IndexError(
                f"row index {hi} is beyond quality artifact {tags.path} ({len(tags.tag)} rows). "
                "The sampler drew a frame outside what the artifact covers."
            )

    new_task: list[str] = []
    n_tagged = 0
    for t, q_raw in zip(task, tags.tag[idx_np].tolist(), strict=True):
        q = int(q_raw)
        if q == NOT_TRAINABLE:
            # Mirrors openpi's tag_for_row KeyError: the tag array and the row plan disagree
            # about which rows exist; defaulting here would silently untag part of the arm.
            raise KeyError(
                f"a sampled row is marked not trainable ({NOT_TRAINABLE}) in {tags.path}, but "
                "the row sampler drew it. The tag array and the row plan disagree about which "
                "rows are trainable."
            )
        if is_tagged(t):
            # DOUBLE TAG: this hook ran twice, or the corpus task strings already carry the
            # marker -- a wiring bug either way, mirrors openpi's exact refusal.
            raise RuntimeError(
                f"task {str(t)[:120]!r} already carries a quality tag ({PROMPT_MARKER!r}), so "
                "tagging it would apply the tag TWICE. The quality-tag hook is wired into the "
                "train loop more than once, or the corpus tasks were built with the tag already "
                "in them."
            )
        if q == NO_TAG:
            new_task.append(t)
        else:
            new_task.append(f"{t}{PROMPT_MARKER}{q}")
            n_tagged += 1

    batch["task"] = new_task
    frac = (n_tagged / len(new_task)) if new_task else 0.0
    return {"cfg_tagged_frac": frac}
