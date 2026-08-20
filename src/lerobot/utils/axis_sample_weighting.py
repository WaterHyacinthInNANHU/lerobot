"""AXIS flat-array sample weighter: weight = weights[global_frame_index].

The array is a shipped AXIS artifact (built by AXIS-Bench from openpi's own AWR join, NaN on
every frame outside the planned row set). NaN here is a GUARD, not a gap: the row sampler and
this weighter must agree about the trainable rows, and a batch index hitting NaN means they
do not — which is fatal, exactly as openpi refuses partially-weighted objectives.
"""
from __future__ import annotations

import numpy as np
import torch

from lerobot.utils.sample_weighting import SampleWeighter, SampleWeightingConfig


class AxisFlatWeighter(SampleWeighter):
    def __init__(self, config: SampleWeightingConfig, device: torch.device) -> None:
        path = config.extra_params["weights_path"]
        expected = int(config.extra_params["expected_frames"])
        flat = np.load(path)["weights"]
        if len(flat) != expected:
            raise ValueError(
                f"weights artifact has {len(flat)} frames but expected_frames={expected}: "
                f"it was built against a different corpus ({path})"
            )
        self._weights = torch.as_tensor(flat, dtype=torch.float32, device=device)
        finite = self._weights[torch.isfinite(self._weights)]
        self._stats = {
            "mean": float(finite.mean()),
            "max": float(finite.max()),
            "frac_ge_cap": float((finite >= finite.max()).float().mean()),
        }

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        idx = batch["index"].to(self._weights.device, dtype=torch.long)
        w = self._weights[idx]
        if not torch.isfinite(w).all():
            bad = idx[~torch.isfinite(w)][:5].tolist()
            raise RuntimeError(
                f"non-finite AWR weight for global indices {bad}: the sampler drew a frame "
                f"outside the planned row set — sampler and weights artifact disagree"
            )
        return w, {"batch_weight_mean": float(w.mean())}

    def get_stats(self) -> dict:
        return dict(self._stats)
