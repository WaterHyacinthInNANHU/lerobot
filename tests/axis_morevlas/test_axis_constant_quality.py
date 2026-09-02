"""Stage-2 constant-tag CFG conditioning (`ConstantQualityTagger`).

The reference for every semantic here is openpi's
`quality_conditioning.LiberoQualityConditioning`: constant tag, two-level dropout
(0.15/0.05 -> tagged marginal 0.8075), and a draw that is PURE in
(seed, presentation, episode_index, frame_index) mixed through SHA-256 -- re-implemented
independently in `_openpi_dropped` below so a drift in the fork's key construction shows
up as a test failure, not a silent train/serve divergence.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from lerobot.utils.axis_quality import (
    DROP_COMPONENT_STAGE2,
    DROP_WHOLE_STAGE2,
    ConstantQualityTagger,
    PROMPT_MARKER,
)


def _openpi_dropped(seed: int, presentation: int, ep: int, fr: int) -> bool:
    """Byte-for-byte re-implementation of openpi LiberoQualityConditioning.dropped."""
    digest = hashlib.sha256(f"{seed}:{presentation}:{ep}:{fr}".encode()).digest()
    u = np.random.default_rng(int.from_bytes(digest[:8], "little")).random(2)
    return bool(u[0] < DROP_WHOLE_STAGE2 or u[1] < DROP_COMPONENT_STAGE2)


def _tagger(**kw) -> ConstantQualityTagger:
    defaults = dict(seed=0, epoch_len=86912, samples_per_step=64)
    defaults.update(kw)
    return ConstantQualityTagger(5, **defaults)


def test_dropped_matches_openpi_reference():
    t = _tagger(seed=7)
    for pres in (0, 1, 4):
        for ep, fr in ((0, 0), (3, 17), (1035, 66), (12, 200)):
            assert t.dropped(pres, ep, fr) == _openpi_dropped(7, pres, ep, fr)


def test_realized_marginal_is_08075():
    t = _tagger()
    draws = [t.dropped(0, ep, fr) for ep in range(50) for fr in range(80)]
    frac_tagged = 1.0 - (sum(draws) / len(draws))
    assert abs(frac_tagged - (1 - DROP_WHOLE_STAGE2) * (1 - DROP_COMPONENT_STAGE2)) < 0.02


def test_presentation_redraws_not_partition():
    """The whole point of the presentation key: a row's fate is re-drawn each pass, so over
    presentations both branches see (statistically) all rows -- not a frozen 19.25%."""
    t = _tagger()
    rows = [(ep, fr) for ep in range(30) for fr in range(30)]
    fates = {(ep, fr): [t.dropped(p, ep, fr) for p in range(5)] for ep, fr in rows}
    changed = sum(1 for v in fates.values() if len(set(v)) > 1)
    # P(same fate all 5 presentations) ~ 0.8075^5 + small; ~66% of rows must change fate.
    assert changed > 0.4 * len(rows)


def test_presentation_for_step_exact_boundaries():
    t = _tagger(epoch_len=86912, samples_per_step=64)  # 1358 steps per epoch, exact
    assert t.boundary_exact
    assert t.presentation_for_step(0) == 0
    assert t.presentation_for_step(1357) == 0
    assert t.presentation_for_step(1358) == 1
    assert t.presentation_for_step(6789) == 4


def test_apply_tags_batch_in_place_and_reports_frac():
    t = _tagger()
    n = 256
    batch = {
        "task": ["Put the Juice on the Napkin Box"] * n,
        "episode_index": torch.arange(n),
        "frame_index": torch.arange(n) * 3,
    }
    stats = t.apply(batch, step=0)
    tagged = [s for s in batch["task"] if s.endswith(f"{PROMPT_MARKER}5")]
    bare = [s for s in batch["task"] if s == "Put the Juice on the Napkin Box"]
    assert len(tagged) + len(bare) == n
    assert stats["cfg_tagged_frac"] == pytest.approx(len(tagged) / n)
    assert 0.6 < stats["cfg_tagged_frac"] < 0.95
    # Deterministic: the same batch at the same step re-tags identically.
    batch2 = {
        "task": ["Put the Juice on the Napkin Box"] * n,
        "episode_index": torch.arange(n),
        "frame_index": torch.arange(n) * 3,
    }
    t.apply(batch2, step=0)
    assert batch2["task"] == batch["task"]


def test_apply_double_tag_refused():
    t = _tagger()
    batch = {
        "task": [f"already{PROMPT_MARKER}5"],
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([0]),
    }
    with pytest.raises(RuntimeError, match="TWICE"):
        t.apply(batch, step=0)


def test_apply_requires_row_keys():
    t = _tagger()
    with pytest.raises(KeyError, match="frame_index"):
        t.apply({"task": ["x"], "episode_index": torch.tensor([0])}, step=0)


def test_constructor_refusals():
    with pytest.raises(ValueError, match="not a bin"):
        ConstantQualityTagger(0, seed=0, epoch_len=10, samples_per_step=2)
    with pytest.raises(ValueError, match="not a bin"):
        ConstantQualityTagger(6, seed=0, epoch_len=10, samples_per_step=2)
    with pytest.raises(ValueError, match="unconditional branch"):
        ConstantQualityTagger(
            5, seed=0, epoch_len=10, samples_per_step=2, drop_whole=0.0, drop_component=0.0
        )


def test_config_field_exists_and_defaults_off():
    """A full validate() needs a complete config (heavy); the field contract is checked here,
    the refusal paths are exercised by the driver's SMOKE run."""
    import dataclasses

    from lerobot.configs.train import TrainPipelineConfig

    fields = {f.name: f for f in dataclasses.fields(TrainPipelineConfig)}
    assert "axis_quality_constant_tag" in fields
    assert fields["axis_quality_constant_tag"].default is None
