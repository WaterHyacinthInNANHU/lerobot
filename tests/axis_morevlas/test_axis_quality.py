from __future__ import annotations

import json

import numpy as np
import pytest
import torch


def _quality_npz(tmp_path, tags, n_rows=None, reward="v2", prompts=("a", "b")):
    p = tmp_path / "q.npz"
    # A pre-built numpy array (as used by the uint8-cast validation test below) is stored
    # VERBATIM -- casting it here would defeat the point of that test, which needs an artifact
    # whose tag array is NOT uint8. Only a plain list/tuple input is cast, for fixture convenience.
    tags = tags if isinstance(tags, np.ndarray) else np.asarray(tags, dtype=np.uint8)
    # bin_row_counts is ALWAYS 5-bin-consistent (keys "1".."5", matching axis_quality.N_BINS) --
    # not derived from the tags actually passed in. The restored bin-count guard compares this
    # against the hardcoded N_BINS constant with equality (mirroring openpi's _check_bin_count
    # line-for-line), so a fixture whose bin_row_counts tracked the tag array's own max would be
    # unable to exercise -- or accidentally trip -- that guard. test_bin_count_mismatch_refused
    # below builds a genuinely mismatched artifact by hand instead of through this helper.
    meta = {
        "reward_id": reward,
        "n_rows": n_rows if n_rows is not None else len(tags),
        "bin_row_counts": {str(i): 1 for i in range(1, 6)},
    }
    np.savez(p, tag=tags, prompts=np.array(list(prompts)), meta=np.array(json.dumps(meta)))
    return p


def test_tags_applied_by_global_index(tmp_path):
    from lerobot.utils.axis_quality import AxisQualityTags, apply_quality_tags

    q = AxisQualityTags(str(_quality_npz(tmp_path, [0, 3, 5, 0])), expected_frames=4)
    batch = {"task": ["pick", "place", "push"], "index": torch.tensor([1, 0, 2])}
    stats = apply_quality_tags(batch, q)
    assert batch["task"] == ["pick\nQuality: 3", "place", "push\nQuality: 5"]
    assert stats["cfg_tagged_frac"] == pytest.approx(2 / 3)


def test_mirrors_openpi_validations(tmp_path):
    from lerobot.utils.axis_quality import AxisQualityTags

    with pytest.raises(ValueError, match="uint8"):
        AxisQualityTags(str(_quality_npz(tmp_path, np.array([1], dtype=np.int64))), expected_frames=1)
    with pytest.raises(ValueError, match="n_rows"):
        AxisQualityTags(str(_quality_npz(tmp_path, [1, 2], n_rows=99)), expected_frames=2)
    with pytest.raises(ValueError, match="expected_frames|n_rows"):
        AxisQualityTags(str(_quality_npz(tmp_path, [1, 2])), expected_frames=99)
    with pytest.raises(ValueError, match="prompts"):
        AxisQualityTags(str(_quality_npz(tmp_path, [1], prompts=())), expected_frames=1)


def test_already_tagged_prompt_is_refused(tmp_path):
    """openpi's is_tagged() guard: double-tagging means the hook ran twice -- a wiring bug."""
    from lerobot.utils.axis_quality import AxisQualityTags, apply_quality_tags

    # tags carries a NO_TAG (0) row too, so construction itself does not trip the >=1-NO_TAG-row
    # guard -- this test is isolating the double-tag refusal, not that guard.
    q = AxisQualityTags(str(_quality_npz(tmp_path, [2, 0])), expected_frames=2)
    batch = {"task": ["pick\nQuality: 2"], "index": torch.tensor([0])}
    with pytest.raises(RuntimeError, match="already"):
        apply_quality_tags(batch, q)


def test_reward_id_required(tmp_path):
    p = tmp_path / "q.npz"
    meta = {"n_rows": 1, "bin_row_counts": {"1": 1}}
    np.savez(p, tag=np.array([1], dtype=np.uint8), prompts=np.array(["a"]), meta=np.array(json.dumps(meta)))
    from lerobot.utils.axis_quality import AxisQualityTags

    with pytest.raises(ValueError, match="reward_id"):
        AxisQualityTags(str(p), expected_frames=1)


def test_not_trainable_row_is_refused(tmp_path):
    """A row sampled from the row plan must never land on the artifact's NOT_TRAINABLE (255)
    sentinel -- if it does, the row plan and the tag array disagree about which rows exist."""
    from lerobot.utils.axis_quality import AxisQualityTags, apply_quality_tags

    # tags carries a NO_TAG (0) row too, so construction does not trip the >=1-NO_TAG-row guard.
    q = AxisQualityTags(str(_quality_npz(tmp_path, [255, 1, 0])), expected_frames=3)
    batch = {"task": ["pick"], "index": torch.tensor([0])}
    with pytest.raises(KeyError, match="not trainable"):
        apply_quality_tags(batch, q)


def test_negative_index_refused(tmp_path):
    from lerobot.utils.axis_quality import AxisQualityTags, apply_quality_tags

    q = AxisQualityTags(str(_quality_npz(tmp_path, [0, 1])), expected_frames=2)
    batch = {"task": ["pick"], "index": torch.tensor([-1])}
    with pytest.raises(IndexError, match="negative"):
        apply_quality_tags(batch, q)


def test_out_of_range_index_refused(tmp_path):
    from lerobot.utils.axis_quality import AxisQualityTags, apply_quality_tags

    q = AxisQualityTags(str(_quality_npz(tmp_path, [0, 1])), expected_frames=2)
    batch = {"task": ["pick"], "index": torch.tensor([2])}
    with pytest.raises(IndexError, match="beyond"):
        apply_quality_tags(batch, q)


def test_bin_count_mismatch_refused(tmp_path):
    """Restored openpi guard (quality_conditioning.py:188-213): the artifact's declared bin
    count must equal the cross-tier N_BINS constant EXACTLY, not merely bound the tags present.
    A SHRUNK offline bin count (here 3 bins, N_BINS=5) is invisible to any self-consistency check
    -- every 3-bin tag is a legal 5-bin tag -- so this must be a hardcoded-constant comparison."""
    from lerobot.utils.axis_quality import AxisQualityTags

    p = tmp_path / "q.npz"
    meta = {
        "reward_id": "v2",
        "n_rows": 4,
        "bin_row_counts": {"1": 1, "2": 1, "3": 1},  # only 3 bins; N_BINS is 5
    }
    np.savez(
        p,
        tag=np.array([0, 1, 2, 3], dtype=np.uint8),
        prompts=np.array(["a", "b"]),
        meta=np.array(json.dumps(meta)),
    )
    with pytest.raises(ValueError, match="N_BINS|bin"):
        AxisQualityTags(str(p), expected_frames=4)


def test_no_untagged_rows_refused(tmp_path):
    """Restored openpi guard (quality_conditioning.py:176-186): a misfired dropout (or
    DROP_WHOLE=0) leaves every trainable row tagged, so CFG never trains the unconditional
    branch and guidance at inference is undefined -- nothing online would notice (loss,
    throughput, every prompt look normal)."""
    from lerobot.utils.axis_quality import AxisQualityTags

    with pytest.raises(ValueError, match="NO_TAG|untagged|bare"):
        AxisQualityTags(str(_quality_npz(tmp_path, [1, 2, 3])), expected_frames=3)


def test_no_tag_leaves_task_untouched_and_frac_zero(tmp_path):
    from lerobot.utils.axis_quality import AxisQualityTags, apply_quality_tags

    q = AxisQualityTags(str(_quality_npz(tmp_path, [0, 0])), expected_frames=2)
    batch = {"task": ["pick", "place"], "index": torch.tensor([0, 1])}
    stats = apply_quality_tags(batch, q)
    assert batch["task"] == ["pick", "place"]
    assert stats["cfg_tagged_frac"] == 0.0


# --- config field + validate() exclusivity ------------------------------------------------------


def test_config_has_axis_quality_field():
    import dataclasses

    from lerobot.configs.train import TrainPipelineConfig

    fields = {f.name for f in dataclasses.fields(TrainPipelineConfig)}
    assert "axis_quality_path" in fields


def test_validate_refuses_quality_without_rows():
    """axis_quality_path requires axis_rows_path: quality-tag conditioning trains a
    row-restricted CFG arm, so there must be a committed row set to bind the tag artifact to."""
    import draccus

    from lerobot.configs.train import TrainPipelineConfig

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--policy.push_to_hub",
            "false",
            "--axis_quality_path",
            "/tmp/q.npz",
        ],
    )
    with pytest.raises(ValueError, match="axis_rows_path"):
        cfg.validate()


def test_validate_refuses_quality_with_schedule():
    """axis_quality_path and axis_schedule_path are two different AXIS regimes at once --
    mirrors openpi config.py's schedule+quality-conditioning refusal."""
    import draccus

    from lerobot.configs.train import TrainPipelineConfig

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--policy.push_to_hub",
            "false",
            "--axis_quality_path",
            "/tmp/q.npz",
            "--axis_schedule_path",
            "/tmp/s.npz",
            "--axis_expected_frames",
            "10",
        ],
    )
    with pytest.raises(ValueError, match="axis_schedule_path|axis_quality_path"):
        cfg.validate()


def test_validate_accepts_quality_with_rows(tmp_path):
    """The requires/excludes rules must not block the intended pairing."""
    import draccus

    from lerobot.configs.train import TrainPipelineConfig

    rows_path = tmp_path / "rows.npz"
    np.savez(rows_path, rows=np.array([0, 1], dtype=np.int64))
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--policy.push_to_hub",
            "false",
            "--axis_rows_path",
            str(rows_path),
            "--axis_expected_frames",
            "10",
            "--axis_quality_path",
            "/tmp/q.npz",
        ],
    )
    cfg.validate()  # must not raise


# --- train-loop wire-in ---------------------------------------------------------------------


def test_make_axis_quality_tags_wire_in(tmp_path):
    """Mirrors _make_axis_sampler's shape: a pure function building the tags object, callable
    without a real dataset."""
    from lerobot.scripts.lerobot_train import _make_axis_quality_tags
    from lerobot.utils.axis_quality import AxisQualityTags

    p = _quality_npz(tmp_path, [0, 1, 2, 0])
    tags = _make_axis_quality_tags(str(p), num_frames=4)
    assert isinstance(tags, AxisQualityTags)


def test_cfg_tagged_frac_reaches_metrics_tracker(tmp_path):
    """The exact mechanism the train loop uses to get cfg_tagged_frac onto the `step:N` log
    line: MetricsTracker.update_metrics auto-registers a meter for a key that was never passed
    to the constructor (the same path sample_weight_* stats take inside update_policy) -- so
    calling it directly with apply_quality_tags' return value must make the key show up both in
    __str__ and to_dict(), with no special-casing needed in the train loop beyond the call.
    """
    from lerobot.utils.axis_quality import AxisQualityTags, apply_quality_tags
    from lerobot.utils.logging_utils import MetricsTracker

    q = AxisQualityTags(str(_quality_npz(tmp_path, [0, 3, 5, 0])), expected_frames=4)
    batch = {"task": ["pick", "place", "push"], "index": torch.tensor([1, 0, 2])}
    stats = apply_quality_tags(batch, q)

    tracker = MetricsTracker(batch_size=3, num_frames=4, num_episodes=1, metrics={})
    tracker.update_metrics(stats)
    assert "cfg_tagged_frac" in str(tracker)
    assert tracker.to_dict()["cfg_tagged_frac"] == pytest.approx(2 / 3)
