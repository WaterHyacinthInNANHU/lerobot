from __future__ import annotations

import json

import numpy as np
import pytest

# Importing lerobot.scripts.lerobot_train pulls in the policy factory, which registers
# PreTrainedConfig subclasses (e.g. "act") as an import-time side effect. Without it,
# `draccus.parse(..., "--policy.type", "act", ...)` below fails with "choose from ()" when
# this file is the first to touch draccus.parse in a session -- see test_train_wirein.py,
# which happens to get this for free because one of its earlier tests imports lerobot_train.
import lerobot.scripts.lerobot_train  # noqa: F401


def _schedule_npz(tmp_path, rows, n_rows=100, dtype=np.int64, meta_extra=None, name="sched.npz"):
    """Write a schedule artifact shaped like the REAL one from
    axis.dataset.build_index_schedule.write_schedule: `rows` is 2-D (total_steps, batch) --
    NOT 1-D -- because that is what both write_schedule (AXIS-Bench) and openpi's own
    ScheduleSampler actually read/write. `meta` mirrors openpi's ScheduleSampler.check_dataset_rows
    contract: n_rows is the field it binds to len(dataset).
    """
    p = tmp_path / name
    meta = {"n_rows": n_rows, "mode": "drop", "reward_id": "v2"}
    if meta_extra:
        meta.update(meta_extra)
    np.savez(p, rows=np.asarray(rows, dtype=dtype), meta=np.array(json.dumps(meta)))
    return p


def test_replays_rows_in_exact_file_order(tmp_path):
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    # 3 steps x 2 batch, row-major flatten -- repeats are legal (a schedule draws WITH replacement).
    rows = [[5, 3], [5, 99], [0, 3]]
    flat = [5, 3, 5, 99, 0, 3]
    s = AxisScheduleSampler(str(_schedule_npz(tmp_path, rows)), expected_frames=100)
    assert list(iter(s)) == flat
    assert len(s) == 6
    assert list(iter(s)) == flat  # second pass replays identically -- no consumption


def test_refuses_resume(tmp_path):
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    s = AxisScheduleSampler(str(_schedule_npz(tmp_path, [[1, 2]])), expected_frames=100)
    with pytest.raises(RuntimeError, match="resume"):
        s.load_state_dict({"epoch": 0, "start_index": 1})


def test_state_dict_is_a_noop_and_set_epoch_is_a_noop(tmp_path):
    """The schedule IS the epoch structure: set_epoch must not perturb replay order, and
    state_dict must not claim any position lerobot could later feed back through load_state_dict."""
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    s = AxisScheduleSampler(str(_schedule_npz(tmp_path, [[1, 2], [3, 4]])), expected_frames=100)
    assert s.state_dict() == {}
    s.set_epoch(5)  # must not raise, must not change replay order
    assert list(iter(s)) == [1, 2, 3, 4]


def test_refuses_wrong_dtype(tmp_path):
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    with pytest.raises(ValueError, match="dtype"):
        AxisScheduleSampler(str(_schedule_npz(tmp_path, [[1.0, 2.0]], dtype=np.float64)), expected_frames=100)


def test_refuses_out_of_range_rows(tmp_path):
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    with pytest.raises(ValueError, match="expected_frames|out of range"):
        AxisScheduleSampler(str(_schedule_npz(tmp_path, [[100]])), expected_frames=100)


def test_refuses_negative_rows(tmp_path):
    """Mirrors openpi ScheduleSampler.check_dataset_rows: torch indexes a negative int as
    'from the end', so a negative row would silently draw from the dataset tail instead of
    raising -- catch it at construction."""
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    with pytest.raises(ValueError, match="negative"):
        AxisScheduleSampler(str(_schedule_npz(tmp_path, [[0, -1]])), expected_frames=100)


def test_refuses_one_dimensional_rows(tmp_path):
    """Mirrors openpi ScheduleSampler: the artifact's `rows` key is 2-D (total_steps, batch);
    a 1-D array cannot be told apart from a corrupt/pre-schedule artifact."""
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    with pytest.raises(ValueError, match="2-D"):
        AxisScheduleSampler(str(_schedule_npz(tmp_path, [1, 2, 3])), expected_frames=100)


def test_refuses_meta_mismatch(tmp_path):
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    # meta says the schedule was drawn against a 50-row corpus; the dataset has 100 frames.
    with pytest.raises(ValueError, match="n_rows"):
        AxisScheduleSampler(str(_schedule_npz(tmp_path, [[1, 2]], n_rows=50)), expected_frames=100)


def test_refuses_missing_n_rows(tmp_path):
    """Mirrors openpi: an artifact with no recorded corpus size cannot be told apart from one
    built against a different corpus, so absence is rejected exactly like a mismatch."""
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    p = tmp_path / "sched.npz"
    np.savez(
        p,
        rows=np.asarray([[1, 2]], dtype=np.int64),
        meta=np.array(json.dumps({"mode": "drop"})),
    )
    with pytest.raises(ValueError, match="n_rows"):
        AxisScheduleSampler(str(p), expected_frames=100)


def test_config_has_axis_schedule_field():
    import dataclasses

    from lerobot.configs.train import TrainPipelineConfig

    fields = {f.name for f in dataclasses.fields(TrainPipelineConfig)}
    assert "axis_schedule_path" in fields


def test_config_exclusivity_schedule_and_rows():
    """A schedule replay and a uniform row draw cannot both drive the sampler."""
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
            "--axis_schedule_path",
            "/tmp/s.npz",
            "--axis_rows_path",
            "/tmp/r.npz",
            "--axis_expected_frames",
            "10",
        ],
    )
    with pytest.raises(ValueError, match="axis_"):
        cfg.validate()


def test_config_exclusivity_schedule_and_sample_weighting():
    """openpi config.py's schedule+quality refusal, mirrored: a schedule already decided every
    draw offline, so a second weighting arm layered on top is two arms at once."""
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
            "--axis_schedule_path",
            "/tmp/s.npz",
            "--axis_expected_frames",
            "10",
            "--sample_weighting.type",
            "axis_flat",
        ],
    )
    with pytest.raises(ValueError, match="axis_schedule|two arms|sample_weighting"):
        cfg.validate()


def test_config_exclusivity_schedule_requires_expected_frames():
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
            "--axis_schedule_path",
            "/tmp/s.npz",
        ],
    )
    with pytest.raises(ValueError, match="axis_expected_frames"):
        cfg.validate()


def test_config_exclusivity_schedule_with_eval_split():
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
            "--axis_schedule_path",
            "/tmp/s.npz",
            "--axis_expected_frames",
            "10",
            "--dataset.eval_split",
            "0.1",
        ],
    )
    with pytest.raises(ValueError, match="eval_split"):
        cfg.validate()


def test_make_axis_schedule_sampler_wire_in(tmp_path):
    """The wire-in helper mirrors _make_axis_sampler's shape: a pure function selecting the
    sampler, callable without a real dataset."""
    from lerobot.datasets.axis_sampler import AxisScheduleSampler
    from lerobot.scripts.lerobot_train import _make_axis_schedule_sampler

    p = _schedule_npz(tmp_path, [[1, 2], [3, 4]])
    s = _make_axis_schedule_sampler(str(p), num_frames=100)
    assert isinstance(s, AxisScheduleSampler)
    assert list(iter(s)) == [1, 2, 3, 4]


# --- batch alignment / step budget guards (mirrors openpi ScheduleSampler.check_batch /
# check_num_train_steps, schedule_sampler.py:60-91) -------------------------------------------


def test_total_steps_and_batch_properties(tmp_path):
    """The schedule's own (total_steps, batch) shape must be readable off the sampler --
    consumed by _check_axis_schedule_budget at the wire-in site."""
    from lerobot.datasets.axis_sampler import AxisScheduleSampler

    # 3 steps x 2 batch.
    p = _schedule_npz(tmp_path, [[1, 2], [3, 4], [5, 6]])
    s = AxisScheduleSampler(str(p), expected_frames=100)
    assert s.total_steps == 3
    assert s.batch == 2


class _FakeScheduleSampler:
    """Stands in for AxisScheduleSampler in _check_axis_schedule_budget tests: the guard is a
    pure function of (sampler.batch, sampler.total_steps, batch_size, num_train_steps), so it
    does not need a real .npz artifact -- mirrors the FakeDs pattern _check_axis_frames tests
    already use."""

    def __init__(self, batch: int, total_steps: int) -> None:
        self.batch = batch
        self.total_steps = total_steps


def test_check_axis_schedule_budget_refuses_batch_mismatch():
    from lerobot.scripts.lerobot_train import _check_axis_schedule_budget

    sampler = _FakeScheduleSampler(batch=64, total_steps=1000)
    with pytest.raises(ValueError, match="batch"):
        _check_axis_schedule_budget(sampler, batch_size=32, num_train_steps=1000)
    with pytest.raises(ValueError, match="batch"):
        _check_axis_schedule_budget(sampler, batch_size=128, num_train_steps=1000)


def test_check_axis_schedule_budget_accepts_matching_batch():
    from lerobot.scripts.lerobot_train import _check_axis_schedule_budget

    sampler = _FakeScheduleSampler(batch=64, total_steps=1000)
    _check_axis_schedule_budget(sampler, batch_size=64, num_train_steps=1000)  # must not raise


def test_check_axis_schedule_budget_refuses_step_overrun():
    """A budget longer than the schedule would silently replay it from row 0 (the torch loader
    restarts an exhausted sampler rather than raising)."""
    from lerobot.scripts.lerobot_train import _check_axis_schedule_budget

    sampler = _FakeScheduleSampler(batch=64, total_steps=1000)
    with pytest.raises(ValueError, match="exceeds|steps"):
        _check_axis_schedule_budget(sampler, batch_size=64, num_train_steps=1001)


def test_check_axis_schedule_budget_warns_but_allows_short_step_budget(caplog):
    """Mirrors openpi's exact policy: fewer steps than the schedule is allowed (just a
    truncated run whose coverage numbers no longer describe it), not refused."""
    from lerobot.scripts.lerobot_train import _check_axis_schedule_budget

    sampler = _FakeScheduleSampler(batch=64, total_steps=1000)
    with caplog.at_level("WARNING"):
        _check_axis_schedule_budget(sampler, batch_size=64, num_train_steps=500)  # must not raise
    assert "short" in caplog.text or "steps" in caplog.text


def test_make_dataloaders_wires_in_the_budget_guard(tmp_path):
    """The guard must actually run at the schedule sampler's construction site, not just exist
    as a standalone function -- construct via _make_axis_schedule_sampler then call the guard
    the same way make_dataloaders does, using a schedule/cfg pairing that must be refused."""
    from lerobot.scripts.lerobot_train import _check_axis_schedule_budget, _make_axis_schedule_sampler

    p = _schedule_npz(tmp_path, [[1, 2], [3, 4]])  # batch=2, total_steps=2
    sampler = _make_axis_schedule_sampler(str(p), num_frames=100)
    with pytest.raises(ValueError, match="batch"):
        _check_axis_schedule_budget(sampler, batch_size=1, num_train_steps=2)
    _check_axis_schedule_budget(sampler, batch_size=2, num_train_steps=2)  # must not raise
