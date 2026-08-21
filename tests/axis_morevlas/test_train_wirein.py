from __future__ import annotations

import numpy as np
import pytest


def test_config_has_axis_fields():
    from lerobot.configs.train import TrainPipelineConfig

    fields = {f.name for f in __import__("dataclasses").fields(TrainPipelineConfig)}
    assert {"axis_rows_path", "axis_expected_frames"} <= fields


def test_sampler_selection_prefers_axis_rows(tmp_path):
    """The wire-in must expose a pure function selecting the sampler, so this test does not
    need a dataset: _make_axis_sampler(rows_path, seed, num_frames) returns AxisRowSampler
    over the rows."""
    from lerobot.datasets.axis_sampler import AxisRowSampler
    from lerobot.scripts.lerobot_train import _make_axis_sampler

    p = tmp_path / "rows.npz"
    np.savez(p, rows=np.array([1, 5, 9], dtype=np.int64))
    s = _make_axis_sampler(str(p), seed=3, num_frames=10)
    assert isinstance(s, AxisRowSampler)
    assert sorted(iter(s)) == [1, 5, 9]


def test_make_axis_sampler_refuses_out_of_range_rows(tmp_path):
    """rows.max() must be strictly less than num_frames: the frame-count guard on the dataset
    total (`_check_axis_frames`) does not by itself confirm every individual row index falls
    inside that range, so `_make_axis_sampler` must check the join directly."""
    from lerobot.scripts.lerobot_train import _make_axis_sampler

    p = tmp_path / "rows.npz"
    np.savez(p, rows=np.array([1, 5, 9], dtype=np.int64))
    with pytest.raises(ValueError, match="out of range"):
        _make_axis_sampler(str(p), seed=3, num_frames=9)  # max row is 9, valid range is [0, 9)
    _make_axis_sampler(str(p), seed=3, num_frames=10)  # ok: max row 9 < num_frames 10


def test_frame_count_guard():
    from lerobot.scripts.lerobot_train import _check_axis_frames

    class FakeDs:
        num_frames = 10

    _check_axis_frames(FakeDs(), 10)  # ok
    with pytest.raises(RuntimeError, match="1906469|expected"):
        _check_axis_frames(FakeDs(), 1906469)


def test_validate_refuses_axis_rows_without_expected_frames():
    """`validate()` must couple axis_rows_path to axis_expected_frames: without the expected
    count, a corpus mismatch would silently land every index on the wrong frame."""
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
            "--axis_rows_path",
            "rows.npz",
        ],
    )
    with pytest.raises(ValueError, match="axis_expected_frames"):
        cfg.validate()


def test_validate_refuses_axis_rows_with_eval_split():
    """axis row sampling requires the full, unfiltered dataset: an eval_split carve-out would
    shift which global frame indices are even reachable, misaligning the rows artifact."""
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
            "--axis_rows_path",
            "rows.npz",
            "--axis_expected_frames",
            "10",
            "--dataset.eval_split",
            "0.1",
        ],
    )
    with pytest.raises(ValueError, match="eval_split"):
        cfg.validate()
