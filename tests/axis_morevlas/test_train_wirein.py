from __future__ import annotations

import numpy as np
import pytest


def test_config_has_axis_fields():
    from lerobot.configs.train import TrainPipelineConfig

    fields = {f.name for f in __import__("dataclasses").fields(TrainPipelineConfig)}
    assert {"axis_rows_path", "axis_expected_frames"} <= fields


def test_sampler_selection_prefers_axis_rows(tmp_path):
    """The wire-in must expose a pure function selecting the sampler, so this test does not
    need a dataset: _make_axis_sampler(rows_path, seed) returns AxisRowSampler over the rows."""
    from lerobot.datasets.axis_sampler import AxisRowSampler
    from lerobot.scripts.lerobot_train import _make_axis_sampler

    p = tmp_path / "rows.npz"
    np.savez(p, rows=np.array([1, 5, 9], dtype=np.int64))
    s = _make_axis_sampler(str(p), seed=3)
    assert isinstance(s, AxisRowSampler)
    assert sorted(iter(s)) == [1, 5, 9]


def test_frame_count_guard():
    from lerobot.scripts.lerobot_train import _check_axis_frames

    class FakeDs:
        num_frames = 10

    _check_axis_frames(FakeDs(), 10)  # ok
    with pytest.raises(RuntimeError, match="1906469|expected"):
        _check_axis_frames(FakeDs(), 1906469)
