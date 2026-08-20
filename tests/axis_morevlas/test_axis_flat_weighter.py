from __future__ import annotations

import numpy as np
import pytest
import torch


def _npz(tmp_path, weights):
    p = tmp_path / "w.npz"
    np.savez(p, weights=np.asarray(weights, dtype=np.float32))
    return p


def _weighter(tmp_path, weights, expected=None):
    from lerobot.utils.axis_sample_weighting import AxisFlatWeighter
    from lerobot.utils.sample_weighting import SampleWeightingConfig

    cfg = SampleWeightingConfig(
        type="axis_flat",
        extra_params={"weights_path": str(_npz(tmp_path, weights)),
                      "expected_frames": expected or len(weights)},
    )
    return AxisFlatWeighter(cfg, torch.device("cpu"))


def test_weights_looked_up_by_global_index(tmp_path):
    w = _weighter(tmp_path, [1.0, 2.0, 3.0, 4.0])
    out, stats = w.compute_batch_weights({"index": torch.tensor([2, 0, 3])})
    assert out.tolist() == [3.0, 1.0, 4.0]


def test_nonfinite_weight_is_fatal(tmp_path):
    w = _weighter(tmp_path, [1.0, float("nan"), 3.0])
    with pytest.raises(RuntimeError, match="non-finite"):
        w.compute_batch_weights({"index": torch.tensor([1])})


def test_frame_count_mismatch_refused(tmp_path):
    with pytest.raises(ValueError, match="expected_frames"):
        _weighter(tmp_path, [1.0, 2.0], expected=99)


def test_factory_dispatches_axis_flat(tmp_path):
    from lerobot.utils.axis_sample_weighting import AxisFlatWeighter
    from lerobot.utils.sample_weighting import SampleWeightingConfig, make_sample_weighter

    cfg = SampleWeightingConfig(
        type="axis_flat",
        extra_params={"weights_path": str(_npz(tmp_path, [1.0])), "expected_frames": 1},
    )
    w = make_sample_weighter(cfg, policy=None, device=torch.device("cpu"))
    assert isinstance(w, AxisFlatWeighter)
