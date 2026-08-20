from __future__ import annotations

import numpy as np
import pytest
import torch


def test_yields_exactly_the_given_rows_shuffled():
    from lerobot.datasets.axis_sampler import AxisRowSampler

    rows = np.array([3, 7, 11, 20, 21], dtype=np.int64)
    s = AxisRowSampler(rows=rows, seed=0)
    got = list(iter(s))
    assert sorted(got) == sorted(rows.tolist())
    assert len(s) == 5


def test_deterministic_in_seed_and_epoch():
    from lerobot.datasets.axis_sampler import AxisRowSampler

    rows = np.arange(0, 1000, 3, dtype=np.int64)
    a = AxisRowSampler(rows=rows, seed=42)
    b = AxisRowSampler(rows=rows, seed=42)
    epoch0_a = list(iter(a))  # a at epoch 0, then auto-increments to 1
    epoch0_b = list(iter(b))  # b at epoch 0, then auto-increments to 1
    assert epoch0_a == epoch0_b  # same seed, same epoch

    epoch1_a = list(iter(a))  # a at epoch 1, then auto-increments to 2
    epoch1_b = list(iter(b))  # b at epoch 1, then auto-increments to 2
    assert epoch1_a == epoch1_b  # same seed, same epoch
    assert epoch0_a != epoch1_a  # different epochs, different permutation

    c = AxisRowSampler(rows=rows, seed=43)
    epoch1_c = list(iter(c))  # c at epoch 0
    assert epoch0_a != epoch1_c  # different seed, different permutation


def test_resume_skips_consumed_prefix():
    from lerobot.datasets.axis_sampler import AxisRowSampler

    rows = np.arange(100, dtype=np.int64)
    full = list(iter(AxisRowSampler(rows=rows, seed=7)))
    resumed = AxisRowSampler(rows=rows, seed=7)
    resumed.load_state_dict({"epoch": 0, "start_index": 40})
    assert list(iter(resumed)) == full[40:]


def test_refuses_out_of_range_rows():
    from lerobot.datasets.axis_sampler import AxisRowSampler

    with pytest.raises(ValueError):
        AxisRowSampler(rows=np.array([-1], dtype=np.int64), seed=0)


def test_len_returns_full_length_during_resume():
    """Regression test: __len__ must return full length, not remaining after resume offset."""
    from lerobot.datasets.axis_sampler import AxisRowSampler

    rows = np.arange(100, dtype=np.int64)
    s = AxisRowSampler(rows=rows, seed=7)
    s.load_state_dict({"epoch": 0, "start_index": 40})
    # Even though we'll skip 40 items, __len__ must report full length (like EpisodeAwareSampler)
    assert len(s) == 100


def test_iter_eagerly_advances_epoch_and_resets_start():
    """Test that __iter__ eagerly mutates state before the generator yields anything.

    This ensures that calling iter(s) immediately advances the epoch and resets _start,
    mirroring EpisodeAwareSampler's contract. state_dict() called right after iter()
    (before consuming any items) must reflect these eager mutations.
    """
    from lerobot.datasets.axis_sampler import AxisRowSampler

    rows = np.arange(10, dtype=np.int64)
    s = AxisRowSampler(rows=rows, seed=99)

    # Before any iteration
    assert s.state_dict() == {"epoch": 0, "start_index": 0}

    # Create an iterator but don't consume it yet
    it = iter(s)

    # state_dict must IMMEDIATELY reflect the eager mutations:
    # - epoch advanced to 1
    # - start_index reset to 0
    assert s.state_dict() == {"epoch": 1, "start_index": 0}

    # Consume the iterator to verify it still works correctly
    result = list(it)
    assert len(result) == 10
    assert sorted(result) == list(range(10))
