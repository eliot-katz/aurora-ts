import numpy as np
import pandas as pd
import pytest

from aurora_ts.data.splits import (
    expanding_window_splits,
    rolling_window_splits,
    walk_forward_splits,
)


def _assert_chronological(split, gap: int):
    # Basic properties
    assert split.train_idx.ndim == 1
    assert split.test_idx.ndim == 1
    assert split.train_idx.size > 0
    assert split.test_idx.size > 0

    # Strictly increasing
    assert np.all(np.diff(split.train_idx) == 1)
    assert np.all(np.diff(split.test_idx) == 1)

    # No overlap
    assert set(split.train_idx).isdisjoint(set(split.test_idx))

    # Chronology with gap:
    # train_end < test_start - gap  (since gap points exist between)
    train_end = split.train_idx[-1]
    test_start = split.test_idx[0]
    assert train_end < test_start
    assert test_start - train_end - 1 == gap


def test_expanding_window_basic():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    splits = list(
        expanding_window_splits(idx, test_size=10, train_min_size=50, step_size=10, gap=0)
    )
    assert len(splits) == 5  # test windows: [50:60], [60:70], [70:80], [80:90], [90:100]

    for s in splits:
        _assert_chronological(s, gap=0)
        assert s.test_idx.size == 10
        assert s.train_idx.size >= 50
        # expanding means the train grows over time
    assert splits[0].train_idx.size == 50
    assert splits[1].train_idx.size == 60
    assert splits[-1].train_idx.size == 90


def test_expanding_window_with_gap():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    splits = list(
        expanding_window_splits(idx, test_size=10, train_min_size=50, step_size=10, gap=2)
    )
    # test starts at 52, 62, 72, 82, 92, 102; last test is [102:112]
    assert len(splits) == 6

    for s in splits:
        _assert_chronological(s, gap=2)
        assert s.test_idx.size == 10
        assert s.train_idx.size >= 50

    # Check first split boundaries explicitly
    s0 = splits[0]
    assert s0.train_idx[0] == 0
    assert s0.train_idx[-1] == 49  # train ends at 49
    assert s0.test_idx[0] == 52  # 50,51 are gap
    assert s0.test_idx[-1] == 61


def test_rolling_window_basic():
    idx = pd.date_range("2021-01-01", periods=100, freq="D")
    splits = list(rolling_window_splits(idx, test_size=10, train_size=30, step_size=10, gap=0))
    # test starts at 30 -> [30:40], [40:50], ... [90:100] => 7 splits
    assert len(splits) == 7

    for s in splits:
        _assert_chronological(s, gap=0)
        assert s.test_idx.size == 10
        assert s.train_idx.size == 30  # fixed size rolling

    # Check train window slides
    assert splits[0].train_idx[0] == 0
    assert splits[0].train_idx[-1] == 29
    assert splits[1].train_idx[0] == 10
    assert splits[1].train_idx[-1] == 39


def test_walk_forward_step_size_1_count():
    idx = list(range(20))
    splits = list(walk_forward_splits(idx, test_size=2, train_min_size=10, step_size=1, gap=0))
    # test starts at 10, last start at 18 (exclusive end 20) => 20-2 -10 +1 = 9
    assert len(splits) == 9
    for s in splits:
        _assert_chronological(s, gap=0)
        assert s.test_idx.size == 2
        assert s.train_idx.size >= 10


def test_invalid_inputs_raise():
    idx = list(range(10))

    with pytest.raises(ValueError):
        list(expanding_window_splits(idx, test_size=0, train_min_size=5))

    with pytest.raises(ValueError):
        list(expanding_window_splits(idx, test_size=5, train_min_size=0))

    with pytest.raises(ValueError):
        list(expanding_window_splits(idx, test_size=5, train_min_size=5, gap=-1))

    # Not enough data for even one split
    with pytest.raises(ValueError):
        list(expanding_window_splits(idx, test_size=6, train_min_size=5, gap=0))

    # Rolling: train_size must be >= train_min_size (enforced internally)
    with pytest.raises(ValueError):
        list(walk_forward_splits(idx, test_size=2, train_min_size=5, train_size=3))
