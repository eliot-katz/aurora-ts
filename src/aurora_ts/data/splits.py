"""
Time-series splitting utilities (walk-forward, expanding/rolling windows).

Design goals:
- Chronological splits only (no shuffling)
- Optional "gap" (a.k.a. embargo) between train and test to reduce leakage
- Test window size fixed; train window can be expanding or rolling
- Returns index arrays (integer positions) to avoid copying data
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeSeriesSplit:
    """A single chronological split defined by integer positions."""

    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def train_start(self) -> int:
        return int(self.train_idx[0])

    @property
    def train_end(self) -> int:
        return int(self.train_idx[-1])

    @property
    def test_start(self) -> int:
        return int(self.test_idx[0])

    @property
    def test_end(self) -> int:
        return int(self.test_idx[-1])


def _n_obs(index: Sequence) -> int:
    # Works for pandas Index, list, np.ndarray, etc.
    return len(index)


def _validate_inputs(
    n: int,
    test_size: int,
    train_min_size: int,
    step_size: int,
    gap: int,
    train_size: int | None,
) -> None:
    if n <= 0:
        raise ValueError("index must contain at least 1 observation.")
    if test_size <= 0:
        raise ValueError("test_size must be > 0.")
    if train_min_size <= 0:
        raise ValueError("train_min_size must be > 0.")
    if step_size <= 0:
        raise ValueError("step_size must be > 0.")
    if gap < 0:
        raise ValueError("gap must be >= 0.")
    if train_size is not None and train_size <= 0:
        raise ValueError("train_size must be > 0 when provided.")
    if train_min_size + gap + test_size > n:
        raise ValueError(
            "Not enough data for a single split with given train_min_size/gap/test_size."
        )


def walk_forward_splits(
    index: Sequence,
    *,
    test_size: int,
    train_min_size: int,
    step_size: int = 1,
    gap: int = 0,
    train_size: int | None = None,
) -> Iterator[TimeSeriesSplit]:
    """
    Generate walk-forward time-series splits.

    Parameters
    ----------
    index:
        Any sequence with length n (e.g., pandas DatetimeIndex, RangeIndex).
        We only use its length; we return integer positions.
    test_size:
        Number of observations in each test window.
    train_min_size:
        Minimum size of training window for the first split.
        For expanding window, train grows over time.
        For rolling window (train_size not None), train window has fixed length.
    step_size:
        How many observations to move forward between successive splits.
    gap:
        Number of observations to skip between the end of train and start of test
        (a.k.a embargo). Helps reduce leakage from label construction or microstructure.
    train_size:
        If provided, use a rolling training window of fixed size.
        If None, use an expanding training window.

    Yields
    ------
    TimeSeriesSplit
        Contains train_idx and test_idx as integer arrays.

    Notes
    -----
    - Split k uses:
        train = [train_start, train_end]
        gap  = (train_end+1) ... (test_start-1)
        test = [test_start, test_end]
    - Ensures strict chronological order and no overlap.
    """
    n = _n_obs(index)
    _validate_inputs(n, test_size, train_min_size, step_size, gap, train_size)

    # First test window starts after initial train + gap
    test_start = train_min_size + gap
    while True:
        test_end = test_start + test_size  # exclusive
        if test_end > n:
            break

        # Train ends right before the gap
        train_end_excl = test_start - gap  # exclusive end for train
        if train_end_excl <= 0:
            break

        if train_size is None:
            # Expanding window from 0
            train_start = 0
            train_end = train_end_excl
            if train_end - train_start < train_min_size:
                # Shouldn't happen with our initialization, but keep safe.
                test_start += step_size
                continue
        else:
            # Rolling window of fixed length train_size, but at least train_min_size
            if train_size < train_min_size:
                raise ValueError("train_size must be >= train_min_size.")
            train_end = train_end_excl
            train_start = max(0, train_end - train_size)

            # Ensure min size
            if train_end - train_start < train_min_size:
                test_start += step_size
                continue

        train_idx = np.arange(train_start, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)

        yield TimeSeriesSplit(train_idx=train_idx, test_idx=test_idx)

        test_start += step_size


def expanding_window_splits(
    index: Sequence,
    *,
    test_size: int,
    train_min_size: int,
    step_size: int = 1,
    gap: int = 0,
) -> Iterator[TimeSeriesSplit]:
    """Convenience wrapper for expanding window walk-forward splits."""
    return walk_forward_splits(
        index,
        test_size=test_size,
        train_min_size=train_min_size,
        step_size=step_size,
        gap=gap,
        train_size=None,
    )


def rolling_window_splits(
    index: Sequence,
    *,
    test_size: int,
    train_size: int,
    step_size: int = 1,
    gap: int = 0,
) -> Iterator[TimeSeriesSplit]:
    """Convenience wrapper for rolling window walk-forward splits."""
    return walk_forward_splits(
        index,
        test_size=test_size,
        train_min_size=train_size,
        step_size=step_size,
        gap=gap,
        train_size=train_size,
    )
