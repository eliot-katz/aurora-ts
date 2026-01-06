from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aurora_ts.data.loaders import OHLCVLoadSpec, load_ohlcv_csv


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "data.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_load_ohlcv_sorts_and_parses_dates(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": ["2020-01-03", "2020-01-01", "2020-01-02"],
            "open": [3, 1, 2],
            "high": [3.1, 1.1, 2.1],
            "low": [2.9, 0.9, 1.9],
            "close": [3.05, 1.05, 2.05],
            "volume": [300, 100, 200],
        }
    )
    path = _write_csv(tmp_path, df)

    out = load_ohlcv_csv(path)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.name == "date"
    assert out.index.is_monotonic_increasing
    assert out.shape == (3, 5)


def test_load_ohlcv_rejects_missing_columns(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "open": [1, 2],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            # "close" missing
            "volume": [100, 200],
        }
    )
    path = _write_csv(tmp_path, df)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_ohlcv_csv(path)


def test_load_ohlcv_duplicate_timestamps_raise_by_default(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "open": [1, 10, 2],
            "high": [1.1, 10.1, 2.1],
            "low": [0.9, 9.9, 1.9],
            "close": [1.05, 10.05, 2.05],
            "volume": [100, 1000, 200],
        }
    )
    path = _write_csv(tmp_path, df)

    with pytest.raises(ValueError, match="Duplicate timestamps"):
        load_ohlcv_csv(path)


def test_load_ohlcv_duplicate_timestamps_drop_keep_last(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "open": [1, 10, 2],
            "high": [1.1, 10.1, 2.1],
            "low": [0.9, 9.9, 1.9],
            "close": [1.05, 10.05, 2.05],
            "volume": [100, 1000, 200],
        }
    )
    path = _write_csv(tmp_path, df)

    spec = OHLCVLoadSpec(duplicate_policy="drop_keep_last")
    out = load_ohlcv_csv(path, spec=spec)

    assert out.shape == (2, 5)
    # For 2020-01-01 we keep the last row (open=10)
    assert float(out.loc[pd.Timestamp("2020-01-01"), "open"]) == 10.0


def test_load_ohlcv_parses_timezone_and_returns_naive_index(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00+00:00"],
            "open": [1, 2],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [100, 200],
        }
    )
    path = _write_csv(tmp_path, df)

    out = load_ohlcv_csv(path)
    assert out.index.tz is None  # timezone-naive
    assert out.index[0] == pd.Timestamp("2020-01-01")
    assert out.index[1] == pd.Timestamp("2020-01-02")


def test_load_ohlcv_symbol_filtering(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "open": [1, 10, 2, 20],
            "high": [1.1, 10.1, 2.1, 20.1],
            "low": [0.9, 9.9, 1.9, 19.9],
            "close": [1.05, 10.05, 2.05, 20.05],
            "volume": [100, 1000, 200, 2000],
        }
    )
    path = _write_csv(tmp_path, df)

    out = load_ohlcv_csv(path, symbol="AAA")
    assert out.shape == (2, 5)
    assert out.index.is_monotonic_increasing
    assert float(out.loc[pd.Timestamp("2020-01-01"), "open"]) == 1.0


def test_load_ohlcv_symbol_filter_requires_symbol_column(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "open": [1, 2],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [100, 200],
        }
    )
    path = _write_csv(tmp_path, df)

    with pytest.raises(ValueError, match="symbol was provided"):
        load_ohlcv_csv(path, symbol="AAA")
