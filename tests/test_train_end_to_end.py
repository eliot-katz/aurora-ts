from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurora_ts.models.train import ExperimentConfig, run_baseline_experiment_from_csv


def _write_synthetic_ohlcv_csv(path: Path, n: int = 600) -> None:
    rng = np.random.default_rng(0)

    # Random walk for close
    steps = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = close * (1.0 + rng.normal(0, 0.001, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.002, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.002, size=n)))
    volume = rng.integers(1000, 5000, size=n)

    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    df = pd.DataFrame(
        {
            "date": dates.astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.to_csv(path, index=False)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_end_to_end_baseline_train_pipeline(tmp_path: Path):
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_ohlcv_csv(csv_path, n=650)

    cfg = ExperimentConfig(
        train_min_size=200,
        test_size=50,
        step_size=50,
        gap=0,
        horizon=1,
        random_state=123,
    )

    results = run_baseline_experiment_from_csv(csv_path, cfg)

    # We should have multiple splits
    assert len(results) >= 3

    # Check each split result is consistent
    for r in results:
        assert r.n_train >= cfg.train_min_size
        assert r.n_test == cfg.test_size

        m = r.metrics
        assert 0.0 <= m.accuracy <= 1.0
        assert 0.0 <= m.balanced_accuracy <= 1.0
        assert 0.0 <= m.f1 <= 1.0
        # AUC can exist and should be between 0 and 1 if present
        if m.auc is not None:
            assert 0.0 <= m.auc <= 1.0
