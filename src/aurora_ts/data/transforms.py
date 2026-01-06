"""
Data transformations with strict anti-leakage guarantees.

Design principles:
- Fit transformations ONLY on training data
- Apply frozen parameters to validation/test
- Rolling statistics must be causal (past-only)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedScaler:
    """A fitted scaler with immutable parameters."""

    mean_: np.ndarray
    scale_: np.ndarray
    columns: list[str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        arr = (df[self.columns].values - self.mean_) / self.scale_
        out = df.copy()
        out[self.columns] = arr
        return out


def fit_standard_scaler(
    df_train: pd.DataFrame,
    columns: Iterable[str],
) -> FittedScaler:
    """
    Fit a standard scaler (z-score) on training data only.

    Parameters
    ----------
    df_train:
        Training dataframe.
    columns:
        Columns to scale.

    Returns
    -------
    FittedScaler
        Frozen scaler parameters.
    """
    cols = list(columns)
    if not cols:
        raise ValueError("No columns provided for scaling.")

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(df_train[cols].values)

    train_std = df_train[cols].std(ddof=0).to_numpy()
    if np.any(train_std == 0):
        raise ValueError("Zero variance detected in scaling columns.")

    return FittedScaler(
        mean_=scaler.mean_.copy(),
        scale_=scaler.scale_.copy(),
        columns=cols,
    )


def apply_scaler(
    df: pd.DataFrame,
    scaler: FittedScaler,
) -> pd.DataFrame:
    """
    Apply a fitted scaler to any dataset (train/val/test).

    Parameters
    ----------
    df:
        Input dataframe.
    scaler:
        FittedScaler obtained from training data.

    Returns
    -------
    pd.DataFrame
        Scaled dataframe.
    """
    missing = [c for c in scaler.columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for scaling: {missing}")

    return scaler.transform(df)


def rolling_zscore(
    df: pd.DataFrame,
    columns: Sequence[str],
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Compute causal rolling z-score (past-only).

    For each time t:
        z_t = (x_t - mean(x_{t-window:t-1})) / std(x_{t-window:t-1})

    Parameters
    ----------
    df:
        Input dataframe (must be time-ordered).
    columns:
        Columns to transform.
    window:
        Rolling window size.
    min_periods:
        Minimum observations required to compute statistics.
        Defaults to window.

    Returns
    -------
    pd.DataFrame
        Dataframe with rolling z-scored columns.
    """
    if window <= 1:
        raise ValueError("window must be > 1.")

    if min_periods is None:
        min_periods = window

    out = df.copy()

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found for rolling z-score.")

        roll_mean = df[col].shift(1).rolling(window=window, min_periods=min_periods).mean()
        roll_std = df[col].shift(1).rolling(window=window, min_periods=min_periods).std()

        out[col] = (df[col] - roll_mean) / roll_std

    return out
