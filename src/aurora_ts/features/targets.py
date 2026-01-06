from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TargetType = Literal["return", "log_return", "direction", "quantile_class"]


@dataclass(frozen=True)
class TargetSpec:
    target_type: TargetType
    horizon: int = 1
    price_col: str = "close"

    # For quantile_class
    n_classes: int = 5  # e.g. quintiles

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0.")
        if self.target_type == "quantile_class" and self.n_classes < 2:
            raise ValueError("n_classes must be >= 2 for quantile_class.")


def _future_return(prices: pd.Series, horizon: int) -> pd.Series:
    # r_{t,h} = P_{t+h} / P_t - 1
    return prices.shift(-horizon) / prices - 1.0


def _future_log_return(prices: pd.Series, horizon: int) -> pd.Series:
    return np.log(prices.shift(-horizon)) - np.log(prices)


def build_target(df: pd.DataFrame, spec: TargetSpec) -> pd.Series:
    """
    Build a target series y_t predicting the future at horizon h.

    Returns a Series indexed like df, with NaN on the last 'horizon' rows.
    """
    if spec.price_col not in df.columns:
        raise ValueError(f"Missing price column '{spec.price_col}'.")

    px = df[spec.price_col].astype(float)

    if spec.target_type == "return":
        y = _future_return(px, spec.horizon)
    elif spec.target_type == "log_return":
        y = _future_log_return(px, spec.horizon)
    elif spec.target_type == "direction":
        r = _future_return(px, spec.horizon)
        y = (r > 0).astype("float")
        y[r.isna()] = np.nan
    elif spec.target_type == "quantile_class":
        r = _future_return(px, spec.horizon)
        # Quantiles computed on available (non-NaN) returns
        # Note: quantiles should be computed on TRAIN only in a pipeline.
        y = pd.qcut(r, q=spec.n_classes, labels=False, duplicates="drop")
    else:
        raise ValueError(f"Unknown target_type: {spec.target_type}")

    y.name = f"y_{spec.target_type}_h{spec.horizon}"
    return y


def align_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align X and y by dropping rows where y is NaN (typically last horizon rows),
    and any rows where X contains NaNs.

    This is the final step before modeling.
    """
    df = X.join(y, how="inner")
    df = df.dropna(axis=0)
    y_aligned = df[y.name]
    X_aligned = df.drop(columns=[y.name])
    return X_aligned, y_aligned
