from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    price_col: str = "close"
    volume_col: str = "volume"
    horizons: tuple[int, ...] = (1, 5, 10, 20)


DEFAULT_FEATURE_SPEC: FeatureSpec = FeatureSpec()


def build_basic_features(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """
    Build a set of causal features from OHLCV.

    All features at time t use information available at or before t.
    """
    if spec is None:
        spec = DEFAULT_FEATURE_SPEC

    if spec.price_col not in df.columns:
        raise ValueError(f"Missing price column '{spec.price_col}'.")
    px = df[spec.price_col].astype(float)

    out = pd.DataFrame(index=df.index)

    # Past returns (known at time t): r_{t-1, h}
    for h in spec.horizons:
        out[f"ret_{h}"] = px.pct_change(h)

    # Rolling volatility of 1-day returns (past-only)
    r1 = px.pct_change(1)
    for w in (5, 10, 20):
        out[f"vol_{w}"] = r1.rolling(w).std()

    # Momentum: price / MA - 1 (MA uses past up to t)
    for w in (10, 20, 50):
        ma = px.rolling(w).mean()
        out[f"mom_{w}"] = px / ma - 1.0

    # Volume features if available
    if spec.volume_col in df.columns:
        vol = df[spec.volume_col].astype(float)
        out["vol_chg_1"] = vol.pct_change(1)
        out["vol_z_20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()

    return out
