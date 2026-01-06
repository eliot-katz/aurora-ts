"""
Data loading utilities (raw market data).

Goals:
- Reproducible, deterministic loading
- Strict validation (columns, datetime index, duplicates)
- Avoid silent leakage-prone behaviors
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pandas as pd

DuplicatePolicy = Literal["raise", "drop_keep_last", "drop_keep_first"]


DEFAULT_REQUIRED_OHLCV: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class OHLCVLoadSpec:
    """Specification to load an OHLCV CSV in a consistent format."""

    date_col: str = "date"
    symbol_col: str = "symbol"
    required_cols: tuple[str, ...] = DEFAULT_REQUIRED_OHLCV

    # If your CSV uses different column names, provide a rename map:
    # e.g. {"Open": "open", "Adj Close": "close"}
    rename_map: Mapping[str, str] | None = None

    duplicate_policy: DuplicatePolicy = "raise"
    allow_missing: bool = False


DEFAULT_OHLCV_SPEC: Final[OHLCVLoadSpec] = OHLCVLoadSpec()


def _ensure_required_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _coerce_datetime_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'")

    dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    if dt.isna().any():
        bad = df.loc[dt.isna(), date_col].head(5).tolist()
        raise ValueError(f"Failed to parse some dates in '{date_col}'. Examples: {bad}")

    # Convert to timezone-naive timestamps (UTC naive)
    df = df.drop(columns=[date_col]).copy()
    df.index = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    df.index.name = "date"
    return df


def _handle_duplicates(df: pd.DataFrame, policy: DuplicatePolicy) -> pd.DataFrame:
    if df.index.is_unique:
        return df

    if policy == "raise":
        dup_count = int(df.index.duplicated().sum())
        raise ValueError(f"Duplicate timestamps found in index (count={dup_count}).")
    if policy == "drop_keep_last":
        return df[~df.index.duplicated(keep="last")]
    if policy == "drop_keep_first":
        return df[~df.index.duplicated(keep="first")]

    raise ValueError(f"Unknown duplicate_policy: {policy}")


def load_ohlcv_csv(
    path: str | Path,
    *,
    spec: OHLCVLoadSpec | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV time series from a CSV file.

    Expected output:
    - pandas.DataFrame indexed by timezone-naive datetime (UTC naive)
    - strictly increasing index
    - columns include: open, high, low, close, volume (float)

    Parameters
    ----------
    path:
        CSV file path.
    spec:
        Loading specification.
    symbol:
        If provided and spec.symbol_col exists in CSV, filter rows to that symbol.

    Returns
    -------
    pd.DataFrame
        Validated OHLCV time series.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if spec is None:
        spec = DEFAULT_OHLCV_SPEC

    df = pd.read_csv(p)

    # Standardize column names to lower-case for consistency
    df.columns = [str(c).strip() for c in df.columns]

    if spec.rename_map is not None:
        df = df.rename(columns=spec.rename_map)

    # Normalize again after rename (in case rename_map introduced spacing)
    df.columns = [str(c).strip() for c in df.columns]

    # Optional symbol filtering
    if symbol is not None:
        if spec.symbol_col not in df.columns:
            raise ValueError(f"symbol was provided but '{spec.symbol_col}' column is missing.")
        df = df[df[spec.symbol_col] == symbol].copy()
        if df.empty:
            raise ValueError(f"No rows found for symbol='{symbol}'")

    # Build datetime index first
    df = _coerce_datetime_index(df, spec.date_col)

    # If symbol column exists, keep it only if user didn't filter (useful for multi-asset)
    # But for single-asset modeling, you typically drop it.
    if symbol is not None and spec.symbol_col in df.columns:
        df = df.drop(columns=[spec.symbol_col])

    # Ensure required OHLCV columns exist
    _ensure_required_columns(df, spec.required_cols)

    # Keep only required columns + (optionally other columns if you want later)
    # Here we keep required + any extra columns, but could be strict. We'll keep extras for flexibility.
    # Convert required cols to numeric
    df = df.sort_index()
    df = _handle_duplicates(df, spec.duplicate_policy)
    df = df.sort_index()

    for c in spec.required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if not spec.allow_missing:
        if df[list(spec.required_cols)].isna().any().any():
            bad_cols = (
                df[list(spec.required_cols)]
                .columns[df[list(spec.required_cols)].isna().any()]
                .tolist()
            )
            raise ValueError(f"Missing/NaN values found in required columns: {bad_cols}")

    # Final safety: strict monotonic increasing
    if not df.index.is_monotonic_increasing:
        raise ValueError("Datetime index is not monotonically increasing after sorting.")

    return df
