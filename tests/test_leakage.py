import numpy as np
import pandas as pd
import pytest

from aurora_ts.data.transforms import apply_scaler, fit_standard_scaler, rolling_zscore


def test_standard_scaler_raises_on_zero_variance_train():
    df_train = pd.DataFrame({"x": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="Zero variance"):
        fit_standard_scaler(df_train, columns=["x"])


def test_standard_scaler_no_leakage_statistics():
    """
    The scaler must be fit ONLY on training data.
    Test values must not influence mean/std.
    """
    df_train = pd.DataFrame({"x": [0.0, 1.0, -1.0]})
    df_test = pd.DataFrame({"x": [100.0, 101.0]})

    scaler = fit_standard_scaler(df_train, columns=["x"])
    transformed_test = apply_scaler(df_test, scaler)

    # Expected based ONLY on train statistics
    mean = df_train["x"].mean()
    std_pop = df_train["x"].std(ddof=0)  # StandardScaler uses population std
    expected = (df_test["x"] - mean) / std_pop

    np.testing.assert_allclose(transformed_test["x"].values, expected.values)


def test_scaler_missing_columns_raises():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler = fit_standard_scaler(df, columns=["x"])

    with pytest.raises(ValueError, match="Missing columns required for scaling"):
        apply_scaler(pd.DataFrame({"y": [1.0]}), scaler)


def test_rolling_zscore_is_causal():
    """
    Ensure rolling z-score only uses past values (shifted by 1).
    """
    df = pd.DataFrame({"x": np.arange(10, dtype=float)})

    out = rolling_zscore(df, columns=["x"], window=3, min_periods=3)

    # At t=3, rolling window uses x[0,1,2] due to shift(1)
    t = 3
    past = df.loc[0:2, "x"]
    expected = (df.loc[t, "x"] - past.mean()) / past.std()

    assert out.loc[t, "x"] == pytest.approx(expected)

    # First indices should be NaN (insufficient past data)
    assert out["x"].iloc[:3].isna().all()


def test_rolling_zscore_handles_zero_std_without_future_leakage():
    """
    With shift(1), at t=3 the past window is [x1, x2] = [0, 0] so std=0.
    We expect NaN or inf, but crucially we must not use x3 itself in the stats.
    """
    df = pd.DataFrame({"x": [0.0, 0.0, 0.0, 100.0]})
    out = rolling_zscore(df, columns=["x"], window=2, min_periods=2)

    v = out.loc[3, "x"]
    assert np.isnan(v) or np.isinf(v)
