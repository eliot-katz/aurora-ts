import numpy as np
import pandas as pd
import pytest

from aurora_ts.features.targets import TargetSpec, align_xy, build_target


def test_return_target_h1():
    df = pd.DataFrame({"close": [100.0, 110.0, 121.0]})
    spec = TargetSpec(target_type="return", horizon=1)
    y = build_target(df, spec)

    # r0 = 110/100 - 1 = 0.1 ; r1 = 121/110 - 1 = 0.1 ; last is NaN
    assert y.iloc[0] == pytest.approx(0.1)
    assert y.iloc[1] == pytest.approx(0.1)
    assert np.isnan(y.iloc[2])


def test_log_return_target_h1():
    df = pd.DataFrame({"close": [100.0, 110.0]})
    spec = TargetSpec(target_type="log_return", horizon=1)
    y = build_target(df, spec)
    assert y.iloc[0] == pytest.approx(np.log(110.0) - np.log(100.0))
    assert np.isnan(y.iloc[1])


def test_direction_target():
    df = pd.DataFrame({"close": [100.0, 90.0, 99.0]})
    spec = TargetSpec(target_type="direction", horizon=1)
    y = build_target(df, spec)
    # 90/100 - 1 negative => 0
    assert y.iloc[0] == 0
    # 99/90 - 1 positive => 1
    assert y.iloc[1] == 1
    assert np.isnan(y.iloc[2])


def test_align_xy_drops_nan_rows():
    X = pd.DataFrame({"f1": [1.0, 2.0, np.nan], "f2": [0.0, 1.0, 2.0]})
    y = pd.Series([1.0, np.nan, 0.0], name="y")

    Xa, ya = align_xy(X, y)
    assert Xa.shape[0] == 1
    assert ya.shape[0] == 1
    assert Xa.iloc[0]["f1"] == 1.0
    assert ya.iloc[0] == 1.0
