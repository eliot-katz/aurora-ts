import numpy as np
import pandas as pd

from aurora_ts.features.build_features import FeatureSpec, build_basic_features


def test_build_basic_features_shapes_and_columns():
    df = pd.DataFrame(
        {
            "close": np.linspace(100, 200, 100),
            "volume": np.linspace(1000, 2000, 100),
        }
    )
    X = build_basic_features(df, FeatureSpec(horizons=(1, 5, 10)))

    assert X.shape[0] == 100
    for col in ["ret_1", "ret_5", "ret_10", "vol_5", "mom_10", "vol_chg_1"]:
        assert col in X.columns


def test_features_are_causal_no_future_shift():
    """
    If we modify the last close price, features at earlier timestamps must not change.
    This is a crude but effective anti-leakage check.
    """
    df = pd.DataFrame(
        {"close": np.arange(1, 51, dtype=float), "volume": np.arange(1, 51, dtype=float)}
    )
    X1 = build_basic_features(df)

    df2 = df.copy()
    df2.loc[df2.index[-1], "close"] = 1e9  # huge change at the end
    X2 = build_basic_features(df2)

    # Features before the last row should be identical (no use of future data)
    pd.testing.assert_frame_equal(X1.iloc[:-1], X2.iloc[:-1])
