import numpy as np
import pandas as pd

from aurora_ts.models.baselines import fit_predict_mlp, naive_momentum_direction


def test_naive_momentum_direction_shape_and_causality():
    df = pd.DataFrame({"close": [100.0, 90.0, 99.0, 98.0]})
    yhat = naive_momentum_direction(df)

    assert len(yhat) == len(df)
    # first return is NaN -> False -> 0
    assert yhat.iloc[0] in (0, 1)

    # check sign logic
    # 90/100 - 1 < 0 => 0
    assert yhat.iloc[1] == 0
    # 99/90 - 1 > 0 => 1
    assert yhat.iloc[2] == 1


def test_fit_predict_mlp_classification_runs_and_is_deterministic():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(50, 5)))
    y_train = pd.Series((rng.normal(size=50) > 0).astype(int))
    X_test = pd.DataFrame(rng.normal(size=(20, 5)))

    r1 = fit_predict_mlp(X_train, y_train, X_test, task="classification", random_state=123)
    r2 = fit_predict_mlp(X_train, y_train, X_test, task="classification", random_state=123)

    assert r1.y_pred.shape == (20,)
    assert np.array_equal(r1.y_pred, r2.y_pred)


def test_fit_predict_mlp_regression_runs():
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame(rng.normal(size=(60, 4)))
    y_train = pd.Series(rng.normal(size=60))
    X_test = pd.DataFrame(rng.normal(size=(10, 4)))

    r = fit_predict_mlp(X_train, y_train, X_test, task="regression", random_state=7)
    assert r.y_pred.shape == (10,)
