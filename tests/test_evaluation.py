import numpy as np
import pytest

from aurora_ts.models.evaluation import classification_metrics, regression_metrics


def test_classification_metrics_basic():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.3, 0.8])

    m = classification_metrics(y_true, y_pred, y_proba)

    assert 0.0 <= m.accuracy <= 1.0
    assert 0.0 <= m.balanced_accuracy <= 1.0
    assert 0.0 <= m.f1 <= 1.0
    assert m.auc is not None
    assert 0.0 <= m.auc <= 1.0


def test_classification_metrics_length_mismatch_raises():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError, match="same length"):
        classification_metrics(y_true, y_pred)


def test_regression_metrics_basic():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.1, 1.9, 3.2])

    m = regression_metrics(y_true, y_pred)
    assert m.mae >= 0.0
    assert m.rmse >= 0.0
    assert -1.0 <= m.pearson_corr <= 1.0
    assert -1.0 <= m.information_coefficient <= 1.0


def test_regression_metrics_constant_series_corr_zero():
    y_true = np.array([1.0, 1.0, 1.0, 1.0])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0])

    m = regression_metrics(y_true, y_pred)
    assert m.pearson_corr == 0.0
    assert m.information_coefficient == 0.0
