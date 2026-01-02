import pytest
import numpy as np
from ds_workflow_suite.utils.evaluation import regression_metrics, classification_metrics, compare_models

def test_regression_metrics():
    y_true = [1, 2, 3, 4]
    y_pred = [1.1, 1.9, 3.2, 3.8]
    metrics = regression_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert metrics["MAE"] >= 0

def test_classification_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    metrics = classification_metrics(y_true, y_pred)
    assert "Accuracy" in metrics
    assert 0 <= metrics["Accuracy"] <= 1

def test_compare_models():
    results = {
        "Linear Regression": {"MAE": 0.5, "MSE": 0.3, "RMSE": 0.55, "R2": 0.9},
        "Random Forest": {"MAE": 0.4, "MSE": 0.25, "RMSE": 0.5, "R2": 0.92}
    }
    df = compare_models(results)
    assert "Linear Regression" in df.index
    assert "Random Forest" in df.index
