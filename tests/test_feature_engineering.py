import pytest
import pandas as pd
from ds_workflow_suite.utils.feature_engineering import scale_features, create_lag_features

def test_scale_features_standard():
    df = pd.DataFrame({"sales": [10, 20, 30]})
    result = scale_features(df, ["sales"], method="standard")
    assert abs(result["sales"].mean()) < 1e-6  # mean ~ 0 after standard scaling

def test_create_lag_features():
    df = pd.DataFrame({"sales": [10, 20, 30]}, index=pd.date_range("2021-01-01", periods=3))
    result = create_lag_features(df, "sales", lags=[1])
    assert "sales_lag1" in result.columns
