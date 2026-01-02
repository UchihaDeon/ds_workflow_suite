import pytest
import pandas as pd
from ds_workflow_suite.utils.preprocessing import preprocess_timeseries

def test_preprocess_timeseries_interpolate():
    df = pd.DataFrame({"sales": [100, None, 120]}, index=pd.date_range("2021-01-01", periods=3))
    result = preprocess_timeseries(df, column="sales", freq="D", method="interpolate")
    assert not result["sales"].isnull().any()
