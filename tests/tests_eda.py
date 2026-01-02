import pytest
import pandas as pd
from utils.eda import summary_statistics

def test_summary_statistics():
    df = pd.DataFrame({"sales": [100, 200, 300]})
    stats = summary_statistics(df)
    assert "sales" in stats.columns
