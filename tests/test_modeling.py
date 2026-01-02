import pytest
import pandas as pd
from ds_workflow_suite.utils.modeling import ModelWrapper

def test_linear_model_fit_predict():
    df = pd.DataFrame({"x": [1,2,3,4], "sales": [2,4,6,8]})
    X = df[["x"]]
    y = df["sales"]

    model = ModelWrapper(model_type="linear")
    model.build()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
