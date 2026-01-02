import logging
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from utils.modeling import ModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Data Science Workflow Suite API")

# Example request schema
class PredictRequest(BaseModel):
    features: list

# Example model (linear regression)
df = pd.DataFrame({"x": [1,2,3,4], "sales": [2,4,6,8]})
X = df[["x"]]
y = df["sales"]

model = ModelWrapper(model_type="linear")
model.build()
model.fit(X, y)

@app.post("/predict")
def predict(request: PredictRequest):
    logging.info("Received prediction request...")
    preds = model.predict(pd.DataFrame([request.features], columns=["x"]))
    return {"prediction": preds.tolist()}
