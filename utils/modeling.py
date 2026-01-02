import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ModelWrapper:
    def __init__(self, model_type="linear", **kwargs):
        """
        Initialize model based on type.
        Supported: 'linear', 'random_forest', 'arima', 'lstm'
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None

    def build(self):
        if self.model_type == "linear":
            self.model = LinearRegression(**self.kwargs)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(**self.kwargs)
        elif self.model_type == "arima":
            order = self.kwargs.get("order", (1,1,1))
            self.model = ARIMA(self.kwargs["series"], order=order)
        elif self.model_type == "lstm":
            timesteps = self.kwargs.get("timesteps", 10)
            features = self.kwargs.get("features", 1)
            self.model = Sequential([
                LSTM(50, activation="relu", input_shape=(timesteps, features)),
                Dense(1)
            ])
            self.model.compile(optimizer="adam", loss="mse")
        else:
            raise ValueError("Unsupported model type")
        logging.info(f"{self.model_type} model built successfully.")

    def fit(self, X, y=None):
        if self.model_type in ["linear", "random_forest"]:
            self.model.fit(X, y)
        elif self.model_type == "arima":
            self.model = self.model.fit()
        elif self.model_type == "lstm":
            self.model.fit(X, y, epochs=10, verbose=0)
        logging.info(f"{self.model_type} model trained successfully.")

    def predict(self, X=None):
        if self.model_type in ["linear", "random_forest"]:
            return self.model.predict(X)
        elif self.model_type == "arima":
            return self.model.forecast(steps=self.kwargs.get("steps", 5))
        elif self.model_type == "lstm":
            return self.model.predict(X, verbose=0)
