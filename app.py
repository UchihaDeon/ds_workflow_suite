from utils.eda import summary_statistics, correlation_matrix, distribution_plot, timeseries_plot
import pandas as pd

# Load data
df = pd.read_csv("data/sales_data.csv", index_col="date", parse_dates=True)

# Run EDA
print(summary_statistics(df))
correlation_matrix(df)
distribution_plot(df, "sales")
timeseries_plot(df, "sales")

from utils.feature_engineering import scale_features, encode_categorical, create_lag_features, rolling_features
import pandas as pd

df = pd.read_csv("data/sales_data.csv", index_col="date", parse_dates=True)

# Apply feature engineering
df = scale_features(df, ["sales"], method="standard")
df = encode_categorical(df, "region", method="onehot")
df = create_lag_features(df, "sales", lags=[1,2,3])
df = rolling_features(df, "sales", windows=[3,7])

print(df.head())

from utils.modeling import ModelWrapper
import pandas as pd

# Example: Linear Regression
df = pd.read_csv("data/sales_data.csv", index_col="date", parse_dates=True)
X = df.drop("sales", axis=1)
y = df["sales"]

model = ModelWrapper(model_type="linear")
model.build()
model.fit(X, y)
preds = model.predict(X)
print(preds[:5])

from utils.evaluation import regression_metrics, classification_metrics, plot_confusion_matrix, compare_models
import pandas as pd

# Example: regression evaluation
y_true = [100, 120, 140, 160]
y_pred = [110, 115, 150, 155]

metrics = regression_metrics(y_true, y_pred)
print(metrics)

# Example: classification evaluation
y_true_cls = [0,1,1,0,1]
y_pred_cls = [0,1,0,0,1]

cls_metrics = classification_metrics(y_true_cls, y_pred_cls)
print(cls_metrics)

# Compare models
results = {
    "Linear Regression": metrics,
    "Random Forest": {"MAE": 5, "MSE": 30, "RMSE": 5.47, "R2": 0.92}
}
print(compare_models(results))
