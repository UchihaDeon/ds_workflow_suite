import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def regression_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    """
    logging.info("Calculating regression metrics...")
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    """
    logging.info("Calculating classification metrics...")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1": f1_score(y_true, y_pred, average="weighted")
    }

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot confusion matrix.
    """
    logging.info("Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def compare_models(results_dict):
    """
    Compare multiple models using a DataFrame.
    results_dict: {model_name: metrics_dict}
    """
    logging.info("Comparing models...")
    return pd.DataFrame(results_dict).T
