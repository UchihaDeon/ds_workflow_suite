import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def summary_statistics(df):
    """
    Generate summary statistics for the dataset.
    """
    logging.info("Generating summary statistics...")
    return df.describe()

def correlation_matrix(df, figsize=(8,6)):
    """
    Plot correlation matrix heatmap.
    """
    logging.info("Plotting correlation matrix...")
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def distribution_plot(df, column):
    """
    Plot distribution of a specific column.
    """
    logging.info(f"Plotting distribution for {column}...")
    plt.figure(figsize=(8,6))
    sns.histplot(df[column], kde=True, color="blue")
    plt.title(f"Distribution of {column}")
    plt.show()

def timeseries_plot(df, column):
    """
    Plot time series data for a given column.
    """
    logging.info(f"Plotting time series for {column}...")
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df[column], color="green")
    plt.title(f"Time Series Plot of {column}")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.show()
