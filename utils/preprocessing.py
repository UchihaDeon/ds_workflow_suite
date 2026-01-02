import pandas as pd
import logging
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_config(config_path="config.yaml"):
    """
    Load preprocessing configuration from a YAML file.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}

def preprocess_timeseries(df, column='sales', freq='D', method='interpolate'):
    """
    Preprocess a time series DataFrame for decomposition/forecasting.

    Parameters:
    - df: pandas DataFrame with datetime index or date column
    - column: target column to clean
    - freq: frequency string ('D', 'W', 'M')
    - method: missing value handling ('interpolate', 'ffill', 'bfill')

    Returns:
    - Cleaned DataFrame
    """

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        logging.info("Converted index to DatetimeIndex.")

    # Set frequency
    df = df.asfreq(freq)
    logging.info(f"Frequency set to {freq}.")

    # Handle missing values
    if method == 'interpolate':
        df[column] = df[column].interpolate()
        logging.info("Missing values interpolated.")
    elif method == 'ffill':
        df[column] = df[column].fillna(method='ffill')
        logging.info("Missing values forward filled.")
    elif method == 'bfill':
        df[column] = df[column].fillna(method='bfill')
        logging.info("Missing values backward filled.")
    else:
        logging.error("Unsupported method.")
        raise ValueError("Choose 'interpolate', 'ffill', or 'bfill'.")

    return df
