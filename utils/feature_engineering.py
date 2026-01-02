import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def scale_features(df, columns, method="standard"):
    """
    Scale numerical features using StandardScaler or MinMaxScaler.
    """
    logging.info(f"Scaling columns {columns} using {method} scaler...")
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Choose 'standard' or 'minmax'")

    df[columns] = scaler.fit_transform(df[columns])
    return df


def encode_categorical(df, column, method="onehot"):
    """
    Encode categorical features using OneHotEncoder or LabelEncoder.
    """
    logging.info(f"Encoding column {column} using {method}...")
    if method == "onehot":
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        df = pd.concat([df.drop(column, axis=1), encoded_df], axis=1)
    elif method == "label":
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
    else:
        raise ValueError("Choose 'onehot' or 'label'")
    return df


def create_lag_features(df, column, lags=[1, 2, 3]):
    """
    Create lag features for time series forecasting.
    """
    logging.info(f"Creating lag features for {column} with lags {lags}...")
    for lag in lags:
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df


def rolling_features(df, column, windows=[3, 7]):
    """
    Create rolling mean features for time series forecasting.
    """
    logging.info(f"Creating rolling features for {column} with windows {windows}...")
    for window in windows:
        df[f"{column}_roll{window}"] = df[column].rolling(window=window).mean()
    return df
