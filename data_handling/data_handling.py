import pandas as pd
import numpy as np
import random
import tensorflow as tf


def add_features(data: pd.DataFrame,
                 data_col: str,
                 feature_list: list,
                 lags: int,
                 window: int):
    cols = []
    df = data.copy()
    df.dropna(inplace=True)
    if 'rets' in feature_list:
        df['rets'] = np.log(df[data_col] / df[data_col].shift())
        cols.append('rets')
    if 'sma' in feature_list:
        df['sma'] = df[data_col].rolling(window).mean()
        cols.append('sma')
    if 'min' in feature_list:
        df['min'] = df[data_col].rolling(window).min()
        cols.append('min')
    if 'max' in feature_list:
        df['max'] = df[data_col].rolling(window).max()
        cols.append('max')
    if 'mom' in feature_list:
        df['mom'] = df['rets'].rolling(window).mean()
        cols.append('mom')
    if 'vol' in feature_list:
        df['vol'] = df['rets'].rolling(window).std()
        cols.append('vol')
    df.dropna(inplace=True)

    df['d'] = np.where(df['rets'] > 0, 1, 0)
    features = [data_col]
    for feat in feature_list:
        features.append(feat)
    if lags > 0:
        for f in features:
            for lag in range(1, lags + 1):
                col = f'{f}_lag_{lag}'
                df[col] = df[f].shift(lag)
                cols.append(col)
        df.dropna(inplace=True)

    return df, cols


def split_train_test(data):
    # Split the whole data set into training data and test data
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    return train, test


def cw(df):
    c0, c1 = np.bincount(df['d'])
    w0 = (1 / c0) * len(df) / 2
    w1 = (1 / c1) * len(df) / 2
    return {0: w0, 1: w1}


def set_seeds(seed=1000):
    # Python numpy and tensorflow random seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normalize(data):
    # Mean and standard deviation for all features
    mu, std = data.mean(), data.std()

    # Normalize training data
    norm = (data - mu) / std

    return norm
