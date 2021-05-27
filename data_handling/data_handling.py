import pandas as pd
import numpy as np
import random
import tensorflow as tf


def add_lags(data: pd.DataFrame,
             data_col: str,
             lags: int,
             window=20):
    cols = []
    df = data.copy()
    df.dropna(inplace=True)
    df['rets'] = np.log(df[data_col] / df[data_col].shift())
    df['sma'] = df[data_col].rolling(window).mean()
    df['min'] = df[data_col].rolling(window).min()
    df['max'] = df[data_col].rolling(window).max()
    df['mom'] = df['rets'].rolling(window).mean()
    df['vol'] = df['rets'].rolling(window).std()
    df.dropna(inplace=True)

    df['d'] = np.where(df['rets'] > 0, 1, 0)
    features = [data_col, 'rets', 'd', 'sma', 'min', 'max', 'mom', 'vol']
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
