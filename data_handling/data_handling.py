import pandas as pd
import numpy as np


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
