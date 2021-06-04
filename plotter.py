import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential


def time_series(df: pd.DataFrame, data_col: str):
    ax = df[data_col].plot(figsize=(10, 6), title=data_col)
    ax.set_ylabel('CLOSE')
    plt.show()


def dnn_accuracy(model: Sequential):
    res = pd.DataFrame(model.history.history)
    ax = res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')
    ax.set_xlabel('EPOCH')
    plt.show()

def rnn_prediction(data, data_col):
    t = data_col + ' rnn prediction'
    data[['rets', 'pred']].iloc[50:100].plot(figsize=(10, 6), style=['b', 'r--'], alpha=0.75, title=t)
    plt.axhline(0, c='grey', ls='--')
    plt.show()
