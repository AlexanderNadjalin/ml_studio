import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential


def time_series(df: pd.DataFrame, data_col: str):
    ax = df[data_col].plot(figsize=(10, 6), title=data_col)
    ax.set_ylabel('CLOSE')
    plt.show()


def accuracy(model: Sequential):
    res = pd.DataFrame(model.history.history)
    ax = res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')
    ax.set_xlabel('EPOCH')
    plt.show()
