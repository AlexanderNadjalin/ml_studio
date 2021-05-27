# Dense Neural Network
import pandas as pd

from file_handling import file_handling as fh
from data_handling import data_handling as dh
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from loguru import logger

# Globals
conf_file_name = 'C:\\Python projects\\ml_studio\\config.ini'
file_name = 'ETF.csv'
data_col = 'XACTOMXS30.ST_CLOSE'


def create_model(optimizer, hl=1, hu=128, dropout=True, rate=0.3):
    model = Sequential()

    # Default layer
    model.add(Dense(hu, input_dim=len(cols), activation='relu'))

    # Add dropout layer
    if dropout:
        model.add(Dropout(rate, seed=1000))
    for _ in range(hl):
        # Additional layer
        model.add(Dense(hu, activation='relu'))

        # Add dropout layer
        if dropout:
            model.add(Dropout(rate, seed=1000))
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        # Loss function
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


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


if __name__ == '__main__':
    conf = fh.config(conf_file_name)
    raw = fh.read_csv(conf, file_name)

    df = raw[data_col].to_frame()

    ax = df[data_col].plot(figsize=(10, 6), title=data_col)
    ax.set_ylabel('CLOSE')
    plt.show()

    lags = 5
    data, cols = dh.add_lags(data=df, data_col=data_col, lags=lags)

    # Standard optimizer
    optimizer = Adam(learning_rate=0.001)

    # Split the whole data set into training data and test data
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    # Mean and standard deviation for all features
    mu, std = train.mean(), train.std()

    # Normalize training data
    train_ = (train - mu) / std

    set_seeds()
    model = create_model(optimizer=optimizer, hl=2, hu=128, rate=0.3)
    history = model.fit(train_[cols], train['d'], epochs=50,
                        verbose=False,
                        class_weight=cw(train),
                        shuffle=False,
                        validation_split=0.15)

    # Evaluate in-sample performance
    logger.info('Evaluation in-sample performance:')
    model.evaluate(train_[cols], train['d'])

    # Normalize test data set
    test_ = (test - mu) / std

    # Evaluate out-of-sample performance
    logger.info('Evaluation out-of-sample performance:')
    model.evaluate(test_[cols], test['d'])

    test['p'] = np.where(model.predict(test_[cols]) > 0.5, 1, 0)
    logger.info(test['p'].value_counts())

    res = pd.DataFrame(history.history)
    res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')
    plt.show()
