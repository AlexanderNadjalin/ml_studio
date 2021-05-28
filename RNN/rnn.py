# Recurrent Neural Network
from file_handling import file_handling as fh
from data_handling import data_handling as dh
import rnn_model as rnnm
import plotter as plt
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

# Globals
conf_file_name = 'C:\\Python projects\\ml_studio\\config.ini'
file_name = 'ETF.csv'
data_col = 'XACTOMXS30.ST_CLOSE'
features_list = ['rets', 'mom', 'vol']
window = 20
lags = 0
ts_lags = 5
batch_size = 5


if __name__ == '__main__':
    # File imports
    conf = fh.config(conf_file_name)
    raw = fh.read_csv(conf, file_name)

    # Select which column to use
    df = raw[data_col].to_frame()

    # Add features
    data, cols = dh.add_features(data=df,
                                 data_col=data_col,
                                 feature_list=features_list,
                                 lags=lags,
                                 window=window)

    # Split into training and test data
    train, test = dh.split_train_test(data)

    # Normalize training data
    train_ = dh.normalize(train)
    test_ = dh.normalize(test)

    # Fit model to training data
    g = TimeseriesGenerator(train.value,
                            train['rets'].values,
                            length=ts_lags,
                            batch_size=batch_size)

    # Set random seeds
    dh.set_seeds()

    model = rnnm.create_rnn_model(lags=ts_lags)
    model.fit_generator(g,
                        epochs=500,
                        steps_per_epoch=10,
                        verbose=False)
    y = rnnm.predict(model, g)
    data['pred'] = np.nan
    data['pred'].iloc[ts_lags:] = y.flatten()
    data.dropna(inplace=True)

    plt.rnn_prediction(data, data_col)
