# Dense Neural Network
from file_handling import file_handling as fh
from data_handling import data_handling as dh
import dnn_model as dnnm
import plotter as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Globals
conf_file_name = '/config.ini'
file_name = 'ETF.csv'
data_col = 'XACTOMXS30.ST_CLOSE'
feature_list = ['rets', 'sma', 'min', 'max', 'mom', 'vol']
lags = 5
window = 20
optimizer = Adam(learning_rate=0.001)
dropout = True
regularize = True
bagging = True
max_features = 0.75


if __name__ == '__main__':
    # File imports
    conf = fh.config(conf_file_name)
    raw = fh.read_csv(conf, file_name)

    # Select which column to use
    df = raw[data_col].to_frame()

    # Add features and lags
    data, cols = dh.add_features(data=df,
                                 data_col=data_col,
                                 feature_list=feature_list,
                                 lags=lags,
                                 window=window)

    # Split into training and test data
    train, test = dh.split_train_test(data)

    # Normalize training data
    train_ = dh.normalize(train)
    test_ = dh.normalize(test)

    # Set random seeds
    dh.set_seeds()

    # Bagging

    model = dnnm.create_model(optimizer=optimizer,
                              reg=l2(0.001),
                              cols=cols,
                              hl=2,
                              hu=128,
                              regularize=regularize,
                              dropout=dropout)

    if bagging:
        model_bag = dnnm.create_model_bag(model=model,
                                          train_=train_,
                                          train=train,
                                          test_=test,
                                          test=test,
                                          dropout=dropout,
                                          regularize=regularize,
                                          cols=cols,
                                          max_features=max_features)

    history = dnnm.model_fit(model=model,
                             train_=train_,
                             train=train,
                             cols=cols)

    # Evaluate in-sample performance
    dnnm.evaluate(model,
                  train_[cols],
                  train,
                  'in-sample')

    # Evaluate out-of-sample performance
    dnnm.evaluate(model,
                  test_[cols],
                  test['d'],
                  'out-of-sample')

    # Prediction
    dnnm.predict(model,
                 test,
                 test_[cols])

    # Plot accuracy
    plt.dnn_accuracy(history)
