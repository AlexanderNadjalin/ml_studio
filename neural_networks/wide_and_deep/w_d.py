# Neural network Wide and Deep
import w_d_model as wdn
import plotter
import os
from file_handling import file_handling as fh
from data_handling import data_handling as dh
from loguru import logger
from sklearn.model_selection import train_test_split
from tensorflow import keras as keras


# Globals
conf_file_name = 'C:\\Python_projects\\ml_studio\\config.ini'
file_name = 'ETF.csv'
data_col = 'XACTOMXS30.ST_CLOSE'
feature_list = ['rets', 'sma', 'min', 'max', 'mom', 'vol']
lags = 0
window = 20


if __name__ == '__main__':
    # File imports
    conf = fh.Settings(cfg_path=conf_file_name)
    log_dir = conf.get_setting('logs',
                               'logs_directory')
    run_log_dir = fh.get_run_logdir(log_dir,
                                    del_old_logs=True)

    raw = fh.read_csv(conf,
                      file_name)

    # Select which column to use
    df = raw[data_col].to_frame()

    # Add features and lags
    data, cols = dh.add_features(data=df,
                                 data_col=data_col,
                                 feature_list=feature_list,
                                 lags=lags,
                                 window=window)

    # Split into training and test data
    X_train_full, X_test, y_train_full, y_test = train_test_split(data.values,
                                                                  data[data_col],
                                                                  shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,
                                                          y_train_full,
                                                          shuffle=False)

    # Scale data sets
    target_scaler, feature_scaler = wdn.set_scalers()

    X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled = \
        wdn.min_max_scaling(X_train,
                            X_valid,
                            X_test,
                            y_train,
                            y_valid,
                            y_test,
                            target_scaler,
                            feature_scaler)

    # Define wide-and-deep network
    input_ = keras.layers.Input(shape=X_train_scaled.shape[1:])
    print(X_train_scaled.shape[1:])
    hidden_1 = keras.layers.Dense(30,
                                  activation='linear'
                                  )(input_)
    hidden_2 = keras.layers.Dense(30,
                                  activation='sigmoid'
                                  )(hidden_1)
    hidden_3 = keras.layers.Dense(30,
                                  activation='sigmoid'
                                  )(hidden_2)
    concat = keras.layers.Concatenate()([input_, hidden_1, hidden_2, hidden_3])
    output = keras.layers.Dense(1)(concat)
    model = keras.Model(inputs=[input_],
                        outputs=[output])

    model.compile(loss='mse',
                  optimizer='Adam')

    tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir,
                                                 profile_batch=0)

    # Train model
    history = model.fit(X_train_scaled,
                        y_train_scaled,
                        epochs=100,
                        validation_data=(X_valid_scaled,
                                         y_valid_scaled),
                        callbacks=[tensorboard_cb])
    mse_test = model.evaluate(X_test_scaled, y_test_scaled)

    # Predictions
    X_new = X_test_scaled
    y_pred = model.predict(X_new)
    y_pred_rescaled = target_scaler.inverse_transform(y_pred)

    # Plot prediction vs. test
    plotter.w_and_d_pred(y_pred_rescaled,
                         y_test)

    # Start Tensorboard server
    string = 'tensorboard --logdir=' + str(run_log_dir) + ' --port=6006'
    os.system(string)

    logger.info('Complete.')
