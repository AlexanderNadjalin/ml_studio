
from file_handling import file_handling as fh
from data_handling import data_handling as dh
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as keras


# Globals
conf_file_name = 'C:\\Python projects\\ml_studio\\config.ini'
file_name = 'ETF.csv'
data_col = 'XACTOMXS30.ST_CLOSE'
feature_list = ['rets', 'sma', 'min', 'max', 'mom', 'vol']
lags = 0
window = 20


if __name__ == '__main__':
    # File imports
    conf = fh.Settings(cfg_path=conf_file_name)

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
    X_train_full, X_test, y_train_full, y_test = train_test_split(data.values, data[cols], shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_test = scaler.fit_transform(X_test)

    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden_1 = keras.layers.Dense(30,
                                  activation='relu'
                                  )(input_)
    hidden_2 = keras.layers.Dense(30,
                                  activation='relu'
                                  )(hidden_1)
    concat = keras.layers.Concatenate()([input_, hidden_2])
    output = keras.layers.Dense(1)(concat)
    model = keras.Model(inputs=[input_], outputs=[output])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.SGD(learning_rate=1e-3))

    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    X_new = X_test[:3]
    y_pred = model.predict(X_new)
    logger.info(str(y_pred))

    logger.info('Complete.')
