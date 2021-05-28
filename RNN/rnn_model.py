from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense


def create_rnn_model(lags,
                     hu=100,
                     layer='SimpleRNN',
                     features=1,
                     algorithm='estimation'):
    model = Sequential()
    if layer == 'SimpleRNN':
        model.add(SimpleRNN(hu,
                            activation='relu',
                            input_shape=(lags, features)))
    else:
        model.add(LSTM(hu,
                       activation='relu',
                       input_shape=(lags, features)))
    if algorithm == 'estimation':
        model.add(Dense(1,
                        activation='linear'))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])
    else:
        model.add(Dense(1,
                        activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model

def predict(model,
            data):
    return model.predict(g, verbose=False)