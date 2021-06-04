from data_handling import data_handling as dh
import numpy as np
import pandas as pd
from loguru import logger
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import BaggingClassifier


def create_model(optimizer,
                 reg,
                 cols,
                 hl=1,
                 hu=128,
                 dropout=True,
                 rate=0.3,
                 regularize=True):
    # Regularization
    if not regularize:
        reg = None

    model = Sequential()

    # Default layer
    model.add(Dense(hu,
                    input_dim=len(cols),
                    activation='relu',
                    activity_regularizer=reg))

    # Add dropout layer
    if dropout:
        model.add(Dropout(rate, seed=1000))
    for _ in range(hl):
        # Additional layer
        model.add(Dense(hu,
                        activation='relu',
                        activity_regularizer=reg))

        # Add dropout layer
        if dropout:
            model.add(Dropout(rate,
                              seed=1000))
        # Output layer
        model.add(Dense(1,
                        activation='sigmoid'))
        # Loss function
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
    return model


def model_fit(model,
              train_,
              train,
              cols):
    m_fit = model.fit(train_[cols], train['d'], epochs=50,
                      verbose=False,
                      class_weight=dh.cw(train),
                      shuffle=False,
                      validation_split=0.2)
    return m_fit


def create_model_bag(model,
                     train_,
                     train,
                     test_,
                     test,
                     dropout,
                     regularize,
                     cols,
                     max_features):

    # TODO Fix bagging
    base_estimator = KerasClassifier(build_fn=model,
                                     verbose=False,
                                     epochs=20,
                                     # hl=1,
                                     # hu=128,
                                     # dropout=dropout,
                                     # regularize=regularize,
                                     # input_dim=int(len(cols)) * max_features)
                                     )

    model_bag = BaggingClassifier(base_estimator=base_estimator,
                                  n_estimators=15,
                                  max_samples=0.75,
                                  max_features=max_features,
                                  bootstrap=True,
                                  bootstrap_features=True,
                                  n_jobs=1,
                                  random_state=100)

    model_bag.fit(train_, train, cols)

    logger.info('Model bagging score (training): ' +
                str(model_bag.score(train_[cols], train['d'])))

    logger.info('Model bagging score (test): ' +
                str(model_bag.score(test_[cols], test['d'])))

    predict(model_bag,
            data=test,
            data_=test_)


def evaluate(model: Sequential,
             data_: pd.DataFrame,
             data: pd.DataFrame,
             sample_type: str):
    logger.info('Evaluation ' + sample_type + ' performance:')
    model.evaluate(data_, data['d'])


def predict(model, data, data_):
    data['p'] = np.where(model.predict(data_) > 0.5, 1, 0)
    preds = data['p'].value_counts()
    logger.info('Number of prediction > 0.5: ' + str(preds) + '.')
