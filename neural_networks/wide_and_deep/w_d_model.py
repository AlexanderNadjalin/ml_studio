from sklearn.preprocessing import MinMaxScaler
import numpy as np


def set_scalers(min_=0.01,
                max_=0.99):
    target_scaler = MinMaxScaler(feature_range=(min_, max_))
    feature_scaler = MinMaxScaler(feature_range=(min_, max_))
    return target_scaler, feature_scaler


def min_max_scaling(x_train,
                    x_valid,
                    x_test,
                    y_train,
                    y_valid,
                    y_test,
                    target_scaler,
                    feature_scaler):
    X_train_scaled = feature_scaler.fit_transform(np.array(x_train))
    X_valid_scaled = feature_scaler.fit_transform(np.array(x_valid))
    X_test_scaled = feature_scaler.fit_transform(np.array(x_test))

    y_train_scaled = target_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    y_valid_scaled = target_scaler.fit_transform(np.array(y_valid).reshape(-1, 1))
    y_test_scaled = target_scaler.fit_transform(np.array(y_test).reshape(-1, 1))

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled
