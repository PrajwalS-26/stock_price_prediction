"""Sequence-model (LSTM/GRU) utilities with train-only scaling."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM
from tensorflow.keras.models import Sequential


def build_windows(scaled_prices, window):
    """Sliding windows: input = `window` days, target = next day.

    Returns (X, y, indices) where indices[i] is the row index of the
    predicted day (the day after the last input day).
    """
    X, y, idx = [], [], []
    for i in range(window, len(scaled_prices)):
        X.append(scaled_prices[i - window:i])
        y.append(scaled_prices[i])
        idx.append(i)
    return np.array(X), np.array(y), np.array(idx)


def build_sequence_model(kind, window):
    """Build an LSTM or GRU regressor with dropout."""
    kind = kind.lower()
    if kind == "lstm":
        layer = LSTM
    elif kind == "gru":
        layer = GRU
    else:
        raise ValueError(f"Unknown sequence model kind: {kind}")

    model = Sequential([
        layer(50, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        layer(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_sequence_model(kind, window, X_train, y_train, X_val=None, y_val=None,
                         epochs=15, batch_size=32, seed=42, verbose=0):
    model = build_sequence_model(kind, window)
    callbacks = []
    if X_val is not None and len(X_val) > 0:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        )
        validation_data = (X_val, y_val)
    else:
        validation_data = None

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,
    )
    return model


def fit_scaler(prices_train):
    """Fit a MinMaxScaler on TRAIN prices only (prevents leakage)."""
    scaler = MinMaxScaler()
    scaler.fit(np.asarray(prices_train).reshape(-1, 1))
    return scaler


def predict_prices(model, scaler, X):
    scaled = model.predict(X, verbose=0)
    return scaler.inverse_transform(scaled).flatten()


def predict_returns(model, scaler, X):
    """Inverse-transform model output to raw next-day returns."""
    scaled = model.predict(X, verbose=0)
    return scaler.inverse_transform(scaled).flatten()
