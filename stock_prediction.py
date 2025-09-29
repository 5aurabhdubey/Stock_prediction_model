# ---------------------------
# STEP 1: Load Dataset
# ---------------------------
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Download stock data
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2023-01-01")

# Use multiple features (Open, High, Low, Close, Volume)
features = ["Open", "High", "Low", "Close", "Volume"]
dataset = data[features]

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

# ---------------------------
# STEP 2: Create Sequences
# ---------------------------
def create_sequences(data, sequence_length=60):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])          # past 60 days (all features)
        y.append(data[i+sequence_length, 3])        # predict "Close" price only (index 3 in features)
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Shape of X_train:", X_train.shape)  # (samples, 60, 5 features)
print("Shape of y_train:", y_train.shape)

# ---------------------------
# STEP 3: Build the LSTM Model
# ---------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)  # output = predicted Close price
])

model.compile(optimizer="adam", loss="mean_squared_error")

# ---------------------------
# STEP 4: Train the Model
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)
