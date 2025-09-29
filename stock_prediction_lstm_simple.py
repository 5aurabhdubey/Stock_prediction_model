import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------
# Download stock data
# -------------------------
ticker = "AAPL"  # you can change to "RELIANCE.NS" for Reliance, etc.
df = yf.download(ticker, period="5y")
data = df[['Close']].values  # we only use Close prices

# -------------------------
# Scale data between 0 and 1
# -------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# -------------------------
# Create sequences for LSTM
# -------------------------
sequence_length = 60  # use last 60 days to predict next
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # reshape for LSTM

# -------------------------
# Train-Test Split
# -------------------------
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------
# Build LSTM model
# -------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))  # output layer
model.compile(optimizer="adam", loss="mean_squared_error")

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# -------------------------
# Make predictions
# -------------------------
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # back to price scale

# Real prices
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(10,6))
plt.plot(real_prices, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
plt.show()
