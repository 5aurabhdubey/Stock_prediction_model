import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Download stock data (AAPL by default, you can change ticker)
ticker = "AAPL"
df = yf.download(ticker, period="5y")

# Use only the 'Close' column
df = df[['Close']]
df['Prediction'] = df[['Close']].shift(-1)  # next-day price

# Drop last row (no label available)
df = df[:-1]

# Features (X) and Labels (y)
X = np.array(df[['Close']])
y = np.array(df['Prediction'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(10,6))
plt.plot(y_test, label="Actual Price")
plt.plot(preds, label="Predicted Price")
plt.legend()
plt.show()
