from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    graph_path = None
    if request.method == "POST":
        ticker = request.form["ticker"]
        
        # Download data
        data = yf.download(ticker, start="2015-01-01", end="2023-01-01")
        features = ["Open", "High", "Low", "Close", "Volume"]
        dataset = data[features]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dataset)
        
        # Create sequences
        sequence_length = 60
        X = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
        X = np.array(X)
        
        # Load model (or train here)
        model = load_model("stock_lstm_model.h5")
        predictions = model.predict(X)
        
        # Inverse scale Close price
        scale_close = scaler.scale_[3]
        min_close = scaler.min_[3]
        predicted_prices = predictions / scale_close - min_close / scale_close
        real_prices = dataset["Close"].values[sequence_length:]
        
        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(real_prices, label="Actual Price")
        plt.plot(predicted_prices, label="Predicted Price")
        plt.title(f"{ticker} Stock Price Prediction")
        plt.legend()
        graph_path = f"static/{ticker}_prediction.png"
        os.makedirs("static", exist_ok=True)
        plt.savefig(graph_path)
        plt.close()
    
    return f'''
        <h1>Stock Prediction</h1>
        <form method="POST">
            Stock Ticker: <input name="ticker" type="text">
            <input type="submit" value="Predict">
        </form>
        {f'<img src="{graph_path}">' if graph_path else ""}
    '''

if __name__ == "__main__":
    app.run(debug=True)
