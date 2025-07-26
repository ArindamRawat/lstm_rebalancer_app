from flask import Flask, request, jsonify
from binance.client import Client
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
client = Client()

LOOKBACK = 7
FEATURES = ["close", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_bandwidth"]

# Load models and scalers
models = {
    "BNB": load_model("models/BNB_model.h5", compile=False),
    "CAKE": load_model("models/CAKE_model.h5", compile=False),
    "BUSD": load_model("models/BUSD_model.h5", compile=False),
}
scalers = {
    "BNB": joblib.load("models/BNB_scaler.pkl"),
    "CAKE": joblib.load("models/CAKE_scaler.pkl"),
    "BUSD": joblib.load("models/BUSD_scaler.pkl"),
}

symbols = {
    "BNB": "BNBUSDT",
    "CAKE": "CAKEUSDT",
    "BUSD": "BUSDUSDT",
}

def get_data(symbol, days=40):
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, f"{days} days ago UTC")
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def preprocess(df):
    df["rsi"] = RSIIndicator(close=df["close"]).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bb = BollingerBands(close=df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_bandwidth"] = df["bb_upper"] - df["bb_lower"]
    df = df.dropna().reset_index(drop=True)
    return df

def predict_7day_return(token):
    df = get_data(symbols[token])
    df = preprocess(df)
    
    if len(df) < LOOKBACK:
        return 0.0

    df_scaled = df.copy()
    df_scaled[FEATURES] = scalers[token].transform(df[FEATURES])

    X = df_scaled[FEATURES].tail(LOOKBACK).values.reshape(1, LOOKBACK, len(FEATURES))
    pred = models[token].predict(X)[0][0]
    return float(pred)

@app.route("/rebalance", methods=["POST"])
def rebalance():
    try:
        data = request.get_json()
        portfolio = data["current_portfolio"]
        strategy = data.get("strategy", "Balanced")

        returns = {token: predict_7day_return(token) for token in portfolio.keys()}

        if strategy == "Preservation":
            weights = {k: 1 / (1 + abs(v)) for k, v in returns.items()}
        elif strategy == "Growth":
            weights = {k: max(0.0, v) for k, v in returns.items()}
        else:  # Balanced
            weights = {k: v + 1 for k, v in returns.items()}

        total = sum(weights.values()) or 1e-6
        rebalanced = {k: round(v / total, 2) for k, v in weights.items()}
        return jsonify(rebalanced)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
