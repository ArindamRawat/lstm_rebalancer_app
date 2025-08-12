from flask import Flask, request, jsonify
from binance.client import Client
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
import math
import os
app = Flask(__name__)
client = Client()

LOOKBACK = 21  # Matches training
FEATURES = [
    "close", "rsi", "macd", "macd_signal",
    "bb_upper", "bb_lower", "bb_bandwidth",
    "sma_7", "sma_21",
    "volume_pct_change", "return_1d",
    "hl_range", "momentum",
    "adx", "candle_body", "price_std"
]

# Load models and scalers
models = {
    "XVS": load_model("models/XVS_model.keras", compile=False),
    "CAKE": load_model("models/CAKE_model.keras", compile=False),
    "TWT": load_model("models/TWT_model.keras", compile=False),
}
scalers = {
    "XVS": joblib.load("models/XVS_scaler.pkl"),
    "CAKE": joblib.load("models/CAKE_scaler.pkl"),
    "TWT": joblib.load("models/TWT_scaler.pkl"),
}
symbols = {
    "XVS": "XVSUSDT",
    "CAKE": "CAKEUSDT",
    "TWT": "TWTUSDT",
}


def get_data(symbol, days=60):
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

    df["sma_7"] = SMAIndicator(close=df["close"], window=7).sma_indicator()
    df["sma_21"] = SMAIndicator(close=df["close"], window=21).sma_indicator()

    df["volume_pct_change"] = df["volume"].pct_change()
    df["return_1d"] = df["close"].pct_change()
    df["hl_range"] = df["high"] - df["low"]
    df["momentum"] = df["close"] / df["close"].shift(7) - 1

    df["adx"] = ADXIndicator(high=df["high"], low=df["low"], close=df["close"]).adx()
    df["candle_body"] = df["close"] - df["open"]
    df["price_std"] = df["close"].rolling(window=5).std()

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

        # Step 1: Predict
        returns = {token: predict_7day_return(token) for token in portfolio.keys()}
        
        # Step 2: Strategy weight logic (Improved)
        if strategy == "Preservation":
            # Inverse volatility-weighted, favoring lower risk
            weights = {k: 1 / (1 + abs(returns[k])) for k in returns}
        elif strategy == "Growth":
            # Softmax over returns to strongly favor high return assets
            exps = {k: math.exp(returns[k]) for k in returns}
            total = sum(exps.values()) or 1e-6
            weights = {k: v / total for k, v in exps.items()}
        else:  # Balanced
            # Linear scaled returns between 0 and 1
            min_ret = min(returns.values())
            max_ret = max(returns.values())
            range_ret = max_ret - min_ret or 1e-6
            weights = {k: (returns[k] - min_ret) / range_ret for k in returns}


        # Step 3: Normalize
        total = sum(weights.values()) or 1e-6
        rebalanced = {k: round(v / total, 4) for k, v in weights.items()}

        # Step 4: Prediction response
        prediction_list = [{"token": token, "return_7d": round(returns[token], 6)} for token in portfolio.keys()]

        return jsonify({
            "new_allocation": rebalanced,
            "predictions": prediction_list
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
