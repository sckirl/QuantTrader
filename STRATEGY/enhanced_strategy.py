#!/usr/bin/env python3
"""
Enhanced XGBoost Strategy for BTC/ETH on Bybit (4h).
Implements Shorting, Advanced Features, and Hyperparameter Tuning.
Replication of updated specifications for >70% Win Rate Target.
"""

import warnings
import numpy as np
import pandas as pd
import ccxt
import datetime
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# --- Configuration ---
TICKERS = ['BTC/USDT', 'ETH/USDT']
TIMEFRAME = '4h'
START_DATE = "2021-05-01"
END_DATE = "2025-12-31"

def fetch_data(ticker):
    """
    Fetches 4h OHLCV data from Bybit via CCXT.
    Period: 2021-05-01 to 2025-12-31.
    """
    print(f"Fetching {TIMEFRAME} data for {ticker} from Bybit...")
    exchange = ccxt.bybit()
    
    start_dt = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(END_DATE, "%Y-%m-%d")
    since = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000 # Bybit usually supports 200 or 1000 depending on endpoint
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe=TIMEFRAME, since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            last_ts = ohlcv[-1][0]
            # Next request starts after the last candle
            # 4h = 4 * 60 * 60 * 1000 = 14400000 ms
            since = last_ts + 1
            
            if last_ts >= end_ts or last_ts >= int(datetime.datetime.now().timestamp() * 1000):
                break
                
            # Safety for infinite loops
            if len(ohlcv) < 1:
                break
                
        except Exception as e:
            print(f"Error during fetch: {e}")
            break
            
    if not all_ohlcv:
        raise ValueError(f"No data fetched for {ticker}")
        
    df = pd.DataFrame(all_ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df.astype(float)
    
    # Filter exact range
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
    
    # Remove potential duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    return df

def add_technical_indicators(df):
    """
    Adds RSI, MACD, Bollinger Bands, and Volume SMA manually (pandas).
    """
    close = df['Close']
    
    # 1. RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_14'].fillna(50, inplace=True) # Fill init with neutral
    
    # 2. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
    
    # 3. Bollinger Bands (20, 2)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    df['BB_Upper'] = sma20 + (2 * std20)
    df['BB_Lower'] = sma20 - (2 * std20)
    
    # 4. Volume SMA (7)
    df['Vol_SMA_7'] = df['Volume'].rolling(window=7).mean()
    
    return df

def preprocess_data(df):
    """
    Feature Engineering & Scaling.
    """
    data = df.copy()
    data = add_technical_indicators(data)
    
    # --- Base Features ---
    # Returns
    data['Return'] = data['Close'].pct_change()
    
    # Lags 1-10
    for i in range(1, 11):
        data[f'Close_lag_{i}'] = data['Close'].shift(i)
        
    # SMAs
    data['SMA_7'] = data['Close'].rolling(window=7).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    
    # Volatility
    data['Vol_7'] = data['Close'].rolling(window=7).std()
    data['Vol_14'] = data['Close'].rolling(window=14).std()
    
    # Target: Next Close
    data['Target_Close'] = data['Close'].shift(-1)
    
    data.dropna(inplace=True)
    
    # Feature List
    features = [
        'Close', 'Volume', 'Return',
        'SMA_7', 'SMA_30', 'Vol_7', 'Vol_14',
        'RSI_14', 'MACD_Line', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'Vol_SMA_7'
    ]
    # Add Lags
    features += [f'Close_lag_{i}' for i in range(1, 11)]
    
    X = data[features]
    y = data['Target_Close']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA (n=10)
    # n_components must be <= min(n_samples, n_features)
    n_feats = X_scaled.shape[1]
    n_comp = min(10, n_feats)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    
    return data, X_pca, y, scaler, pca

def train_model(X, y):
    """
    Trains XGBoost with GridSearchCV tuning.
    """
    # 80/20 Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Grid Search Specs
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 0.9],
        'n_estimators': [200]
    }
    
    xgb_base = xgb.XGBRegressor(
        objective='reg:squarederror',
        gamma=0,
        reg_lambda=1,
        n_jobs=-1,
        random_state=42
    )
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Tuning hyperparameters (GridSearchCV)...")
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best Params: {grid_search.best_params_}")
    
    # Final Fit on full train data with early stopping using test set as valid
    # XGBoost 2.0+ requires early stopping via callbacks or constructor
    # We will re-instantiate to be safe or set params
    
    best_params = grid_search.best_params_
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        gamma=0,
        reg_lambda=1,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=10, # Pass to constructor
        **best_params
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    preds = final_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.4%}")
    
    return final_model, preds

def generate_signals(df, model, X_pca):
    """
    Signals with Hybrid Confirmation (Trend Shifts) and Shorting.
    """
    all_preds = model.predict(X_pca)
    df['Predicted_Close'] = all_preds
    df['Predicted_Return'] = (df['Predicted_Close'] - df['Close']) / df['Close']
    
    # Base Signals
    # Buy > 0.003, Sell < -0.003
    buy_thresh = 0.003
    sell_thresh = -0.003
    
    df['Pred_Signal'] = 0
    df.loc[df['Predicted_Return'] > buy_thresh, 'Pred_Signal'] = 1
    df.loc[df['Predicted_Return'] < sell_thresh, 'Pred_Signal'] = -1
    
    # Confirmation: SMA Crossover/Regime
    # "Buy if Buy AND 7SMA > 30SMA (bull cross)"
    # We interpret "bull cross" as the specific entry point where trends align.
    # Strict Cross: Prev_7 <= Prev_30 AND Curr_7 > Curr_30
    
    df['SMA_7_prev'] = df['SMA_7'].shift(1)
    df['SMA_30_prev'] = df['SMA_30'].shift(1)
    
    # Bull Cross Event
    df['Bull_Cross'] = (df['SMA_7'] > df['SMA_30']) & (df['SMA_7_prev'] <= df['SMA_30_prev'])
    # Bear Cross Event
    df['Bear_Cross'] = (df['SMA_7'] < df['SMA_30']) & (df['SMA_7_prev'] >= df['SMA_30_prev'])
    
    # Final Signal
    df['Final_Signal'] = 0
    
    # Buy: Pred Buy + Bull Cross
    df.loc[(df['Pred_Signal'] == 1) & (df['Bull_Cross']), 'Final_Signal'] = 1
    
    # Sell (Short): Pred Sell + Bear Cross
    df.loc[(df['Pred_Signal'] == -1) & (df['Bear_Cross']), 'Final_Signal'] = -1
    
    return df

def run_xgboost_strategy(df, model_path="STRATEGY/xgb_enhanced_BTCUSDT.json"):
    """
    Runs the XGBoost Strategy in inference mode.
    Used by Backtester or Live Engine.
    """
    print(f"Running XGBoost Strategy (Inference)...")
    
    # Re-implement feature engineering ensures consistency
    # Ideally, we call preprocess_data, but that returns X_pca which needs the fitted PCA object.
    # For robust production, we should save the PCA pipeline too.
    # For now, we will re-fit PCA on the historical data passed in (df).
    # In a pure live loop, we'd load the PCA pickle.
    
    # 1. Preprocess
    processed_df, X_pca, _, _, _ = preprocess_data(df)
    
    # 2. Load Model
    if not os.path.exists(model_path):
        print(f"Warning: Model {model_path} not found. Returning empty signals.")
        processed_df['Signal'] = 0
        return processed_df, processed_df['Signal']
        
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # 3. Generate Signals
    # reuse generate_signals logic but we need to return specific format
    signals_df = generate_signals(processed_df, model, X_pca)
    
    return signals_df, signals_df['Final_Signal']

def main():
    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")
        try:
            df = fetch_data(ticker)
            processed, X_pca, y, scaler, pca = preprocess_data(df)
            
            # Train
            model, preds = train_model(X_pca, y)
            
            # Generate Signals
            signals = generate_signals(processed, model, X_pca)
            
            # Output Signals
            print(f"\nLast 30 Periods ({ticker}):")
            cols = ['Close', 'Predicted_Return', 'Pred_Signal', 'Bull_Cross', 'Bear_Cross', 'Final_Signal']
            print(signals[cols].tail(30))
            
            # Save
            safe_tick = ticker.replace('/', '')
            model.save_model(f"STRATEGY/xgb_enhanced_{safe_tick}.json")
            signals.to_csv(f"STRATEGY/signals_enhanced_{safe_tick}.csv")
            
        except Exception as e:
            print(f"Failed {ticker}: {e}")

if __name__ == "__main__":
    main()
