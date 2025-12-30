#!/usr/bin/env python3
"""
High-Accuracy XGBoost Strategy for BTC-USD and ETH-USD.
Replication of "Machine learning approaches to cryptocurrency trading optimization" (2025).

Strategy: XGBoost Regressor (Next-day Return) + Hybrid SMA Confirmation
Assets: BTC-USD, ETH-USD
Interval: Daily
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
import datetime
import os

warnings.filterwarnings("ignore")

import ccxt

def fetch_data(ticker):
    """
    Fetches daily OHLCV data from Binance via CCXT.
    Period: 2021-05-01 to present.
    Adapts Yahoo Finance tickers (BTC-USD) to CCXT format (BTC/USDT).
    """
    start_date = "2021-05-01"
    start_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    
    # Map Tickers
    symbol_map = {
        'BTC-USD': 'BTC/USDT',
        'ETH-USD': 'ETH/USDT'
    }
    symbol = symbol_map.get(ticker, ticker)
    
    print(f"Fetching data for {symbol} via CCXT (Binance) from {start_date}...")
    try:
        exchange = ccxt.binance()
        # Fetch OHLCV
        # Limit is usually 1000 per call, need pagination for 4 years
        all_ohlcv = []
        since = start_ts
        
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update since to last timestamp + 1 day (86400000 ms)
            last_ts = ohlcv[-1][0]
            since = last_ts + 86400000
            
            # Break if we reached current time
            if last_ts >= int(datetime.datetime.now().timestamp() * 1000):
                break
                
            # Safety break for loop
            if len(ohlcv) < 1000:
                break
        
        if not all_ohlcv:
            raise ValueError(f"No data fetched for {symbol}")
            
        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        
        # Keep relevant columns and ensure float
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols].astype(float)
        
        # Filter strict date range
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        df = df.loc[start_date:end_date]
        
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def preprocess_data(df):
    """
    Generates features, targets, and prepares data for training.
    """
    data = df.copy()
    
    # --- Feature Engineering ---
    
    # 1. Daily Returns
    data['Return'] = data['Close'].pct_change()
    
    # 2. Lagged Prices (Close_t-1, t-2, t-3)
    for lag in range(1, 4):
        data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
        
    # 3. Momentum: SMAs
    data['SMA_7'] = data['Close'].rolling(window=7).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    
    # 4. Volatility: Rolling Std Dev
    data['Vol_7'] = data['Close'].rolling(window=7).std()
    data['Vol_14'] = data['Close'].rolling(window=14).std()
    
    # 5. Target: Next Day Close
    data['Target_Close'] = data['Close'].shift(-1)
    
    # Drop NaNs created by lags/rolling/shifting
    data.dropna(inplace=True)
    
    # --- Feature Selection for Model ---
    features = [
        'Close', 'Return', 
        'Close_lag_1', 'Close_lag_2', 'Close_lag_3',
        'SMA_7', 'SMA_30', 'Vol_7', 'Vol_14'
    ]
    
    X = data[features]
    y = data['Target_Close']
    
    # --- Standardization ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Dimensionality Reduction (PCA) ---
    # Study specifies n_components=10, but we have 9 features.
    # We will use min(n_features, 10) -> 9 components (retaining 100% variance of input space).
    # If we added more technical indicators (RSI, MACD etc) we would need PCA 10.
    # Given the constraint to strictly follow features list:
    n_comp = min(X_scaled.shape[1], 10)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    
    return data, X_pca, y, scaler, pca

def train_model(X, y):
    """
    Trains XGBoost Regressor using TimeSeriesSplit.
    """
    # 80/20 Split based on time
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Hyperparameters from specifications
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,      # Lower end of 0.1-0.3 range for stability
        max_depth=4,            # Middle of 3-6 range
        subsample=0.8,
        gamma=0,
        reg_lambda=1,           # lambda is reserved keyword
        n_estimators=100,
        early_stopping_rounds=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit
    # Validation set is required for early_stopping
    # We take a small slice of training data as validation for early stopping 
    # OR use the test set strictly for evaluation. 
    # To respect TimeSeries nature:
    val_split = int(len(X_train) * 0.9)
    X_tr_final, X_val = X_train[:val_split], X_train[val_split:]
    y_tr_final, y_val = y_train[:val_split], y_train[val_split:]

    model.fit(
        X_tr_final, y_tr_final,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluation
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print("\n--- Model Evaluation (Test Set) ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4%}")
    print(f"R²:   {r2:.4f}")
    
    return model, X_test, y_test, preds

def generate_signals(df, model, X_pca):
    """
    Generates trading signals based on predicted return and SMA confirmation.
    """
    # Predict all data points
    all_preds = model.predict(X_pca)
    
    df['Predicted_Close'] = all_preds
    # Predicted Return: (Pred_Close_t+1 - Close_t) / Close_t
    df['Predicted_Return'] = (df['Predicted_Close'] - df['Close']) / df['Close']
    
    # 1. Base Signals based on Predicted Return
    # Buy > 0.5%, Sell < -0.5%
    conditions = [
        (df['Predicted_Return'] > 0.005),
        (df['Predicted_Return'] < -0.005)
    ]
    choices = [1, -1] # 1: Buy Candidate, -1: Sell Candidate
    df['Model_Signal'] = np.select(conditions, choices, default=0)
    
    # 2. Hybrid Confirmation (SMA Crossover)
    # Bullish Crossover: 7SMA crosses ABOVE 30SMA
    # Bearish Crossover: 7SMA crosses BELOW 30SMA
    
    df['SMA_7_prev'] = df['SMA_7'].shift(1)
    df['SMA_30_prev'] = df['SMA_30'].shift(1)
    
    # Bullish Cross confirmation logic (Current > Long AND Prev <= PrevLong)
    # OR just Trend Confirmation (Short > Long)?
    # Study says "Detect crossovers...". 
    # Hybrid confirmation usually means: Model says Buy AND Trend is Up/Turning Up.
    # Implementing strict crossover detection as requested:
    
    df['Bull_Cross'] = (df['SMA_7'] > df['SMA_30']) & (df['SMA_7_prev'] <= df['SMA_30_prev'])
    df['Bear_Cross'] = (df['SMA_7'] < df['SMA_30']) & (df['SMA_7_prev'] >= df['SMA_30_prev'])
    
    # However, crossovers are rare events. If we require a crossover EXACTLY on the day 
    # the model predicts a return > 0.5%, trades will be almost zero.
    # Re-reading: "Buy only if signal is Buy AND 7-day SMA crosses above..."
    # This implies a trigger. 
    # Alternative interpretation: Buy if Model Buy AND Trend is Bullish (SMA7 > SMA30).
    # Given "Detect crossovers" instruction, strict implementation might be too restrictive.
    # BUT "Implement Exactly As Described".
    # I will implement strict crossover check. If volume is low, it's low.
    
    # Relaxed interpretation for functionality: 
    # Often "Crossover" in these papers implies "Trend Regime".
    # But text says: "Detect crossovers: SMA_short.shift(1) < SMA_long.shift(1)..."
    # I will stick to the STRICT crossover event for entry.
    
    signals = []
    
    # Vectorized signal logic
    buy_cond = (df['Model_Signal'] == 1) & (df['Bull_Cross'])
    sell_cond = (df['Model_Signal'] == -1) & (df['Bear_Cross'])
    
    df['Final_Signal'] = 0
    df.loc[buy_cond, 'Final_Signal'] = 1
    df.loc[sell_cond, 'Final_Signal'] = -1
    
    return df

def backtest(df):
    """
    Backtests the strategy.
    
    Params:
    - Initial Capital: $10,000
    - Position Sizing: 1% per trade
    - Stop Loss: 3%
    - Fee: 0.1%
    - Exit: Opposite signal or 7 days hold
    """
    initial_capital = 10000.0
    cash = initial_capital
    position = 0.0 # Amount of asset
    entry_price = 0.0
    entry_day = 0
    
    equity_curve = []
    trades = 0
    winning_trades = 0
    
    # Stop Loss / Fee
    sl_pct = 0.03
    fee_pct = 0.001
    
    # Logic loop
    # We iterate through the dataframe
    indices = df.index
    
    for i in range(len(df) - 1): # -1 because we look ahead for PnL sometimes or execute at next open
        today = df.iloc[i]
        tomorrow = df.iloc[i+1] # Execution price (Open)
        
        signal = today['Final_Signal']
        price = tomorrow['Open']
        
        # Check Stop Loss if in position
        if position > 0:
            # Check Low of tomorrow to see if SL hit
            if tomorrow['Low'] < entry_price * (1 - sl_pct):
                # SL Hit
                exit_p = entry_price * (1 - sl_pct)
                cash += position * exit_p * (1 - fee_pct)
                
                # Stats
                if exit_p > entry_price: winning_trades += 1 # Unlikely for SL but logical check
                
                position = 0.0
                entry_day = 0
                
            # Check Time Exit (7 days)
            elif (i - entry_day) >= 7:
                exit_p = price
                cash += position * exit_p * (1 - fee_pct)
                if exit_p > entry_price: winning_trades += 1
                position = 0.0
                entry_day = 0
                
            # Check Opposite Signal (Sell)
            elif signal == -1:
                exit_p = price
                cash += position * exit_p * (1 - fee_pct)
                if exit_p > entry_price: winning_trades += 1
                position = 0.0
                entry_day = 0
        
        # Entry Logic
        if position == 0 and signal == 1:
            # Position Sizing: 1% of Portfolio
            # Current Portfolio Value = Cash
            invest_amount = cash * 0.01
            cost = invest_amount * (1 + fee_pct)
            
            if cash >= cost:
                position = invest_amount / price
                cash -= cost
                entry_price = price
                entry_day = i
                trades += 1
        
        # Record Equity
        current_val = cash + (position * df.iloc[i]['Close']) # Mark to market with close
        equity_curve.append(current_val)

    # Metrics
    final_equity = equity_curve[-1] if equity_curve else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital
    win_rate = (winning_trades / trades) if trades > 0 else 0.0
    
    # Max Drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    print("\n--- Backtest Results ---")
    print(f"Total Trades: {trades}")
    print(f"Win Rate:     {win_rate:.2%}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Final Equity: ${final_equity:.2f}")

    return df

def main():
    tickers = ['BTC-USD', 'ETH-USD']
    
    for ticker in tickers:
        print(f"\n{'='*40}")
        print(f"Running Strategy for {ticker}")
        print(f"{'='*40}")
        
        # 1. Fetch
        df = fetch_data(ticker)
        if df is None: continue
        
        # 2. Preprocess
        processed_df, X_pca, y, scaler, pca = preprocess_data(df)
        
        # 3. Train
        model, _, _, _ = train_model(X_pca, y)
        
        # 4. Signals
        signal_df = generate_signals(processed_df, model, X_pca)
        
        # 5. Backtest
        final_df = backtest(signal_df)
        
        # Output Signals (Last 30 Days)
        print(f"\n--- Recent Signals ({ticker}) ---")
        recent = final_df.tail(30)[['Close', 'Predicted_Return', 'Model_Signal', 'Bull_Cross', 'Final_Signal']]
        # Filter only non-zero signals for clarity if sparse
        active_signals = recent[recent['Final_Signal'] != 0]
        if not active_signals.empty:
            print(active_signals)
        else:
            print("No trade signals in the last 30 days (Strict Crossover Logic).")
            
        # Save Model
        model.save_model(f"STRATEGY/xgboost_model_{ticker}.json")
        final_df.to_csv(f"STRATEGY/signals_{ticker}.csv")
        print(f"Model and signals saved to STRATEGY/ folder.")

if __name__ == "__main__":
    main()
