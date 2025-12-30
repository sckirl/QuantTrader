#!/usr/bin/env python3
"""
STRATEGY_BACKTEST.py
Comprehensive Backtesting Suite
Comparies Strategies from /STRATEGY/ folder.
"""

import pandas as pd
import numpy as np
import ccxt
import datetime
import warnings
import sys
import os

# Add root to path to find modules
sys.path.append(os.getcwd())

from STRATEGY.enhanced_strategy import run_xgboost_strategy
from STRATEGY.institutional_strategy import run_institutional_strategy

warnings.filterwarnings("ignore")

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '4h'
START_DATE = "2021-05-01"

def fetch_data():
    """
    Fetches 4h OHLCV data from Binance (Reliable Source).
    """
    print(f"--- Fetching Data for {SYMBOL} ---")
    exchange = ccxt.binance()
    start_ts = int(datetime.datetime.strptime(START_DATE, "%Y-%m-%d").timestamp() * 1000)
    
    all_ohlcv = []
    since = start_ts
    
    while True:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=1000)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if since > int(datetime.datetime.now().timestamp() * 1000): break
        if len(ohlcv) < 1000: break
            
    df = pd.DataFrame(all_ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)
    return df.astype(float)

# ==========================================
# Advanced Backtest Engine
# ==========================================
def backtest_engine(data, signals, name):
    capital = 10000.0
    cash = capital
    position = 0.0
    entry_price = 0.0
    equity_curve = []
    trade_log = []
    
    fee = 0.00075
    sl_pct = 0.02
    tp_pct = 0.06
    
    for i in range(len(data) - 1):
        today = data.iloc[i]
        tomorrow = data.iloc[i+1]
        sig = signals.iloc[i]
        curr_price = tomorrow['Open']
        
        # Position Management
        if position != 0:
            exit_type = None
            if position > 0:
                if tomorrow['Low'] < entry_price * (1 - sl_pct): exit_type = 'SL'
                elif tomorrow['High'] > entry_price * (1 + tp_pct): exit_type = 'TP'
                elif sig == -1: exit_type = 'Signal'
            elif position < 0:
                if tomorrow['High'] > entry_price * (1 + sl_pct): exit_type = 'SL'
                elif tomorrow['Low'] < entry_price * (1 - tp_pct): exit_type = 'TP'
                elif sig == 1: exit_type = 'Signal'
            
            if exit_type:
                if position > 0:
                    exit_p = entry_price * (1 - sl_pct) if exit_type == 'SL' else (entry_price * (1 + tp_pct) if exit_type == 'TP' else curr_price)
                    pnl = position * (exit_p - entry_price)
                    cash += position * entry_price
                else:
                    exit_p = entry_price * (1 + sl_pct) if exit_type == 'SL' else (entry_price * (1 - tp_pct) if exit_type == 'TP' else curr_price)
                    pnl = abs(position) * (entry_price - exit_p)
                    cash += pnl
                
                # Deduct Fee
                trade_fee = abs(position) * exit_p * fee
                cash -= trade_fee
                pnl -= trade_fee 
                
                trade_log.append(pnl)
                position = 0.0
        
        # Entry
        if position == 0 and sig != 0:
            invest = cash * 0.10 # 10%
            cost = invest * (1 + fee)
            if cash >= cost:
                if sig == 1:
                    position = invest / curr_price
                    cash -= cost
                else:
                    position = -(invest / curr_price)
                    cash -= (invest * fee) 
                entry_price = curr_price
        
        # Equity Tracking
        pos_val = 0
        if position > 0: pos_val = position * data.iloc[i]['Close']
        elif position < 0: pos_val = abs(position) * entry_price + (abs(position) * (entry_price - data.iloc[i]['Close']))
        equity_curve.append(cash + pos_val)

    # --- Advanced Metrics ---
    eq = pd.Series(equity_curve)
    ret_series = eq.pct_change().dropna()
    
    total_ret = (eq.iloc[-1] - capital) / capital
    # Approx Annualized
    years = len(data) / (6 * 365) # 4h candles
    cagr = ((eq.iloc[-1] / capital) ** (1 / years)) - 1 if years > 0 else 0
    
    sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(365 * 6) if ret_series.std() > 0 else 0
    sortino = (ret_series.mean() / ret_series[ret_series < 0].std()) * np.sqrt(365 * 6) if len(ret_series[ret_series < 0]) > 0 else 0
    
    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = dd.min()
    
    trades = len(trade_log)
    wins = len([t for t in trade_log if t > 0])
    win_rate = wins / trades if trades > 0 else 0
    
    gross_profit = sum([t for t in trade_log if t > 0])
    gross_loss = abs(sum([t for t in trade_log if t < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'Name': name,
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max DD': max_dd,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Trades': trades,
        'Final Equity': eq.iloc[-1]
    }

def main():
    df = fetch_data()
    
    # 1. ML Strategy
    # Point to the BTC model we trained in STRATEGY/
    _, sig_ml = run_xgboost_strategy(df, model_path="STRATEGY/xgb_enhanced_BTCUSDT.json")
    
    # 2. Institutional Strategy
    _, sig_inst = run_institutional_strategy(df)
    
    # Align Signals
    common_idx = df.index.intersection(sig_ml.index).intersection(sig_inst.index)
    df_clean = df.loc[common_idx]
    sig_ml = sig_ml.loc[common_idx]
    sig_inst = sig_inst.loc[common_idx]
    
    # Backtest
    res_ml = backtest_engine(df_clean, sig_ml, "ML (XGBoost)")
    res_inst = backtest_engine(df_clean, sig_inst, "Institutional (Triple-Conf)")
    
    # Print Comparison
    print("\n" + "="*70)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("="*70)
    
    metrics_list = [res_ml, res_inst]
    results = pd.DataFrame(metrics_list).set_index('Name')
    
    format_mapping = {
        'CAGR': "{:.2%}",
        'Sharpe': "{:.2f}",
        'Sortino': "{:.2f}",
        'Max DD': "{:.2%}",
        'Win Rate': "{:.2%}",
        'Profit Factor': "{:.2f}",
        'Final Equity': "${:,.2f}"
    }
    
    for col, fmt in format_mapping.items():
        if col in results.columns:
            results[col] = results[col].apply(lambda x: fmt.format(x) if pd.notnull(x) and not np.isinf(x) else str(x))
            
    print(results.T)
    print("="*70)

if __name__ == "__main__":
    main()