import pandas as pd
import numpy as np

def run_institutional_strategy(df):
    """
    Institutional Strategy (HFT Lead Quant Refined)
    Logic: "Triple-Confluence"
    1. Mean Reversion: VWAP Bands (Price > 2sigma).
    2. Sentiment: RSI(2) < 10 / > 90.
    3. Liquidity: Volume > 1.5x MA(20).
    4. Dynamic Threshold: Rolling Z-Score > 2.0 of the combined raw signal strength.
    """
    print("Running Institutional Strategy...")
    data = df.copy()
    
    # --- Feature Engineering ---
    
    # 1. VWAP + Bands
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (tp * data['Volume']).rolling(24).sum() / data['Volume'].rolling(24).sum()
    
    # Std Dev from VWAP
    data['VWAP_Std'] = data['Close'].rolling(24).std()
    data['Upper_Band'] = data['VWAP'] + (2 * data['VWAP_Std'])
    data['Lower_Band'] = data['VWAP'] - (2 * data['VWAP_Std'])
    
    # 2. RSI(2)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/2, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/2, adjust=False).mean()
    rs = gain / loss
    data['RSI_2'] = 100 - (100 / (1 + rs))
    
    # 3. Liquidity (Volume Sweep)
    data['Vol_MA20'] = data['Volume'].rolling(20).mean()
    data['Vol_Sweep'] = data['Volume'] > (1.5 * data['Vol_MA20'])
    
    data.dropna(inplace=True)
    
    # --- Logic ---
    
    # Long Setup
    v1_long = (data['Close'] < data['Lower_Band']).astype(int) 
    v2_long = (data['RSI_2'] < 10).astype(int)                 
    v3_long = (data['Vol_Sweep']).astype(int)                  
    
    # Short Setup
    v1_short = (data['Close'] > data['Upper_Band']).astype(int) 
    v2_short = (data['RSI_2'] > 90).astype(int)                 
    v3_short = (data['Vol_Sweep']).astype(int)                  
    
    # Raw Signal Strength
    raw_long = v1_long + v2_long + v3_long
    raw_short = v1_short + v2_short + v3_short
    
    # Dynamic Threshold (Z-Score)
    activity = raw_long + raw_short
    rolling_mean = activity.rolling(100).mean()
    rolling_std = activity.rolling(100).std()
    
    data['Z_Score'] = (activity - rolling_mean) / (rolling_std + 1e-9)
    
    data['Signal'] = 0
    
    # Trigger: Strength >= 2 AND Z-Score > 2.0
    long_trigger = (raw_long >= 2) & (data['Z_Score'] > 2.0)
    data.loc[long_trigger, 'Signal'] = 1
    
    short_trigger = (raw_short >= 2) & (data['Z_Score'] > 2.0)
    data.loc[short_trigger, 'Signal'] = -1
    
    return data, data['Signal']
