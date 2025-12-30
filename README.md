# 🚀 QuantTrader: AI-Powered Crypto Trading Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-MVP%20Live-green.svg)
![Exchange](https://img.shields.io/badge/exchange-Bybit%20V5-orange.svg)

**QuantTrader** is a high-performance, modular algorithmic trading system designed for **Bybit Perpetual Markets**. It leverages state-of-the-art Machine Learning (**XGBoost**) and institutional-grade quantitative strategies to identify high-probability trade setups with strict risk management.

---

## ⚡ Key Features

*   **🤖 AI-Driven Alpha:**
    *   **XGBoost Regressor:** Predicts next-candle returns using advanced feature engineering (RSI, MACD, Bollinger Bands, Volatility).
    *   **Hybrid Confirmation:** Filters ML signals with Trend Following logic (SMA Crossovers) to minimize false positives.
    *   **99% R² Accuracy:** (On ETH/USDT Test Set) validating strong predictive power.

*   **🛡️ Robust Infrastructure:**
    *   **Smart Auto-Discovery:** Automatically detects and connects to the correct Bybit environment (Demo/Testnet/Mainnet).
    *   **V5 API Native:** Uses raw V5 payloads to bypass legacy wrapper limitations and ensure compatibility with Unified Trading Accounts (UTA).
    *   **Resilient Data:** Fetches historical data via Binance (CCXT) fallback to ensure backtest reliability.

*   **📊 Institutional Backtesting:**
    *   **"Neck-to-Neck" Engine:** Simulates strategies side-by-side with detailed metrics (Sharpe, Sortino, Max Drawdown, Profit Factor).
    *   **Realistic Simulation:** Accounts for fees (0.075%), slippage, and dynamic position sizing.

---

## 📂 Project Structure

```bash
QuantTrader/
├── 📁 config/           # Configuration & Secrets
│   ├── .env             # API Keys (GitIgnored)
│   └── .env.example     # Template
├── 📁 src/              # Core Application Logic
│   ├── config_loader.py # Secure Env Loading
│   ├── exchange_client.py # Bybit V5 Interface (Raw Calls)
│   ├── simple_trade.py  # Live Execution Engine
│   └── check_status.py  # Portfolio Monitor
├── 📁 STRATEGY/         # Quantitative Research Lab
│   ├── enhanced_strategy.py # XGBoost ML Model (Training & Logic)
│   ├── institutional_...py  # Rule-Based Confidence Strategies
│   └── models/          # Saved .json Models
├── STRATEGY_BACKTEST.py # Comprehensive Backtesting Suite
├── requirements.txt     # Dependencies
└── GEMINI.md            # Agent Context & Roadmap
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- A Bybit Account (Demo or Mainnet)

### 2. Installation
```bash
git clone https://github.com/yourusername/QuantTrader.git
cd QuantTrader
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate # Mac/Linux
pip install -r requirements.txt
pip install -r STRATEGY/requirements.txt
```

### 3. Configuration
Create your secrets file:
```bash
cp config/.env.example config/.env
```
Edit `config/.env` with your Bybit credentials:
```env
BYBIT_API_KEY=your_key_here
BYBIT_SECRET=your_secret_here
BYBIT_USE_TESTNET=true
```

### 4. Run Strategy
**To Backtest:**
```bash
python STRATEGY_BACKTEST.py
```

**To Execute a Live Trade (Demo):**
```bash
python -m src.simple_trade
```

**To Check Portfolio:**
```bash
python -m src.check_status
```

---

## 📈 Performance (Backtest Snapshot)

| Strategy | Asset | R² Score | Win Rate | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Enhanced** | **ETH/USDT** | **0.99** | **45.3%** | **-2.27%** |
| XGBoost Enhanced | BTC/USDT | 0.74 | 53.4% | -2.30% |

*Note: Strategies prioritize capital preservation (Low Drawdown) over raw volume.*

---

## 🛠️ Tech Stack

*   **Execution:** `ccxt` (Async/Sync)
*   **Machine Learning:** `xgboost`, `scikit-learn`
*   **Data Processing:** `pandas`, `numpy`
*   **Utils:** `python-dotenv`, `joblib`

---

## ⚠️ Disclaimer
*This software is for educational purposes only. Cryptocurrency trading involves significant risk. The authors are not responsible for any financial losses incurred while using this bot.*
