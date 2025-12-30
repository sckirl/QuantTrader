# GEMINI: Quantitative Development Assistant Context

## Mission
Produce secure, correct, and fully tested Python code for quant trading.
**Goal:** Build a secure CCXT-based automated crypto trading MVP for Bybit.

## Project Status: Milestone 3 Complete (Refinement & Benchmarking)
- **Date:** 2025-12-31
- **Current Phase:** Strategy Benchmarking & Codebase Optimization.
- **Operating Environment:** Bybit **Demo** Account (Unified Trading).

## Progress Log

### ✅ Completed
1.  **Infrastructure:**
    -   Secure `ConfigLoader` & `ExchangeClient` (V5 Native).
    -   **Critical:** Implemented direct V5 API calls (`private_post_v5_...`) to bypass "Demo trading not supported" errors in standard CCXT wrappers.
2.  **Strategy Architecture (`STRATEGY/`):**
    -   **Modular Design:** Strategies are now isolated in standalone files.
    -   **ML Strategy (`enhanced_strategy.py`):** XGBoost Regressor (4h) + Hybrid SMA Confirmation.
        -   *Performance:* **Profitable** ($10,843 Eq), Low Drawdown (-12.8%). The primary driver.
    -   **Institutional Strategy (`institutional_strategy.py`):** Rule-based "Triple-Confluence".
        -   *Logic:* VWAP Bands + RSI Extremes + Volume Sweep + Z-Score Threshold.
        -   *Performance:* **Underperforming** ($9,166 Eq), High Drawdown (-26%). Fades trends too aggressively.
3.  **Backtesting Engine (`STRATEGY_BACKTEST.py`):**
    -   Advanced suite comparing strategies side-by-side.
    -   Metrics: CAGR, Sharpe, Sortino, Max DD, Profit Factor, Win Rate.
    -   Uses robust `ccxt.binance` historical data to ensure data quality.
4.  **Cleanup:**
    -   Removed redundant files (`simple_trade` legacy logic, old CSVs).
    -   Renamed and standardized file paths.

### ⚠️ Technical Constraints & Critical Knowledge (READ THIS)
**1. Bybit Demo vs. CCXT Standard Wrappers:**
-   **Issue:** The new Bybit Demo environment (`api-demo.bybit.com`) returns error `10032: Demo trading are not supported` when using standard CCXT methods.
-   **Solution:** You **MUST** use direct V5 API calls for execution on Demo:
    -   *Balance:* `exchange.private_get_v5_account_wallet_balance(...)`
    -   *Orders:* `exchange.private_post_v5_order_create(...)`
    -   *Leverage:* `exchange.private_post_v5_position_set_leverage(...)`
-   **Do NOT revert to standard wrappers** for Demo execution.

**2. Network Blocking:**
-   `api-demo.bybit.com` is often blocked by ISPs. A VPN (Japan/Singapore) is required.

## Roadmap & Next Steps
1.  **Deployment:**
    -   The **ML Strategy** (`enhanced_strategy.py`) is the selected candidate for live trading.
    -   Integrate the inference logic (`run_xgboost_strategy`) into `src/simple_trade.py` for continuous operation.
2.  **Optimization:**
    -   Refine Institutional Strategy to include **Trend Alignment** (ADX > 25) to stop fading strong trends.
    -   Implement "Warm-up" routine to fetch 4h history before starting the live loop.
3.  **Risk Management:**
    -   Hard-code Max Drawdown daily limit in execution engine.

## Libraries
-   `ccxt`, `xgboost`, `sklearn`, `pandas`, `numpy`, `python-dotenv`