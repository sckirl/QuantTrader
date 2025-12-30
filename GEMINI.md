# GEMINI: Quantitative Development Assistant Context

## Mission
Produce secure, correct, and fully tested Python code for quant trading.
**Goal:** Build a secure CCXT-based automated crypto trading MVP for Bybit.

## Project Status: Milestone 1 Complete
- **Date:** 2025-12-30
- **Current Phase:** Core Infrastructure & Connectivity Verified.
- **Operating Environment:** Bybit **Demo** Account (Unified Trading).

## Progress Log

### ✅ Completed
1.  **Project Architecture:**
    -   Established `src/` (logic), `tests/` (unit tests), `config/` (secrets) structure.
    -   Implemented secure `ConfigLoader` using `.env`.
2.  **Connectivity & Authentication:**
    -   Implemented `ExchangeClient` with **Environment Auto-Detection**.
    -   System automatically probes `api-demo.bybit.com`, `api-testnet.bybit.com`, and `api-testnet.bytick.com` to find the valid endpoint for the provided keys.
3.  **Trade Execution (MVP):**
    -   Successfully executed first live trade on Demo: Long BTC/USDT @ 10x Leverage.
    -   Verified position data retrieval.
4.  **Refactoring:**
    -   Codebase converted to **CamelCase** per user preference.
    -   Code is minimal and efficient, stripping unnecessary abstractions.

### ⚠️ Technical Constraints & Critical Knowledge (READ THIS)
**1. Bybit Demo vs. CCXT Standard Wrappers:**
-   **Issue:** The new Bybit Demo environment (`api-demo.bybit.com`) returns error `10032: Demo trading are not supported` when using standard CCXT methods like `exchange.fetch_balance()` or `exchange.create_order()`.
-   **Solution:** You **MUST** use direct V5 API calls for execution on Demo.
    -   *Balance:* `exchange.private_get_v5_account_wallet_balance(...)`
    -   *Orders:* `exchange.private_post_v5_order_create(...)`
    -   *Leverage:* `exchange.private_post_v5_position_set_leverage(...)`
    -   *Positions:* `exchange.private_get_v5_position_list(...)`
-   **Do NOT revert to standard wrappers** without verifying they support the Demo URL first.

**2. Network Blocking:**
-   `api-demo.bybit.com` is often blocked by ISPs. A VPN (Japan/Singapore) is required for connectivity.

## Roadmap & Next Steps
1.  **Risk Management Module:**
    -   Implement position sizing rules (e.g., max % of equity).
    -   Add Stop Loss / Take Profit logic to order execution.
2.  **Order Service Expansion:**
    -   Support Limit Orders.
    -   Support Short Selling.
    -   Implement "Close All Positions" functionality.
3.  **Strategy Implementation:**
    -   Develop a basic signal generator (e.g., Moving Average Crossover) to trigger the Order Service.

## Mandatory Workflow (Persistent)
1.  **Library Documentation:** Verify APIs (especially V5 specific endpoints).
2.  **Code Generation:** PEP8 (modified for CamelCase preference), modular, secure.
3.  **Self-Testing:** Run `src/check_status.py` after changes to verify connectivity.
4.  **Security:** Keep secrets in `.env`.

## Libraries Used
-   `ccxt` (Bybit V5)
-   `python-dotenv`
-   `requests`
-   `pytest`
