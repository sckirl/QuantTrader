import ccxt
import logging
import requests

class ExchangeClient:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        
        options = {
            'apiKey': config['apiKey'],
            'secret': config['secret'],
            'enableRateLimit': True,
        }
        
        self.exchange = ccxt.bybit(options)
        
        if config.get('isTestnet', False):
            self.logger.info("Mode: Testnet/Demo")
            self.exchange.set_sandbox_mode(True)
            self.detectEnvironment()
        else:
            self.logger.info("Mode: Mainnet")

    def detectEnvironment(self):
        """
        Probes endpoints to find the correct Testnet/Demo URL.
        """
        candidates = [
            "https://api-demo.bybit.com",      # Priority: New Demo
            "https://api-testnet.bybit.com",   # Standard Testnet
            "https://api-testnet.bytick.com",  # Mirror
        ]
        
        self.logger.info("Auto-detecting environment...")
        
        for url in candidates:
            try:
                # 1. Connectivity
                if requests.get(f"{url}/v5/market/time", timeout=3).status_code != 200:
                    continue
                    
                # 2. Auth Check
                self.exchange.urls['api'] = {k: url for k in ['spot', 'futures', 'v2', 'public', 'private']}
                
                try:
                    self.exchange.private_get_v5_account_info()
                    self.logger.info(f"Connected to: {url}")
                    return
                except Exception as e:
                    # Error 10032 means we connected to Demo but hit a specific endpoint issue.
                    # We accept this as a valid connection because auth worked enough to give a specific error.
                    if "10032" in str(e): 
                        self.logger.info(f"Connected to: {url} (Demo Limitation)")
                        return

            except Exception:
                continue
        
        self.logger.error("CRITICAL: No reachable environment found for these keys.")

    def validateKeys(self):
        try:
            self.exchange.private_get_v5_account_info()
            return True
        except Exception:
            # Fallback for Demo quirks
            try:
                self.exchange.fetch_balance()
                return True
            except Exception as e:
                self.logger.error(f"Auth Failed: {e}")
                raise

    def setLeverage(self, symbol, leverage):
        try:
            marketSymbol = symbol.split(':')[0].replace('/', '')
            self.exchange.private_post_v5_position_set_leverage({
                'category': 'linear',
                'symbol': marketSymbol,
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage),
            })
            self.logger.info(f"Leverage set: {leverage}x")
        except Exception as e:
            if "110043" in str(e) or "10032" in str(e): # Already set or Demo quirk
                self.logger.info(f"Leverage verified: {leverage}x")
            else:
                self.logger.warning(f"Leverage update failed: {e}")

    def placeMarketOrder(self, symbol, side, amount):
        try:
            marketSymbol = symbol.split(':')[0].replace('/', '')
            payload = {
                'category': 'linear',
                'symbol': marketSymbol,
                'side': side.capitalize(),
                'orderType': 'Market',
                'qty': str(amount),
            }
            response = self.exchange.private_post_v5_order_create(payload)
            orderId = response.get('result', {}).get('orderId')
            self.logger.info(f"Order Placed: {orderId}")
            return {'id': orderId, 'info': response}
        except Exception as e:
            self.logger.error(f"Order Failed: {e}")
            raise

    def getBalance(self):
        try:
            response = self.exchange.private_get_v5_account_wallet_balance({'accountType': 'UNIFIED', 'coin': 'USDT'})
            result = response.get('result', {}).get('list', [])
            
            if not result: return {'USDT': {'free': 0.0, 'total': 0.0}}
            
            coinData = result[0].get('coin', [])
            usdtData = next((c for c in coinData if c['coin'] == 'USDT'), None)
            
            if usdtData:
                return {
                    'USDT': {
                        'free': float(usdtData.get('availableToWithdraw', 0) or 0),
                        'total': float(usdtData.get('walletBalance', 0) or 0)
                    }
                }
            return {'USDT': {'free': 0.0, 'total': 0.0}}
        except Exception as e:
            self.logger.error(f"Balance fetch failed: {e}")
            raise

    def getPositions(self, symbols=None):
        try:
            params = {'category': 'linear', 'limit': 50, 'settleCoin': 'USDT'}
            if symbols:
                params['symbol'] = symbols[0].split(':')[0].replace('/', '')
            
            response = self.exchange.private_get_v5_position_list(params)
            rawPositions = response.get('result', {}).get('list', [])
            
            return [{
                'symbol': p['symbol'],
                'side': p['side'],
                'contracts': float(p['size']),
                'entryPrice': float(p['avgPrice']),
                'leverage': p['leverage'],
                'pnl': p['unrealisedPnl']
            } for p in rawPositions]
            
        except Exception as e:
            self.logger.error(f"Position fetch failed: {e}")
            raise