import logging
from src.config_loader import ConfigLoader
from src.exchange_client import ExchangeClient

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def executeTrade():
    logger.info("--- Executing Trade ---")
    
    loader = ConfigLoader()
    config = loader.getConfig()

    if not config['isTestnet']:
        logger.error("SAFETY: Testnet ONLY.")
        return

    client = ExchangeClient(config)
    
    if not client.validateKeys():
        return

    symbol = "BTC/USDT:USDT"
    leverage = 10
    amount = 0.001 
    
    client.setLeverage(symbol, leverage)
    client.placeMarketOrder(symbol, 'buy', amount)
    logger.info("--- Done ---")

if __name__ == "__main__":
    executeTrade()