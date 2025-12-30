import logging
from src.config_loader import ConfigLoader
from src.exchange_client import ExchangeClient

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def checkAccount():
    logger.info("--- Account Status ---")
    
    loader = ConfigLoader()
    config = loader.getConfig()
    client = ExchangeClient(config)
    
    # Balance
    bal = client.getBalance()
    usdt = bal.get('USDT', {})
    logger.info(f"USDT Total: {usdt.get('total'):.2f}")

    # Positions
    positions = client.getPositions(['BTC/USDT:USDT'])
    active = [p for p in positions if p['contracts'] > 0]
    
    if not active:
        logger.info("No Active Positions.")
    else:
        for p in active:
            logger.info(f"Position: {p['symbol']} | {p['side']} | Size: {p['contracts']} | Entry: {p['entryPrice']} | {p['leverage']}x")

if __name__ == "__main__":
    checkAccount()