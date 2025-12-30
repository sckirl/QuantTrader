import os
from dotenv import load_dotenv

class ConfigLoader:
    def __init__(self, envPath="config/.env"):
        load_dotenv(dotenv_path=envPath)
    
    def getConfig(self):
        """
        Retrieves Bybit configuration.
        """
        apiKey = os.getenv("BYBIT_API_KEY")
        secret = os.getenv("BYBIT_SECRET")
        useTestnet = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"
        
        if not apiKey or not secret:
            raise ValueError("BYBIT_API_KEY and BYBIT_SECRET must be set.")
            
        return {
            "apiKey": apiKey.strip(),
            "secret": secret.strip(),
            "isTestnet": useTestnet
        }