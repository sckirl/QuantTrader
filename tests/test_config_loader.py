import os
import pytest
from unittest.mock import patch
from src.config_loader import ConfigLoader

def test_get_bybit_config_valid_env():
    with patch.dict(os.environ, {
        "BYBIT_API_KEY": "test_key",
        "BYBIT_SECRET": "test_secret",
        "BYBIT_USE_TESTNET": "true"
    }):
        loader = ConfigLoader()
        config = loader.get_bybit_config()
        
        assert config["apiKey"] == "test_key"
        assert config["secret"] == "test_secret"
        assert config["is_testnet"] is True

def test_get_bybit_config_missing_keys():
    with patch.dict(os.environ, {}, clear=True):
        loader = ConfigLoader()
        with pytest.raises(ValueError, match="must be set"):
            loader.get_bybit_config()

def test_get_bybit_config_default_testnet():
    with patch.dict(os.environ, {
        "BYBIT_API_KEY": "test_key",
        "BYBIT_SECRET": "test_secret"
        # BYBIT_USE_TESTNET missing, should default to true or handle it
    }):
        loader = ConfigLoader()
        config = loader.get_bybit_config()
        assert config["is_testnet"] is True # Based on our implementation default
