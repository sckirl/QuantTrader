import pytest
from unittest.mock import MagicMock
from src.exchange_client import ExchangeClient
import ccxt

def test_exchange_client_init_testnet(mocker):
    mock_ccxt_bybit = mocker.patch('ccxt.bybit')
    mock_exchange_instance = mock_ccxt_bybit.return_value
    
    config = {
        'apiKey': 'test_key',
        'secret': 'test_secret',
        'is_testnet': True
    }
    
    client = ExchangeClient(config)
    
    mock_ccxt_bybit.assert_called_once()
    mock_exchange_instance.set_sandbox_mode.assert_called_once_with(True)

def test_exchange_client_init_mainnet(mocker):
    mock_ccxt_bybit = mocker.patch('ccxt.bybit')
    mock_exchange_instance = mock_ccxt_bybit.return_value
    
    config = {
        'apiKey': 'test_key',
        'secret': 'test_secret',
        'is_testnet': False
    }
    
    client = ExchangeClient(config)
    
    mock_exchange_instance.set_sandbox_mode.assert_not_called()

def test_validate_api_keys_success(mocker):
    mock_ccxt_bybit = mocker.patch('ccxt.bybit')
    mock_exchange_instance = mock_ccxt_bybit.return_value
    mock_exchange_instance.fetch_balance.return_value = {} # Success
    
    config = {'apiKey': 'k', 'secret': 's', 'is_testnet': True}
    client = ExchangeClient(config)
    
    assert client.validate_api_keys() is True

def test_validate_api_keys_failure(mocker):
    mock_ccxt_bybit = mocker.patch('ccxt.bybit')
    mock_exchange_instance = mock_ccxt_bybit.return_value
    mock_exchange_instance.fetch_balance.side_effect = ccxt.AuthenticationError("Invalid keys")
    
    config = {'apiKey': 'k', 'secret': 's', 'is_testnet': True}
    client = ExchangeClient(config)
    
    with pytest.raises(ccxt.AuthenticationError):
        client.validate_api_keys()
