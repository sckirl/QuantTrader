import pytest
from unittest.mock import MagicMock
from src.exchange_client import ExchangeClient

def test_set_leverage_success(mocker):
    mock_ccxt = mocker.patch('ccxt.bybit')
    exchange_instance = mock_ccxt.return_value
    
    client = ExchangeClient({'apiKey': 'k', 'secret': 's'})
    client.set_leverage('BTC/USDT:USDT', 10)
    
    exchange_instance.set_leverage.assert_called_once_with(10, 'BTC/USDT:USDT')

def test_set_leverage_failure_handled(mocker):
    mock_ccxt = mocker.patch('ccxt.bybit')
    exchange_instance = mock_ccxt.return_value
    exchange_instance.set_leverage.side_effect = Exception("Already set")
    
    client = ExchangeClient({'apiKey': 'k', 'secret': 's'})
    # Should not raise exception, just log warning
    client.set_leverage('BTC/USDT:USDT', 10) 
    
    exchange_instance.set_leverage.assert_called_once()

def test_place_market_order_success(mocker):
    mock_ccxt = mocker.patch('ccxt.bybit')
    exchange_instance = mock_ccxt.return_value
    exchange_instance.create_order.return_value = {'id': '123'}
    
    client = ExchangeClient({'apiKey': 'k', 'secret': 's'})
    result = client.place_market_order('BTC/USDT:USDT', 'buy', 0.001)
    
    assert result['id'] == '123'
    exchange_instance.create_order.assert_called_once_with(
        'BTC/USDT:USDT', 'market', 'buy', 0.001, params={}
    )
