"""
Delta Exchange API Client - FIXED Symbol Conversion
"""

import logging
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger("DELTA")

class DeltaClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.delta.exchange"
        self.ws_url = "wss://socket.delta.exchange"
        
        logger.info("✅ Delta Exchange connected")
    
    def _convert_symbol(self, binance_symbol: str) -> str:
        """
        Convert Binance symbol to Delta format
        Binance: BTCUSDT -> Delta: BTCUSD
        """
        # Delta uses different naming
        if binance_symbol.endswith('USDT'):
            return binance_symbol.replace('USDT', 'USD')
        return binance_symbol
    
    async def get_ohlcv(self, symbol: str, resolution: str = '5', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV"""
        try:
            delta_symbol = self._convert_symbol(symbol)
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/history/candles"
                params = {
                    'symbol': delta_symbol,
                    'resolution': resolution,
                    'limit': limit
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.debug(f"Delta OHLCV failed for {symbol}: HTTP {resp.status}")
                        return None
                    
                    data = await resp.json()
                    
                    if not data or 'result' not in data or not data['result']:
                        logger.debug(f"Delta OHLCV no data for {symbol}")
                        return None
                    
                    candles = data['result']
                    df = pd.DataFrame(candles)
                    
                    if df.empty:
                        return None
                    
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
        except Exception as e:
            logger.debug(f"Delta OHLCV error for {symbol}: {e}")
            return None
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker - FIXED"""
        try:
            delta_symbol = self._convert_symbol(symbol)
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/tickers/{delta_symbol}"
                
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.debug(f"Delta ticker failed for {symbol}: HTTP {resp.status}")
                        return {}
                    
                    data = await resp.json()
                    
                    # Validate response structure
                    if not data or 'result' not in data:
                        logger.debug(f"Delta ticker no data for {symbol}")
                        return {}
                    
                    result = data['result']
                    
                    # Check if result is a dict and has required fields
                    if not isinstance(result, dict):
                        logger.debug(f"Delta ticker invalid format for {symbol}")
                        return {}
                    
                    # Return properly formatted ticker
                    return {
                        'symbol': symbol,
                        'last': float(result.get('close', 0)),
                        'bid': float(result.get('bid', 0)),
                        'ask': float(result.get('ask', 0)),
                        'volume': float(result.get('volume', 0)),
                        'change_24h': float(result.get('price_change_24h', 0))
                    }
                    
        except Exception as e:
            logger.debug(f"Delta ticker error for {symbol}: {e}")
            return {}
    
    async def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            return {'USDT': 0}  # Placeholder
        except Exception as e:
            logger.error(f"Delta balance error: {e}")
            return {}
    
    async def get_best_price(self, symbol: str) -> Dict:
        """Get best bid/ask"""
        ticker = await self.get_ticker(symbol)
        if not ticker:
            return {}
        
        return {
            'best_bid': {'price': ticker.get('bid', 0), 'size': 0},
            'best_ask': {'price': ticker.get('ask', 0), 'size': 0}
        }
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate - FIXED"""
        try:
            delta_symbol = self._convert_symbol(symbol)
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/products/{delta_symbol}"
                
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return 0.0
                    
                    data = await resp.json()
                    
                    if not data or 'result' not in data:
                        return 0.0
                    
                    result = data['result']
                    
                    if isinstance(result, dict):
                        return float(result.get('funding_rate', 0))
                    
                    return 0.0
                    
        except Exception as e:
            logger.debug(f"Delta funding error for {symbol}: {e}")
            return 0.0
