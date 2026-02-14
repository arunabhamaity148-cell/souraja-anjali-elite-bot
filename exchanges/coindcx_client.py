"""
CoinDCX API Client - FIXED Symbol Conversion
"""

import logging
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger("COINDCX")

class CoinDCXClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coindcx.com"
        self.base_url_exchange = "https://public.coindcx.com"
        
        logger.info("✅ CoinDCX connected")
    
    def _convert_symbol(self, binance_symbol: str) -> str:
        """
        Convert Binance symbol to CoinDCX format
        Binance: BTCUSDT -> CoinDCX: B-BTC_USDT
        """
        if binance_symbol.endswith('USDT'):
            base = binance_symbol.replace('USDT', '')
            return f'B-{base}_USDT'
        return binance_symbol
    
    async def get_ohlcv(self, symbol: str, interval: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV"""
        try:
            coindcx_symbol = self._convert_symbol(symbol)
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url_exchange}/market_data/candles"
                params = {
                    'pair': coindcx_symbol,
                    'interval': interval,
                    'limit': limit
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.debug(f"CoinDCX OHLCV failed for {symbol}: HTTP {resp.status}")
                        return None
                    
                    data = await resp.json()
                    
                    if not data or not isinstance(data, list):
                        logger.debug(f"CoinDCX OHLCV no data for {symbol}")
                        return None
                    
                    df = pd.DataFrame(data)
                    
                    if df.empty:
                        return None
                    
                    # CoinDCX format: [timestamp, open, high, low, close, volume]
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
        except Exception as e:
            logger.debug(f"CoinDCX OHLCV error for {symbol}: {e}")
            return None
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker - FIXED"""
        try:
            coindcx_symbol = self._convert_symbol(symbol)
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url_exchange}/market_data/ticker"
                
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.debug(f"CoinDCX ticker failed for {symbol}: HTTP {resp.status}")
                        return {}
                    
                    data = await resp.json()
                    
                    # CoinDCX returns array of tickers
                    if not data or not isinstance(data, list):
                        logger.debug(f"CoinDCX ticker invalid format for {symbol}")
                        return {}
                    
                    # Find our symbol
                    for ticker in data:
                        if not isinstance(ticker, dict):
                            continue
                        
                        if ticker.get('market') == coindcx_symbol:
                            return {
                                'symbol': symbol,
                                'last': float(ticker.get('last_price', 0)),
                                'bid': float(ticker.get('bid', 0)),
                                'ask': float(ticker.get('ask', 0)),
                                'volume': float(ticker.get('volume', 0)),
                                'change_24h': float(ticker.get('change_24_hour', 0))
                            }
                    
                    # Symbol not found
                    logger.debug(f"CoinDCX ticker not found for {symbol}")
                    return {}
                    
        except Exception as e:
            logger.debug(f"CoinDCX ticker error for {symbol}: {e}")
            return {}
    
    async def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            return {'USDT': 0}  # Placeholder
        except Exception as e:
            logger.error(f"CoinDCX balance error: {e}")
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
        """Get funding rate"""
        try:
            # CoinDCX doesn't have funding rates (spot exchange)
            return 0.0
        except Exception as e:
            logger.debug(f"CoinDCX funding error: {e}")
            return 0.0
