"""
Delta Exchange API Client
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
    
    async def get_ohlcv(self, symbol: str, resolution: str = '5', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV"""
        try:
            # Delta uses different symbol format
            delta_symbol = symbol.replace('USDT', '')
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/history/candles"
                params = {
                    'symbol': delta_symbol,
                    'resolution': resolution,
                    'limit': limit
                }
                
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                    
                    if 'result' not in data:
                        return None
                    
                    candles = data['result']
                    df = pd.DataFrame(candles)
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
        except Exception as e:
            logger.error(f"Delta OHLCV error: {e}")
            return None
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker"""
        try:
            delta_symbol = symbol.replace('USDT', '')
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/tickers/{delta_symbol}"
                
                async with session.get(url) as resp:
                    data = await resp.json()
                    
                    if 'result' not in data:
                        return {}
                    
                    ticker = data['result']
                    return {
                        'symbol': symbol,
                        'last': float(ticker['close']),
                        'bid': float(ticker['bid']),
                        'ask': float(ticker['ask']),
                        'volume': float(ticker['volume']),
                        'change_24h': float(ticker['change_24h'])
                    }
                    
        except Exception as e:
            logger.error(f"Delta ticker error: {e}")
            return {}
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get L2 orderbook"""
        try:
            delta_symbol = symbol.replace('USDT', '')
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/l2orderbook/{delta_symbol}"
                
                async with session.get(url) as resp:
                    data = await resp.json()
                    
                    if 'result' not in data:
                        return {'bids': [], 'asks': []}
                    
                    result = data['result']
                    return {
                        'bids': [[float(b['price']), float(b['size'])] for b in result['buy'][:limit]],
                        'asks': [[float(a['price']), float(a['size'])] for a in result['sell'][:limit]]
                    }
                    
        except Exception as e:
            logger.error(f"Delta OB error: {e}")
            return {'bids': [], 'asks': []}
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate"""
        try:
            delta_symbol = symbol.replace('USDT', '')
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/tickers/{delta_symbol}"
                
                async with session.get(url) as resp:
                    data = await resp.json()
                    result = data.get('result', {})
                    return float(result.get('funding_rate', 0))
                    
        except Exception as e:
            logger.error(f"Delta funding error: {e}")
            return 0.0
