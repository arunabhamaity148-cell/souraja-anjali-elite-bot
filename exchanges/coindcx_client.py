"""
CoinDCX API Client
"""

import logging
import asyncio
import aiohttp
import pandas as pd
import hmac
import hashlib
import json
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger("COINDCX")

class CoinDCXClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coindcx.com"
        self.base_url_exchange = "https://api.coindcx.com/exchange/v1"
        
        logger.info("✅ CoinDCX connected")
    
    def _generate_signature(self, body: str) -> str:
        """Generate HMAC signature"""
        secret_bytes = self.api_secret.encode()
        signature = hmac.new(secret_bytes, body.encode(), hashlib.sha256).hexdigest()
        return signature
    
    async def get_ohlcv(self, symbol: str, interval: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV"""
        try:
            # CoinDCX uses market pair format
            pair = symbol.replace('USDT', 'USDT')  # B-BTC_USDT
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url_exchange}/markets/candles"
                params = {
                    'pair': f'B-{pair}',
                    'interval': interval,
                    'limit': limit
                }
                
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                    
                    if not isinstance(data, list):
                        return None
                    
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
                    
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
        except Exception as e:
            logger.error(f"CoinDCX OHLCV error: {e}")
            return None
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker"""
        try:
            pair = symbol.replace('USDT', 'USDT')
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url_exchange}/markets/tickers"
                
                async with session.get(url) as resp:
                    data = await resp.json()
                    
                    for ticker in data:
                        if ticker['market'] == f'B-{pair}':
                            return {
                                'symbol': symbol,
                                'last': float(ticker['last_price']),
                                'bid': float(ticker['bid']),
                                'ask': float(ticker['ask']),
                                'volume': float(ticker['volume']),
                                'change_24h': float(ticker['change_24_hour'])
                            }
                    
                    return {}
                    
        except Exception as e:
            logger.error(f"CoinDCX ticker error: {e}")
            return {}
    
    async def get_balance(self) -> Dict:
        """Get balance (requires auth)"""
        try:
            timestamp = int(datetime.now().timestamp() * 1000)
            body = {
                "timestamp": timestamp
            }
            
            json_body = json.dumps(body, separators=(',', ':'))
            signature = self._generate_signature(json_body)
            
            headers = {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url_exchange}/users/balances"
                
                async with session.post(url, headers=headers, json=body) as resp:
                    data = await resp.json()
                    return data
                    
        except Exception as e:
            logger.error(f"CoinDCX balance error: {e}")
            return {}
    
    async def create_order(self, symbol: str, side: str, amount: float, 
                          price: float = None, order_type: str = 'market_order') -> Dict:
        """Create order (requires auth)"""
        try:
            timestamp = int(datetime.now().timestamp() * 1000)
            pair = symbol.replace('USDT', 'USDT')
            
            body = {
                "timestamp": timestamp,
                "market": f"B-{pair}",
                "side": side,  # buy or sell
                "order_type": order_type,
                "total_quantity": amount,
                "price_per_unit": price if price else 0
            }
            
            json_body = json.dumps(body, separators=(',', ':'))
            signature = self._generate_signature(json_body)
            
            headers = {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url_exchange}/orders/create"
                
                async with session.post(url, headers=headers, json=body) as resp:
                    return await resp.json()
                    
        except Exception as e:
            logger.error(f"CoinDCX order error: {e}")
            return {}
