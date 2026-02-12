"""
Binance API Client - Spot + Futures
"""

import logging
import asyncio
import ccxt
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger("BINANCE")

class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # futures trading
                'testnet': testnet
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        logger.info(f"✅ Binance {'Testnet' if testnet else 'Live'} connected")
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles"""
        try:
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv, 
                symbol, 
                timeframe, 
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Binance OHLCV error: {e}")
            return None
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current price"""
        try:
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
            return {
                'symbol': symbol,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['quoteVolume'],
                'change_24h': ticker['percentage']
            }
        except Exception as e:
            logger.error(f"Binance ticker error: {e}")
            return {}
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get orderbook"""
        try:
            ob = await asyncio.to_thread(self.exchange.fetch_order_book, symbol, limit)
            return {
                'bids': ob['bids'][:limit],
                'asks': ob['asks'][:limit],
                'timestamp': ob['timestamp']
            }
        except Exception as e:
            logger.error(f"Binance OB error: {e}")
            return {'bids': [], 'asks': []}
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate"""
        try:
            funding = await asyncio.to_thread(self.exchange.fetch_funding_rate, symbol)
            return funding['fundingRate']
        except Exception as e:
            logger.error(f"Funding error: {e}")
            return 0.0
    
    async def get_open_interest(self, symbol: str) -> Dict:
        """Get open interest"""
        try:
            # Binance specific endpoint
            markets = await asyncio.to_thread(self.exchange.load_markets)
            market = self.exchange.market(symbol)
            
            # Use futures API for OI
            response = await asyncio.to_thread(
                self.exchange.fapiPublic_get_openinterest,
                {'symbol': market['id']}
            )
            
            return {
                'openInterest': float(response['openInterest']),
                'timestamp': response['time']
            }
        except Exception as e:
            logger.error(f"OI error: {e}")
            return {'openInterest': 0}
    
    async def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            balance = await asyncio.to_thread(self.exchange.fetch_balance)
            return {
                'USDT': balance.get('USDT', {}).get('free', 0),
                'total': balance.get('total', {})
            }
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return {}
    
    async def create_order(self, symbol: str, side: str, amount: float, 
                          price: float = None, order_type: str = 'market') -> Dict:
        """Create order"""
        try:
            order = await asyncio.to_thread(
                self.exchange.create_order,
                symbol,
                order_type,
                side,
                amount,
                price
            )
            return order
        except Exception as e:
            logger.error(f"Order error: {e}")
            return {}
    
    async def close(self):
        """Close connection"""
        await asyncio.to_thread(self.exchange.close)
