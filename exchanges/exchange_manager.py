"""
Multi-Exchange Manager - Primary + Fallback
"""

import logging
import pandas as pd
from typing import Dict, Optional
from exchanges.binance_client import BinanceClient
from exchanges.delta_client import DeltaClient
from exchanges.coindcx_client import CoinDCXClient

logger = logging.getLogger("EXCHANGE_MGR")

class ExchangeManager:
    def __init__(self, config: Dict):
        self.clients = {}
        self.primary = 'binance'
        
        # Initialize Binance
        if config.get('binance_api_key'):
            self.clients['binance'] = BinanceClient(
                config['binance_api_key'],
                config['binance_api_secret'],
                config.get('binance_testnet', False)
            )
        
        # Initialize Delta
        if config.get('delta_api_key'):
            self.clients['delta'] = DeltaClient(
                config['delta_api_key'],
                config['delta_api_secret']
            )
        
        # Initialize CoinDCX
        if config.get('coindcx_api_key'):
            self.clients['coindcx'] = CoinDCXClient(
                config['coindcx_api_key'],
                config['coindcx_api_secret']
            )
        
        logger.info(f"✅ Exchange Manager: {list(self.clients.keys())}")
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV with fallback"""
        # Try primary first
        for exchange_name in [self.primary] + [e for e in self.clients if e != self.primary]:
            if exchange_name not in self.clients:
                continue
            
            client = self.clients[exchange_name]
            df = await client.get_ohlcv(symbol, timeframe, limit)
            
            if df is not None and len(df) > 0:
                logger.debug(f"✅ {exchange_name} data for {symbol}")
                return df
            
            logger.warning(f"⚠️ {exchange_name} failed for {symbol}, trying fallback")
        
        logger.error(f"❌ All exchanges failed for {symbol}")
        return None
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker with fallback"""
        for exchange_name in [self.primary] + [e for e in self.clients if e != self.primary]:
            if exchange_name not in self.clients:
                continue
            
            ticker = await self.clients[exchange_name].get_ticker(symbol)
            
            if ticker and 'last' in ticker:
                return {**ticker, 'exchange': exchange_name}
        
        return {}
    
    async def get_best_price(self, symbol: str) -> Dict:
        """Get best price across exchanges"""
        prices = []
        
        for name, client in self.clients.items():
            ticker = await client.get_ticker(symbol)
            if ticker and 'last' in ticker:
                prices.append({
                    'exchange': name,
                    'price': ticker['last'],
                    'bid': ticker.get('bid', 0),
                    'ask': ticker.get('ask', 0)
                })
        
        if not prices:
            return {}
        
        # Find best bid and ask
        best_bid = max(prices, key=lambda x: x['bid'])
        best_ask = min(prices, key=lambda x: x['ask'])
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'all_prices': prices
        }
    
    async def get_funding_rates(self, symbol: str) -> Dict:
        """Get funding rates from all exchanges"""
        rates = {}
        
        if 'binance' in self.clients:
            rates['binance'] = await self.clients['binance'].get_funding_rate(symbol)
        
        if 'delta' in self.clients:
            rates['delta'] = await self.clients['delta'].get_funding_rate(symbol)
        
        return rates
    
    def get_primary_client(self):
        """Get primary exchange client"""
        return self.clients.get(self.primary)
    
    async def close_all(self):
        """Close all connections"""
        for name, client in self.clients.items():
            if hasattr(client, 'close'):
                await client.close()
