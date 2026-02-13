"""
Multi-Exchange Manager - Primary + Fallback
"""

import logging
import asyncio
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
        """Get OHLCV with fallback and retry"""
        max_retries = 3
        
        for attempt in range(max_retries):
            # Try primary first
            for exchange_name in [self.primary] + [e for e in self.clients if e != self.primary]:
                if exchange_name not in self.clients:
                    continue
                
                try:
                    client = self.clients[exchange_name]
                    df = await client.get_ohlcv(symbol, timeframe, limit)
                    
                    # FIX: Validate data
                    if df is not None and len(df) > 0 and not df.empty:
                        # Check required columns exist
                        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        if all(col in df.columns for col in required_cols):
                            # Check for NaN values
                            if not df[['open', 'high', 'low', 'close', 'volume']].isna().all().all():
                                logger.debug(f"✅ {exchange_name} data for {symbol}")
                                return df
                    
                    logger.warning(f"⚠️ {exchange_name} invalid data for {symbol}, trying fallback")
                    
                except Exception as e:
                    logger.warning(f"⚠️ {exchange_name} error (attempt {attempt+1}): {e}")
                    continue
            
            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
        
        logger.error(f"❌ All exchanges failed for {symbol}")
        return None
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker with fallback and retry"""
        max_retries = 3
        
        for attempt in range(max_retries):
            for exchange_name in [self.primary] + [e for e in self.clients if e != self.primary]:
                if exchange_name not in self.clients:
                    continue
                
                try:
                    ticker = await self.clients[exchange_name].get_ticker(symbol)
                    
                    # FIX: Validate ticker data
                    if ticker and isinstance(ticker, dict):
                        last = ticker.get('last')
                        bid = ticker.get('bid')
                        ask = ticker.get('ask')
                        
                        # Check all required fields exist and are valid numbers
                        if (last is not None and bid is not None and ask is not None and
                            isinstance(last, (int, float)) and isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and
                            last > 0 and bid > 0 and ask > 0):
                            return {**ticker, 'exchange': exchange_name}
                    
                    logger.warning(f"⚠️ {exchange_name} invalid ticker for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ {exchange_name} ticker error (attempt {attempt+1}): {e}")
                    continue
            
            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
        
        logger.error(f"❌ All exchanges failed for {symbol} ticker")
        return {}
    
    async def get_best_price(self, symbol: str) -> Dict:
        """Get best price across exchanges"""
        prices = []
        max_retries = 2
        
        for attempt in range(max_retries):
            for name, client in self.clients.items():
                try:
                    ticker = await client.get_ticker(symbol)
                    
                    # FIX: Validate ticker
                    if ticker and isinstance(ticker, dict):
                        last = ticker.get('last')
                        bid = ticker.get('bid')
                        ask = ticker.get('ask')
                        
                        if (last is not None and bid is not None and ask is not None and
                            isinstance(last, (int, float)) and isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and
                            last > 0 and bid > 0 and ask > 0):
                            prices.append({
                                'exchange': name,
                                'price': float(last),
                                'bid': float(bid),
                                'ask': float(ask)
                            })
                            
                except Exception as e:
                    logger.debug(f"Price check error {name}: {e}")
                    continue
            
            if prices:
                break
            
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
        
        if not prices:
            logger.warning(f"No valid prices for {symbol}")
            return {}
        
        # Find best bid and ask
        try:
            best_bid = max(prices, key=lambda x: x['bid'])
            best_ask = min(prices, key=lambda x: x['ask'])
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'all_prices': prices
            }
        except Exception as e:
            logger.error(f"Error calculating best price: {e}")
            return {}
    
    async def get_funding_rates(self, symbol: str) -> Dict:
        """Get funding rates from all exchanges"""
        rates = {}
        
        if 'binance' in self.clients:
            try:
                rates['binance'] = await self.clients['binance'].get_funding_rate(symbol)
            except Exception as e:
                logger.debug(f"Binance funding error: {e}")
        
        if 'delta' in self.clients:
            try:
                rates['delta'] = await self.clients['delta'].get_funding_rate(symbol)
            except Exception as e:
                logger.debug(f"Delta funding error: {e}")
        
        return rates
    
    def get_primary_client(self):
        """Get primary exchange client"""
        return self.clients.get(self.primary)
    
    async def close_all(self):
        """Close all connections"""
        for name, client in self.clients.items():
            if hasattr(client, 'close'):
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing {name}: {e}")
