"""
Exchange Manager - FIXED Ticker Validation
Multi-exchange support with proper fallback
"""

import logging
import asyncio
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger("EXCHANGE_MGR")

class ExchangeManager:
    def __init__(self, config: Dict):
        self.clients = {}
        self.primary = None
        
        # Initialize Binance
        if config.get('binance_api_key'):
            from exchanges.binance_client import BinanceClient
            self.clients['binance'] = BinanceClient(
                config['binance_api_key'],
                config['binance_api_secret'],
                testnet=config.get('binance_testnet', False)
            )
            if not self.primary:
                self.primary = 'binance'
        
        # Initialize Delta
        if config.get('delta_api_key'):
            from exchanges.delta_client import DeltaClient
            self.clients['delta'] = DeltaClient(
                config['delta_api_key'],
                config['delta_api_secret']
            )
            if not self.primary:
                self.primary = 'delta'
        
        # Initialize CoinDCX
        if config.get('coindcx_api_key'):
            from exchanges.coindcx_client import CoinDCXClient
            self.clients['coindcx'] = CoinDCXClient(
                config['coindcx_api_key'],
                config['coindcx_api_secret']
            )
            if not self.primary:
                self.primary = 'coindcx'
        
        if self.clients:
            logger.info(f"✅ Exchange Manager: {list(self.clients.keys())}")
        else:
            logger.error("❌ No exchanges configured")
    
    def get_primary_client(self):
        """Get primary exchange client"""
        return self.clients.get(self.primary)
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV with fallback and retry"""
        max_retries = 3
        
        for attempt in range(max_retries):
            # Try primary first, then fallback
            for exchange_name in [self.primary] + [e for e in self.clients if e != self.primary]:
                if exchange_name not in self.clients:
                    continue
                
                try:
                    df = await self.clients[exchange_name].get_ohlcv(symbol, timeframe, limit)
                    
                    if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
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
        """Get ticker with RELAXED validation"""
        max_retries = 3
        
        for attempt in range(max_retries):
            for exchange_name in [self.primary] + [e for e in self.clients if e != self.primary]:
                if exchange_name not in self.clients:
                    continue
                
                try:
                    ticker = await self.clients[exchange_name].get_ticker(symbol)
                    
                    # ✅ RELAXED validation - only check 'last' price
                    if ticker and isinstance(ticker, dict):
                        last = ticker.get('last', 0)
                        
                        # Only require 'last' price to be valid
                        if last and isinstance(last, (int, float)) and last > 0:
                            # Fill in missing bid/ask with last price if needed
                            if not ticker.get('bid') or ticker.get('bid', 0) <= 0:
                                ticker['bid'] = last * 0.9999  # Slightly below last
                            if not ticker.get('ask') or ticker.get('ask', 0) <= 0:
                                ticker['ask'] = last * 1.0001  # Slightly above last
                            
                            logger.debug(f"✅ {exchange_name} ticker for {symbol}: {last}")
                            return {**ticker, 'exchange': exchange_name}
                    
                    logger.debug(f"⚠️ {exchange_name} invalid ticker for {symbol}")
                    
                except Exception as e:
                    logger.debug(f"⚠️ {exchange_name} ticker error: {e}")
                    continue
            
            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
        
        logger.warning(f"❌ All exchanges failed for {symbol} ticker")
        return {}
    
    async def get_best_price(self, symbol: str) -> Dict:
        """Get best bid/ask"""
        for exchange_name in [self.primary] + [e for e in self.clients if e != self.primary]:
            if exchange_name not in self.clients:
                continue
            
            try:
                prices = await self.clients[exchange_name].get_best_price(symbol)
                if prices:
                    return prices
            except Exception as e:
                logger.debug(f"{exchange_name} best price error: {e}")
                continue
        
        logger.warning(f"No valid prices for {symbol}")
        return {}
    
    async def get_funding_rates(self, symbol: str) -> Dict:
        """Get funding rates from all exchanges"""
        rates = {}
        
        for exchange_name, client in self.clients.items():
            try:
                rate = await client.get_funding_rate(symbol)
                if rate is not None:
                    rates[exchange_name] = rate
            except Exception as e:
                logger.debug(f"{exchange_name} funding error: {e}")
                continue
        
        return rates
