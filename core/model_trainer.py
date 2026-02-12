"""
Model Trainer - Daily Retraining - FIXED (No Data Leakage)
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict
import pandas as pd
from core.ml_engine import MLEngine

logger = logging.getLogger("TRAINER")

class ModelTrainer:
    """Handle daily model retraining - FIXED VERSION"""
    
    def __init__(self):
        self.ml_engine = MLEngine()
        self.last_train_time = None
        
    async def train_daily(self, bot_instance):
        """Train model on last 6 months of data - NO FUTURE DATA"""
        try:
            logger.info("=" * 50)
            logger.info("🎓 DAILY MODEL TRAINING - FIXED")
            logger.info("=" * 50)
            
            # FIX #1: Use yesterday's date, not today (no future data)
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            start_date = end_date - timedelta(days=180)  # 6 months
            
            logger.info(f"Training period: {start_date.date()} to {end_date.date()}")
            
            # Fetch 6 months of data for all pairs
            historical_data = {}
            
            for symbol in bot_instance.symbols:
                try:
                    # Fetch from exchange
                    df = await self._fetch_historical_data(
                        bot_instance.exchange_mgr, 
                        symbol, 
                        start_date, 
                        end_date
                    )
                    
                    if df is not None and len(df) > 200:
                        historical_data[symbol] = df
                        logger.info(f"✅ {symbol}: {len(df)} candles")
                    else:
                        logger.warning(f"⚠️ {symbol}: Insufficient data")
                    
                    # Rate limit
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Fetch error {symbol}: {e}")
                    continue
            
            if not historical_data:
                logger.error("❌ No historical data available for training")
                return False
            
            # Train model
            success = self.ml_engine.train(historical_data)
            
            if success:
                self.last_train_time = datetime.now()
                logger.info("✅ Daily training completed successfully")
                
                # Log feature importance
                importance = self.ml_engine.get_feature_importance()
                logger.info("Top 5 important features:")
                for feat, imp in list(importance.items())[:5]:
                    logger.info(f"  {feat}: {imp:.3f}")
            else:
                logger.error("❌ Training failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Daily training error: {e}")
            return False
    
    async def _fetch_historical_data(self, exchange_mgr, symbol: str, 
                                    start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            if not exchange_mgr:
                return None
            
            # Calculate required candles
            days = (end - start).days
            candles_needed = days * 24 * 12  # 5-min candles per day
            
            # Fetch from exchange
            df = await exchange_mgr.get_ohlcv(
                symbol, 
                timeframe='5m', 
                limit=min(candles_needed, 1000)
            )
            
            # Filter to date range (extra safety)
            if df is not None and 'timestamp' in df.columns:
                df = df[df['timestamp'] <= end]
            
            return df
            
        except Exception as e:
            logger.error(f"Historical data fetch error: {e}")
            return False
