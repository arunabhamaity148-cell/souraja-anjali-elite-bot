"""
Auto Model Trainer
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from core.ml_engine import MLEngine
from core.feature_engineering import FeatureEngineer

logger = logging.getLogger("TRAINER")

class ModelTrainer:
    def __init__(self):
        self.ml_engine = MLEngine()
        self.feature_eng = FeatureEngineer()
        
    async def prepare_training_data(self, historical_df: pd.DataFrame) -> tuple:
        """Prepare labeled dataset"""
        features = self.feature_eng.create_features(historical_df)
        
        features['future_return'] = features['close'].shift(-5) / features['close'] - 1
        features['label'] = np.where(features['future_return'] > 0.005, 1,
                                   np.where(features['future_return'] < -0.005, 0, 2))
        
        features = features[features['label'] != 2]
        
        feature_cols = [c for c in features.columns if c not in 
                       ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                        'future_return', 'label']]
        
        X = features[feature_cols].fillna(0).values
        y = features['label'].values
        
        return X, y, feature_cols
    
    async def train_daily(self, data_fetcher):
        """Daily retraining"""
        try:
            logger.info("🚀 Starting daily model training")
            
            # Use last 30 days data
            all_data = []
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                df = await data_fetcher.get_ohlcv_data(symbol)
                if df is not None:
                    all_data.append(df)
            
            if not all_data:
                logger.warning("No data for training")
                return
            
            combined = pd.concat(all_data, ignore_index=True)
            
            X, y, features = await self.prepare_training_data(combined)
            
            if len(X) > 500:
                self.ml_engine.train(X, y)
                logger.info(f"✅ Trained on {len(X)} samples")
            else:
                logger.warning(f"Insufficient data: {len(X)} samples")
                
        except Exception as e:
            logger.error(f"Training error: {e}")
