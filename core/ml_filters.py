"""
ML-Based Filters - Filter 9 & 10
"""

import logging
from typing import Dict
from core.ml_engine import MLEngine
from core.feature_engineering import FeatureEngineer

logger = logging.getLogger("ML_FILTERS")

class MLFilterManager:
    def __init__(self):
        self.ml_engine = MLEngine()
        self.feature_eng = FeatureEngineer()
        
    async def ml_direction_filter(self, symbol: str, signal: Dict, features_df) -> bool:
        """Filter 9: ML Direction Prediction"""
        try:
            prediction = await self.ml_engine.predict_direction(features_df)
            
            if prediction['direction'] == 'NEUTRAL':
                return False
            
            if prediction['direction'] != signal['direction']:
                return False
            
            return prediction['confidence'] >= 0.65
            
        except Exception as e:
            logger.error(f"ML filter error: {e}")
            return True
    
    async def ml_volatility_filter(self, symbol: str, signal: Dict, features_df) -> bool:
        """Filter 10: ML Volatility Prediction"""
        try:
            vol_pred = await self.ml_engine.predict_volatility(features_df)
            
            if signal.get('tier') == 'TIER_1' and vol_pred['regime'] == 'HIGH_VOL':
                return False
            
            return True
            
        except Exception as e:
            return True
    
    async def ml_optimal_entry(self, symbol: str, features_df) -> Dict:
        """Filter 11: Optimal entry timing"""
        try:
            hold_time = await self.ml_engine.predict_optimal_hold_time(features_df)
            
            return {
                'optimal_hold_minutes': hold_time,
                'entry_score': 0.8
            }
            
        except:
            return {'optimal_hold_minutes': 60, 'entry_score': 0.5}
    
    async def ensemble_prediction(self, symbol: str, features_df) -> Dict:
        """Combined ML prediction"""
        direction = await self.ml_engine.predict_direction(features_df)
        volatility = await self.ml_engine.predict_volatility(features_df)
        optimal = await self.ml_optimal_entry(symbol, features_df)
        
        score = (
            direction['confidence'] * 0.5 +
            volatility['score'] * 0.3 +
            optimal['entry_score'] * 0.2
        )
        
        return {
            'ensemble_score': round(score, 3),
            'direction': direction['direction'],
            'confidence': direction['confidence'],
            'volatility_regime': volatility['regime'],
            'hold_time': optimal['optimal_hold_minutes']
        }
