"""
ML Engine - Random Forest + Ensemble
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger("ML_ENGINE")

class MLEngine:
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.rf_model = None
        self.scaler = StandardScaler()
        self.sequence_length = 60
        self.prediction_threshold = 0.65
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            rf_path = os.path.join(self.model_path, "rf_model.pkl")
            
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                logger.info("✅ RF Model loaded")
            else:
                self.rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    random_state=42
                )
                logger.info("🆕 New RF Model created")
        except Exception as e:
            logger.error(f"Model load error: {e}")
            self.rf_model = RandomForestClassifier(n_estimators=100)
    
    async def predict_direction(self, features_df: pd.DataFrame) -> Dict:
        """Predict price direction"""
        try:
            if len(features_df) < self.sequence_length:
                return {'direction': 'NEUTRAL', 'confidence': 0.5}
            
            feature_cols = [c for c in features_df.columns 
                          if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            
            X = features_df[feature_cols].fillna(0).values
            X_scaled = self.scaler.fit_transform(X)
            
            if hasattr(self.rf_model, 'classes_'):
                prob = self.rf_model.predict_proba(X_scaled[-1:])[0]
                long_prob = prob[1] if len(prob) > 1 else prob[0]
            else:
                long_prob = 0.5
            
            if long_prob > self.prediction_threshold:
                direction = 'LONG'
                confidence = long_prob
            elif long_prob < (1 - self.prediction_threshold):
                direction = 'SHORT'
                confidence = 1 - long_prob
            else:
                direction = 'NEUTRAL'
                confidence = 0.5
            
            return {
                'direction': direction,
                'confidence': round(confidence, 3),
                'long_probability': round(long_prob, 3),
                'features_used': len(feature_cols)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'direction': 'NEUTRAL', 'confidence': 0.5}
    
    async def predict_volatility(self, features_df: pd.DataFrame) -> Dict:
        """Predict volatility regime"""
        try:
            recent_vol = features_df['realized_vol'].iloc[-1]
            vol_trend = features_df.get('vol_of_vol', pd.Series([0])).iloc[-1]
            
            if recent_vol > 0.8 and vol_trend > 0.1:
                regime = 'HIGH_VOL'
                score = 0.9
            elif recent_vol < 0.3:
                regime = 'LOW_VOL'
                score = 0.9
            else:
                regime = 'MEDIUM_VOL'
                score = 0.7
            
            return {'regime': regime, 'score': score}
            
        except Exception as e:
            return {'regime': 'UNKNOWN', 'score': 0.5}
    
    async def predict_optimal_hold_time(self, features_df: pd.DataFrame) -> int:
        """Predict optimal trade duration"""
        try:
            volatility = features_df.get('volatility_regime', pd.Series([0.5])).iloc[-1]
            trend_strength = features_df.get('trend_strength', pd.Series([1])).iloc[-1]
            
            if volatility > 0.5 and trend_strength > 2:
                return 120
            elif volatility > 0.5:
                return 60
            else:
                return 30
                
        except:
            return 60
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train RF model"""
        logger.info(f"Training RF on {len(X)} samples")
        self.rf_model.fit(X, y)
        
        os.makedirs(self.model_path, exist_ok=True)
        joblib.dump(self.rf_model, os.path.join(self.model_path, "rf_model.pkl"))
        logger.info("✅ Model saved")
