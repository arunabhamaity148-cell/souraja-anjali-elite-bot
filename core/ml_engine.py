"""
ML Engine - Real Training on Historical Data
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

logger = logging.getLogger("ML_ENGINE")

class MLEngine:
    """Real ML with training capability"""
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.rf_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        os.makedirs(model_path, exist_ok=True)
        self._load_or_init_model()
    
    def _load_or_init_model(self):
        """Load existing model or initialize new"""
        model_file = os.path.join(self.model_path, "rf_model.pkl")
        scaler_file = os.path.join(self.model_path, "scaler.pkl")
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            try:
                self.rf_model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                self.is_trained = True
                logger.info("✅ Loaded trained model")
            except Exception as e:
                logger.error(f"Model load error: {e}")
                self._init_new_model()
        else:
            self._init_new_model()
    
    def _init_new_model(self):
        """Initialize new untrained model"""
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        logger.info("🆕 Initialized new model")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare ML features from OHLCV data"""
        features = pd.DataFrame()
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # EMA ratios
        ema_9 = df['close'].ewm(span=9).mean()
        ema_21 = df['close'].ewm(span=21).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        features['ema_9_21_ratio'] = ema_9 / ema_21
        features['ema_21_50_ratio'] = ema_21 / ema_50
        
        # Volume
        features['volume_ma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        features['atr'] = true_range.rolling(14).mean()
        features['atr_ratio'] = features['atr'] / df['close']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Price position
        features['price_vs_high'] = df['close'] / df['high'].rolling(20).max()
        features['price_vs_low'] = df['close'] / df['low'].rolling(20).min()
        
        return features.dropna()
    
    def create_target(self, df: pd.DataFrame, lookahead: int = 5) -> pd.Series:
        """Create target variable: next 5 candle direction"""
        future_return = df['close'].shift(-lookahead) / df['close'] - 1
        
        # 1 = Up > 0.5%, 0 = Down > 0.5%, exclude small moves
        target = pd.Series(2, index=df.index)  # 2 = neutral
        target[future_return > 0.005] = 1
        target[future_return < -0.005] = 0
        
        return target
    
    def train(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Train model on historical data
        historical_data: dict of {symbol: df}
        """
        try:
            logger.info("🎓 Starting model training...")
            
            all_features = []
            all_targets = []
            
            for symbol, df in historical_data.items():
                if len(df) < 200:
                    continue
                
                features = self.prepare_features(df)
                target = self.create_target(df)
                
                # Align indices
                aligned = pd.concat([features, target], axis=1).dropna()
                aligned = aligned[aligned.iloc[:, -1] != 2]  # Remove neutral
                
                if len(aligned) < 100:
                    continue
                
                X = aligned.iloc[:, :-1]
                y = aligned.iloc[:, -1]
                
                all_features.append(X)
                all_targets.append(y)
                
                logger.info(f"  {symbol}: {len(X)} samples")
            
            if not all_features:
                logger.error("No training data available")
                return False
            
            # Combine all data
            X = pd.concat(all_features, ignore_index=True)
            y = pd.concat(all_targets, ignore_index=True)
            
            if len(X) < 1000:
                logger.error(f"Insufficient training data: {len(X)} samples")
                return False
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train
            logger.info(f"Training on {len(X_train)} samples...")
            self.rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.rf_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"✅ Training complete!")
            logger.info(f"   Accuracy: {accuracy:.2%}")
            logger.info(f"   Test samples: {len(y_test)}")
            
            # Save
            joblib.dump(self.rf_model, os.path.join(self.model_path, "rf_model.pkl"))
            joblib.dump(self.scaler, os.path.join(self.model_path, "scaler.pkl"))
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    async def predict(self, features_df: pd.DataFrame) -> Dict:
        """Make prediction on new data"""
        try:
            if not self.is_trained:
                return {'direction': 'NEUTRAL', 'confidence': 0.5, 'prob_up': 0.5}
            
            # Prepare features same as training
            features = self.prepare_features(features_df)
            
            if len(features) < 1:
                return {'direction': 'NEUTRAL', 'confidence': 0.5, 'prob_up': 0.5}
            
            X = features.iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prob = self.rf_model.predict_proba(X_scaled)[0]
            prob_up = prob[1] if len(prob) > 1 else 0.5
            
            # Determine direction
            if prob_up > 0.65:
                direction = 'LONG'
                confidence = prob_up
            elif prob_up < 0.35:
                direction = 'SHORT'
                confidence = 1 - prob_up
            else:
                direction = 'NEUTRAL'
                confidence = 0.5
            
            return {
                'direction': direction,
                'confidence': round(confidence, 3),
                'prob_up': round(prob_up, 3)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'direction': 'NEUTRAL', 'confidence': 0.5, 'prob_up': 0.5}
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance"""
        if not self.is_trained:
            return {}
        
        importance = self.rf_model.feature_importances_
        feature_names = [
            'returns', 'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'ema_9_21_ratio', 'ema_21_50_ratio', 'volume_ma', 'volume_ratio',
            'atr', 'atr_ratio', 'bb_upper', 'bb_lower', 'bb_position',
            'price_vs_high', 'price_vs_low'
        ]
        
        return dict(sorted(zip(feature_names, importance), 
                          key=lambda x: x[1], reverse=True))
