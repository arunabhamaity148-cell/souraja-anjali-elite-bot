"""
ML Engine - Complete Implementation with Real Training
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

logger = logging.getLogger("ML_ENGINE")

class MLEngine:
    """Complete ML engine with training and prediction"""
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.rf_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
        os.makedirs(model_path, exist_ok=True)
        self._load_or_init_model()
    
    def _load_or_init_model(self):
        """Load existing model or initialize new"""
        model_file = os.path.join(self.model_path, "rf_model.pkl")
        scaler_file = os.path.join(self.model_path, "scaler.pkl")
        features_file = os.path.join(self.model_path, "features.pkl")
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            try:
                self.rf_model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                if os.path.exists(features_file):
                    self.feature_names = joblib.load(features_file)
                self.is_trained = True
                logger.info("✅ Loaded trained model from disk")
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
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        logger.info("🆕 Initialized new untrained model")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML"""
        features = pd.DataFrame(index=df.index)
        
        # Returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_slope'] = features['rsi'].diff(3)
        
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
        features['ema_9_21'] = ema_9 / ema_21
        features['ema_21_50'] = ema_21 / ema_50
        
        # Volume
        features['volume_ma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        features['volume_trend'] = features['volume_ma'].diff(5)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        features['atr_ratio'] = features['atr'] / df['close']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20
        
        # Price position
        features['price_vs_high'] = df['close'] / df['high'].rolling(20).max()
        features['price_vs_low'] = df['close'] / df['low'].rolling(20).min()
        
        self.feature_names = list(features.columns)
        
        return features.dropna()
    
    def create_target(self, df: pd.DataFrame, lookahead: int = 5) -> pd.Series:
        """Create target: next 5 candle direction"""
        future_return = df['close'].shift(-lookahead) / df['close'] - 1
        
        target = pd.Series(2, index=df.index)  # 2 = neutral
        
        # Strong moves only (>0.5%)
        target[future_return > 0.005] = 1   # Up
        target[future_return < -0.005] = 0  # Down
        
        return target
    
    def train(self, historical_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Train model on historical data
        """
        try:
            logger.info("=" * 50)
            logger.info("🎓 TRAINING MODEL")
            logger.info("=" * 50)
            
            all_X = []
            all_y = []
            
            for symbol, df in historical_data.items():
                if len(df) < 200:
                    logger.warning(f"{symbol}: Insufficient data ({len(df)})")
                    continue
                
                # Prepare features
                features = self.prepare_features(df)
                target = self.create_target(df)
                
                # Align and combine
                combined = pd.concat([features, target], axis=1).dropna()
                combined = combined[combined.iloc[:, -1] != 2]  # Remove neutral
                
                if len(combined) < 100:
                    logger.warning(f"{symbol}: Not enough labeled data")
                    continue
                
                X = combined.iloc[:, :-1]
                y = combined.iloc[:, -1]
                
                all_X.append(X)
                all_y.append(y)
                
                logger.info(f"  {symbol}: {len(X)} samples")
            
            if not all_X:
                logger.error("No training data available")
                return False
            
            # Combine all data
            X = pd.concat(all_X, ignore_index=True)
            y = pd.concat(all_y, ignore_index=True)
            
            logger.info(f"Total samples: {len(X)}")
            logger.info(f"Features: {len(X.columns)}")
            logger.info(f"Class distribution: {y.value_counts().to_dict()}")
            
            if len(X) < 1000:
                logger.error(f"Insufficient total samples: {len(X)}")
                return False
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train
            logger.info("Training Random Forest...")
            self.rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.rf_model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            logger.info("✅ Training complete!")
            logger.info(f"   Accuracy:  {accuracy:.2%}")
            logger.info(f"   Precision: {precision:.2%}")
            logger.info(f"   Recall:    {recall:.2%}")
            
            # Feature importance
            importance = dict(zip(
                self.feature_names,
                self.rf_model.feature_importances_
            ))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("Top 5 features:")
            for feat, imp in top_features:
                logger.info(f"   {feat}: {imp:.3f}")
            
            # Save
            joblib.dump(self.rf_model, os.path.join(self.model_path, "rf_model.pkl"))
            joblib.dump(self.scaler, os.path.join(self.model_path, "scaler.pkl"))
            joblib.dump(self.feature_names, os.path.join(self.model_path, "features.pkl"))
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    async def predict(self, df: pd.DataFrame) -> Dict:
        """Make prediction"""
        try:
            if not self.is_trained:
                return {'direction': 'NEUTRAL', 'confidence': 0.5, 'prob_up': 0.5}
            
            features = self.prepare_features(df)
            
            if len(features) < 1:
                return {'direction': 'NEUTRAL', 'confidence': 0.5, 'prob_up': 0.5}
            
            X = features.iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            prob = self.rf_model.predict_proba(X_scaled)[0]
            prob_up = prob[1] if len(prob) > 1 else 0.5
            
            if prob_up > 0.65:
                direction = 'LONG'
                confidence = prob_up
            elif prob_up < 0.35:
                direction = 'SHORT'
                confidence = 1 - prob_up
            else:
                direction = 'NEUTRAL'
                confidence = max(prob_up, 1 - prob_up)
            
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
        if not self.is_trained or not self.feature_names:
            return {}
        
        return dict(sorted(zip(
            self.feature_names,
            self.rf_model.feature_importances_
        ), key=lambda x: x[1], reverse=True))
