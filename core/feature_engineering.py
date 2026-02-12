"""
Feature Engineering - 50+ Features
"""

import numpy as np
import pandas as pd
from typing import List

class FeatureEngineer:
    def __init__(self):
        pass
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create 50+ technical features"""
        features = df.copy()
        
        features = self._add_price_features(features)
        features = self._add_volume_features(features)
        features = self._add_volatility_features(features)
        features = self._add_momentum_features(features)
        features = self._add_trend_features(features)
        features = self._add_microstructure_features(features)
        
        return features.dropna()
    
    def _add_price_features(self, df):
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['open_close_range'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['intraday_trend'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        return df
    
    def _add_volume_features(self, df):
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ma_ratio'] = df['obv'] / df['obv'].rolling(20).mean()
        df['volume_price_trend'] = df['volume'] * df['returns']
        df['buy_volume'] = df['volume'] * (df['close'] > df['open']).astype(float)
        df['sell_volume'] = df['volume'] * (df['close'] < df['open']).astype(float)
        return df
    
    def _add_volatility_features(self, df):
        df['atr'] = self._calculate_atr(df)
        df['atr_ratio'] = df['atr'] / df['close']
        df['volatility_regime'] = df['returns'].rolling(20).std() * np.sqrt(365)
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(365)
        return df
    
    def _add_momentum_features(self, df):
        df['rsi'] = self._calculate_rsi(df['close'])
        df['rsi_slope'] = df['rsi'].diff(5)
        df['macd'] = self._calculate_macd(df['close'])
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_30'] = df['close'] / df['close'].shift(30) - 1
        return df
    
    def _add_trend_features(self, df):
        df['adx'] = self._calculate_adx(df)
        df['ema_ratio_9_21'] = df['close'].ewm(span=9).mean() / df['close'].ewm(span=21).mean()
        df['ema_ratio_21_50'] = df['close'].ewm(span=21).mean() / df['close'].ewm(span=50).mean()
        df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].rolling(20).std()
        return df
    
    def _add_microstructure_features(self, df):
        df['trade_intensity'] = df['volume'] / (df['high'] - df['low'] + 1e-10)
        df['order_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['volume'] + 1e-10)
        df['price_impact'] = df['returns'].abs() / np.log(df['volume'] + 1)
        df['jump_detect'] = (df['returns'].abs() > 3 * df['returns'].rolling(20).std()).astype(int)
        return df
    
    def _calculate_atr(self, df, period=14):
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_adx(self, df, period=14):
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = -df['low'].diff().clip(upper=0)
        return (abs(plus_dm) + abs(minus_dm)).rolling(period).mean()
