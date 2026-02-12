"""
Real Technical Analysis - No Random Data
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class TechnicalAnalysis:
    """Pure technical analysis, no randomness"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_ema(prices: pd.Series, span: int) -> float:
        """Calculate EMA"""
        return prices.ewm(span=span, adjust=False).mean().iloc[-1]
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        rolling_std = prices.rolling(period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        width = (upper - lower) / sma
        
        return {
            'upper': upper.iloc[-1],
            'lower': lower.iloc[-1],
            'middle': sma.iloc[-1],
            'width': width.iloc[-1],
            'percent_b': (prices.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        }
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1],
            'crossover': macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
        }
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, lookback: int = 100) -> Dict:
        """Find support and resistance levels"""
        recent = df.tail(lookback)
        
        # Local minima (support)
        lows = recent['low'].rolling(window=5, center=True).min()
        support_levels = recent['low'][recent['low'] == lows].tail(3).values
        
        # Local maxima (resistance)
        highs = recent['high'].rolling(window=5, center=True).max()
        resistance_levels = recent['high'][recent['high'] == highs].tail(3).values
        
        current_price = df['close'].iloc[-1]
        
        # Nearest levels
        support_below = [s for s in support_levels if s < current_price]
        resistance_above = [r for r in resistance_levels if r > current_price]
        
        nearest_support = max(support_below) if support_below else current_price * 0.98
        nearest_resistance = min(resistance_above) if resistance_above else current_price * 1.02
        
        return {
            'supports': support_levels,
            'resistances': resistance_levels,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }
    
    @staticmethod
    def detect_volume_spike(df: pd.DataFrame, multiplier: float = 2.0) -> bool:
        """Detect volume spike"""
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        return current_volume > (avg_volume * multiplier)
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX"""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0
    
    @staticmethod
    def detect_bos_choch(df: pd.DataFrame) -> str:
        """Detect Break of Structure or Change of Character"""
        # Get swing highs and lows
        highs = df['high'].rolling(window=5, center=True).max()
        lows = df['low'].rolling(window=5, center=True).min()
        
        swing_highs = df[df['high'] == highs]['high'].tail(3).values
        swing_lows = df[df['low'] == lows]['low'].tail(3).values
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'NONE'
        
        current_price = df['close'].iloc[-1]
        prev_high = swing_highs[-2]
        prev_low = swing_lows[-2]
        
        # Break of Structure
        if current_price > prev_high:
            return 'BOS_UP'
        elif current_price < prev_low:
            return 'BOS_DOWN'
        
        # Change of Character
        last_high = swing_highs[-1]
        last_low = swing_lows[-1]
        
        if last_low > prev_low and current_price > last_high:
            return 'CHoCH_UP'
        elif last_high < prev_high and current_price < last_low:
            return 'CHoCH_DOWN'
        
        return 'NONE'
    
    @staticmethod
    def get_trend_alignment(df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> Dict:
        """Check trend alignment across timeframes"""
        ema_5m = df_5m['close'].ewm(span=9).mean().iloc[-1] > df_5m['close'].ewm(span=21).mean().iloc[-1]
        ema_15m = df_15m['close'].ewm(span=9).mean().iloc[-1] > df_15m['close'].ewm(span=21).mean().iloc[-1]
        ema_1h = df_1h['close'].ewm(span=9).mean().iloc[-1] > df_1h['close'].ewm(span=21).mean().iloc[-1]
        
        bullish_count = sum([ema_5m, ema_15m, ema_1h])
        
        if bullish_count >= 2:
            return {'direction': 'BULLISH', 'alignment': bullish_count / 3}
        elif bullish_count <= 1:
            return {'direction': 'BEARISH', 'alignment': (3 - bullish_count) / 3}
        
        return {'direction': 'NEUTRAL', 'alignment': 0.5}
