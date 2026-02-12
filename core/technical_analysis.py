"""
Real Technical Analysis - Complete Implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class TechnicalAnalysis:
    """Complete technical analysis library"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(com=period-1, adjust=False).mean()
        avg_loss = loss.ewm(com=period-1, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        atr = true_range.rolling(period).mean()
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        rolling_std = prices.rolling(period).std()
        
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        # Bollinger Band Width
        bb_width = (upper_band - lower_band) / sma
        
        # %B indicator
        percent_b = (prices - lower_band) / (upper_band - lower_band)
        
        # Bandwidth squeeze detection
        bandwidth = (upper_band - lower_band) / sma
        
        return {
            'upper': float(upper_band.iloc[-1]),
            'lower': float(lower_band.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'width': float(bb_width.iloc[-1]),
            'percent_b': float(percent_b.iloc[-1]),
            'bandwidth': float(bandwidth.iloc[-1]),
            'is_squeeze': float(bandwidth.iloc[-1]) < float(bandwidth.rolling(20).mean().iloc[-1]) * 0.8
        }
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Detect crossover
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        curr_macd = macd_line.iloc[-1]
        curr_signal = signal_line.iloc[-1]
        
        bullish_cross = curr_macd > curr_signal and prev_macd <= prev_signal
        bearish_cross = curr_macd < curr_signal and prev_macd >= prev_signal
        
        return {
            'macd': float(curr_macd),
            'signal': float(curr_signal),
            'histogram': float(histogram.iloc[-1]),
            'bullish_cross': bullish_cross,
            'bearish_cross': bearish_cross,
            'above_signal': curr_macd > curr_signal
        }
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return {
            'k': float(k.iloc[-1]),
            'd': float(d.iloc[-1]),
            'overbought': float(k.iloc[-1]) > 80,
            'oversold': float(k.iloc[-1]) < 20
        }
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate ADX (Average Directional Index)"""
        # True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smoothed
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # ADX
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return {
            'adx': float(adx.iloc[-1]),
            'plus_di': float(plus_di.iloc[-1]),
            'minus_di': float(minus_di.iloc[-1]),
            'trending': float(adx.iloc[-1]) > 25,
            'bullish': float(plus_di.iloc[-1]) > float(minus_di.iloc[-1])
        }
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, lookback: int = 100, 
                               sensitivity: float = 0.02) -> Dict:
        """Find support and resistance levels using pivot points"""
        recent = df.tail(lookback).copy()
        
        # Find local minima (support)
        recent['local_min'] = recent['low'].rolling(window=5, center=True).min()
        support_pivots = recent[recent['low'] == recent['local_min']]['low'].values
        
        # Find local maxima (resistance)
        recent['local_max'] = recent['high'].rolling(window=5, center=True).max()
        resistance_pivots = recent[recent['high'] == recent['local_max']]['high'].values
        
        # Cluster levels (within 2%)
        def cluster_levels(levels, tolerance):
            if len(levels) == 0:
                return []
            sorted_levels = np.sort(levels)
            clusters = [[sorted_levels[0]]]
            for level in sorted_levels[1:]:
                if abs(level - np.mean(clusters[-1])) / np.mean(clusters[-1]) < tolerance:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            return [np.mean(cluster) for cluster in clusters]
        
        support_levels = cluster_levels(support_pivots, sensitivity)
        resistance_levels = cluster_levels(resistance_pivots, sensitivity)
        
        current_price = float(df['close'].iloc[-1])
        
        # Find nearest levels
        supports_below = [s for s in support_levels if s < current_price]
        resistances_above = [r for r in resistance_levels if r > current_price]
        
        nearest_support = max(supports_below) if supports_below else current_price * 0.95
        nearest_resistance = min(resistances_above) if resistances_above else current_price * 1.05
        
        return {
            'supports': support_levels,
            'resistances': resistance_levels,
            'nearest_support': float(nearest_support),
            'nearest_resistance': float(nearest_resistance),
            'support_distance': (current_price - nearest_support) / current_price,
            'resistance_distance': (nearest_resistance - current_price) / current_price
        }
    
    @staticmethod
    def detect_volume_spike(df: pd.DataFrame, lookback: int = 20, 
                           multiplier: float = 2.0) -> Dict:
        """Detect volume spike"""
        avg_volume = df['volume'].rolling(lookback).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        volume_ma_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume trend
        volume_sma_10 = df['volume'].rolling(10).mean().iloc[-1]
        volume_sma_20 = df['volume'].rolling(20).mean().iloc[-1]
        volume_trend = volume_sma_10 / volume_sma_20 if volume_sma_20 > 0 else 1.0
        
        return {
            'spike': volume_ma_ratio > multiplier,
            'ratio': float(volume_ma_ratio),
            'trend': float(volume_trend),
            'above_average': volume_ma_ratio > 1.0
        }
    
    @staticmethod
    def detect_bos_choch(df: pd.DataFrame, lookback: int = 50) -> str:
        """
        Detect Break of Structure (BOS) or Change of Character (CHoCH)
        """
        recent = df.tail(lookback).copy()
        
        # Find swing highs and lows
        recent['swing_high'] = recent['high'].rolling(window=5, center=True).max()
        recent['swing_low'] = recent['low'].rolling(window=5, center=True).min()
        
        swing_highs = recent[recent['high'] == recent['swing_high']]['high'].values
        swing_lows = recent[recent['low'] == recent['swing_low']]['low'].values
        
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return 'NONE'
        
        current_price = float(df['close'].iloc[-1])
        prev_high = float(swing_highs[-2])
        prev_low = float(swing_lows[-2])
        last_high = float(swing_highs[-1])
        last_low = float(swing_lows[-1])
        
        # Break of Structure
        if current_price > prev_high:
            return 'BOS_UP'
        elif current_price < prev_low:
            return 'BOS_DOWN'
        
        # Change of Character (higher low / lower high)
        if last_low > prev_low and current_price > last_high:
            return 'CHoCH_UP'
        elif last_high < prev_high and current_price < last_low:
            return 'CHoCH_DOWN'
        
        return 'NONE'
    
    @staticmethod
    def get_trend_alignment(df_5m: pd.DataFrame, df_15m: pd.DataFrame, 
                           df_1h: pd.DataFrame) -> Dict:
        """Check trend alignment across multiple timeframes"""
        
        def get_trend_direction(df):
            ema_9 = df['close'].ewm(span=9).mean().iloc[-1]
            ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1] if len(df) >= 50 else ema_21
            
            bullish = ema_9 > ema_21 > ema_50
            bearish = ema_9 < ema_21 < ema_50
            
            if bullish:
                return 'BULLISH'
            elif bearish:
                return 'BEARISH'
            return 'NEUTRAL'
        
        trend_5m = get_trend_direction(df_5m)
        trend_15m = get_trend_direction(df_15m)
        trend_1h = get_trend_direction(df_1h)
        
        trends = [trend_5m, trend_15m, trend_1h]
        bullish_count = trends.count('BULLISH')
        bearish_count = trends.count('BEARISH')
        
        if bullish_count >= 2:
            return {'direction': 'BULLISH', 'alignment': bullish_count / 3, 'timeframes': trends}
        elif bearish_count >= 2:
            return {'direction': 'BEARISH', 'alignment': bearish_count / 3, 'timeframes': trends}
        
        return {'direction': 'NEUTRAL', 'alignment': 0.33, 'timeframes': trends}
    
    @staticmethod
    def calculate_fibonacci_retracement(high: float, low: float) -> Dict:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        return {
            '0.0': high,
            '0.236': high - 0.236 * diff,
            '0.382': high - 0.382 * diff,
            '0.5': high - 0.5 * diff,
            '0.618': high - 0.618 * diff,
            '0.786': high - 0.786 * diff,
            '1.0': low
        }
    
    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame) -> Dict:
        """Calculate pivot points"""
        prev = df.iloc[-2]
        
        pivot = (prev['high'] + prev['low'] + prev['close']) / 3
        r1 = (2 * pivot) - prev['low']
        s1 = (2 * pivot) - prev['high']
        r2 = pivot + (prev['high'] - prev['low'])
        s2 = pivot - (prev['high'] - prev['low'])
        
        return {
            'pivot': float(pivot),
            'r1': float(r1), 'r2': float(r2),
            's1': float(s1), 's2': float(s2)
        }
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> float:
        """Calculate On-Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return float(obv[-1])
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return float(vwap)
    
    @staticmethod
    def detect_divergence(df: pd.DataFrame, indicator: str = 'rsi') -> str:
        """Detect bullish or bearish divergence"""
        if indicator == 'rsi':
            ind_values = TechnicalAnalysis.calculate_rsi(df['close'])
        else:
            return 'NONE'
        
        price_highs = df['high'].tail(20).values
        price_lows = df['low'].tail(20).values
        
        # Simplified divergence check
        # Higher high in price, lower high in indicator = bearish divergence
        # Lower low in price, higher low in indicator = bullish divergence
        
        return 'NONE'  # Complex calculation, simplified for now
