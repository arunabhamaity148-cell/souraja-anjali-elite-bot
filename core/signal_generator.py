"""
Signal Generator - Real Technical Analysis - FIXED (No Repainting)
"""

import logging
import pandas as pd
from typing import Dict, Optional
from core.technical_analysis import TechnicalAnalysis

logger = logging.getLogger("SIGNAL_GEN")

class EliteSignalGenerator:
    """Generate signals using pure technical analysis - FIXED VERSION"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
        
    async def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal based on multiple technical indicators
        NO RANDOMNESS, NO REPAINTING
        """
        try:
            # FIX #2: Remove forming candle to prevent repainting
            if len(df) < 100:
                logger.warning(f"{symbol}: Insufficient data ({len(df)} candles)")
                return None
            
            # CRITICAL FIX: Remove last (forming) candle
            df = df.iloc[:-1].copy()
            
            current_price = float(df['close'].iloc[-1])
            prev_price = float(df['close'].iloc[-2])
            
            # Calculate all indicators
            rsi = self.ta.calculate_rsi(df['close'])
            ema_9 = self.ta.calculate_ema(df['close'], 9).iloc[-1]
            ema_21 = self.ta.calculate_ema(df['close'], 21).iloc[-1]
            ema_50 = self.ta.calculate_ema(df['close'], 50).iloc[-1] if len(df) >= 50 else ema_21
            atr = self.ta.calculate_atr(df)
            bb = self.ta.calculate_bollinger_bands(df['close'])
            macd = self.ta.calculate_macd(df['close'])
            stoch = self.ta.calculate_stochastic(df)
            adx = self.ta.calculate_adx(df)
            sr_levels = self.ta.find_support_resistance(df)
            volume = self.ta.detect_volume_spike(df)
            structure = self.ta.detect_bos_choch(df)
            
            # Skip if ATR is too small (low volatility)
            if atr < current_price * 0.0005:
                logger.debug(f"{symbol}: ATR too small")
                return None
            
            # Scoring system for LONG
            long_score = 0
            long_conditions = []
            
            # RSI oversold bounce
            if rsi < 35:
                long_score += 2
                long_conditions.append('RSI_OVERSOLD')
            elif rsi < 50:
                long_score += 1
            
            # EMA alignment
            if ema_9 > ema_21 > ema_50:
                long_score += 2
                long_conditions.append('EMA_BULLISH')
            elif ema_9 > ema_21:
                long_score += 1
            
            # MACD bullish
            if macd['bullish_cross']:
                long_score += 2
                long_conditions.append('MACD_CROSS')
            elif macd['above_signal'] and macd['histogram'] > 0:
                long_score += 1
            
            # Bollinger Bands
            if bb['percent_b'] < 0.2:
                long_score += 1
                long_conditions.append('BB_OVERSOLD')
            elif bb['is_squeeze'] and current_price > bb['middle']:
                long_score += 1
            
            # Structure
            if structure in ['BOS_UP', 'CHoCH_UP']:
                long_score += 2
                long_conditions.append(f'STRUCTURE_{structure}')
            
            # Stochastic
            if stoch['oversold']:
                long_score += 1
            
            # Trend strength
            if adx['trending'] and adx['bullish']:
                long_score += 1
            
            # Volume confirmation
            if volume['above_average'] and current_price > prev_price:
                long_score += 1
            
            # Support bounce
            if sr_levels['support_distance'] < 0.01:  # Within 1% of support
                long_score += 1
                long_conditions.append('NEAR_SUPPORT')
            
            # Scoring system for SHORT
            short_score = 0
            short_conditions = []
            
            # RSI overbought
            if rsi > 65:
                short_score += 2
                short_conditions.append('RSI_OVERBOUGHT')
            elif rsi > 50:
                short_score += 1
            
            # EMA alignment
            if ema_9 < ema_21 < ema_50:
                short_score += 2
                short_conditions.append('EMA_BEARISH')
            elif ema_9 < ema_21:
                short_score += 1
            
            # MACD bearish
            if macd['bearish_cross']:
                short_score += 2
                short_conditions.append('MACD_CROSS')
            elif not macd['above_signal'] and macd['histogram'] < 0:
                short_score += 1
            
            # Bollinger Bands
            if bb['percent_b'] > 0.8:
                short_score += 1
                short_conditions.append('BB_OVERBOUGHT')
            elif bb['is_squeeze'] and current_price < bb['middle']:
                short_score += 1
            
            # Structure
            if structure in ['BOS_DOWN', 'CHoCH_DOWN']:
                short_score += 2
                short_conditions.append(f'STRUCTURE_{structure}')
            
            # Stochastic
            if stoch['overbought']:
                short_score += 1
            
            # Trend strength
            if adx['trending'] and not adx['bullish']:
                short_score += 1
            
            # Volume confirmation
            if volume['above_average'] and current_price < prev_price:
                short_score += 1
            
            # Resistance rejection
            if sr_levels['resistance_distance'] < 0.01:  # Within 1% of resistance
                short_score += 1
                short_conditions.append('NEAR_RESISTANCE')
            
            # Determine direction (need strong consensus)
            MIN_SCORE = 5
            
            if long_score >= MIN_SCORE and long_score > short_score + 2:
                direction = 'LONG'
                score = long_score
                conditions = long_conditions
            elif short_score >= MIN_SCORE and short_score > long_score + 2:
                direction = 'SHORT'
                score = short_score
                conditions = short_conditions
            else:
                logger.debug(f"{symbol}: No clear signal (L:{long_score}, S:{short_score})")
                return None
            
            # Calculate ATR-based levels
            if direction == 'LONG':
                sl = current_price - (1.5 * atr)
                tp1 = current_price + (2 * atr)
                tp2 = current_price + (3 * atr)
                tp3 = current_price + (4 * atr)
            else:
                sl = current_price + (1.5 * atr)
                tp1 = current_price - (2 * atr)
                tp2 = current_price - (3 * atr)
                tp3 = current_price - (4 * atr)
            
            # Validate R:R
            risk = abs(current_price - sl)
            reward = abs(tp1 - current_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < 1.2:
                logger.debug(f"{symbol}: Poor R:R {rr_ratio:.2f}")
                return None
            
            # Ensure SL is not beyond support/resistance
            if direction == 'LONG':
                if sl < sr_levels['nearest_support'] * 0.99:
                    sl = sr_levels['nearest_support'] * 0.995
            else:
                if sl > sr_levels['nearest_resistance'] * 1.01:
                    sl = sr_levels['nearest_resistance'] * 1.005
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry': round(current_price, 4),
                'sl': round(sl, 4),
                'tp1': round(tp1, 4),
                'tp2': round(tp2, 4),
                'tp3': round(tp3, 4),
                'atr': round(atr, 4),
                'rr_ratio': round(rr_ratio, 2),
                'score': score,
                'conditions': conditions,
                'indicators': {
                    'rsi': round(rsi, 2),
                    'ema_9': round(ema_9, 4),
                    'ema_21': round(ema_21, 4),
                    'macd_hist': round(macd['histogram'], 6),
                    'bb_percent_b': round(bb['percent_b'], 4),
                    'adx': round(adx['adx'], 2),
                    'stoch_k': round(stoch['k'], 2),
                    'volume_ratio': round(volume['ratio'], 2),
                    'structure': structure
                },
                'confidence_score': long_score if direction == 'LONG' else short_score
            }
            
        except Exception as e:
            logger.error(f"Signal generation error {symbol}: {e}")
            return None
