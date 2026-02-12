"""
Real Signal Generator - No Random, Pure TA
"""

import logging
import pandas as pd
from typing import Dict, Optional, Tuple
from core.technical_analysis import TechnicalAnalysis

logger = logging.getLogger("SIGNAL_GEN")

class EliteSignalGenerator:
    """Generate signals based on real technical analysis"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
        
    async def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate signal based on technical analysis
        Returns None if no valid setup
        """
        try:
            if len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} candles")
                return None
            
            # Calculate indicators
            rsi = self.ta.calculate_rsi(df['close'])
            ema_9 = self.ta.calculate_ema(df['close'], 9)
            ema_21 = self.ta.calculate_ema(df['close'], 21)
            atr = self.ta.calculate_atr(df)
            bb = self.ta.calculate_bollinger_bands(df['close'])
            macd = self.ta.calculate_macd(df['close'])
            sr_levels = self.ta.find_support_resistance(df)
            volume_spike = self.ta.detect_volume_spike(df)
            structure = self.ta.detect_bos_choch(df)
            
            current_price = df['close'].iloc[-1]
            
            # Determine direction based on multiple factors
            long_signals = 0
            short_signals = 0
            
            # RSI check
            if rsi < 30:
                long_signals += 2
            elif rsi > 70:
                short_signals += 2
            
            # EMA crossover
            if ema_9 > ema_21:
                long_signals += 1
            else:
                short_signals += 1
            
            # MACD
            if macd['histogram'] > 0:
                long_signals += 1
            else:
                short_signals += 1
            
            # Bollinger Bands
            if bb['percent_b'] < 0.2:
                long_signals += 1
            elif bb['percent_b'] > 0.8:
                short_signals += 1
            
            # Structure
            if structure in ['BOS_UP', 'CHoCH_UP']:
                long_signals += 2
            elif structure in ['BOS_DOWN', 'CHoCH_DOWN']:
                short_signals += 2
            
            # Volume confirmation
            if volume_spike:
                if current_price > df['open'].iloc[-1]:
                    long_signals += 1
                else:
                    short_signals += 1
            
            # Need strong consensus
            if long_signals >= 5 and long_signals > short_signals + 2:
                direction = 'LONG'
            elif short_signals >= 5 and short_signals > long_signals + 2:
                direction = 'SHORT'
            else:
                return None  # No clear signal
            
            # Calculate entry, SL, TP based on ATR
            entry = current_price
            
            if direction == 'LONG':
                sl = entry - (1.5 * atr)
                tp1 = entry + (2 * atr)
                tp2 = entry + (3 * atr)
                tp3 = entry + (4 * atr)
            else:
                sl = entry + (1.5 * atr)
                tp1 = entry - (2 * atr)
                tp2 = entry - (3 * atr)
                tp3 = entry - (4 * atr)
            
            # Validate R:R
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < 1.2:
                logger.debug(f"Poor R:R for {symbol}: {rr_ratio}")
                return None
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp1': round(tp1, 4),
                'tp2': round(tp2, 4),
                'tp3': round(tp3, 4),
                'atr': round(atr, 4),
                'rr_ratio': round(rr_ratio, 2),
                'indicators': {
                    'rsi': round(rsi, 2),
                    'ema_9': round(ema_9, 4),
                    'ema_21': round(ema_21, 4),
                    'macd_hist': round(macd['histogram'], 6),
                    'bb_width': round(bb['width'], 4),
                    'structure': structure,
                    'volume_spike': volume_spike
                },
                'confidence_score': long_signals if direction == 'LONG' else short_signals
            }
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None
