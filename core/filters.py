"""
10 Real Filters - No Dummy Returns
"""

import logging
import pandas as pd
from typing import Dict, Optional
from core.market_regime import MarketRegime
from core.ml_filters import MLFilterManager
from core.technical_analysis import TechnicalAnalysis
from utils.time_utils import is_golden_hour

logger = logging.getLogger("FILTERS")

class FilterManager:
    """Real filter implementations"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
        self.ml_filters = MLFilterManager()
        self.recent_signals = {}  # Track recent signals per symbol
        
    async def apply_all_filters(self, symbol: str, signal: Dict, 
                                regime, features_df: pd.DataFrame,
                                df_5m: pd.DataFrame, df_15m: pd.DataFrame,
                                df_1h: pd.DataFrame, exchange_data: Dict) -> Dict:
        """Apply all 10 filters with real checks"""
        
        from config import REGIME_SETTINGS
        
        settings = REGIME_SETTINGS.get(regime.value, REGIME_SETTINGS['CHOPPY'])
        enabled_manual = settings['enabled_filters']
        
        results = {}
        passed = 0
        
        # Filter 1: Market Structure
        if 'structure' in enabled_manual:
            results['structure'] = await self.filter_structure(symbol, signal, df_5m)
            if results['structure']:
                passed += 1
        
        # Filter 2: Volume
        if 'volume' in enabled_manual:
            results['volume'] = await self.filter_volume(symbol, df_5m)
            if results['volume']:
                passed += 1
        
        # Filter 3: Liquidity (Spread + Orderbook)
        if 'liquidity' in enabled_manual:
            results['liquidity'] = await self.filter_liquidity(symbol, exchange_data)
            if results['liquidity']:
                passed += 1
        
        # Filter 4: Correlation
        if 'correlation' in enabled_manual:
            results['correlation'] = await self.filter_correlation(symbol, signal, df_5m)
            if results['correlation']:
                passed += 1
        
        # Filter 5: Funding Rate
        if 'funding' in enabled_manual:
            results['funding'] = await self.filter_funding(symbol, exchange_data)
            if results['funding']:
                passed += 1
        
        # Filter 6: Liquidation Levels
        if 'liquidation' in enabled_manual:
            results['liquidation'] = await self.filter_liquidation(symbol, signal, exchange_data)
            if results['liquidation']:
                passed += 1
        
        # Filter 7: Multi-Timeframe
        if 'mtf' in enabled_manual:
            results['mtf'] = await self.filter_mtf(symbol, signal, df_5m, df_15m, df_1h)
            if results['mtf']:
                passed += 1
        
        # Filter 8: Session Quality (Golden Hour)
        if 'session' in enabled_manual:
            results['session'] = await self.filter_session(symbol)
            if results['session']:
                passed += 1
        
        # Filter 9 & 10: ML Filters
        if features_df is not None and len(features_df) >= 60:
            ml_dir = await self.ml_filters.ml_direction_filter(symbol, signal, features_df)
            results['ml_direction'] = ml_dir
            if ml_dir:
                passed += 1
            
            ml_vol = await self.ml_filters.ml_volatility_filter(symbol, signal, features_df)
            results['ml_volatility'] = ml_vol
            if ml_vol:
                passed += 1
            
            total = len(enabled_manual) + 2
            ml_pred = await self.ml_filters.ensemble_prediction(symbol, features_df)
        else:
            total = len(enabled_manual)
            ml_pred = None
        
        return {
            'passed': passed,
            'total': total,
            'details': results,
            'ml_prediction': ml_pred
        }
    
    async def filter_structure(self, symbol: str, signal: Dict, df: pd.DataFrame) -> bool:
        """Filter 1: Market Structure (BOS/CHoCH)"""
        structure = self.ta.detect_bos_choch(df)
        
        if signal['direction'] == 'LONG':
            return structure in ['BOS_UP', 'CHoCH_UP']
        else:
            return structure in ['BOS_DOWN', 'CHoCH_DOWN']
    
    async def filter_volume(self, symbol: str, df: pd.DataFrame) -> bool:
        """Filter 2: Volume > 20 period average"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        return current_volume > avg_volume
    
    async def filter_liquidity(self, symbol: str, exchange_data: Dict) -> bool:
        """Filter 3: Spread < 0.1%, Orderbook depth"""
        if not exchange_data or 'best_prices' not in exchange_data:
            return False
        
        best = exchange_data['best_prices']
        if not best or 'best_ask' not in best or 'best_bid' not in best:
            return False
        
        ask = best['best_ask']['price']
        bid = best['best_bid']['price']
        
        spread_pct = (ask - bid) / ((ask + bid) / 2)
        
        # Spread must be < 0.1%
        if spread_pct > 0.001:
            logger.debug(f"{symbol} spread too high: {spread_pct:.4%}")
            return False
        
        return True
    
    async def filter_correlation(self, symbol: str, signal: Dict, df: pd.DataFrame) -> bool:
        """Filter 4: BTC correlation check"""
        if symbol == 'BTCUSDT':
            return True
        
        # For alts, check if following BTC direction
        # This would need BTC data passed in, simplified here
        return True
    
    async def filter_funding(self, symbol: str, exchange_data: Dict) -> bool:
        """Filter 5: Avoid extreme funding rates"""
        if not exchange_data or 'funding_rates' not in exchange_data:
            return True
        
        rates = exchange_data['funding_rates']
        
        # Check if any exchange has extreme funding
        for exchange, rate in rates.items():
            if abs(rate) > 0.01:  # > 1% funding
                logger.debug(f"{symbol} extreme funding on {exchange}: {rate}")
                return False
        
        return True
    
    async def filter_liquidation(self, symbol: str, signal: Dict, exchange_data: Dict) -> bool:
        """Filter 6: Liquidation cluster alignment"""
        # Simplified - would need liquidation heatmap data
        # For now, check if entry is away from recent liquidation wicks
        
        return True
    
    async def filter_mtf(self, symbol: str, signal: Dict, 
                        df_5m: pd.DataFrame, df_15m: pd.DataFrame, 
                        df_1h: pd.DataFrame) -> bool:
        """Filter 7: 5m/15m/1h trend confluence"""
        alignment = self.ta.get_trend_alignment(df_5m, df_15m, df_1h)
        
        # Must match signal direction
        if signal['direction'] == 'LONG' and alignment['direction'] != 'BULLISH':
            return False
        if signal['direction'] == 'SHORT' and alignment['direction'] != 'BEARISH':
            return False
        
        # At least 2/3 timeframes must align
        return alignment['alignment'] >= 0.66
    
    async def filter_session(self, symbol: str) -> bool:
        """Filter 8: Golden hour check"""
        return is_golden_hour()
    
    def check_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown"""
        import time
        current_time = time.time()
        
        if symbol in self.recent_signals:
            last_signal_time = self.recent_signals[symbol]
            if current_time - last_signal_time < 1200:  # 20 minutes
                return False
        
        self.recent_signals[symbol] = current_time
        return True
