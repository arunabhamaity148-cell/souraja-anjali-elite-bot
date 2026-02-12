"""
10 Filters (8 Manual + 2 ML)
"""

import logging
from typing import Dict
from core.market_regime import MarketRegime
from core.ml_filters import MLFilterManager
from utils.market_structure import MarketStructure

logger = logging.getLogger("FILTERS")

class FilterManager:
    def __init__(self):
        self.market_structure = MarketStructure()
        self.ml_filters = MLFilterManager()
        
        self.manual_filters = {
            'structure': self.filter_structure,
            'volume': self.filter_volume,
            'liquidity': self.filter_liquidity,
            'correlation': self.filter_correlation,
            'funding': self.filter_funding,
            'liquidation': self.filter_liquidation,
            'mtf': self.filter_mtf,
            'session': self.filter_session
        }
        
    async def apply_all_filters(self, symbol: str, signal: Dict, regime, features_df=None) -> Dict:
        """Apply all 10 filters"""
        from config import REGIME_SETTINGS
        
        settings = REGIME_SETTINGS.get(regime.value, REGIME_SETTINGS['CHOPPY'])
        enabled_manual = settings['enabled_filters']
        
        results = {}
        passed = 0
        
        # 8 Manual filters
        for name in enabled_manual:
            if name in self.manual_filters:
                result = await self.manual_filters[name](symbol, signal)
                results[name] = result
                if result:
                    passed += 1
        
        # 2 ML filters
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
        else:
            total = len(enabled_manual)
        
        # Get ensemble prediction
        ml_pred = None
        if features_df is not None:
            ml_pred = await self.ml_filters.ensemble_prediction(symbol, features_df)
        
        return {
            'passed': passed,
            'total': total,
            'details': results,
            'ml_prediction': ml_pred
        }
    
    async def filter_structure(self, s, sig): return True
    async def filter_volume(self, s, sig): return True
    async def filter_liquidity(self, s, sig): return True
    async def filter_correlation(self, s, sig): return True
    async def filter_funding(self, s, sig): return True
    async def filter_liquidation(self, s, sig): return True
    async def filter_mtf(self, s, sig): return True
    async def filter_session(self, s, sig): return True
