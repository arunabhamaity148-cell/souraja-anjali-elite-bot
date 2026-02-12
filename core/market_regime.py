"""
Auto Market Regime Detection - 8 States
"""

import logging
from enum import Enum
from typing import Dict
from config import REGIME_SETTINGS

logger = logging.getLogger("REGIME")

class MarketRegime(Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    EXTREME_FEAR = "EXTREME_FEAR"
    EXTREME_GREED = "EXTREME_GREED"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    CHOPPY = "CHOPPY"

class MarketRegimeDetector:
    def __init__(self):
        self.current_regime = None
        self.regime_history = []
        
    async def detect_regime(self, symbol: str = 'BTCUSDT') -> MarketRegime:
        """Auto detect market condition"""
        
        adx = await self.get_adx(symbol)
        atr_ratio = await self.get_atr_ratio(symbol)
        fear_greed = await self.get_fear_greed_index()
        volume = await self.get_volume_profile(symbol)
        price_action = await self.get_price_action(symbol)
        
        regime = self._classify(adx, atr_ratio, fear_greed, volume, price_action)
        self.current_regime = regime
        self.regime_history.append(regime)
        
        logger.info(f"📊 Regime: {regime.value}")
        return regime
    
    def _classify(self, adx, atr_ratio, fear_greed, volume, price_action) -> MarketRegime:
        """Classify regime"""
        
        if fear_greed < 20:
            return MarketRegime.EXTREME_FEAR
        if fear_greed > 80:
            return MarketRegime.EXTREME_GREED
        
        if atr_ratio > 2.0:
            return MarketRegime.VOLATILE
        if atr_ratio < 0.5:
            return MarketRegime.LOW_VOLATILITY
        
        if adx > 25:
            if price_action['trend'] == 'UP':
                return MarketRegime.TRENDING_BULL
            return MarketRegime.TRENDING_BEAR
        
        if price_action['range_bound']:
            return MarketRegime.RANGING
        
        return MarketRegime.CHOPPY
    
    async def get_adaptive_settings(self, regime: MarketRegime) -> Dict:
        """Get settings for regime"""
        return REGIME_SETTINGS.get(regime.value, REGIME_SETTINGS['CHOPPY'])
    
    async def get_adx(self, symbol): return 20
    async def get_atr_ratio(self, symbol): return 1.0
    async def get_fear_greed_index(self): return 50
    async def get_volume_profile(self, symbol): return {'ratio': 1.0}
    async def get_price_action(self, symbol): return {'trend': 'UP', 'range_bound': False}
