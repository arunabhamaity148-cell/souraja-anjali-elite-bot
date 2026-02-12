#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║     ARUNABHA ELITE v8.0 ML FINAL - 10 FILTERS                    ║
║     8 Pairs | 10 Filters | 3 Tiers | Auto ML | 92/100 Rating     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from core.signal_generator import EliteSignalGenerator
from core.filters import FilterManager
from core.market_regime import MarketRegimeDetector
from core.tier_system import TierManager
from core.risk_manager import EliteRiskManager
from core.websocket_handler import WebSocketManager
from core.feature_engineering import FeatureEngineer
from core.model_trainer import ModelTrainer
from alerts.telegram_alerts import HumanStyleAlerts
from utils.time_utils import is_golden_hour, get_ist_time

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ARUNABHA_ELITE")

class ArunabhaEliteBot:
    def __init__(self):
        self.signal_gen = EliteSignalGenerator()
        self.filters = FilterManager()
        self.regime_detector = MarketRegimeDetector()
        self.tiers = TierManager()
        self.risk_mgr = EliteRiskManager()
        self.alerts = HumanStyleAlerts()
        self.ws_manager = WebSocketManager()
        self.feature_eng = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT',
            'BNBUSDT', 'XRPUSDT', 'LINKUSDT', 'ADAUSDT'
        ]
        
        self.current_regime = None
        self.adaptive_settings = None
        self.last_regime_check = None
        self.last_training = None
        self.daily_stats = {
            'total': 0, 
            'by_tier': {'TIER_1': 0, 'TIER_2': 0, 'TIER_3': 0},
            'by_regime': {}
        }
        
        logger.info("🚀 ARUNABHA ELITE v8.0 ML FINAL INITIALIZED")
        logger.info(f"📊 {len(self.symbols)} Pairs | 10 Filters | Auto ML")
        
    async def run(self):
        await self.alerts.send_startup()
        
        while True:
            try:
                now = datetime.now()
                
                # Daily model training at 00:00 IST
                if (not self.last_training or 
                    (now.hour == 0 and now.minute < 5)):
                    await self.model_trainer.train_daily(self)
                    self.last_training = now
                
                # Update regime every 5 minutes
                if (not self.last_regime_check or 
                    (now - self.last_regime_check).seconds > 300):
                    
                    self.current_regime = await self.regime_detector.detect_regime()
                    self.adaptive_settings = await self.regime_detector.get_adaptive_settings(
                        self.current_regime
                    )
                    self.last_regime_check = now
                    
                    regime_name = self.current_regime.value
                    self.daily_stats['by_regime'][regime_name] = \
                        self.daily_stats['by_regime'].get(regime_name, 0) + 1
                    
                    await self.alerts.regime_alert(self.current_regime, self.adaptive_settings)
                
                if not is_golden_hour():
                    await asyncio.sleep(60)
                    continue
                
                await self.run_adaptive_trading()
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                await asyncio.sleep(10)
    
    async def run_adaptive_trading(self):
        settings = self.adaptive_settings
        max_signals = settings['max_signals']
        strategy = settings['strategy']
        min_tier = settings['min_tier']
        
        logger.info(f"🔥 {self.current_regime.value} | {strategy} | Max: {max_signals}")
        
        if strategy == 'NO_TRADE':
            await self.alerts.skip_alert(f"Market: {self.current_regime.value}")
            return
        
        signals_sent = 0
        
        for symbol in self.symbols:
            if signals_sent >= max_signals:
                break
            
            if self.daily_stats['total'] >= 12:
                break
            
            can_trade, reason = await self.risk_mgr.check_trade_allowed()
            if not can_trade:
                continue
            
            # Generate raw signal
            raw_signal = await self.signal_gen.generate_signal(symbol)
            if not raw_signal:
                continue
            
            # Check direction bias
            direction_bias = settings['direction_bias']
            if direction_bias:
                if direction_bias == 'LONG_ONLY' and raw_signal['direction'] != 'LONG':
                    continue
                if direction_bias == 'SHORT_ONLY' and raw_signal['direction'] != 'SHORT':
                    continue
            
            # Get features for ML
            raw_data = await self.get_ohlcv_data(symbol)
            if raw_data is None or len(raw_data) < 60:
                continue
            
            features_df = self.feature_eng.create_features(raw_data)
            
            # Apply 10 filters (8 + 2 ML)
            filter_result = await self.filters.apply_all_filters(
                symbol, raw_signal, self.current_regime, features_df
            )
            
            if filter_result.get('blocked'):
                continue
            
            passed = filter_result['passed']
            total = filter_result['total']
            
            # Get ML prediction
            ml_pred = filter_result.get('ml_prediction', {})
            
            # Determine tier
            tier = self.tiers.determine_tier_adaptive(passed, total, min_tier)
            if not tier:
                continue
            
            # Final signal with ML
            signal = {
                **raw_signal,
                'tier': tier['tier'],
                'confidence': tier['confidence'],
                'filters_passed': f"{passed}/{total}",
                'win_rate': tier['expected_win_rate'],
                'regime': self.current_regime.value,
                'strategy': strategy,
                'ml_score': ml_pred.get('ensemble_score', 0),
                'ml_hold_time': ml_pred.get('hold_time', 60),
                'ml_direction_prob': ml_pred.get('confidence', 0.5)
            }
            
            await self.alerts.signal_alert(signal)
            self.update_stats(tier['tier'])
            signals_sent += 1
            
            asyncio.create_task(self.monitor_position(signal))
            await asyncio.sleep(5)
    
    async def monitor_position(self, signal):
        entry_time = datetime.now()
        tp1_hit = tp2_hit = tp3_hit = False
        
        # Use ML predicted hold time
        max_hold = signal.get('ml_hold_time', 120) * 60  # Convert to seconds
        
        while True:
            try:
                current_price = await self.ws_manager.get_price(signal['symbol'])
                
                if not tp1_hit and self._hit_tp(signal, current_price, 'tp1'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp1', signal, profit)
                    tp1_hit = True
                
                if tp1_hit and not tp2_hit and self._hit_tp(signal, current_price, 'tp2'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp2', signal, profit)
                    tp2_hit = True
                
                if tp2_hit and not tp3_hit and self._hit_tp(signal, current_price, 'tp3'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp3', signal, profit)
                    return
                
                if self._hit_sl(signal, current_price):
                    await self.alerts.sl_alert(signal)
                    return
                
                be_action = self.risk_mgr.check_breakeven(signal, current_price)
                if be_action:
                    await self.alerts.breakeven_alert(be_action)
                
                if (datetime.now() - entry_time).seconds > max_hold:
                    await self.alerts.timeout_alert(signal)
                    return
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)
    
    def _hit_tp(self, signal, price, tp_level):
        if signal['direction'] == 'LONG':
            return price >= signal[tp_level]
        return price <= signal[tp_level]
    
    def _hit_sl(self, signal, price):
        if signal['direction'] == 'LONG':
            return price <= signal['sl']
        return price >= signal['sl']
    
    def _calc_profit(self, signal, current_price):
        diff = abs(current_price - signal['entry'])
        qty = (1000 * 15) / signal['entry']
        return round(diff * qty, 2)
    
    def update_stats(self, tier_name):
        self.daily_stats['total'] += 1
        self.daily_stats['by_tier'][tier_name] = \
            self.daily_stats['by_tier'].get(tier_name, 0) + 1
    
    async def get_ohlcv_data(self, symbol):
        """Fetch OHLCV - implement with real API"""
        try:
            # Stub - replace with Binance API
            import pandas as pd
            import numpy as np
            
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
            base = {'BTCUSDT': 97000, 'ETHUSDT': 2650, 'SOLUSDT': 195,
                   'DOGEUSDT': 0.25, 'BNBUSDT': 720, 'XRPUSDT': 2.45,
                   'LINKUSDT': 18.5, 'ADAUSDT': 0.85}.get(symbol, 100)
            
            data = {
                'timestamp': dates,
                'open': base + np.random.randn(100).cumsum() * 50,
                'high': base + np.random.randn(100).cumsum() * 50 + 30,
                'low': base + np.random.randn(100).cumsum() * 50 - 30,
                'close': base + np.random.randn(100).cumsum() * 50,
                'volume': np.random.rand(100) * 10000
            }
            
            df = pd.DataFrame(data)
            df['high'] = df[['open', 'close', 'high']].max(axis=1)
            df['low'] = df[['open', 'close', 'low']].min(axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return None

if __name__ == "__main__":
    bot = ArunabhaEliteBot()
    asyncio.run(bot.run())
