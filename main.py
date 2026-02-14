#!/usr/bin/env python3
"""
ARUNABHA ELITE v8.3 FINAL - PRODUCTION READY
Real Money Trading Bot with Telegram Commands
24/7 Trading Mode - Sleep Hours: 1 AM - 7 AM IST
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.signal_generator import EliteSignalGenerator
from core.filters import FilterManager
from core.market_regime import MarketRegimeDetector
from core.tier_system import TierManager
from core.risk_manager import EliteRiskManager
from core.position_sizing import PositionSizer
from core.feature_engineering import FeatureEngineer
from core.model_trainer import ModelTrainer
from core.technical_analysis import TechnicalAnalysis
from exchanges.exchange_manager import ExchangeManager
from alerts.telegram_alerts import HumanStyleAlerts
from utils.time_utils import is_sleep_hours, get_ist_time
from config import TELEGRAM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ARUNABHA_ELITE")

class ArunabhaEliteBot:
    def __init__(self):
        logger.info("=" * 70)
        logger.info("🚀 ARUNABHA ELITE v8.3 FINAL - REAL MONEY MODE")
        logger.info("=" * 70)
        
        # Core components
        self.signal_gen = EliteSignalGenerator()
        self.filters = FilterManager()
        self.regime_detector = MarketRegimeDetector()
        self.tiers = TierManager()
        self.risk_mgr = EliteRiskManager()
        self.ta = TechnicalAnalysis()
        self.feature_eng = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.alerts = HumanStyleAlerts()
        
        # Exchange manager - NO MOCK DATA
        self.exchange_mgr = self._init_exchange_manager()
        if not self.exchange_mgr or not self.exchange_mgr.clients:
            logger.error("❌ NO EXCHANGE CONNECTED - BOT CANNOT START")
            raise Exception("Exchange API keys required")
        
        # Position sizing with real balance
        self.position_sizer = PositionSizer(self.exchange_mgr)
        
        # ML engine (accessible for commands)
        from core.ml_engine import MLEngine
        self.ml_engine = MLEngine()
        
        # Pass bot reference to alerts for commands
        self.alerts.bot_ref = self
        
        # 8 pairs
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT',
            'BNBUSDT', 'XRPUSDT', 'LINKUSDT', 'ADAUSDT'
        ]
        
        # State
        self.current_regime = None
        self.adaptive_settings = None
        self.last_regime_check = None
        self.last_training = None
        self.hourly_trade_count = {}
        self.last_hour_reset = datetime.now().hour
        self.is_paused = False
        
        # Stats
        self.daily_stats = {
            'total': 0, 'by_tier': {'TIER_1': 0, 'TIER_2': 0, 'TIER_3': 0},
            'by_regime': {}, 'pnl': 0, 'wins': 0, 'losses': 0
        }
        self.active_positions = {}
        
        logger.info(f"✅ {len(self.symbols)} pairs configured")
        logger.info(f"✅ Exchanges: {list(self.exchange_mgr.clients.keys())}")
        
    def _init_exchange_manager(self):
        """Initialize with real API keys only - NO TESTNET"""
        config = {
            'binance_api_key': os.getenv('BINANCE_API_KEY'),
            'binance_api_secret': os.getenv('BINANCE_API_SECRET'),
            'binance_testnet': False,
            'delta_api_key': os.getenv('DELTA_API_KEY'),
            'delta_api_secret': os.getenv('DELTA_API_SECRET'),
            'coindcx_api_key': os.getenv('COINDCX_API_KEY'),
            'coindcx_api_secret': os.getenv('COINDCX_API_SECRET')
        }
        
        has_keys = any([
            config['binance_api_key'],
            config['delta_api_key'],
            config['coindcx_api_key']
        ])
        
        if not has_keys:
            logger.error("NO API KEYS FOUND IN ENVIRONMENT")
            return None
        
        return ExchangeManager(config)
    
    async def run(self):
        """Main loop with telegram commands"""
        # Send startup alerts
        await self.alerts.send_startup()
        
        # Start telegram command polling from alerts class
        if self.alerts.app:
            asyncio.create_task(self.alerts.start_polling())
            logger.info("✅ Telegram polling started")
        
        # Main trading loop
        while True:
            try:
                now = get_ist_time()
                
                # Reset hourly trade count
                if now.hour != self.last_hour_reset:
                    self.hourly_trade_count = {}
                    self.last_hour_reset = now.hour
                    logger.info(f"⏰ Hourly reset at {now.strftime('%H:%M')}")
                
                # Daily reset at midnight
                if now.hour == 0 and now.minute < 5:
                    self._reset_daily()
                
                # Daily training at 00:10
                if now.hour == 0 and 10 <= now.minute < 15:
                    if not self.last_training or (now - self.last_training).days >= 1:
                        await self.model_trainer.train_daily(self)
                        self.last_training = now
                
                # Update regime every 5 min
                if not self.last_regime_check or (now - self.last_regime_check).seconds >= 300:
                    await self._update_regime()
                
                # Check if paused
                if self.is_paused:
                    logger.info("⏸️ Bot is PAUSED - use /resume to continue")
                    await asyncio.sleep(60)
                    continue
                
                # ✅ CHECK SLEEP HOURS (1 AM - 7 AM IST)
                is_sleeping, sleep_reason = is_sleep_hours()
                if is_sleeping:
                    logger.info(f"💤 {sleep_reason}")
                    await asyncio.sleep(300)
                    continue
                
                # ✅ TRADING 24/7 (except sleep hours)
                await self._trading_session()
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Main error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await asyncio.sleep(10)
    
    async def _update_regime(self):
        """Update market regime"""
        self.current_regime = await self.regime_detector.detect_regime()
        self.adaptive_settings = await self.regime_detector.get_adaptive_settings(
            self.current_regime
        )
        self.last_regime_check = get_ist_time()
        
        regime_name = self.current_regime.value
        self.daily_stats['by_regime'][regime_name] = \
            self.daily_stats['by_regime'].get(regime_name, 0) + 1
        
        logger.info(f"📊 Regime: {regime_name}")
    async def _trading_session(self):
        """Execute trading session with detailed logging - 24/7 mode"""
        settings = self.adaptive_settings
        
        logger.info("=" * 70)
        logger.info(f"🎯 TRADING SESSION START")
        logger.info(f"   Time: {get_ist_time().strftime('%H:%M:%S IST')}")
        logger.info(f"   Regime: {self.current_regime.value if self.current_regime else 'Unknown'}")
        logger.info(f"   Strategy: {settings.get('strategy') if settings else 'None'}")
        logger.info(f"   Max Signals: {settings.get('max_signals') if settings else 0}")
        logger.info(f"   Min Tier: {settings.get('min_tier') if settings else 'None'}")
        logger.info(f"   Enabled Filters: {len(settings.get('enabled_filters', []))} filters")
        logger.info(f"   Daily Count: {self.daily_stats['total']}/12")
        logger.info(f"   Mode: 24/7 Trading (Sleep: 1-7 AM)")
        logger.info("=" * 70)
        
        if not settings:
            logger.warning("⚠️ No adaptive settings available")
            return
        
        if settings['strategy'] == 'NO_TRADE':
            logger.warning(f"⚠️ Trading SKIPPED - Strategy: NO_TRADE")
            return
        
        if self.daily_stats['total'] >= 12:
            logger.warning(f"⚠️ Daily signal limit reached (12/12)")
            return
        
        max_signals = settings['max_signals']
        signals_sent = 0
        
        for symbol in self.symbols:
            if signals_sent >= max_signals:
                break
            
            if self.daily_stats['total'] >= 12:
                break
            
            if self.hourly_trade_count.get(symbol, 0) >= 3:
                continue
            
            can_trade, reason = await self.risk_mgr.check_trade_allowed()
            if not can_trade:
                continue
            
            success = await self._process_symbol(symbol, settings)
            if success:
                signals_sent += 1
                self.hourly_trade_count[symbol] = self.hourly_trade_count.get(symbol, 0) + 1
            
            await asyncio.sleep(2)
        
        logger.info(f"📊 Session: {signals_sent} signals sent")
    
    async def _process_symbol(self, symbol: str, settings: dict) -> bool:
        """Process single symbol with real data"""
        try:
            logger.info(f"🔍 Processing {symbol}...")
            
            df_5m = await self.exchange_mgr.get_ohlcv(symbol, '5m', 100)
            df_15m = await self.exchange_mgr.get_ohlcv(symbol, '15m', 100)
            df_1h = await self.exchange_mgr.get_ohlcv(symbol, '1h', 100)
            
            if any(df is None or len(df) < 60 for df in [df_5m, df_15m, df_1h]):
                return False
            
            ticker = await self.exchange_mgr.get_ticker(symbol)
            if ticker:
                spread = (ticker.get('ask', 0) - ticker.get('bid', 0)) / ticker.get('last', 1)
                if spread > 0.002:
                    return False
            
            raw_signal = await self.signal_gen.generate_signal(symbol, df_5m)
            
            if not raw_signal:
                return False
            
            bias = settings.get('direction_bias')
            if bias and ((bias == 'LONG_ONLY' and raw_signal['direction'] != 'LONG') or
                        (bias == 'SHORT_ONLY' and raw_signal['direction'] != 'SHORT')):
                return False
            
            features_df = self.feature_eng.create_features(df_5m)
            
            exchange_data = {
                'best_prices': await self.exchange_mgr.get_best_price(symbol),
                'funding_rates': await self.exchange_mgr.get_funding_rates(symbol)
            }
            
            filter_result = await self.filters.apply_all_filters(
                symbol, raw_signal, self.current_regime, 
                features_df, df_5m, df_15m, df_1h, exchange_data
            )
            
            if filter_result.get('blocked'):
                return False
            
            passed = filter_result['passed']
            total = filter_result['total']
            
            logger.info(f"   📊 {symbol}: Filters {passed}/{total} passed")
            
            tier = self.tiers.determine_tier_adaptive(
                passed, total, settings['min_tier']
            )
            
            if not tier:
                return False
            
            position = await self.position_sizer.calculate_position_size(
                symbol, raw_signal['entry'], raw_signal['sl'], tier['tier']
            )
            
            if not position:
                return False
            
            signal = {
                **raw_signal,
                'tier': tier['tier'],
                'confidence': tier['confidence'],
                'filters_passed': f"{passed}/{total}",
                'win_rate': tier['expected_win_rate'],
                'position_size': position['position_size'],
                'margin_required': position['margin_required'],
                'risk_amount': position['risk_amount'],
                'balance': position['balance'],
                'ml_score': filter_result.get('ml_prediction', {}).get('ensemble_score', 0),
                'timestamp': get_ist_time().isoformat()
            }
            
            await self.alerts.signal_alert(signal)
            self.daily_stats['total'] += 1
            self.daily_stats['by_tier'][tier['tier']] += 1
            
            logger.info("=" * 70)
            logger.info(f"✅ SIGNAL SENT: {symbol} {signal['direction']}")
            logger.info(f"   Entry: {signal['entry']}, SL: {signal['sl']}, TP1: {signal['tp1']}")
            logger.info(f"   Tier: {signal['tier']}, Confidence: {signal['confidence']}%")
            logger.info("=" * 70)
            
            asyncio.create_task(self._monitor_position(signal))
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ {symbol}: {e}")
            return False
    
    async def _monitor_position(self, signal: dict):
        """Monitor with real price updates"""
        entry_time = get_ist_time()
        tp1_hit = tp2_hit = tp3_hit = False
        
        while True:
            try:
                ticker = await self.exchange_mgr.get_ticker(signal['symbol'])
                current_price = ticker.get('last', 0)
                
                if current_price == 0:
                    await asyncio.sleep(5)
                    continue
                
                if not tp1_hit and self._hit_tp(signal, current_price, 'tp1'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp1', signal, profit)
                    tp1_hit = True
                    self._update_pnl(profit, True)
                
                if tp1_hit and not tp2_hit and self._hit_tp(signal, current_price, 'tp2'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp2', signal, profit)
                    tp2_hit = True
                
                if tp2_hit and not tp3_hit and self._hit_tp(signal, current_price, 'tp3'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp3', signal, profit)
                    self._update_pnl(profit, True)
                    return
                
                if self._hit_sl(signal, current_price):
                    loss = self._calc_profit(signal, current_price)
                    await self.alerts.sl_alert(signal)
                    self._update_pnl(loss, False)
                    return
                
                be = self.risk_mgr.check_breakeven(signal, current_price)
                if be:
                    await self.alerts.breakeven_alert(be)
                    signal['sl'] = signal['entry']
                
                if (get_ist_time() - entry_time).seconds > 7200:
                    pnl = self._calc_profit(signal, current_price)
                    await self.alerts.timeout_alert(signal)
                    self._update_pnl(pnl, pnl > 0)
                    return
                
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)
    
    def _hit_tp(self, signal, price, tp):
        return (signal['direction'] == 'LONG' and price >= signal[tp]) or \
               (signal['direction'] == 'SHORT' and price <= signal[tp])
    
    def _hit_sl(self, signal, price):
        return (signal['direction'] == 'LONG' and price <= signal['sl']) or \
               (signal['direction'] == 'SHORT' and price >= signal['sl'])
    
    def _calc_profit(self, signal, current_price):
        diff = current_price - signal['entry']
        if signal['direction'] == 'SHORT':
            diff = -diff
        qty = signal.get('position_size', 0)
        return round(diff * qty, 2)
    
    def _update_pnl(self, pnl: float, is_win: bool):
        self.daily_stats['pnl'] += pnl
        if is_win:
            self.daily_stats['wins'] += 1
        else:
            self.daily_stats['losses'] += 1
    
    def _reset_daily(self):
        logger.info("=" * 70)
        logger.info("🌅 DAILY RESET")
        logger.info(f"   Yesterday: {self.daily_stats['wins']}W/{self.daily_stats['losses']}L | PnL: ₹{self.daily_stats['pnl']:.2f}")
        logger.info("=" * 70)
        
        self.daily_stats = {
            'total': 0, 'by_tier': {'TIER_1': 0, 'TIER_2': 0, 'TIER_3': 0},
            'by_regime': {}, 'pnl': 0, 'wins': 0, 'losses': 0
        }
        self.tiers.reset_daily()
        self.position_sizer.reset_daily_stats()

async def main():
    bot = ArunabhaEliteBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
