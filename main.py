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
from alerts.telegram_commands import TelegramCommands
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
        
        # Telegram commands (must be after bot initialization)
        self.commands = None
        if TELEGRAM.get('bot_token'):
            try:
                self.commands = TelegramCommands(self)
                logger.info("✅ Telegram commands initialized")
            except Exception as e:
                logger.warning(f"Telegram commands init failed: {e}")
        
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
        self.is_paused = False  # For /pause command
        
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
            'binance_testnet': False,  # LIVE ONLY
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
        
        # Start telegram bot commands in background
        if self.commands and self.commands.app:
            asyncio.create_task(self._run_telegram_commands())
        
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
                    await asyncio.sleep(300)  # Check every 5 minutes
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
    
    async def _run_telegram_commands(self):
        """Run telegram bot commands polling"""
        try:
            logger.info("🤖 Starting Telegram bot polling...")
            await self.commands.app.initialize()
            await self.commands.app.start()
            await self.commands.app.updater.start_polling()
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Telegram bot error: {e}")
    
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
        
        # ✅ DETAILED SESSION LOGGING
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
            logger.info(f"   Reason: {self.current_regime.value} regime set to NO_TRADE")
            return
        
        # ✅ Check if already hit daily limit
        if self.daily_stats['total'] >= 12:
            logger.warning(f"⚠️ Daily signal limit reached (12/12)")
            logger.info(f"   Wins: {self.daily_stats['wins']}, Losses: {self.daily_stats['losses']}")
            logger.info(f"   PnL Today: ₹{self.daily_stats['pnl']:.2f}")
            return
        
        max_signals = settings['max_signals']
        signals_sent = 0
        symbols_processed = 0
        symbols_with_signals = 0
        
        for symbol in self.symbols:
            symbols_processed += 1
            
            if signals_sent >= max_signals:
                logger.info(f"✋ Max signals reached ({signals_sent}/{max_signals})")
                break
            
            if self.daily_stats['total'] >= 12:
                logger.info(f"✋ Daily limit reached (12/12)")
                break
            
            # Max 3 trades per hour per pair
            if self.hourly_trade_count.get(symbol, 0) >= 3:
                logger.debug(f"   ⏭️ {symbol}: Hourly limit reached (3/3)")
                continue
            
            # Check risk
            can_trade, reason = await self.risk_mgr.check_trade_allowed()
            if not can_trade:
                logger.warning(f"   ⚠️ Risk check failed: {reason}")
                logger.info(f"      Daily PnL: ₹{self.daily_stats['pnl']:.2f}")
                logger.info(f"      Trades today: {self.daily_stats['total']}")
                continue
            
            success = await self._process_symbol(symbol, settings)
            if success:
                signals_sent += 1
                symbols_with_signals += 1
                self.hourly_trade_count[symbol] = self.hourly_trade_count.get(symbol, 0) + 1
            
            await asyncio.sleep(2)
        
        # ✅ SESSION SUMMARY
        logger.info("=" * 70)
        logger.info(f"📊 SESSION SUMMARY")
        logger.info(f"   Symbols Processed: {symbols_processed}/{len(self.symbols)}")
        logger.info(f"   Signals Generated: {signals_sent}")
        logger.info(f"   Symbols with Signals: {symbols_with_signals}")
        logger.info(f"   Daily Total: {self.daily_stats['total']}/12")
        if signals_sent == 0:
            logger.info(f"   Status: No signals met tier requirements")
        logger.info("=" * 70)
    
    async def _process_symbol(self, symbol: str, settings: dict) -> bool:
        """Process single symbol with real data and detailed logging"""
        try:
            logger.info(f"🔍 Processing {symbol}...")
            
            # Fetch multi-timeframe data from REAL exchange
            df_5m = await self.exchange_mgr.get_ohlcv(symbol, '5m', 100)
            df_15m = await self.exchange_mgr.get_ohlcv(symbol, '15m', 100)
            df_1h = await self.exchange_mgr.get_ohlcv(symbol, '1h', 100)
            
            if any(df is None or len(df) < 60 for df in [df_5m, df_15m, df_1h]):
                logger.debug(f"   ❌ {symbol}: Insufficient data from exchange")
                return False
            
            # Check spread
            ticker = await self.exchange_mgr.get_ticker(symbol)
            if ticker:
                spread = (ticker.get('ask', 0) - ticker.get('bid', 0)) / ticker.get('last', 1)
                if spread > 0.002:  # > 0.2% spread
                    logger.debug(f"   ❌ {symbol}: Spread too high {spread:.4%}")
                    return False
            
            # Generate signal with REAL TA
            raw_signal = await self.signal_gen.generate_signal(symbol, df_5m)
            
            if not raw_signal:
                logger.debug(f"   ❌ {symbol}: No technical signal generated")
                return False
            else:
                logger.info(f"   ✅ {symbol}: RAW Signal {raw_signal['direction']} @ {raw_signal['entry']}")
                logger.debug(f"      Score: {raw_signal['score']}, RR: {raw_signal['rr_ratio']}")
            
            # Check direction bias
            bias = settings.get('direction_bias')
            if bias and ((bias == 'LONG_ONLY' and raw_signal['direction'] != 'LONG') or
                        (bias == 'SHORT_ONLY' and raw_signal['direction'] != 'SHORT')):
                logger.debug(f"   ❌ {symbol}: Direction bias mismatch (need {bias})")
                return False
            
            # Create features for ML
            features_df = self.feature_eng.create_features(df_5m)
            
            # Get exchange data
            exchange_data = {
                'best_prices': await self.exchange_mgr.get_best_price(symbol),
                'funding_rates': await self.exchange_mgr.get_funding_rates(symbol)
            }
            
            # Apply 10 filters
            filter_result = await self.filters.apply_all_filters(
                symbol, raw_signal, self.current_regime, 
                features_df, df_5m, df_15m, df_1h, exchange_data
            )
            
            if filter_result.get('blocked'):
                logger.warning(f"   ⛔ {symbol}: BLOCKED by filters")
                return False
            
            passed = filter_result['passed']
            total = filter_result['total']
            
            logger.info(f"   📊 {symbol}: Filters {passed}/{total} passed")
            
            # Debug: Show individual filter results
            if logger.level <= logging.DEBUG:
                for fname, fresult in filter_result.get('details', {}).items():
                    logger.debug(f"      {fname}: {'✅ PASS' if fresult else '❌ FAIL'}")
            
            # Determine tier
            tier = self.tiers.determine_tier_adaptive(
                passed, total, settings['min_tier']
            )
            
            if not tier:
                logger.warning(f"   ❌ {symbol}: Did not meet tier requirement (min: {settings['min_tier']}, got {passed}/{total})")
                return False
            else:
                logger.info(f"   🏆 {symbol}: {tier['tier']} assigned (confidence: {tier['confidence']}%)")
            
            # Calculate position size with REAL balance
            position = await self.position_sizer.calculate_position_size(
                symbol, raw_signal['entry'], raw_signal['sl'], tier['tier']
            )
            
            if not position:
                logger.warning(f"   ❌ {symbol}: Position sizing failed")
                return False
            
            # Final signal
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
            
            # Send alert
            await self.alerts.signal_alert(signal)
            self.daily_stats['total'] += 1
            self.daily_stats['by_tier'][tier['tier']] += 1
            
            # ✅ SUCCESS LOG
            logger.info("=" * 70)
            logger.info(f"✅ SIGNAL SENT: {symbol} {signal['direction']}")
            logger.info(f"   Entry: {signal['entry']}, SL: {signal['sl']}, TP1: {signal['tp1']}")
            logger.info(f"   TP2: {signal['tp2']}, TP3: {signal['tp3']}")
            logger.info(f"   Tier: {signal['tier']}, Confidence: {signal['confidence']}%")
            logger.info(f"   Filters: {passed}/{total}, RR: {signal['rr_ratio']}")
            logger.info(f"   Position: {signal['position_size']}, Risk: ₹{signal['risk_amount']}")
            logger.info(f"   Balance: ₹{signal['balance']}, Margin: ₹{signal['margin_required']}")
            logger.info("=" * 70)
            
            # Monitor position
            asyncio.create_task(self._monitor_position(signal))
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ {symbol}: Processing error - {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    async def _monitor_position(self, signal: dict):
        """Monitor with real price updates"""
        entry_time = get_ist_time()
        tp1_hit = tp2_hit = tp3_hit = False
        
        logger.info(f"👁️ Monitoring {signal['symbol']} position...")
        
        while True:
            try:
                # Get REAL price from exchange
                ticker = await self.exchange_mgr.get_ticker(signal['symbol'])
                current_price = ticker.get('last', 0)
                
                if current_price == 0:
                    await asyncio.sleep(5)
                    continue
                
                # Check TPs
                if not tp1_hit and self._hit_tp(signal, current_price, 'tp1'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp1', signal, profit)
                    tp1_hit = True
                    self._update_pnl(profit, True)
                    logger.info(f"✅ {signal['symbol']}: TP1 HIT! Profit: ₹{profit}")
                
                if tp1_hit and not tp2_hit and self._hit_tp(signal, current_price, 'tp2'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp2', signal, profit)
                    tp2_hit = True
                    logger.info(f"✅ {signal['symbol']}: TP2 HIT! Profit: ₹{profit}")
                
                if tp2_hit and not tp3_hit and self._hit_tp(signal, current_price, 'tp3'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp3', signal, profit)
                    self._update_pnl(profit, True)
                    logger.info(f"✅ {signal['symbol']}: TP3 HIT! Total Profit: ₹{profit}")
                    return
                
                # Check SL
                if self._hit_sl(signal, current_price):
                    loss = self._calc_profit(signal, current_price)
                    await self.alerts.sl_alert(signal)
                    self._update_pnl(loss, False)
                    logger.warning(f"❌ {signal['symbol']}: SL HIT. Loss: ₹{loss}")
                    return
                
                # Breakeven
                be = self.risk_mgr.check_breakeven(signal, current_price)
                if be:
                    await self.alerts.breakeven_alert(be)
                    signal['sl'] = signal['entry']
                    logger.info(f"🔒 {signal['symbol']}: Moved to BREAKEVEN")
                
                # Timeout (2 hours)
                if (get_ist_time() - entry_time).seconds > 7200:
                    pnl = self._calc_profit(signal, current_price)
                    await self.alerts.timeout_alert(signal)
                    self._update_pnl(pnl, pnl > 0)
                    logger.info(f"⏱️ {signal['symbol']}: TIMEOUT. PnL: ₹{pnl}")
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
        logger.info("🌅 DAILY RESET - New Trading Day")
        logger.info(f"   Yesterday's Stats:")
        logger.info(f"   Total Signals: {self.daily_stats['total']}")
        logger.info(f"   Wins: {self.daily_stats['wins']}, Losses: {self.daily_stats['losses']}")
        logger.info(f"   PnL: ₹{self.daily_stats['pnl']:.2f}")
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
