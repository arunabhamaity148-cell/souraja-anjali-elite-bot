#!/usr/bin/env python3
"""
ARUNABHA ELITE v8.2 - FIXED TELEGRAM COMMANDS
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
from utils.time_utils import is_golden_hour, get_ist_time
from telegram import Bot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ARUNABHA_ELITE")

class ArunabhaEliteBot:
    def __init__(self):
        logger.info("=" * 70)
        logger.info("🚀 ARUNABHA ELITE v8.2 - COMMANDS FIXED")
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
        
        # Exchange manager
        self.exchange_mgr = self._init_exchange_manager()
        if not self.exchange_mgr or not self.exchange_mgr.clients:
            raise Exception("Exchange API keys required")
        
        self.position_sizer = PositionSizer(self.exchange_mgr)
        
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT',
                       'BNBUSDT', 'XRPUSDT', 'LINKUSDT', 'ADAUSDT']
        
        self.current_regime = None
        self.adaptive_settings = None
        self.last_regime_check = None
        self.last_training = None
        self.hourly_trade_count = {}
        self.last_hour_reset = datetime.now().hour
        self.last_update_id = 0
        
        self.daily_stats = {
            'total': 0, 'by_tier': {'TIER_1': 0, 'TIER_2': 0, 'TIER_3': 0},
            'by_regime': {}, 'pnl': 0, 'wins': 0, 'losses': 0
        }
        self.active_positions = {}
        
        # Telegram bot
        from config import TELEGRAM
        self.telegram_bot = Bot(token=TELEGRAM['bot_token'])
        self.chat_id = TELEGRAM['chat_id']
        
        logger.info(f"✅ Exchanges: {list(self.exchange_mgr.clients.keys())}")
        
    def _init_exchange_manager(self):
        config = {
            'binance_api_key': os.getenv('BINANCE_API_KEY'),
            'binance_api_secret': os.getenv('BINANCE_API_SECRET'),
            'binance_testnet': False,
            'delta_api_key': os.getenv('DELTA_API_KEY'),
            'delta_api_secret': os.getenv('DELTA_API_SECRET'),
            'coindcx_api_key': os.getenv('COINDCX_API_KEY'),
            'coindcx_api_secret': os.getenv('COINDCX_API_SECRET')
        }
        has_keys = any([config['binance_api_key'], config['delta_api_key'], config['coindcx_api_key']])
        if not has_keys:
            return None
        return ExchangeManager(config)
    
    async def check_commands(self):
        """Check Telegram commands every 5 seconds"""
        try:
            updates = await self.telegram_bot.get_updates(offset=self.last_update_id, limit=10)
            
            for update in updates:
                self.last_update_id = update.update_id + 1
                
                if not update.message or not update.message.text:
                    continue
                
                text = update.message.text.strip()
                chat_id = update.message.chat_id
                
                if str(chat_id) != str(self.chat_id):
                    continue
                
                if text.startswith('/train'):
                    await self.cmd_train(chat_id)
                elif text.startswith('/status'):
                    await self.cmd_status(chat_id)
                elif text.startswith('/scan'):
                    await self.cmd_scan(chat_id)
                elif text.startswith('/balance'):
                    await self.cmd_balance(chat_id)
                elif text.startswith('/help') or text.startswith('/start'):
                    await self.cmd_help(chat_id)
                    
        except Exception as e:
            logger.error(f"Command error: {e}")
    
    async def cmd_train(self, chat_id):
        await self.telegram_bot.send_message(chat_id=chat_id, text="🎓 Training started... 5-10 min")
        try:
            success = await self.model_trainer.train_daily(self)
            msg = "✅ Training complete!" if success else "❌ Training failed"
            await self.telegram_bot.send_message(chat_id=chat_id, text=msg)
        except Exception as e:
            await self.telegram_bot.send_message(chat_id=chat_id, text=f"❌ Error: {str(e)}")
    
    async def cmd_status(self, chat_id):
        regime = self.current_regime.value if self.current_regime else 'Unknown'
        ml = '✅ Trained' if self.model_trainer.ml_engine.is_trained else '❌ Untrained'
        status = f"""🤖 *Status*
📊 Regime: `{regime}`
📈 Signals: {self.daily_stats['total']}/12
🏆 {self.daily_stats.get('wins', 0)}W / {self.daily_stats.get('losses', 0)}L
💰 PNL: ₹{self.daily_stats.get('pnl', 0):,.2f}
🧠 ML: {ml}"""
        await self.telegram_bot.send_message(chat_id=chat_id, text=status, parse_mode='Markdown')
    
    async def cmd_scan(self, chat_id):
        await self.telegram_bot.send_message(chat_id=chat_id, text="🔍 Scanning...")
        if not self.adaptive_settings:
            await self.telegram_bot.send_message(chat_id=chat_id, text="❌ No regime")
            return
        count = 0
        for symbol in self.symbols:
            if await self._process_symbol(symbol, self.adaptive_settings):
                count += 1
            await asyncio.sleep(1)
        await self.telegram_bot.send_message(chat_id=chat_id, text=f"✅ {count} signals")
    
    async def cmd_balance(self, chat_id):
        try:
            client = self.exchange_mgr.get_primary_client()
            if client:
                bal = await client.get_balance()
                usdt = bal.get('USDT', 0)
                await self.telegram_bot.send_message(chat_id=chat_id, text=f"💰 `{usdt:,.2f} USDT`", parse_mode='Markdown')
        except Exception as e:
            await self.telegram_bot.send_message(chat_id=chat_id, text=f"❌ {str(e)}")
    
    async def cmd_help(self, chat_id):
        help_text = """🤖 *Commands*
/train - ML train
/status - Status
/scan - Force scan
/balance - Balance
/help - Help"""
        await self.telegram_bot.send_message(chat_id=chat_id, text=help_text, parse_mode='Markdown')
    
    async def run(self):
        await self.alerts.send_startup()
        await self.telegram_bot.send_message(chat_id=self.chat_id, text="✅ Commands: /train /status /scan /balance /help")
        
        while True:
            try:
                now = get_ist_time()
                
                # Check commands every 5 seconds
                await self.check_commands()
                
                if now.hour != self.last_hour_reset:
                    self.hourly_trade_count = {}
                    self.last_hour_reset = now.hour
                
                if now.hour == 0 and now.minute < 5:
                    self._reset_daily()
                
                if now.hour == 0 and 10 <= now.minute < 15:
                    if not self.last_training or (now - self.last_training).days >= 1:
                        await self.model_trainer.train_daily(self)
                        self.last_training = now
                
                if not self.last_regime_check or (now - self.last_regime_check).seconds >= 300:
                    await self._update_regime()
                
                if not is_golden_hour():
                    await asyncio.sleep(5)
                    continue
                
                await self._trading_session()
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(10)
    
    async def _update_regime(self):
        self.current_regime = await self.regime_detector.detect_regime()
        self.adaptive_settings = await self.regime_detector.get_adaptive_settings(self.current_regime)
        self.last_regime_check = get_ist_time()
        self.daily_stats['by_regime'][self.current_regime.value] = self.daily_stats['by_regime'].get(self.current_regime.value, 0) + 1
    
    async def _trading_session(self):
        settings = self.adaptive_settings
        if not settings or settings['strategy'] == 'NO_TRADE':
            return
        
        max_signals = settings['max_signals']
        signals_sent = 0
        
        for symbol in self.symbols:
            if signals_sent >= max_signals or self.daily_stats['total'] >= 12:
                break
            
            if self.hourly_trade_count.get(symbol, 0) >= 3:
                continue
            
            can_trade, _ = await self.risk_mgr.check_trade_allowed()
            if not can_trade:
                continue
            
            if await self._process_symbol(symbol, settings):
                signals_sent += 1
                self.hourly_trade_count[symbol] = self.hourly_trade_count.get(symbol, 0) + 1
            
            await asyncio.sleep(2)
    
    async def _process_symbol(self, symbol: str, settings: dict) -> bool:
        try:
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
            
            tier = self.tiers.determine_tier_adaptive(passed, total, settings['min_tier'])
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
                'regime': self.current_regime.value if self.current_regime else 'UNKNOWN',
                'timestamp': get_ist_time().isoformat()
            }
            
            await self.alerts.signal_alert(signal)
            self.daily_stats['total'] += 1
            self.daily_stats['by_tier'][tier['tier']] += 1
            
            asyncio.create_task(self._monitor_position(signal))
            
            return True
            
        except Exception as e:
            logger.error(f"Process error {symbol}: {e}")
            return False
    
    async def _monitor_position(self, signal: dict):
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
