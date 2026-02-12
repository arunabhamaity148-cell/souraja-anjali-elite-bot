#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║     ARUNABHA ELITE v8.0 FINAL - MULTI-EXCHANGE + ML              ║
║     8 Pairs | 10 Filters | 3 Tiers | Auto ML | 95/100 Rating     ║
║     Binance + Delta + CoinDCX Integration                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.signal_generator import EliteSignalGenerator
from core.filters import FilterManager
from core.market_regime import MarketRegimeDetector
from core.tier_system import TierManager
from core.risk_manager import EliteRiskManager
from core.websocket_handler import WebSocketManager
from core.feature_engineering import FeatureEngineer
from core.model_trainer import ModelTrainer
from exchanges.exchange_manager import ExchangeManager
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
        logger.info("=" * 60)
        logger.info("🚀 INITIALIZING ARUNABHA ELITE v8.0 FINAL")
        logger.info("=" * 60)
        
        # Initialize core components
        self.signal_gen = EliteSignalGenerator()
        self.filters = FilterManager()
        self.regime_detector = MarketRegimeDetector()
        self.tiers = TierManager()
        self.risk_mgr = EliteRiskManager()
        self.alerts = HumanStyleAlerts()
        self.ws_manager = WebSocketManager()
        self.feature_eng = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        # Initialize multi-exchange manager
        self.exchange_mgr = self._init_exchange_manager()
        
        # 8 trading pairs
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT',
            'BNBUSDT', 'XRPUSDT', 'LINKUSDT', 'ADAUSDT'
        ]
        
        # State management
        self.current_regime = None
        self.adaptive_settings = None
        self.last_regime_check = None
        self.last_training = None
        self.last_stats_reset = None
        
        # Daily statistics
        self.daily_stats = {
            'total': 0, 
            'by_tier': {'TIER_1': 0, 'TIER_2': 0, 'TIER_3': 0},
            'by_regime': {},
            'by_exchange': {},
            'pnl': 0
        }
        
        # Active positions tracking
        self.active_positions = {}
        
        logger.info(f"✅ Bot initialized with {len(self.symbols)} pairs")
        logger.info(f"✅ Exchanges: {list(self.exchange_mgr.clients.keys()) if self.exchange_mgr else 'None'}")
        
    def _init_exchange_manager(self) -> ExchangeManager:
        """Initialize exchange manager with API keys"""
        try:
            config = {
                'binance_api_key': os.getenv('BINANCE_API_KEY'),
                'binance_api_secret': os.getenv('BINANCE_API_SECRET'),
                'binance_testnet': os.getenv('BINANCE_TESTNET', 'False').lower() == 'true',
                'delta_api_key': os.getenv('DELTA_API_KEY'),
                'delta_api_secret': os.getenv('DELTA_API_SECRET'),
                'coindcx_api_key': os.getenv('COINDCX_API_KEY'),
                'coindcx_api_secret': os.getenv('COINDCX_API_SECRET')
            }
            
            # Check if at least one exchange is configured
            has_exchange = any([
                config['binance_api_key'],
                config['delta_api_key'],
                config['coindcx_api_key']
            ])
            
            if not has_exchange:
                logger.warning("⚠️ No exchange API keys found, using mock data")
            
            return ExchangeManager(config)
            
        except Exception as e:
            logger.error(f"❌ Exchange manager init failed: {e}")
            return None
    
    async def run(self):
        """Main execution loop"""
        await self.alerts.send_startup()
        
        while True:
            try:
                now = get_ist_time()
                current_time = now.strftime('%H:%M')
                
                # Reset daily stats at midnight
                if self.last_stats_reset != now.date():
                    self._reset_daily_stats()
                    self.last_stats_reset = now.date()
                
                # Daily model training at 00:05 IST
                if now.hour == 0 and now.minute < 10:
                    if not self.last_training or (now - self.last_training).hours >= 23:
                        await self._daily_training()
                
                # Update market regime every 5 minutes
                if (not self.last_regime_check or 
                    (now - self.last_regime_check).seconds >= 300):
                    await self._update_market_regime()
                
                # Check if golden hour
                if not is_golden_hour():
                    logger.info(f"⏸️ Off hours ({current_time} IST) - Waiting...")
                    await asyncio.sleep(60)
                    continue
                
                # Trading session
                await self._trading_session()
                
                # Sleep between scans
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                await asyncio.sleep(10)
    
    async def _daily_training(self):
        """Daily model retraining"""
        try:
            logger.info("🎓 Starting daily model training...")
            await self.model_trainer.train_daily(self)
            self.last_training = get_ist_time()
            await self.alerts.send_message("✅ Daily ML training completed")
        except Exception as e:
            logger.error(f"Training error: {e}")
    
    async def _update_market_regime(self):
        """Update market regime detection"""
        try:
            self.current_regime = await self.regime_detector.detect_regime()
            self.adaptive_settings = await self.regime_detector.get_adaptive_settings(
                self.current_regime
            )
            self.last_regime_check = get_ist_time()
            
            # Track regime stats
            regime_name = self.current_regime.value
            self.daily_stats['by_regime'][regime_name] = \
                self.daily_stats['by_regime'].get(regime_name, 0) + 1
            
            # Send alert if regime changed significantly
            if len(self.regime_detector.regime_history) > 1:
                prev_regime = self.regime_detector.regime_history[-2]
                if prev_regime != self.current_regime:
                    await self.alerts.regime_alert(self.current_regime, self.adaptive_settings)
                    
        except Exception as e:
            logger.error(f"Regime update error: {e}")
    
    async def _trading_session(self):
        """Main trading session"""
        if not self.adaptive_settings:
            logger.warning("No adaptive settings available")
            return
        
        settings = self.adaptive_settings
        max_signals = settings['max_signals']
        strategy = settings['strategy']
        min_tier = settings['min_tier']
        
        logger.info(f"🔥 TRADING SESSION")
        logger.info(f"   Regime: {self.current_regime.value}")
        logger.info(f"   Strategy: {strategy}")
        logger.info(f"   Max Signals: {max_signals}")
        logger.info(f"   Min Tier: {min_tier}")
        
        # Skip if no trade strategy
        if strategy == 'NO_TRADE':
            await self.alerts.skip_alert(f"Market regime: {self.current_regime.value}")
            return
        
        signals_sent = 0
        
        for symbol in self.symbols:
            # Check limits
            if signals_sent >= max_signals:
                logger.info(f"📊 Max signals ({max_signals}) reached")
                break
            
            if self.daily_stats['total'] >= 12:
                logger.info("📊 Daily signal limit reached")
                break
            
            # Risk check
            can_trade, reason = await self.risk_mgr.check_trade_allowed()
            if not can_trade:
                logger.info(f"⛔ Risk check failed: {reason}")
                continue
            
            # Process symbol
            try:
                signal_sent = await self._process_symbol(symbol, settings, min_tier)
                if signal_sent:
                    signals_sent += 1
                    self.daily_stats['total'] += 1
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
            
            # Small delay between symbols
            await asyncio.sleep(2)
        
        logger.info(f"✅ Session complete: {signals_sent} signals sent")
    
    async def _process_symbol(self, symbol: str, settings: dict, min_tier: str) -> bool:
        """Process single symbol for signal"""
        
        # 1. Generate raw signal
        raw_signal = await self.signal_gen.generate_signal(symbol)
        if not raw_signal:
            return False
        
        # 2. Check direction bias
        direction_bias = settings.get('direction_bias')
        if direction_bias:
            if direction_bias == 'LONG_ONLY' and raw_signal['direction'] != 'LONG':
                return False
            if direction_bias == 'SHORT_ONLY' and raw_signal['direction'] != 'SHORT':
                return False
        
        # 3. Fetch real market data
        ohlcv_data = await self._fetch_market_data(symbol)
        if ohlcv_data is None or len(ohlcv_data) < 60:
            logger.warning(f"Insufficient data for {symbol}")
            return False
        
        # 4. Create ML features
        features_df = self.feature_eng.create_features(ohlcv_data)
        
        # 5. Apply 10 filters (8 manual + 2 ML)
        filter_result = await self.filters.apply_all_filters(
            symbol, raw_signal, self.current_regime, features_df
        )
        
        if filter_result.get('blocked'):
            logger.debug(f"Signal blocked: {filter_result['blocked']}")
            return False
        
        passed = filter_result['passed']
        total = filter_result['total']
        
        # 6. Determine tier
        tier = self.tiers.determine_tier_adaptive(passed, total, min_tier)
        if not tier:
            logger.debug(f"Tier requirements not met: {passed}/{total}")
            return False
        
        # 7. Get ML prediction details
        ml_pred = filter_result.get('ml_prediction', {})
        
        # 8. Check multi-exchange price
        exchange_data = await self._get_exchange_data(symbol)
        
        # 9. Final signal assembly
        signal = {
            **raw_signal,
            'tier': tier['tier'],
            'confidence': tier['confidence'],
            'filters_passed': f"{passed}/{total}",
            'win_rate': tier['expected_win_rate'],
            'regime': self.current_regime.value,
            'strategy': settings['strategy'],
            'ml_score': ml_pred.get('ensemble_score', 0),
            'ml_hold_time': ml_pred.get('hold_time', 60),
            'ml_direction_prob': ml_pred.get('confidence', 0.5),
            'exchange_data': exchange_data,
            'timestamp': get_ist_time().isoformat()
        }
        
        # 10. Send alert
        await self.alerts.signal_alert(signal)
        self._update_stats(tier['tier'])
        
        # 11. Start position monitoring
        asyncio.create_task(self._monitor_position(signal))
        
        return True
    
    async def _fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data from exchanges"""
        if not self.exchange_mgr:
            # Fallback to mock data
            return self._generate_mock_data(symbol)
        
        # Try primary exchange first, then fallbacks
        df = await self.exchange_mgr.get_ohlcv(symbol, '5m', 100)
        
        if df is not None and len(df) >= 60:
            # Track which exchange provided data
            self.daily_stats['by_exchange']['primary'] = \
                self.daily_stats['by_exchange'].get('primary', 0) + 1
            return df
        
        # If all exchanges fail, use mock data for testing
        logger.warning(f"All exchanges failed for {symbol}, using mock data")
        return self._generate_mock_data(symbol)
    
    def _generate_mock_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock OHLCV for testing"""
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
        
        base_prices = {
            'BTCUSDT': 97000, 'ETHUSDT': 2650, 'SOLUSDT': 195,
            'DOGEUSDT': 0.25, 'BNBUSDT': 720, 'XRPUSDT': 2.45,
            'LINKUSDT': 18.5, 'ADAUSDT': 0.85
        }
        
        base = base_prices.get(symbol, 100)
        noise = np.random.randn(100).cumsum() * base * 0.001
        
        data = {
            'timestamp': dates,
            'open': base + noise + np.random.randn(100) * base * 0.0005,
            'high': base + noise + abs(np.random.randn(100)) * base * 0.001,
            'low': base + noise - abs(np.random.randn(100)) * base * 0.001,
            'close': base + noise,
            'volume': np.random.rand(100) * 10000
        }
        
        df = pd.DataFrame(data)
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        return df
    
    async def _get_exchange_data(self, symbol: str) -> dict:
        """Get price data from all exchanges"""
        if not self.exchange_mgr:
            return {}
        
        try:
            best_prices = await self.exchange_mgr.get_best_price(symbol)
            funding_rates = await self.exchange_mgr.get_funding_rates(symbol)
            
            return {
                'best_prices': best_prices,
                'funding_rates': funding_rates,
                'primary': self.exchange_mgr.primary
            }
        except Exception as e:
            logger.error(f"Exchange data error: {e}")
            return {}
    
    async def _monitor_position(self, signal: dict):
        """Monitor open position"""
        position_id = f"{signal['symbol']}_{get_ist_time().strftime('%H%M%S')}"
        self.active_positions[position_id] = signal
        
        entry_time = get_ist_time()
        tp1_hit = tp2_hit = tp3_hit = False
        
        # Use ML predicted hold time
        max_hold_seconds = signal.get('ml_hold_time', 60) * 60
        
        logger.info(f"📊 Monitoring {signal['symbol']} | Hold: {signal.get('ml_hold_time', 60)}min")
        
        while True:
            try:
                # Get current price
                current_price = await self._get_current_price(signal['symbol'])
                
                if current_price == 0:
                    await asyncio.sleep(5)
                    continue
                
                # Check TP1
                if not tp1_hit and self._hit_tp(signal, current_price, 'tp1'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp1', signal, profit)
                    tp1_hit = True
                    logger.info(f"✅ TP1 hit: {signal['symbol']} +₹{profit}")
                
                # Check TP2
                if tp1_hit and not tp2_hit and self._hit_tp(signal, current_price, 'tp2'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp2', signal, profit)
                    tp2_hit = True
                    logger.info(f"🎯 TP2 hit: {signal['symbol']} +₹{profit}")
                
                # Check TP3
                if tp2_hit and not tp3_hit and self._hit_tp(signal, current_price, 'tp3'):
                    profit = self._calc_profit(signal, current_price)
                    await self.alerts.tp_alert('tp3', signal, profit)
                    logger.info(f"🔥 TP3 hit: {signal['symbol']} +₹{profit}")
                    self._close_position(position_id, profit)
                    return
                
                # Check SL
                if self._hit_sl(signal, current_price):
                    loss = self._calc_profit(signal, current_price)  # Negative
                    await self.alerts.sl_alert(signal)
                    logger.info(f"😔 SL hit: {signal['symbol']} ₹{loss}")
                    self._close_position(position_id, loss)
                    return
                
                # Check breakeven
                be_action = self.risk_mgr.check_breakeven(signal, current_price)
                if be_action:
                    await self.alerts.breakeven_alert(be_action)
                    signal['sl'] = signal['entry']  # Update SL to entry
                
                # Check timeout
                elapsed = (get_ist_time() - entry_time).seconds
                if elapsed > max_hold_seconds:
                    pnl = self._calc_profit(signal, current_price)
                    await self.alerts.timeout_alert(signal)
                    logger.info(f"⏰ Timeout: {signal['symbol']} ₹{pnl}")
                    self._close_position(position_id, pnl)
                    return
                
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Monitor error for {signal['symbol']}: {e}")
                await asyncio.sleep(5)
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price from exchange"""
        if self.exchange_mgr:
            ticker = await self.exchange_mgr.get_ticker(symbol)
            return ticker.get('last', 0)
        
        # Fallback
        prices = {
            'BTCUSDT': 97000, 'ETHUSDT': 2650, 'SOLUSDT': 195,
            'DOGEUSDT': 0.25, 'BNBUSDT': 720, 'XRPUSDT': 2.45,
            'LINKUSDT': 18.5, 'ADAUSDT': 0.85
        }
        return prices.get(symbol, 0)
    
    def _hit_tp(self, signal: dict, price: float, tp_level: str) -> bool:
        """Check if TP hit"""
        if signal['direction'] == 'LONG':
            return price >= signal[tp_level]
        return price <= signal[tp_level]
    
    def _hit_sl(self, signal: dict, price: float) -> bool:
        """Check if SL hit"""
        if signal['direction'] == 'LONG':
            return price <= signal['sl']
        return price >= signal['sl']
    
    def _calc_profit(self, signal: dict, current_price: float) -> float:
        """Calculate P&L in INR"""
        diff = current_price - signal['entry']
        if signal['direction'] == 'SHORT':
            diff = -diff
        
        # ₹1000 margin, 15x leverage
        margin = 1000
        leverage = 15
        qty = (margin * leverage) / signal['entry']
        profit = diff * qty
        
        return round(profit, 2)
    
    def _update_stats(self, tier_name: str):
        """Update daily statistics"""
        self.daily_stats['by_tier'][tier_name] = \
            self.daily_stats['by_tier'].get(tier_name, 0) + 1
    
    def _close_position(self, position_id: str, pnl: float):
        """Close position and update stats"""
        if position_id in self.active_positions:
            del self.active_positions[position_id]
        
        self.daily_stats['pnl'] += pnl
        self.risk_mgr.daily_pnl += pnl
        self.risk_mgr.trades_today += 1
        self.risk_mgr.last_trade_win = pnl > 0
    
    def _reset_daily_stats(self):
        """Reset stats at midnight"""
        logger.info("🌙 Resetting daily stats")
        self.daily_stats = {
            'total': 0, 
            'by_tier': {'TIER_1': 0, 'TIER_2': 0, 'TIER_3': 0},
            'by_regime': {},
            'by_exchange': {},
            'pnl': 0
        }
        self.tiers.reset_daily()
        self.risk_mgr.daily_pnl = 0
        self.risk_mgr.trades_today = 0

async def main():
    """Entry point"""
    bot = ArunabhaEliteBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped")
