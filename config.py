"""
ARUNABHA ELITE v8.3 FINAL - PRODUCTION CONFIGURATION
24/7 Trading Mode - Sleep Hours: 1 AM - 7 AM IST
UPDATED: Lenient TIER_3 for signal generation - TESTING MODE
"""

import os
from dotenv import load_dotenv

load_dotenv()

BOT_CONFIG = {
    'name': 'ARUNABHA ELITE v8.3 FINAL',
    'version': '8.3.0',
    'mode': 'LIVE',
    'timezone': 'Asia/Kolkata',
    'currency': 'INR',
    'rating': '95/100',
    'description': 'Production-grade ML trading bot - 24/7 mode - TESTING'
}

TELEGRAM = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
    'admin_id': os.getenv('TELEGRAM_ADMIN_ID', '')
}

EXCHANGE_CONFIG = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY', ''),
        'api_secret': os.getenv('BINANCE_API_SECRET', ''),
        'testnet': False,
        'enabled': bool(os.getenv('BINANCE_API_KEY'))
    },
    'delta': {
        'api_key': os.getenv('DELTA_API_KEY', ''),
        'api_secret': os.getenv('DELTA_API_SECRET', ''),
        'enabled': bool(os.getenv('DELTA_API_KEY'))
    },
    'coindcx': {
        'api_key': os.getenv('COINDCX_API_KEY', ''),
        'api_secret': os.getenv('COINDCX_API_SECRET', ''),
        'enabled': bool(os.getenv('COINDCX_API_KEY'))
    }
}

TRADING = {
    'symbols': [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT',
        'BNBUSDT', 'XRPUSDT', 'LINKUSDT', 'ADAUSDT'
    ],
    'timeframes': {
        'primary': '5m',
        'confirmation': '15m',
        'trend': '1h'
    },
    'leverage': 15,
    'max_daily_signals': 12,
    'max_hourly_trades_per_pair': 3,
    'daily_loss_limit': 0.03,
    'daily_loss_limit_percent': 0.03,
    'max_spread_percent': 0.002,
    'risk_per_trade_tier1': 0.01,
    'risk_per_trade_tier2': 0.008,
    'risk_per_trade_tier3': 0.005,
    'atr_multiplier_sl': 1.5,
    'atr_multiplier_tp1': 2.0,
    'atr_multiplier_tp2': 3.0,
    'atr_multiplier_tp3': 4.0,
    'breakeven_trigger': 0.015
}

# ✅ UPDATED: Lenient tier requirements for TESTING
TIER_SETTINGS = {
    'TIER_1': {
        'label': '💎 DIAMOND',
        'min_filters': 8,
        'max_filters': 10,
        'confidence': 92,
        'expected_win_rate': '88%',
        'risk_per_trade': 0.01,
        'max_daily': 3
    },
    'TIER_2': {
        'label': '🥇 GOLD',
        'min_filters': 5,
        'max_filters': 7,
        'confidence': 82,
        'expected_win_rate': '78%',
        'risk_per_trade': 0.008,
        'max_daily': 4
    },
    'TIER_3': {
        'label': '🥈 SILVER',
        'min_filters': 2,  # ✅ Lowered from 3 to 2
        'max_filters': 6,  # ✅ Increased max
        'confidence': 60,  # ✅ Lowered for more signals
        'expected_win_rate': '55%',  # ✅ Realistic for testing
        'risk_per_trade': 0.005,
        'max_daily': 10  # ✅ Increased for testing
    }
}

# ✅ UPDATED: Very lenient regime settings for TESTING
REGIME_SETTINGS = {
    'TRENDING_BULL': {
        'enabled_filters': ['structure', 'volume', 'liquidity', 'correlation', 'funding', 'liquidation', 'mtf', 'session'],
        'min_tier': 'TIER_2',
        'max_signals': 10,
        'strategy': 'TREND_FOLLOW',
        'direction_bias': None
    },
    'TRENDING_BEAR': {
        'enabled_filters': ['structure', 'volume', 'liquidity', 'correlation', 'funding', 'liquidation', 'mtf', 'session'],
        'min_tier': 'TIER_2',
        'max_signals': 10,
        'strategy': 'TREND_FOLLOW',
        'direction_bias': None
    },
    'RANGING': {
        'enabled_filters': ['structure', 'volume', 'liquidity', 'funding', 'mtf', 'session'],
        'min_tier': 'TIER_2',
        'max_signals': 8,
        'strategy': 'MEAN_REVERSION',
        'direction_bias': None
    },
    'VOLATILE': {
        'enabled_filters': ['structure', 'liquidity', 'liquidation', 'mtf', 'session'],
        'min_tier': 'TIER_1',
        'max_signals': 5,
        'strategy': 'BREAKOUT',
        'direction_bias': None
    },
    'EXTREME_FEAR': {
        'enabled_filters': ['structure', 'liquidity', 'liquidation', 'mtf', 'session'],
        'direction_bias': 'LONG_ONLY',
        'min_tier': 'TIER_2',
        'max_signals': 6,
        'strategy': 'CONTRARIAN_LONG'
    },
    'EXTREME_GREED': {
        'enabled_filters': ['structure', 'liquidity', 'liquidation', 'mtf', 'session'],
        'direction_bias': 'SHORT_ONLY',
        'min_tier': 'TIER_2',
        'max_signals': 6,
        'strategy': 'CONTRARIAN_SHORT'
    },
    'LOW_VOLATILITY': {
        'enabled_filters': ['structure', 'volume', 'liquidity', 'funding', 'mtf', 'session'],
        'min_tier': 'TIER_1',
        'max_signals': 4,
        'strategy': 'SCALP_ONLY',
        'direction_bias': None
    },
    # ✅ CHOPPY: Very lenient for TESTING - only 3 filters, TIER_3 accepted
    'CHOPPY': {
        'enabled_filters': ['structure', 'volume', 'session'],  # Only 3 critical filters
        'min_tier': 'TIER_3',  # ✅ Accept TIER_3 (2+ filters passed)
        'max_signals': 8,      # ✅ More signals allowed
        'strategy': 'MEAN_REVERSION',
        'direction_bias': None
    }
}

# DEPRECATED: No longer used in 24/7 mode
GOLDEN_HOURS = {
    'enabled': False,
    'london_open': ('13:30', '14:30'),
    'ny_open': ('19:00', '20:30'),
    'london_close': ('21:30', '22:30'),
    'weekend_enabled': True
}

# Sleep hours configuration
SLEEP_HOURS = {
    'enabled': True,
    'start_hour': 1,
    'end_hour': 7,
    'reason': 'Low liquidity and high volatility risk during these hours'
}

ML_CONFIG = {
    'model_type': 'random_forest',
    'sequence_length': 60,
    'prediction_threshold': 0.60,  # ✅ Lowered from 0.65 for more signals
    'retrain_interval_hours': 24,
    'min_training_samples': 1000,
    'model_path': 'models/',
    'features': [
        'returns', 'log_returns', 'rsi', 'macd', 'macd_hist',
        'ema_9_21', 'ema_21_50', 'volume_ratio', 'atr', 'atr_ratio',
        'bb_width', 'price_vs_high', 'price_vs_low'
    ]
}

SAFETY = {
    'max_drawdown_percent': 10,
    'daily_loss_limit_percent': 3,
    'max_spread_percent': 0.2,
    'emergency_stop_pnl_percent': -5,
    'max_position_size_percent': 50
}

# ✅ UPDATED: Debug mode for testing
DEBUG = {
    'verbose_logging': True,
    'log_signal_attempts': True,
    'log_filter_results': True,
    'log_exchange_calls': True,  # ✅ Enabled for testing
    'test_mode': False
}
