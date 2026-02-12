"""
╔══════════════════════════════════════════════════════════════════╗
║           ARUNABHA ELITE v8.0 FINAL - CONFIGURATION              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ═════════════════════════════════════════════════════════════════
# BOT IDENTITY
# ═════════════════════════════════════════════════════════════════

BOT_CONFIG = {
    'name': 'ARUNABHA ELITE v8.0 FINAL',
    'version': '8.0.0',
    'mode': 'LIVE',
    'timezone': 'Asia/Kolkata',
    'currency': 'INR',
    'rating': '95/100',
    'description': 'Multi-Exchange ML Trading Bot'
}

# ═════════════════════════════════════════════════════════════════
# TELEGRAM SETTINGS
# ═════════════════════════════════════════════════════════════════

TELEGRAM = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
    'admin_id': os.getenv('TELEGRAM_ADMIN_ID', ''),
    'alert_cooldown_seconds': 30
}

# ═════════════════════════════════════════════════════════════════
# EXCHANGE API CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════

EXCHANGE_CONFIG = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY', ''),
        'api_secret': os.getenv('BINANCE_API_SECRET', ''),
        'testnet': os.getenv('BINANCE_TESTNET', 'False').lower() == 'true',
        'enabled': bool(os.getenv('BINANCE_API_KEY')),
        'weight': 1.0  # Priority weight
    },
    'delta': {
        'api_key': os.getenv('DELTA_API_KEY', ''),
        'api_secret': os.getenv('DELTA_API_SECRET', ''),
        'enabled': bool(os.getenv('DELTA_API_KEY')),
        'weight': 0.8
    },
    'coindcx': {
        'api_key': os.getenv('COINDCX_API_KEY', ''),
        'api_secret': os.getenv('COINDCX_API_SECRET', ''),
        'enabled': bool(os.getenv('COINDCX_API_KEY')),
        'weight': 0.6
    }
}

# Primary exchange for data
PRIMARY_EXCHANGE = 'binance'

# Fallback priority
EXCHANGE_PRIORITY = ['binance', 'delta', 'coindcx']

# ═════════════════════════════════════════════════════════════════
# TRADING PARAMETERS
# ═════════════════════════════════════════════════════════════════

TRADING = {
    # 8 Trading pairs
    'symbols': [
        'BTCUSDT',   # Bitcoin
        'ETHUSDT',   # Ethereum
        'SOLUSDT',   # Solana
        'DOGEUSDT',  # Dogecoin
        'BNBUSDT',   # Binance Coin
        'XRPUSDT',   # Ripple
        'LINKUSDT',  # Chainlink
        'ADAUSDT'    # Cardano
    ],
    
    # Timeframes
    'timeframes': {
        'primary': '5m',
        'confirmation': '15m',
        'trend': '1h'
    },
    
    # Position sizing
    'leverage': 15,
    'margin_per_trade': 1000,  # INR
    'max_position_size': 5000,  # INR
    
    # Daily limits
    'max_daily_signals': 12,
    'max_daily_loss': 300,  # INR
    'max_daily_trades': 15,
    
    # Risk per trade
    'risk_per_trade_tier1': 0.005,  # 0.5%
    'risk_per_trade_tier2': 0.004,  # 0.4%
    'risk_per_trade_tier3': 0.003,  # 0.3%
    
    # Breakeven settings
    'breakeven_trigger': 0.005,  # 0.5% profit
    'trailing_trigger': 0.01,    # 1% profit
    
    # Time limits
    'max_hold_time_minutes': 120,
    'default_hold_time': 60,
    
    # Cooldowns
    'symbol_cooldown_minutes': 30,
    'loss_cooldown_minutes': 15
}

# ═════════════════════════════════════════════════════════════════
# TIER SYSTEM CONFIGURATION
# ═════════════════════════════════════════════════════════════════

TIER_SETTINGS = {
    'TIER_1': {
        'name': 'DIAMOND',
        'label': '💎 DIAMOND',
        'min_filters': 8,      # 8-10 filters
        'max_filters': 10,
        'confidence': 92,
        'expected_win_rate': '88%',
        'risk_per_trade': 0.005,
        'max_daily_signals': 3,
        'min_rr': 1.5,
        'color': '🔵'
    },
    'TIER_2': {
        'name': 'GOLD',
        'label': '🥇 GOLD',
        'min_filters': 6,      # 6-7 filters
        'max_filters': 7,
        'confidence': 82,
        'expected_win_rate': '78%',
        'risk_per_trade': 0.004,
        'max_daily_signals': 4,
        'min_rr': 1.3,
        'color': '🟡'
    },
    'TIER_3': {
        'name': 'SILVER',
        'label': '🥈 SILVER',
        'min_filters': 5,      # 5 filters
        'max_filters': 5,
        'confidence': 72,
        'expected_win_rate': '68%',
        'risk_per_trade': 0.003,
        'max_daily_signals': 3,
        'min_rr': 1.2,
        'color': '⚪'
    }
}

# ═════════════════════════════════════════════════════════════════
# MARKET REGIME SETTINGS (8 STATES)
# ═════════════════════════════════════════════════════════════════

REGIME_SETTINGS = {
    'TRENDING_BULL': {
        'name': 'Trending Bullish',
        'emoji': '📈',
        'enabled_filters': ['structure', 'volume', 'liquidity', 'correlation', 'funding', 'liquidation', 'mtf', 'session'],
        'disabled_filters': [],
        'direction_bias': None,
        'min_tier': 'TIER_2',
        'max_signals': 10,
        'risk_multiplier': 1.0,
        'strategy': 'TREND_FOLLOW',
        'description': 'Strong uptrend, follow the trend'
    },
    'TRENDING_BEAR': {
        'name': 'Trending Bearish',
        'emoji': '📉',
        'enabled_filters': ['structure', 'volume', 'liquidity', 'correlation', 'funding', 'liquidation', 'mtf', 'session'],
        'disabled_filters': [],
        'direction_bias': None,
        'min_tier': 'TIER_2',
        'max_signals': 10,
        'risk_multiplier': 1.0,
        'strategy': 'TREND_FOLLOW',
        'description': 'Strong downtrend, follow the trend'
    },
    'RANGING': {
        'name': 'Range Bound',
        'emoji': '↔️',
        'enabled_filters': ['structure', 'volume', 'liquidity', 'funding', 'session'],
        'disabled_filters': ['correlation', 'mtf'],
        'direction_bias': None,
        'min_tier': 'TIER_2',
        'max_signals': 8,
        'risk_multiplier': 0.8,
        'strategy': 'MEAN_REVERSION',
        'description': 'Buy low, sell high in range'
    },
    'VOLATILE': {
        'name': 'High Volatility',
        'emoji': '⚡',
        'enabled_filters': ['structure', 'liquidity', 'liquidation', 'session'],
        'disabled_filters': ['volume', 'correlation', 'funding', 'mtf'],
        'direction_bias': None,
        'min_tier': 'TIER_1',
        'max_signals': 5,
        'risk_multiplier': 0.5,
        'strategy': 'BREAKOUT',
        'description': 'Wait for confirmation, small size'
    },
    'EXTREME_FEAR': {
        'name': 'Extreme Fear',
        'emoji': '😱',
        'enabled_filters': ['structure', 'liquidity', 'liquidation', 'session'],
        'disabled_filters': ['volume', 'correlation', 'funding', 'mtf'],
        'direction_bias': 'LONG_ONLY',
        'min_tier': 'TIER_2',
        'max_signals': 6,
        'risk_multiplier': 0.6,
        'strategy': 'CONTRARIAN_LONG',
        'description': 'Fear & Greed < 20, only longs'
    },
    'EXTREME_GREED': {
        'name': 'Extreme Greed',
        'emoji': '🤑',
        'enabled_filters': ['structure', 'liquidity', 'liquidation', 'session'],
        'disabled_filters': ['volume', 'correlation', 'funding', 'mtf'],
        'direction_bias': 'SHORT_ONLY',
        'min_tier': 'TIER_2',
        'max_signals': 6,
        'risk_multiplier': 0.6,
        'strategy': 'CONTRARIAN_SHORT',
        'description': 'Fear & Greed > 80, only shorts'
    },
    'LOW_VOLATILITY': {
        'name': 'Low Volatility',
        'emoji': '😴',
        'enabled_filters': ['structure', 'volume', 'funding', 'session'],
        'disabled_filters': ['liquidity', 'correlation', 'liquidation', 'mtf'],
        'direction_bias': None,
        'min_tier': 'TIER_1',
        'max_signals': 4,
        'risk_multiplier': 0.4,
        'strategy': 'SCALP_ONLY',
        'description': 'Tight stops, quick profits'
    },
    'CHOPPY': {
        'name': 'Choppy/Sideways',
        'emoji': '🌊',
        'enabled_filters': ['structure', 'session'],
        'disabled_filters': ['volume', 'liquidity', 'correlation', 'funding', 'liquidation', 'mtf'],
        'direction_bias': None,
        'min_tier': 'TIER_1',
        'max_signals': 2,
        'risk_multiplier': 0.3,
        'strategy': 'NO_TRADE',
        'description': 'Avoid trading, stay out'
    }
}

# ═════════════════════════════════════════════════════════════════
# FILTER CONFIGURATION (10 FILTERS)
# ═════════════════════════════════════════════════════════════════

FILTER_CONFIG = {
    # Manual Filters (8)
    'structure': {
        'name': 'Market Structure',
        'weight': 1.2,
        'description': 'BOS/CHoCH alignment'
    },
    'volume': {
        'name': 'Volume Profile',
        'weight': 1.0,
        'description': 'VWAP/POC confirmation'
    },
    'liquidity': {
        'name': 'Liquidity Analysis',
        'weight': 1.1,
        'description': 'Stop hunt detection'
    },
    'correlation': {
        'name': 'Correlation Matrix',
        'weight': 0.9,
        'description': 'BTC correlation check'
    },
    'funding': {
        'name': 'Funding Rate',
        'weight': 0.8,
        'description': 'Funding arbitrage'
    },
    'liquidation': {
        'name': 'Liquidation Levels',
        'weight': 1.0,
        'description': 'Liquidation cluster alignment'
    },
    'mtf': {
        'name': 'Multi-Timeframe',
        'weight': 1.1,
        'description': '5m/15m/1h confluence'
    },
    'session': {
        'name': 'Session Quality',
        'weight': 0.9,
        'description': 'Volume/volatility check'
    },
    
    # ML Filters (2)
    'ml_direction': {
        'name': 'ML Direction',
        'weight': 1.3,
        'description': 'Random Forest prediction'
    },
    'ml_volatility': {
        'name': 'ML Volatility',
        'weight': 1.0,
        'description': 'Volatility regime prediction'
    }
}

# ═════════════════════════════════════════════════════════════════
# GOLDEN HOURS (IST)
# ═════════════════════════════════════════════════════════════════

GOLDEN_HOURS = {
    'london_open': {
        'start': '13:30',
        'end': '14:30',
        'weight': 1.5,
        'description': 'London market open'
    },
    'ny_open': {
        'start': '19:00',
        'end': '20:30',
        'weight': 1.8,
        'description': 'New York market open'
    },
    'london_close': {
        'start': '21:30',
        'end': '22:30',
        'weight': 1.3,
        'description': 'London market close'
    },
    'weekend_enabled': False,
    'off_hours_monitoring': True  # Monitor for >5% moves
}

# ═════════════════════════════════════════════════════════════════
# MACHINE LEARNING CONFIG
# ═════════════════════════════════════════════════════════════════

ML_CONFIG = {
    'model_type': 'random_forest',
    'sequence_length': 60,
    'prediction_threshold': 0.65,
    'retrain_interval_hours': 24,
    'min_training_samples': 1000,
    'feature_count': 50,
    'model_path': 'models/',
    
    # Feature groups
    'feature_groups': {
        'price': 10,
        'volume': 8,
        'volatility': 8,
        'momentum': 10,
        'trend': 8,
        'microstructure': 6
    },
    
    # Hyperparameters
    'rf_n_estimators': 100,
    'rf_max_depth': 10,
    'rf_min_samples_split': 20
}

# ═════════════════════════════════════════════════════════════════
# ALERT MESSAGES (BENGALI)
# ═════════════════════════════════════════════════════════════════

MESSAGES = {
    'startup': "🚀 Bot চালু হয়েছে!",
    'golden_hour': "🔥 Golden hour শুরু!",
    'no_trade': "⏸️ আজ trade নেই",
    'tp_hit': "✅ Profit! 💪",
    'sl_hit': "😔 SL hit, কাল ঠিক হবে",
    'breakeven': "🛡️ Breakeven set",
    'timeout': "⏰ Time up",
    'new_signal': "🎯 নতুন signal!",
    'regime_change': "📊 Market change detected"
}

# ═════════════════════════════════════════════════════════════════
# LOGGING CONFIG
# ═════════════════════════════════════════════════════════════════

LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    'file': 'logs/bot.log',
    'max_size_mb': 10,
    'backup_count': 5
}

# ═════════════════════════════════════════════════════════════════
# SAFETY LIMITS
# ═════════════════════════════════════════════════════════════════

SAFETY = {
    'max_drawdown_percent': 10,
    'daily_loss_limit_percent': 5,
    'single_trade_max_loss': 200,  # INR
    'consecutive_losses_limit': 3,
    'cooldown_after_loss_minutes': 30,
    'emergency_stop_pnl': -500  # Stop bot if reached
}
