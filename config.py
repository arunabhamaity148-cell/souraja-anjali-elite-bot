"""
ARUNABHA ELITE v8.0 ML FINAL - CONFIG
"""

import os
from dotenv import load_dotenv

load_dotenv()

BOT_CONFIG = {
    'name': 'ARUNABHA ELITE v8.0 ML FINAL',
    'version': '8.0.0',
    'mode': 'LIVE',
    'timezone': 'Asia/Kolkata',
    'currency': 'INR',
    'rating': '92/100'
}

TELEGRAM = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'admin_id': os.getenv('TELEGRAM_ADMIN_ID')
}

BINANCE = {
    'api_key': os.getenv('BINANCE_API_KEY'),
    'api_secret': os.getenv('BINANCE_API_SECRET'),
    'testnet': False,
    'futures': True,
    'websocket_url': 'wss://fstream.binance.com/ws'
}

TRADING = {
    'symbols': [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT',
        'BNBUSDT', 'XRPUSDT', 'LINKUSDT', 'ADAUSDT'
    ],
    'timeframes': ['5m', '15m', '1h'],
    'leverage': 15,
    'margin_per_trade': 1000,
    'max_daily_signals': 12,
    'daily_loss_limit': 300,
    'breakeven_trigger': 0.005
}

TIER_SETTINGS = {
    'TIER_1': {
        'min_filters': 8,
        'max_filters': 10,
        'confidence': 92,
        'win_rate': '88%',
        'risk': 0.005,
        'label': '💎 DIAMOND',
        'max_daily': 3
    },
    'TIER_2': {
        'min_filters': 6,
        'max_filters': 7,
        'confidence': 82,
        'win_rate': '78%',
        'risk': 0.004,
        'label': '🥇 GOLD',
        'max_daily': 4
    },
    'TIER_3': {
        'min_filters': 5,
        'max_filters': 5,
        'confidence': 72,
        'win_rate': '68%',
        'risk': 0.003,
        'label': '🥈 SILVER',
        'max_daily': 3
    }
}

REGIME_SETTINGS = {
    'TRENDING_BULL': {
        'enabled_filters': ['structure', 'volume', 'liquidity', 'correlation', 'funding', 'liquidation', 'mtf', 'session'],
        'disabled_filters': [],
        'direction_bias': None,
        'min_tier': 'TIER_2',
        'max_signals': 10,
        'risk_multiplier': 1.0,
        'strategy': 'TREND_FOLLOW'
    },
    'TRENDING_BEAR': {
        'enabled_filters': ['structure', 'volume', 'liquidity', 'correlation', 'funding', 'liquidation', 'mtf', 'session'],
        'disabled_filters': [],
        'direction_bias': None,
        'min_tier': 'TIER_2',
        'max_signals': 10,
        'risk_multiplier': 1.0,
        'strategy': 'TREND_FOLLOW'
    },
    'RANGING': {
        'enabled_filters': ['structure', 'volume', 'liquidity', 'funding', 'session'],
        'disabled_filters': ['correlation', 'mtf'],
        'direction_bias': None,
        'min_tier': 'TIER_2',
        'max_signals': 8,
        'risk_multiplier': 0.8,
        'strategy': 'MEAN_REVERSION'
    },
    'VOLATILE': {
        'enabled_filters': ['structure', 'liquidity', 'liquidation', 'session'],
        'disabled_filters': ['volume', 'correlation', 'funding', 'mtf'],
        'direction_bias': None,
        'min_tier': 'TIER_1',
        'max_signals': 5,
        'risk_multiplier': 0.5,
        'strategy': 'BREAKOUT'
    },
    'EXTREME_FEAR': {
        'enabled_filters': ['structure', 'liquidity', 'liquidation', 'session'],
        'disabled_filters': ['volume', 'correlation', 'funding', 'mtf'],
        'direction_bias': 'LONG_ONLY',
        'min_tier': 'TIER_2',
        'max_signals': 6,
        'risk_multiplier': 0.6,
        'strategy': 'CONTRARIAN_LONG'
    },
    'EXTREME_GREED': {
        'enabled_filters': ['structure', 'liquidity', 'liquidation', 'session'],
        'disabled_filters': ['volume', 'correlation', 'funding', 'mtf'],
        'direction_bias': 'SHORT_ONLY',
        'min_tier': 'TIER_2',
        'max_signals': 6,
        'risk_multiplier': 0.6,
        'strategy': 'CONTRARIAN_SHORT'
    },
    'LOW_VOLATILITY': {
        'enabled_filters': ['structure', 'volume', 'funding', 'session'],
        'disabled_filters': ['liquidity', 'correlation', 'liquidation', 'mtf'],
        'direction_bias': None,
        'min_tier': 'TIER_1',
        'max_signals': 4,
        'risk_multiplier': 0.4,
        'strategy': 'SCALP_ONLY'
    },
    'CHOPPY': {
        'enabled_filters': ['structure', 'session'],
        'disabled_filters': ['volume', 'liquidity', 'correlation', 'funding', 'liquidation', 'mtf'],
        'direction_bias': None,
        'min_tier': 'TIER_1',
        'max_signals': 2,
        'risk_multiplier': 0.3,
        'strategy': 'NO_TRADE'
    }
}

GOLDEN_HOURS = {
    'london_open': ('13:30', '14:30'),
    'ny_open': ('19:00', '20:30'),
    'london_close': ('21:30', '22:30'),
    'weekend_enabled': False
}

ML_CONFIG = {
    'sequence_length': 60,
    'prediction_threshold': 0.65,
    'retrain_interval_hours': 24,
    'min_training_samples': 1000,
    'feature_count': 50
}
