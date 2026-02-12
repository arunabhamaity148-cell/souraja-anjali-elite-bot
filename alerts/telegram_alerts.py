"""
Human-Style Telegram Alerts
"""

import logging
import random
from telegram import Bot
from config import TELEGRAM, TIER_SETTINGS

logger = logging.getLogger("ALERTS")

class HumanStyleAlerts:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM['bot_token'])
        self.chat_id = TELEGRAM['chat_id']
        self.signal_count = 0
        
    async def send_startup(self):
        msg = """
🚀 *ARUNABHA ELITE v8.0 ML FINAL*

✅ 10 Filters (8+2 ML)
✅ Auto Regime Detection
✅ 50+ Features
✅ Daily Auto-Train
✅ 92/100 Rating

Bot ready! 💪
        """
        await self._send(msg)
    
    async def regime_alert(self, regime, settings):
        emoji_map = {
            'TRENDING_BULL': '📈', 'TRENDING_BEAR': '📉',
            'RANGING': '↔️', 'VOLATILE': '⚡',
            'EXTREME_FEAR': '😱', 'EXTREME_GREED': '🤑',
            'LOW_VOLATILITY': '😴', 'CHOPPY': '🌊'
        }
        emoji = emoji_map.get(regime.value, '⚠️')
        
        msg = f"""
{emoji} *REGIME: {regime.value}*

Strategy: {settings['strategy']}
Max Signals: {settings['max_signals']}
Direction: {settings['direction_bias'] or 'Both'}
Min Tier: {settings['min_tier']}

Bot auto-adjusted! 💪
        """
        await self._send(msg)
    
    async def signal_alert(self, signal):
        self.signal_count += 1
        emoji = '🚀' if signal['direction'] == 'LONG' else '🔴'
        tier_cfg = TIER_SETTINGS[signal['tier']]
        
        msg = f"""
{emoji} *#{self.signal_count}* {tier_cfg['label']}
*{signal['symbol']} {signal['direction']}*

Regime: {signal['regime']}
Confidence: {signal['confidence']}%
Win Rate: {signal['win_rate']}
Filters: {signal['filters_passed']}
ML Score: {signal.get('ml_score', 0)}

🎯 *ENTRY:* `{signal['entry']}`
🛑 *SL:* `{signal['sl']}`
✅ *TP1:* `{signal['tp1']}`
✅ *TP2:* `{signal['tp2']}`
✅ *TP3:* `{signal['tp3']}`

Hold: {signal.get('ml_hold_time', 60)}min | Leverage: 15x

{random.choice([
    "এইটা win করবে, trust me 💪",
    "Smart Money আমাদের সাথে 🐋",
    "Top 1% setup, miss করিস না 🔥",
    "আমি বলি এইটা win 💯"
])}
        """
        await self._send(msg)
    
    async def tp_alert(self, level, signal, profit):
        msgs = {
            'tp1': f"✅ *TP1!* +₹{profit}\nSL breakeven করে দাও! 💪",
            'tp2': f"🎯 *TP2!* +₹{profit}\nPartial close করো 🚀",
            'tp3': f"🔥 *TP3!* +₹{profit}\nFull close, king! 👑"
        }
        await self._send(msgs.get(level, "TP hit!"))
    
    async def sl_alert(self, signal):
        await self._send(f"😔 *SL* - {signal['symbol']}\nকাল ঠিক হবে 💪")
    
    async def breakeven_alert(self, action):
        await self._send(f"🛡️ {action['message']}")
    
    async def timeout_alert(self, signal):
        await self._send(f"⏰ *Timeout* - {signal['symbol']}")
    
    async def skip_alert(self, reason):
        await self._send(f"⏸️ *Skip*: {reason}")
    
    async def _send(self, msg):
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Telegram error: {e}")
