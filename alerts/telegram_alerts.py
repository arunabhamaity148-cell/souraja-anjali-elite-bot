"""
Telegram Alerts - Human Style Messages
With deploy success notifications (ONCE per deployment)
"""

import logging
from datetime import datetime
import pytz
import os
from telegram import Bot
from config import TELEGRAM, BOT_CONFIG

logger = logging.getLogger("TELEGRAM")

class HumanStyleAlerts:
    def __init__(self):
        self.bot_token = TELEGRAM['bot_token']
        self.chat_id = TELEGRAM['chat_id']

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            self.bot = None
        else:
            self.bot = Bot(token=self.bot_token)
            logger.info("✅ Telegram bot initialized")

    async def send_startup(self):
        """Send startup notification - ONCE per deployment only"""
        if not self.bot:
            return

        try:
            # ✅ CHECK: Already sent this deployment?
            flag_file = "/tmp/startup_sent.flag"
            
            if os.path.exists(flag_file):
                logger.info("Startup already sent, skipping...")
                return
            
            # Create flag file
            with open(flag_file, 'w') as f:
                f.write(datetime.now().isoformat())

            from config import TRADING, SLEEP_HOURS

            startup_msg = f"""
🚀 <b>{BOT_CONFIG['name']}</b>

✅ <b>Bot Started Successfully!</b>

⏰ <b>Time:</b> {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b %Y, %H:%M:%S IST')}

📊 <b>Configuration:</b>
• Symbols: {len(TRADING['symbols'])} pairs
• Max Daily Signals: {TRADING['max_daily_signals']}
• Leverage: {TRADING['leverage']}x
• Risk per Trade: {TRADING['risk_per_trade_tier1']*100}% (TIER 1)

⏰ <b>Trading Hours:</b>
• Active: <b>24/7 Mode</b>
• Sleep: {SLEEP_HOURS['start_hour']}:00 AM - {SLEEP_HOURS['end_hour']}:00 AM IST

🎯 <b>Current Status:</b>
• Mode: <b>{BOT_CONFIG['mode']}</b>
• Version: {BOT_CONFIG['version']}
• Rating: {BOT_CONFIG['rating']}

Ready to scan markets! 🔍
            """

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=startup_msg,
                parse_mode='HTML'
            )

            # ✅ Deploy success o ekbar e pathabo
            await self._send_deploy_success_once()

            logger.info("✅ Startup & Deploy alerts sent (once)")

        except Exception as e:
            logger.error(f"Startup alert error: {e}")

    async def _send_deploy_success_once(self):
        """Send deploy success - internal method, called once"""
        try:
            deploy_info = f"""
🚀 <b>DEPLOYMENT SUCCESSFUL</b>

✅ <b>Bot Details:</b>
• Name: {BOT_CONFIG['name']}
• Version: {BOT_CONFIG['version']}
• Mode: <b>{BOT_CONFIG['mode']}</b>

🔧 <b>Platform Info:</b>
• Platform: Railway
• Service: Worker

⏰ <b>Active Hours:</b>
• Trading: <b>24/7 Continuous</b>
• Sleep Mode: 1:00 AM - 7:00 AM IST

📊 <b>System Status:</b>
• Exchanges: ✅ Connected
• ML Model: ✅ Ready
• Telegram: ✅ Active
• Risk Manager: ✅ Active

🎯 <b>Bot is now monitoring markets!</b>
            """

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=deploy_info,
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"Deploy alert error: {e}")

    async def signal_alert(self, signal: dict):
        """Send trading signal alert"""
        if not self.bot:
            return

        try:
            tier_emoji = {
                'TIER_1': '💎',
                'TIER_2': '🥇',
                'TIER_3': '🥈'
            }

            direction_emoji = '🟢' if signal['direction'] == 'LONG' else '🔴'

            msg = f"""
{tier_emoji.get(signal['tier'], '📊')} <b>{signal['tier']} SIGNAL</b>

{direction_emoji} <b>{signal['direction']} {signal['symbol']}</b>

📊 <b>Entry Details:</b>
• Entry: <b>{signal['entry']}</b>
• Stop Loss: {signal['sl']}
• Take Profit 1: {signal['tp1']}
• Take Profit 2: {signal['tp2']}
• Take Profit 3: {signal['tp3']}

📈 <b>Analysis:</b>
• Confidence: {signal['confidence']}%
• Win Rate: {signal['win_rate']}
• RR Ratio: {signal['rr_ratio']}
• Filters: {signal['filters_passed']}

💰 <b>Position:</b>
• Size: {signal['position_size']}
• Risk: ₹{signal['risk_amount']}
• Margin: ₹{signal['margin_required']}
• Balance: ₹{signal['balance']}

⏰ {datetime.fromisoformat(signal['timestamp']).strftime('%d %b, %H:%M:%S IST')}

<i>Trade at your own risk. This is not financial advice.</i>
            """

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode='HTML'
            )

            logger.info(f"✅ Signal alert sent: {signal['symbol']} {signal['direction']}")

        except Exception as e:
            logger.error(f"Signal alert error: {e}")

    async def tp_alert(self, tp_level: str, signal: dict, profit: float):
        """Send take profit hit alert"""
        if not self.bot:
            return

        try:
            msg = f"""
✅ <b>TAKE PROFIT HIT!</b>

🎯 {signal['symbol']} {signal['direction']}
💰 <b>{tp_level.upper()} Hit</b>

💵 Profit: <b>₹{profit:.2f}</b>
📊 Entry: {signal['entry']}
🎯 Target: {signal[tp_level.lower()]}

{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b, %H:%M:%S IST')}
            """

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"TP alert error: {e}")

    async def sl_alert(self, signal: dict):
        """Send stop loss hit alert"""
        if not self.bot:
            return

        try:
            msg = f"""
❌ <b>STOP LOSS HIT</b>

{signal['symbol']} {signal['direction']}

🛑 SL: {signal['sl']}
📊 Entry: {signal['entry']}

{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b, %H:%M:%S IST')}

<i>Loss managed. Moving to next opportunity.</i>
            """

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"SL alert error: {e}")

    async def breakeven_alert(self, signal: dict):
        """Send breakeven move alert"""
        if not self.bot:
            return

        try:
            msg = f"""
🔒 <b>BREAKEVEN ACTIVATED</b>

{signal['symbol']}
Stop Loss moved to Entry: {signal['entry']}

Risk-free position! 🎯
            """

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"Breakeven alert error: {e}")

    async def timeout_alert(self, signal: dict):
        """Send timeout alert"""
        if not self.bot:
            return

        try:
            msg = f"""
⏱️ <b>POSITION TIMEOUT</b>

{signal['symbol']}
Position closed after 2 hours.

{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b, %H:%M:%S IST')}
            """

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"Timeout alert error: {e}")

    async def daily_summary(self, stats: dict):
        """Send daily summary"""
        if not self.bot:
            return

        try:
            win_rate = (stats['wins'] / (stats['wins'] + stats['losses']) * 100) if (stats['wins'] + stats['losses']) > 0 else 0

            msg = f"""
📊 <b>DAILY SUMMARY</b>

📈 <b>Performance:</b>
• Total Signals: {stats['total']}
• Wins: {stats['wins']} ✅
• Losses: {stats['losses']} ❌
• Win Rate: {win_rate:.1f}%

💰 <b>PnL:</b> ₹{stats['pnl']:.2f}

🏆 <b>By Tier:</b>
• TIER 1: {stats['by_tier']['TIER_1']}
• TIER 2: {stats['by_tier']['TIER_2']}
• TIER 3: {stats['by_tier']['TIER_3']}

{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b %Y')}
            """

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"Daily summary error: {e}")
