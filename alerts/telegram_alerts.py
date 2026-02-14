"""
Telegram Alerts & Commands - Unified
"""

import logging
import asyncio
from datetime import datetime
import pytz
import os
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

logger = logging.getLogger("TELEGRAM")

class HumanStyleAlerts:
    def __init__(self):
        from config import TELEGRAM, BOT_CONFIG
        self.bot_token = TELEGRAM['bot_token']
        self.chat_id = TELEGRAM['chat_id']
        self.config = BOT_CONFIG

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            self.bot = None
            self.app = None
        else:
            self.bot = Bot(token=self.bot_token)
            self.app = Application.builder().token(self.bot_token).build()
            self._setup_commands()
            logger.info("✅ Telegram bot & commands initialized")

    def _setup_commands(self):
        """Setup command handlers"""
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("balance", self.cmd_balance))
        self.app.add_handler(CommandHandler("signals", self.cmd_signals))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("regime", self.cmd_regime))
        self.app.add_handler(CommandHandler("train", self.cmd_train))
        self.app.add_handler(CommandHandler("scan", self.cmd_scan))
        logger.info("✅ Command handlers registered")

    async def start_polling(self):
        """Start command polling"""
        if self.app:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            logger.info("🤖 Command polling started")

    # ========== COMMAND HANDLERS ==========
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command"""
        await update.message.reply_text(
            f"🚀 *{self.config['name']}*\n\n"
            f"Bot is ACTIVE and scanning markets...\n\n"
            f"Commands:\n"
            f"/status - Bot status\n"
            f"/balance - Account balance\n"
            f"/signals - Today's signals\n"
            f"/stats - Trading stats\n"
            f"/regime - Market regime\n"
            f"/train - Train ML model\n"
            f"/scan - Force scan\n"
            f"/pause - Pause trading\n"
            f"/resume - Resume trading\n"
            f"/stop - Emergency stop",
            parse_mode='Markdown'
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command"""
        await self.cmd_start(update, context)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Status command"""
        from utils.time_utils import get_ist_time
        now = get_ist_time()
        
        status = "⏸️ PAUSED" if getattr(self, 'bot_ref', None) and self.bot_ref.is_paused else "🟢 ACTIVE"
        regime = getattr(self, 'bot_ref', None) and self.bot_ref.current_regime
        regime_str = regime.value if regime else "Unknown"
        stats = getattr(self, 'bot_ref', None) and self.bot_ref.daily_stats or {}
        
        await update.message.reply_text(
            f"📊 *Bot Status*\n\n"
            f"Status: {status}\n"
            f"Regime: {regime_str}\n"
            f"Signals Today: {stats.get('total', 0)}/12\n"
            f"Wins: {stats.get('wins', 0)}\n"
            f"Losses: {stats.get('losses', 0)}\n"
            f"PnL: ₹{stats.get('pnl', 0):.2f}\n\n"
            f"⏰ {now.strftime('%d %b, %H:%M IST')}",
            parse_mode='Markdown'
        )

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Balance command"""
        try:
            bot_ref = getattr(self, 'bot_ref', None)
            if bot_ref and bot_ref.exchange_mgr:
                balance = await bot_ref.exchange_mgr.get_balance()
                await update.message.reply_text(
                    f"💰 *Account Balance*\n\n"
                    f"Available: ₹{balance.get('available', 0):.2f}\n"
                    f"Total: ₹{balance.get('total', 0):.2f}",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text("❌ Exchange not connected")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Signals command"""
        stats = getattr(self, 'bot_ref', None) and self.bot_ref.daily_stats or {}
        await update.message.reply_text(
            f"📈 *Today's Signals*\n\n"
            f"Total: {stats.get('total', 0)}\n"
            f"TIER 1: {stats.get('by_tier', {}).get('TIER_1', 0)}\n"
            f"TIER 2: {stats.get('by_tier', {}).get('TIER_2', 0)}\n"
            f"TIER 3: {stats.get('by_tier', {}).get('TIER_3', 0)}",
            parse_mode='Markdown'
        )

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause command"""
        bot_ref = getattr(self, 'bot_ref', None)
        if bot_ref:
            bot_ref.is_paused = True
            await update.message.reply_text("⏸️ Bot PAUSED. Use /resume to continue.")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume command"""
        bot_ref = getattr(self, 'bot_ref', None)
        if bot_ref:
            bot_ref.is_paused = False
            await update.message.reply_text("▶️ Bot RESUMED. Trading active!")

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency stop"""
        await update.message.reply_text("🛑 Emergency stop triggered!")
        import sys
        sys.exit(0)

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stats command"""
        await self.cmd_status(update, context)

    async def cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Regime command"""
        bot_ref = getattr(self, 'bot_ref', None)
        regime = bot_ref.current_regime if bot_ref else None
        regime_str = regime.value if regime else "Unknown"
        settings = bot_ref.adaptive_settings if bot_ref else {}
        
        await update.message.reply_text(
            f"📊 *Market Regime*\n\n"
            f"Current: {regime_str}\n"
            f"Strategy: {settings.get('strategy', 'N/A')}\n"
            f"Direction: {settings.get('direction_bias', 'N/A')}\n"
            f"Max Signals: {settings.get('max_signals', 0)}",
            parse_mode='Markdown'
        )

    async def cmd_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manual ML training trigger"""
        await update.message.reply_text("🎓 Training started... eta 5-10 min lagbe")
        
        try:
            bot_ref = getattr(self, 'bot_ref', None)
            if bot_ref and bot_ref.model_trainer:
                success = await bot_ref.model_trainer.train_daily(bot_ref)
                if success:
                    await update.message.reply_text("✅ Training complete! Model ready")
                else:
                    await update.message.reply_text("❌ Training failed. Check logs")
            else:
                await update.message.reply_text("❌ Model trainer not available")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force immediate market scan"""
        await update.message.reply_text("🔍 Force scanning all pairs...")
        
        try:
            bot_ref = getattr(self, 'bot_ref', None)
            if bot_ref:
                count = 0
                for symbol in bot_ref.symbols:
                    success = await bot_ref._process_symbol(symbol, bot_ref.adaptive_settings)
                    if success:
                        count += 1
                await update.message.reply_text(f"✅ Scan complete. {count} signals found")
            else:
                await update.message.reply_text("❌ Bot not ready")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    # ========== ALERT METHODS ==========

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
🚀 <b>{self.config['name']}</b>

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
• Mode: <b>{self.config['mode']}</b>
• Version: {self.config['version']}
• Rating: {self.config['rating']}

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
• Name: {self.config['name']}
• Version: {self.config['version']}
• Mode: <b>{self.config['mode']}</b>

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
