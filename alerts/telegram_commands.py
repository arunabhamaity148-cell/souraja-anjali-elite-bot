"""
Telegram Commands Handler
"""

import logging
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger("COMMANDS")

class CommandHandler:
    def __init__(self, bot_instance):
        self.bot = bot_instance

    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manual ML training trigger"""
        await update.message.reply_text("🎓 Training started... eta 5-10 min lagbe")

        try:
            success = await self.bot.model_trainer.train_daily(self.bot)
            if success:
                await update.message.reply_text("✅ Training complete! Model ready")
            else:
                await update.message.reply_text("❌ Training failed. Check logs")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Bot status check"""
        status = f"""
🤖 *ARUNABHA ELITE Status*

Regime: {self.bot.current_regime.value if self.bot.current_regime else 'Unknown'}
Daily Signals: {self.bot.daily_stats['total']}/12
Trades Today: {self.bot.daily_stats.get('wins', 0)}W / {self.bot.daily_stats.get('losses', 0)}L
PNL: ₹{self.bot.daily_stats.get('pnl', 0):,.2f}

ML Model: {'✅ Trained' if self.bot.model_trainer.ml_engine.is_trained else '❌ Untrained'}
        """
        await update.message.reply_text(status, parse_mode='Markdown')

    async def force_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force immediate market scan"""
        await update.message.reply_text("🔍 Force scanning all pairs...")

        count = 0
        for symbol in self.bot.symbols:
            success = await self.bot._process_symbol(symbol, self.bot.adaptive_settings)
            if success:
                count += 1

        await update.message.reply_text(f"✅ Scan complete. {count} signals found")

# Alias for main.py import
TelegramCommands = CommandHandler
