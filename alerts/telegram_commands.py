"""
Telegram Commands Handler
"""

import logging
from telegram import Update
from telegram.ext import Application, CommandHandler as TelegramCommandHandler, ContextTypes

logger = logging.getLogger("COMMANDS")

class TelegramCommands:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.app = None
        
        # Initialize telegram application
        from config import TELEGRAM
        if TELEGRAM.get('bot_token'):
            try:
                self.app = Application.builder().token(TELEGRAM['bot_token']).build()
                self._setup_handlers()
                logger.info("✅ Telegram commands app initialized")
            except Exception as e:
                logger.error(f"Failed to init telegram app: {e}")

    def _setup_handlers(self):
        """Setup command handlers"""
        self.app.add_handler(TelegramCommandHandler("start", self.cmd_start))
        self.app.add_handler(TelegramCommandHandler("help", self.cmd_help))
        self.app.add_handler(TelegramCommandHandler("status", self.cmd_status))
        self.app.add_handler(TelegramCommandHandler("balance", self.cmd_balance))
        self.app.add_handler(TelegramCommandHandler("signals", self.cmd_signals))
        self.app.add_handler(TelegramCommandHandler("pause", self.cmd_pause))
        self.app.add_handler(TelegramCommandHandler("resume", self.cmd_resume))
        self.app.add_handler(TelegramCommandHandler("stop", self.cmd_stop))
        self.app.add_handler(TelegramCommandHandler("stats", self.cmd_stats))
        self.app.add_handler(TelegramCommandHandler("regime", self.cmd_regime))
        self.app.add_handler(TelegramCommandHandler("train", self.train_command))
        self.app.add_handler(TelegramCommandHandler("scan", self.force_scan))
        logger.info("✅ Telegram command handlers registered")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command"""
        await update.message.reply_text(
            "🚀 *ARUNABHA ELITE v8.3*\n\n"
            "Bot is ACTIVE and scanning markets...\n\n"
            "Commands:\n"
            "/status - Bot status\n"
            "/balance - Account balance\n"
            "/signals - Today's signals\n"
            "/stats - Trading stats\n"
            "/regime - Market regime\n"
            "/train - Train ML model\n"
            "/scan - Force scan\n"
            "/pause - Pause trading\n"
            "/resume - Resume trading\n"
            "/stop - Emergency stop",
            parse_mode='Markdown'
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command"""
        await self.cmd_start(update, context)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Status command"""
        status = "⏸️ PAUSED" if self.bot.is_paused else "🟢 ACTIVE"
        regime = self.bot.current_regime.value if self.bot.current_regime else "Unknown"
        
        await update.message.reply_text(
            f"📊 *Bot Status*\n\n"
            f"Status: {status}\n"
            f"Regime: {regime}\n"
            f"Signals Today: {self.bot.daily_stats['total']}/12\n"
            f"Wins: {self.bot.daily_stats['wins']}\n"
            f"Losses: {self.bot.daily_stats['losses']}\n"
            f"PnL: ₹{self.bot.daily_stats['pnl']:.2f}",
            parse_mode='Markdown'
        )

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Balance command"""
        try:
            balance = await self.bot.exchange_mgr.get_balance()
            await update.message.reply_text(
                f"💰 *Account Balance*\n\n"
                f"Available: ₹{balance.get('available', 0):.2f}\n"
                f"Total: ₹{balance.get('total', 0):.2f}",
                parse_mode='Markdown'
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Signals command"""
        await update.message.reply_text(
            f"📈 *Today's Signals*\n\n"
            f"Total: {self.bot.daily_stats['total']}\n"
            f"TIER 1: {self.bot.daily_stats['by_tier'].get('TIER_1', 0)}\n"
            f"TIER 2: {self.bot.daily_stats['by_tier'].get('TIER_2', 0)}\n"
            f"TIER 3: {self.bot.daily_stats['by_tier'].get('TIER_3', 0)}",
            parse_mode='Markdown'
        )

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause command"""
        self.bot.is_paused = True
        await update.message.reply_text("⏸️ Bot PAUSED. Use /resume to continue.")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume command"""
        self.bot.is_paused = False
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
        regime = self.bot.current_regime.value if self.bot.current_regime else "Unknown"
        settings = self.bot.adaptive_settings or {}
        
        await update.message.reply_text(
            f"📊 *Market Regime*\n\n"
            f"Current: {regime}\n"
            f"Strategy: {settings.get('strategy', 'N/A')}\n"
            f"Direction: {settings.get('direction_bias', 'N/A')}\n"
            f"Max Signals: {settings.get('max_signals', 0)}",
            parse_mode='Markdown'
        )

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

    async def force_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force immediate market scan"""
        await update.message.reply_text("🔍 Force scanning all pairs...")

        count = 0
        for symbol in self.bot.symbols:
            success = await self.bot._process_symbol(symbol, self.bot.adaptive_settings)
            if success:
                count += 1

        await update.message.reply_text(f"✅ Scan complete. {count} signals found")
