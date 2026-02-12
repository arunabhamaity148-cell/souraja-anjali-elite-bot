"""
Risk Manager
"""

import logging
from config import TRADING

logger = logging.getLogger("RISK")

class EliteRiskManager:
    def __init__(self):
        self.daily_pnl = 0
        self.trades_today = 0
        self.last_trade_win = True
        
    async def check_trade_allowed(self):
        """Check if can trade"""
        if self.daily_pnl <= -TRADING['daily_loss_limit']:
            return False, "Daily loss limit"
        if self.trades_today >= TRADING['max_daily_signals']:
            return False, "Max signals"
        return True, "OK"
    
    def check_breakeven(self, signal, current_price):
        """Check breakeven"""
        entry = signal['entry']
        trigger = TRADING['breakeven_trigger']
        
        if signal['direction'] == 'LONG':
            profit_pct = (current_price - entry) / entry
            if profit_pct >= trigger:
                return {
                    'action': 'MOVE_BE',
                    'new_sl': entry,
                    'message': f"✅ Breakeven SL at {entry}"
                }
        else:
            profit_pct = (entry - current_price) / entry
            if profit_pct >= trigger:
                return {
                    'action': 'MOVE_BE',
                    'new_sl': entry,
                    'message': f"✅ Breakeven SL at {entry}"
                }
        return None
