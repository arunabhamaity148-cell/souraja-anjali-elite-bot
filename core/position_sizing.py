"""
Position Sizing - Kelly Criterion + Risk Management
"""

import logging
from typing import Dict, Optional
from exchanges.exchange_manager import ExchangeManager

logger = logging.getLogger("POSITION")

class PositionSizer:
    """Calculate position size based on account balance and risk"""
    
    def __init__(self, exchange_mgr: ExchangeManager):
        self.exchange_mgr = exchange_mgr
        self.daily_stats = {
            'total_risk': 0,
            'trades_today': 0,
            'last_reset': None
        }
    
    async def calculate_position_size(self, symbol: str, entry: float, 
                                     sl: float, tier: str) -> Optional[Dict]:
        """
        Calculate position size based on:
        - Account balance
        - Risk per trade (1% of balance)
        - Tier-based adjustments
        """
        try:
            # Get account balance
            balance = await self._get_account_balance()
            if balance <= 0:
                logger.error("Could not fetch balance")
                return None
            
            # Risk per trade: 1% of balance
            risk_percent = 0.01
            
            # Tier adjustment
            if tier == 'TIER_1':
                risk_percent = 0.01  # 1%
            elif tier == 'TIER_2':
                risk_percent = 0.008  # 0.8%
            else:
                risk_percent = 0.005  # 0.5%
            
            risk_amount = balance * risk_percent
            
            # Check daily loss limit (3% of balance)
            daily_limit = balance * 0.03
            if self.daily_stats['total_risk'] + risk_amount > daily_limit:
                logger.warning("Daily risk limit reached")
                return None
            
            # Calculate position size
            price_risk = abs(entry - sl)
            if price_risk == 0:
                logger.error("Invalid SL distance")
                return None
            
            # Position size in base currency
            position_size = risk_amount / price_risk
            
            # Apply leverage (15x)
            leveraged_position = position_size * 15
            
            # Margin required
            margin_required = leveraged_position * entry / 15
            
            # Check if we have enough balance
            if margin_required > balance * 0.5:  # Max 50% of balance in one trade
                logger.warning(f"Position too large, reducing size")
                leveraged_position = (balance * 0.5 * 15) / entry
                margin_required = balance * 0.5
            
            # Update daily stats
            self.daily_stats['total_risk'] += risk_amount
            self.daily_stats['trades_today'] += 1
            
            return {
                'position_size': round(leveraged_position, 6),
                'margin_required': round(margin_required, 2),
                'risk_amount': round(risk_amount, 2),
                'risk_percent': risk_percent,
                'leverage': 15,
                'balance': round(balance, 2)
            }
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return None
    
    async def _get_account_balance(self) -> float:
        """Get USDT balance from primary exchange"""
        try:
            if not self.exchange_mgr:
                return 10000  # Fallback for testing
            
            client = self.exchange_mgr.get_primary_client()
            if not client:
                return 10000
            
            balance = await client.get_balance()
            return float(balance.get('USDT', 0))
            
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return 0
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_stats = {
            'total_risk': 0,
            'trades_today': 0,
            'last_reset': None
        }
