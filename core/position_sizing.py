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
            'total_risk': 0.0,
            'trades_today': 0,
            'last_reset': None
        }
    
    async def calculate_position_size(self, symbol: str, entry: float, 
                                     sl: float, tier: str) -> Optional[Dict]:
        """
        Calculate position size:
        - Risk per trade: 1% of balance (tier-adjusted)
        - Position size = Risk Amount / (Entry - SL)
        - Apply 15x leverage
        """
        try:
            # Get real account balance
            balance = await self._get_account_balance()
            if balance <= 0:
                logger.error("Could not fetch balance")
                return None
            
            # Risk per trade based on tier
            risk_percents = {
                'TIER_1': 0.01,   # 1%
                'TIER_2': 0.008,  # 0.8%
                'TIER_3': 0.005   # 0.5%
            }
            
            risk_percent = risk_percents.get(tier, 0.005)
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
            
            # Position size in base currency (BTC, ETH, etc.)
            position_size = risk_amount / price_risk
            
            # Apply leverage (15x)
            leveraged_position = position_size * 15
            
            # Margin required
            margin_required = (leveraged_position * entry) / 15
            
            # Max 50% of balance in one trade
            max_margin = balance * 0.5
            if margin_required > max_margin:
                logger.warning(f"Reducing position size to fit margin limit")
                margin_required = max_margin
                leveraged_position = (max_margin * 15) / entry
                risk_amount = leveraged_position * price_risk / 15
            
            # Update stats
            self.daily_stats['total_risk'] += risk_amount
            self.daily_stats['trades_today'] += 1
            
            logger.info(f"{symbol}: Position size calculated")
            logger.info(f"  Balance: ₹{balance:,.2f}")
            logger.info(f"  Risk: ₹{risk_amount:,.2f} ({risk_percent:.2%})")
            logger.info(f"  Position: {leveraged_position:.6f}")
            logger.info(f"  Margin: ₹{margin_required:,.2f}")
            
            return {
                'position_size': round(leveraged_position, 6),
                'margin_required': round(margin_required, 2),
                'risk_amount': round(risk_amount, 2),
                'risk_percent': risk_percent,
                'leverage': 15,
                'balance': round(balance, 2),
                'price_risk': round(price_risk, 4)
            }
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return None
    
    async def _get_account_balance(self) -> float:
        """Fetch real account balance from exchange"""
        try:
            if not self.exchange_mgr:
                logger.error("No exchange manager")
                return 0
            
            client = self.exchange_mgr.get_primary_client()
            if not client:
                logger.error("No primary client")
                return 0
            
            balance = await client.get_balance()
            usdt_balance = float(balance.get('USDT', 0))
            
            logger.debug(f"Fetched balance: {usdt_balance} USDT")
            return usdt_balance
            
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return 0
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_stats = {
            'total_risk': 0.0,
            'trades_today': 0,
            'last_reset': None
        }
