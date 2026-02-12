"""
3 Tier System
"""

import logging
from config import TIER_SETTINGS

logger = logging.getLogger("TIERS")

class TierManager:
    def __init__(self):
        self.daily_count = {'TIER_1': 0, 'TIER_2': 0, 'TIER_3': 0}
        
    def determine_tier_adaptive(self, passed: int, total: int, min_tier: str) -> dict:
        """Determine tier with adaptive minimum"""
        
        tier_order = ['TIER_3', 'TIER_2', 'TIER_1']
        min_idx = tier_order.index(min_tier)
        
        # TIER_1: 8-10 filters
        if passed >= 8 and min_idx <= 2:
            if self.daily_count['TIER_1'] < TIER_SETTINGS['TIER_1']['max_daily']:
                self.daily_count['TIER_1'] += 1
                return {
                    'tier': 'TIER_1',
                    'confidence': 92,
                    'expected_win_rate': '88%'
                }
        
        # TIER_2: 6-7 filters
        if passed >= 6 and min_idx <= 1:
            if self.daily_count['TIER_2'] < TIER_SETTINGS['TIER_2']['max_daily']:
                self.daily_count['TIER_2'] += 1
                return {
                    'tier': 'TIER_2',
                    'confidence': 82,
                    'expected_win_rate': '78%'
                }
        
        # TIER_3: 5 filters
        if passed >= 5 and min_idx == 0:
            if self.daily_count['TIER_3'] < TIER_SETTINGS['TIER_3']['max_daily']:
                self.daily_count['TIER_3'] += 1
                return {
                    'tier': 'TIER_3',
                    'confidence': 72,
                    'expected_win_rate': '68%'
                }
        
        return None
    
    def reset_daily(self):
        self.daily_count = {'TIER_1': 0, 'TIER_2': 0, 'TIER_3': 0}
