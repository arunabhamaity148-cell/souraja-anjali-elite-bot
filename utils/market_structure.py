"""
Market Structure
"""

class MarketStructure:
    def __init__(self):
        pass
    
    async def get_trend(self, symbol):
        return 'BULLISH'
    
    async def get_structure(self, symbol):
        return 'BOS_UP'
    
    async def get_trend_tf(self, symbol, tf):
        return 'BULLISH'
