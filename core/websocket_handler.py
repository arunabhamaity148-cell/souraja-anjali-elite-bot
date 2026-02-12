"""
WebSocket Handler
"""

import logging

logger = logging.getLogger("WEBSOCKET")

class WebSocketManager:
    def __init__(self):
        self.prices = {}
        
    async def get_price(self, symbol: str) -> float:
        """Get current price"""
        if symbol in self.prices:
            return self.prices[symbol]
        
        prices = {
            'BTCUSDT': 97000, 'ETHUSDT': 2650, 'SOLUSDT': 195,
            'DOGEUSDT': 0.25, 'BNBUSDT': 720, 'XRPUSDT': 2.45,
            'LINKUSDT': 18.5, 'ADAUSDT': 0.85
        }
        return prices.get(symbol, 100)
