"""
Math helpers
"""

def calculate_rr(entry, sl, tp):
    """Calculate risk reward"""
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return round(reward / risk, 2) if risk > 0 else 0
