"""
Time utilities
"""

from datetime import datetime
import pytz

def get_ist_time():
    """Get IST time"""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def is_golden_hour():
    """Check if golden hour"""
    now = get_ist_time()
    time_str = now.strftime('%H:%M')
    
    golden = [
        ('13:30', '14:30'),
        ('19:00', '20:30'),
        ('21:30', '22:30')
    ]
    
    for start, end in golden:
        if start <= time_str <= end:
            return True
    return False
