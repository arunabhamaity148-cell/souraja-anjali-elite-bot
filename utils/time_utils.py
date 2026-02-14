"""
Time utilities with sleep hours
24/7 Trading Mode - Sleep only between 1 AM - 7 AM IST
"""

from datetime import datetime, time
import pytz

def get_ist_time():
    """Get current IST time"""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def is_sleep_hours():
    """
    Check if bot should sleep (1 AM - 7 AM IST)
    
    Returns:
        tuple: (bool, str) - (is_sleeping, reason)
    """
    now = get_ist_time()
    current_time = now.time()
    
    # Sleep between 1:00 AM and 7:00 AM IST
    sleep_start = time(1, 0)   # 1:00 AM
    sleep_end = time(7, 0)     # 7:00 AM
    
    if sleep_start <= current_time < sleep_end:
        return True, f"Sleep hours (1 AM - 7 AM IST) - Current: {now.strftime('%H:%M')}"
    
    return False, "Active trading hours"

def is_golden_hour():
    """
    DEPRECATED - No longer used in 24/7 mode
    Bot now trades 24/7 except sleep hours (1-7 AM IST)
    
    Kept for backwards compatibility
    """
    return True  # Always return True, sleep hours checked separately

def get_next_active_time():
    """Get next active trading time if currently in sleep hours"""
    now = get_ist_time()
    is_sleeping, _ = is_sleep_hours()
    
    if is_sleeping:
        # Return 7:00 AM today
        next_active = now.replace(hour=7, minute=0, second=0, microsecond=0)
        return next_active
    
    return now

def format_time_until_active():
    """Format human-readable time until next active period"""
    next_active = get_next_active_time()
    now = get_ist_time()
    
    if next_active <= now:
        return "Active now"
    
    delta = next_active - now
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    
    return f"{hours}h {minutes}m until active"
