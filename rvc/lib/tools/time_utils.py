def format_time(seconds):
    """
    Convert seconds to time format.
    Returns HH:MM:SS if duration >= 1 hour, otherwise MM:SS
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}" 