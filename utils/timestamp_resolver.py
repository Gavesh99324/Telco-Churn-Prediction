"""Timestamp utilities"""
from datetime import datetime


class TimestampResolver:
    """Handle timestamp operations"""

    @staticmethod
    def get_current_timestamp():
        """Get current timestamp"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    @staticmethod
    def format_timestamp(dt, fmt='%Y-%m-%d %H:%M:%S'):
        """Format datetime object"""
        return dt.strftime(fmt)
