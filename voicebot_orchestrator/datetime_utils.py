"""
Date-Time Utilities for Voicebot Orchestrator

Standardized date-time formatting for all reports, logs, and exports.
Format: mmddyyyy_seconds (e.g., 08282025_1756395400)
"""

import time
from datetime import datetime
from typing import Optional, Union


class DateTimeFormatter:
    """Standardized date-time formatting utilities."""
    
    @staticmethod
    def get_timestamp_string(timestamp: Optional[Union[float, int]] = None) -> str:
        """
        Generate standardized timestamp string in format: mmddyyyy_seconds
        
        Args:
            timestamp: Unix timestamp (optional, defaults to current time)
            
        Returns:
            Formatted timestamp string (e.g., "08282025_1756395400")
        """
        if timestamp is None:
            timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        
        # Format: mmddyyyy_seconds
        date_part = dt.strftime("%m%d%Y")
        seconds_part = str(int(timestamp))
        
        return f"{date_part}_{seconds_part}"
    
    @staticmethod
    def get_current_timestamp_string() -> str:
        """
        Get current timestamp in standardized format.
        
        Returns:
            Current timestamp string (e.g., "08282025_1756395400")
        """
        return DateTimeFormatter.get_timestamp_string()
    
    @staticmethod
    def get_report_filename(prefix: str, extension: str = "csv", timestamp: Optional[Union[float, int]] = None) -> str:
        """
        Generate standardized report filename.
        
        Args:
            prefix: File prefix (e.g., "analytics_export", "performance_report")
            extension: File extension without dot (e.g., "csv", "json", "log")
            timestamp: Unix timestamp (optional, defaults to current time)
            
        Returns:
            Formatted filename (e.g., "analytics_export_08282025_1756395400.csv")
        """
        timestamp_str = DateTimeFormatter.get_timestamp_string(timestamp)
        return f"{prefix}_{timestamp_str}.{extension}"
    
    @staticmethod
    def get_log_filename(log_type: str, timestamp: Optional[Union[float, int]] = None) -> str:
        """
        Generate standardized log filename.
        
        Args:
            log_type: Type of log (e.g., "session", "error", "performance")
            timestamp: Unix timestamp (optional, defaults to current time)
            
        Returns:
            Formatted log filename (e.g., "session_log_08282025_1756395400.log")
        """
        return DateTimeFormatter.get_report_filename(f"{log_type}_log", "log", timestamp)
    
    @staticmethod
    def get_audio_filename(prefix: str, timestamp: Optional[Union[float, int]] = None) -> str:
        """
        Generate standardized audio filename.
        
        Args:
            prefix: Audio prefix (e.g., "kokoro_output", "demo_audio")
            timestamp: Unix timestamp (optional, defaults to current time)
            
        Returns:
            Formatted audio filename (e.g., "kokoro_output_08282025_1756395400.wav")
        """
        return DateTimeFormatter.get_report_filename(prefix, "wav", timestamp)
    
    @staticmethod
    def parse_timestamp_from_filename(filename: str) -> Optional[float]:
        """
        Extract timestamp from standardized filename.
        
        Args:
            filename: Filename with embedded timestamp
            
        Returns:
            Unix timestamp as float, or None if not found
        """
        try:
            # Look for pattern: mmddyyyy_seconds
            parts = filename.split('_')
            if len(parts) >= 2:
                # Get the last part that contains the seconds
                for i in range(len(parts) - 1, -1, -1):
                    part = parts[i].split('.')[0]  # Remove extension
                    if part.isdigit() and len(part) >= 10:  # Unix timestamp length
                        return float(part)
            return None
        except (ValueError, IndexError):
            return None
    
    @staticmethod
    def get_readable_timestamp(timestamp: Union[float, int, str]) -> str:
        """
        Convert timestamp to human-readable format.
        
        Args:
            timestamp: Unix timestamp or filename with timestamp
            
        Returns:
            Human-readable datetime string
        """
        try:
            if isinstance(timestamp, str):
                # Extract from filename
                ts = DateTimeFormatter.parse_timestamp_from_filename(timestamp)
                if ts is None:
                    return "Unknown"
                timestamp = ts
            
            dt = datetime.fromtimestamp(float(timestamp))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return "Invalid timestamp"
    
    @staticmethod
    def get_analytics_export_filename(timestamp: Optional[Union[float, int]] = None) -> str:
        """Get standardized analytics export filename."""
        return DateTimeFormatter.get_report_filename("analytics_export", "csv", timestamp)
    
    @staticmethod
    def get_performance_chart_filename(timestamp: Optional[Union[float, int]] = None) -> str:
        """Get standardized performance chart filename."""
        return DateTimeFormatter.get_report_filename("performance_chart", "png", timestamp)
    
    @staticmethod
    def get_cache_export_filename(timestamp: Optional[Union[float, int]] = None) -> str:
        """Get standardized cache export filename."""
        return DateTimeFormatter.get_report_filename("cache_export", "json", timestamp)


# Convenience functions for backward compatibility
def get_timestamp_string() -> str:
    """Get current timestamp string in standardized format."""
    return DateTimeFormatter.get_current_timestamp_string()

def get_report_filename(prefix: str, extension: str = "csv") -> str:
    """Generate standardized report filename."""
    return DateTimeFormatter.get_report_filename(prefix, extension)

def get_readable_timestamp(timestamp: Union[float, int, str]) -> str:
    """Convert timestamp to human-readable format."""
    return DateTimeFormatter.get_readable_timestamp(timestamp)


# Example usage and testing
if __name__ == "__main__":
    print("Date-Time Formatter Examples:")
    print("=" * 50)
    
    # Current timestamp
    current = DateTimeFormatter.get_current_timestamp_string()
    print(f"Current timestamp: {current}")
    
    # Generate report filenames
    analytics_file = DateTimeFormatter.get_analytics_export_filename()
    print(f"Analytics export: {analytics_file}")
    
    performance_file = DateTimeFormatter.get_performance_chart_filename()
    print(f"Performance chart: {performance_file}")
    
    audio_file = DateTimeFormatter.get_audio_filename("kokoro_output")
    print(f"Audio file: {audio_file}")
    
    log_file = DateTimeFormatter.get_log_filename("session")
    print(f"Log file: {log_file}")
    
    # Parse timestamp from filename
    timestamp = DateTimeFormatter.parse_timestamp_from_filename(analytics_file)
    readable = DateTimeFormatter.get_readable_timestamp(timestamp)
    print(f"Readable time: {readable}")
    
    print("\n✅ All examples generated successfully!")
