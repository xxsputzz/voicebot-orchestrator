#!/usr/bin/env python3
"""
Tortoise TTS Timeout Configuration
Dynamic timeout calculation based on hardware performance and synthesis parameters
"""

import os
import json
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TimeoutConfig:
    """Configuration for Tortoise TTS timeout calculations"""
    base_overhead: float = 60.0          # Model loading and setup time
    char_processing_time: float = 6.0    # Seconds per character for ultra-quality
    safety_buffer: float = 1.5           # Safety multiplier (50% extra time)
    min_timeout: float = 300.0           # Minimum 5 minutes
    max_timeout: float = 3600.0          # Maximum 60 minutes for very long texts
    retry_multipliers: list = None       # Progressive retry timeouts
    
    def __post_init__(self):
        if self.retry_multipliers is None:
            self.retry_multipliers = [1.0, 1.5, 2.0]

class TortoiseTimeoutManager:
    """Manages timeout calculations for Tortoise TTS synthesis"""
    
    # Voice complexity multipliers based on actual performance
    VOICE_COMPLEXITY = {
        'angie': 1.3,      # Complex voice with emotional range
        'tom': 1.0,        # Standard baseline voice
        'pat': 1.1,        # Medium complexity
        'william': 1.2,    # Complex male voice
        'deniro': 1.4,     # Very complex emotional voice
        'freeman': 1.3,    # Complex narrative voice
        'halle': 1.2,      # Complex female voice
        'jlaw': 1.35,      # Complex celebrity voice
        'lj': 1.1,         # Standard female voice
        'mol': 1.15,       # Medium complexity
        'myself': 1.0,     # Baseline custom voice
        'rainbow': 1.25,   # Complex character voice
        'tim_reynolds': 1.2, # Complex narrator voice
        'weaver': 1.3,     # Complex character voice
        'default': 1.1     # Default fallback
    }
    
    # Preset complexity multipliers
    PRESET_COMPLEXITY = {
        'ultra_fast': 0.3,     # Very fast, lower quality
        'fast': 0.5,           # Fast synthesis
        'standard': 1.0,       # Standard quality
        'high_quality': 1.8,   # High quality
        'ultra_high_quality': 2.5  # Ultra high quality
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize timeout manager with optional config file"""
        self.config = TimeoutConfig()
        self.performance_history = {}
        self.config_file = config_file or "tortoise_timeout_config.json"
        self.load_config()
    
    def load_config(self):
        """Load timeout configuration from file"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    
                # Update config with saved values
                for key, value in data.get('timeout_config', {}).items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                # Load performance history
                self.performance_history = data.get('performance_history', {})
                
                print(f"[TIMEOUT] Loaded configuration from {config_path}")
            except Exception as e:
                print(f"[TIMEOUT] Error loading config: {e}, using defaults")
    
    def save_config(self):
        """Save current configuration and performance data"""
        try:
            data = {
                'timeout_config': {
                    'base_overhead': self.config.base_overhead,
                    'char_processing_time': self.config.char_processing_time,
                    'safety_buffer': self.config.safety_buffer,
                    'min_timeout': self.config.min_timeout,
                    'max_timeout': self.config.max_timeout,
                    'retry_multipliers': self.config.retry_multipliers
                },
                'performance_history': self.performance_history
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"[TIMEOUT] Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"[TIMEOUT] Error saving config: {e}")
    
    def calculate_timeout(self, 
                         text_length: int, 
                         voice: str = 'angie', 
                         preset: str = 'ultra_fast') -> float:
        """
        Calculate realistic timeout based on text length, voice complexity, and preset
        
        Args:
            text_length: Number of characters in text
            voice: Voice name
            preset: Quality preset
            
        Returns:
            Calculated timeout in seconds
        """
        # Get multipliers
        voice_multiplier = self.VOICE_COMPLEXITY.get(voice, self.VOICE_COMPLEXITY['default'])
        preset_multiplier = self.PRESET_COMPLEXITY.get(preset, 1.0)
        
        # Calculate base time
        base_time = (self.config.base_overhead + 
                    (text_length * self.config.char_processing_time * 
                     voice_multiplier * preset_multiplier))
        
        # Apply safety buffer
        estimated_timeout = base_time * self.config.safety_buffer
        
        # Apply min/max constraints
        final_timeout = max(self.config.min_timeout, 
                           min(estimated_timeout, self.config.max_timeout))
        
        return final_timeout
    
    def get_retry_timeouts(self, base_timeout: float) -> list:
        """Get progressive retry timeouts"""
        return [base_timeout * multiplier for multiplier in self.config.retry_multipliers]
    
    def record_performance(self, 
                          text_length: int, 
                          voice: str, 
                          preset: str, 
                          actual_time: float):
        """Record actual performance for future timeout improvements"""
        key = f"{voice}_{preset}"
        
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        # Store performance data point
        self.performance_history[key].append({
            'text_length': text_length,
            'actual_time': actual_time,
            'char_per_second': actual_time / max(text_length, 1)
        })
        
        # Keep only recent 10 entries per configuration
        self.performance_history[key] = self.performance_history[key][-10:]
        
        # Auto-adjust char_processing_time based on performance history
        self.auto_adjust_parameters()
    
    def auto_adjust_parameters(self):
        """Auto-adjust timeout parameters based on performance history"""
        if not self.performance_history:
            return
            
        # Calculate average character processing time across all configurations
        total_char_times = []
        for config_data in self.performance_history.values():
            for entry in config_data:
                total_char_times.append(entry['char_per_second'])
        
        if total_char_times:
            avg_char_time = sum(total_char_times) / len(total_char_times)
            
            # Adjust if significantly different from current setting
            if abs(avg_char_time - self.config.char_processing_time) > 1.0:
                old_time = self.config.char_processing_time
                # Use 90th percentile for safety
                total_char_times.sort()
                percentile_90 = total_char_times[int(len(total_char_times) * 0.9)]
                self.config.char_processing_time = percentile_90
                
                print(f"[TIMEOUT] Auto-adjusted char processing time: {old_time:.1f}s → {percentile_90:.1f}s")
                self.save_config()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        
        for config, data in self.performance_history.items():
            if data:
                times = [entry['actual_time'] for entry in data]
                char_times = [entry['char_per_second'] for entry in data]
                
                stats[config] = {
                    'samples': len(data),
                    'avg_total_time': sum(times) / len(times),
                    'avg_char_time': sum(char_times) / len(char_times),
                    'min_char_time': min(char_times),
                    'max_char_time': max(char_times)
                }
        
        return stats

# Global timeout manager instance
timeout_manager = TortoiseTimeoutManager()

def get_timeout_manager() -> TortoiseTimeoutManager:
    """Get the global timeout manager instance"""
    return timeout_manager

def calculate_synthesis_timeout(text_length: int, 
                              voice: str = 'angie', 
                              preset: str = 'ultra_fast') -> float:
    """Convenience function for timeout calculation"""
    return timeout_manager.calculate_timeout(text_length, voice, preset)
