"""
Tortoise TTS Progress Tracking
Enhanced progress monitoring and timeout handling for Tortoise TTS synthesis
"""

import time
import threading
from typing import Optional, Callable, Any
from contextlib import contextmanager
import sys


class TortoiseProgressTracker:
    """
    Progress tracking for Tortoise TTS synthesis operations
    Provides progress bars, timeout handling, and cancellation support
    """
    
    def __init__(self):
        self.current_operation = None
        self.start_time = None
        self.timeout_seconds = None
        self.progress_callback = None
        self.cancel_flag = threading.Event()
        self.progress_thread = None
        
    def start_operation(self, operation_name: str, timeout_seconds: Optional[float] = None, 
                       progress_callback: Optional[Callable] = None):
        """
        Start tracking a new operation
        
        Args:
            operation_name: Name of the operation
            timeout_seconds: Maximum time allowed (None for no timeout)
            progress_callback: Function to call with progress updates
        """
        self.current_operation = operation_name
        self.start_time = time.time()
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback
        self.cancel_flag.clear()
        
        print(f"[PROGRESS] Starting: {operation_name}")
        if timeout_seconds:
            print(f"[PROGRESS] Timeout: {timeout_seconds}s")
        else:
            print(f"[PROGRESS] No timeout - synthesis will complete naturally")
        
        # Start progress monitoring thread
        self.progress_thread = threading.Thread(target=self._monitor_progress, daemon=True)
        self.progress_thread.start()
    
    def _monitor_progress(self):
        """Monitor progress and handle timeout"""
        last_update = 0
        update_interval = 15  # Update every 15 seconds
        
        while not self.cancel_flag.is_set():
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Check timeout
            if self.timeout_seconds and elapsed > self.timeout_seconds:
                print(f"\n[PROGRESS] ⏰ Operation timed out after {elapsed:.1f}s")
                self.cancel_flag.set()
                break
            
            # Update progress
            if elapsed - last_update >= update_interval:
                self._print_progress_update(elapsed)
                last_update = elapsed
                
                # Call progress callback if provided
                if self.progress_callback:
                    try:
                        self.progress_callback(elapsed, self.timeout_seconds)
                    except Exception as e:
                        print(f"[PROGRESS] Callback error: {e}")
            
            time.sleep(1)
    
    def _print_progress_update(self, elapsed: float):
        """Print a progress update"""
        if self.timeout_seconds:
            progress_percent = min((elapsed / self.timeout_seconds) * 100, 100)
            remaining = max(self.timeout_seconds - elapsed, 0)
            print(f"[PROGRESS] {self.current_operation}: {elapsed:.1f}s elapsed, "
                  f"{progress_percent:.1f}% of timeout, {remaining:.1f}s remaining")
        else:
            # No timeout - show quality stages
            if elapsed < 30:
                stage = "Initializing neural models"
            elif elapsed < 60:
                stage = "Generating autoregressive samples"
            elif elapsed < 120:
                stage = "Processing diffusion layers"
            elif elapsed < 180:
                stage = "Neural vocoding"
            else:
                stage = "Final quality enhancement"
            
            print(f"[PROGRESS] {self.current_operation}: {elapsed:.1f}s - {stage}")
    
    def finish_operation(self, success: bool = True):
        """
        Finish the current operation
        
        Args:
            success: Whether the operation completed successfully
        """
        if self.start_time:
            elapsed = time.time() - self.start_time
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"[PROGRESS] {status}: {self.current_operation} completed in {elapsed:.1f}s")
        
        # Stop monitoring
        self.cancel_flag.set()
        if self.progress_thread and self.progress_thread.is_alive():
            self.progress_thread.join(timeout=2)
        
        # Reset state
        self.current_operation = None
        self.start_time = None
        self.timeout_seconds = None
        self.progress_callback = None
    
    def is_cancelled(self) -> bool:
        """Check if the operation was cancelled"""
        return self.cancel_flag.is_set()
    
    def cancel_operation(self):
        """Cancel the current operation"""
        if self.current_operation:
            print(f"[PROGRESS] 🛑 Cancelling: {self.current_operation}")
            self.cancel_flag.set()
    
    @contextmanager
    def track_operation(self, operation_name: str, timeout_seconds: Optional[float] = None):
        """
        Context manager for tracking an operation
        
        Usage:
            with progress_tracker.track_operation("Synthesis", timeout_seconds=300):
                # Your operation here
                pass
        """
        self.start_operation(operation_name, timeout_seconds)
        success = False
        
        try:
            yield self
            success = True
            
        except Exception as e:
            print(f"[PROGRESS] Operation failed: {e}")
            raise
            
        finally:
            self.finish_operation(success)


class ProgressDisplay:
    """Enhanced progress display with visual indicators"""
    
    @staticmethod
    def show_synthesis_progress(elapsed: float, timeout: Optional[float] = None):
        """Show a visual progress bar for synthesis"""
        if timeout:
            # Progress bar with timeout
            progress = min(elapsed / timeout, 1.0)
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            percent = progress * 100
            
            remaining = max(timeout - elapsed, 0)
            print(f"\r[SYNTHESIS] |{bar}| {percent:.1f}% ({elapsed:.1f}s/{timeout:.1f}s, {remaining:.1f}s left)", end='')
        else:
            # Infinite progress indicator
            spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            spinner = spinner_chars[int(elapsed) % len(spinner_chars)]
            
            # Estimate stage based on time
            if elapsed < 30:
                stage = "Initializing"
            elif elapsed < 60:
                stage = "Autoregressive"
            elif elapsed < 120:
                stage = "Diffusion"
            elif elapsed < 180:
                stage = "Vocoding"
            else:
                stage = "Enhancement"
            
            print(f"\r[SYNTHESIS] {spinner} {stage} - {elapsed:.1f}s elapsed (ultra-quality synthesis)", end='')
    
    @staticmethod
    def clear_progress():
        """Clear the progress line"""
        print()


# Global progress tracker instance
progress_tracker = TortoiseProgressTracker()


def get_progress_tracker() -> TortoiseProgressTracker:
    """Get the global progress tracker instance"""
    return progress_tracker


if __name__ == "__main__":
    # Test progress tracking
    import time
    
    tracker = TortoiseProgressTracker()
    
    print("Testing progress tracker...")
    
    with tracker.track_operation("Test Synthesis", timeout_seconds=10):
        for i in range(10):
            if tracker.is_cancelled():
                print("Operation was cancelled!")
                break
            time.sleep(1)
            print(f"Working... step {i+1}")
    
    print("Test completed!")
