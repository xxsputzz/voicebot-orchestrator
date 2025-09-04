"""
Tortoise TTS GPU Memory Management System
Comprehensive GPU memory management, cleanup, and monitoring for Tortoise TTS
"""

import torch
import gc
import time
import threading
import atexit
import weakref
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import psutil
import os
import signal


# Import system resource management
try:
    from system_resource_manager import get_system_manager
    SYSTEM_MANAGER_AVAILABLE = True
except ImportError:
    print("Warning: System resource manager not available")
    SYSTEM_MANAGER_AVAILABLE = False


class TortoiseGPUManager:
    """
    Comprehensive GPU memory management for Tortoise TTS
    Handles memory allocation, cleanup, monitoring, and process management
    """
    
    def __init__(self):
        self.device = None
        self.allocated_models = []  # Track allocated models
        self.memory_snapshots = []  # Track memory usage over time
        self.cleanup_registered = False
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Initialize system resource manager
        self.system_manager = get_system_manager() if SYSTEM_MANAGER_AVAILABLE else None
        
        # Auto-initialize device (prefer CUDA if available)
        self.initialize_device()
        
        # Register cleanup handlers
        self._register_cleanup_handlers()
    
    def _register_cleanup_handlers(self):
        """Register various cleanup handlers for different shutdown scenarios"""
        if not self.cleanup_registered:
            # Register atexit handler
            atexit.register(self.cleanup_all)
            
            # Register signal handlers for graceful shutdown
            def signal_cleanup_handler(signum, frame):
                print(f"\n[GPU_MANAGER] Received signal {signum}, cleaning up GPU...")
                self.cleanup_all()
                os._exit(0)
            
            # Handle common termination signals
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, signal_cleanup_handler)
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, signal_cleanup_handler)
            
            self.cleanup_registered = True
            print("[GPU_MANAGER] Cleanup handlers registered")
    
    def initialize_device(self, device=None, force_gpu=False) -> str:
        """
        Initialize and verify GPU device with comprehensive error handling
        
        Args:
            device: Preferred device ('cuda', 'cpu', etc.)
            force_gpu: If True, fail if GPU is not available
            
        Returns:
            Actual device that will be used
        """
        if device == "cuda" or (device is None and torch.cuda.is_available()):
            if not torch.cuda.is_available():
                if force_gpu:
                    raise RuntimeError("GPU forced but CUDA not available")
                print("[GPU_MANAGER] CUDA not available, using CPU")
                self.device = "cpu"
                return self.device
            
            try:
                # Test CUDA functionality
                test_tensor = torch.randn(2, 2).to('cuda')
                del test_tensor
                torch.cuda.empty_cache()
                
                device_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                self.device = "cuda"
                print(f"[GPU_MANAGER] GPU initialized: {device_name} ({total_memory:.1f}GB)")
                
                # Start memory monitoring
                self.start_memory_monitoring()
                
                return self.device
                
            except Exception as e:
                if force_gpu:
                    raise RuntimeError(f"GPU forced but initialization failed: {e}")
                print(f"[GPU_MANAGER] GPU test failed: {e}, using CPU")
                self.device = "cpu"
                return self.device
        else:
            self.device = device or "cpu"
            print(f"[GPU_MANAGER] Using device: {self.device}")
            return self.device
    
    def track_model(self, model, model_name: str = "unknown"):
        """
        Track a model for memory management
        
        Args:
            model: PyTorch model to track
            model_name: Name/identifier for the model
        """
        with self.lock:
            # Use weak references to avoid circular references
            model_ref = weakref.ref(model, lambda ref: self._model_cleanup_callback(model_name))
            
            model_info = {
                'name': model_name,
                'ref': model_ref,
                'device': str(next(model.parameters()).device) if hasattr(model, 'parameters') else 'unknown',
                'timestamp': time.time()
            }
            
            self.allocated_models.append(model_info)
            print(f"[GPU_MANAGER] Tracking model: {model_name} on {model_info['device']}")
    
    def _model_cleanup_callback(self, model_name: str):
        """Callback when a tracked model is garbage collected"""
        print(f"[GPU_MANAGER] Model {model_name} was garbage collected")
    
    def force_models_to_device(self, tts_service, target_device: str = None):
        """
        Force all TTS models to specified device with error handling
        
        Args:
            tts_service: Tortoise TTS service instance
            target_device: Target device ('cuda', 'cpu', etc.)
        """
        if target_device is None:
            target_device = self.device
        
        if not hasattr(tts_service, 'tts') or tts_service.tts is None:
            print("[GPU_MANAGER] No TTS service to move")
            return
        
        try:
            print(f"[GPU_MANAGER] Moving models to {target_device}...")
            
            # Track and move autoregressive model
            if hasattr(tts_service.tts, 'autoregressive') and tts_service.tts.autoregressive is not None:
                tts_service.tts.autoregressive = tts_service.tts.autoregressive.to(target_device)
                self.track_model(tts_service.tts.autoregressive, "autoregressive")
                print(f"[GPU_MANAGER] Autoregressive model moved to {target_device}")
            
            # Track and move diffusion model
            if hasattr(tts_service.tts, 'diffusion') and tts_service.tts.diffusion is not None:
                tts_service.tts.diffusion = tts_service.tts.diffusion.to(target_device)
                self.track_model(tts_service.tts.diffusion, "diffusion")
                print(f"[GPU_MANAGER] Diffusion model moved to {target_device}")
            
            # Track and move vocoder
            if hasattr(tts_service.tts, 'vocoder') and tts_service.tts.vocoder is not None:
                tts_service.tts.vocoder = tts_service.tts.vocoder.to(target_device)
                self.track_model(tts_service.tts.vocoder, "vocoder")
                print(f"[GPU_MANAGER] Vocoder model moved to {target_device}")
            
            # Track and move CLVP model
            if hasattr(tts_service.tts, 'clvp') and tts_service.tts.clvp is not None:
                tts_service.tts.clvp = tts_service.tts.clvp.to(target_device)
                self.track_model(tts_service.tts.clvp, "clvp")
                print(f"[GPU_MANAGER] CLVP model moved to {target_device}")
            
            # Force garbage collection and cache clearing
            self.clear_gpu_cache()
            
        except Exception as e:
            print(f"[GPU_MANAGER] Error moving models to {target_device}: {e}")
            raise
    
    def clear_gpu_cache(self, aggressive: bool = False):
        """
        Clear GPU cache and force garbage collection with timeout protection
        
        Args:
            aggressive: If True, perform more aggressive cleanup
        """
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                # Get memory before cleanup
                before_allocated = torch.cuda.memory_allocated(0) / (1024**2)
                before_reserved = torch.cuda.memory_reserved(0) / (1024**2)
                
                # Force garbage collection first (safe operation)
                gc.collect()
                
                # Basic cache clear (usually safe)
                torch.cuda.empty_cache()
                
                if aggressive:
                    # Use timeout to prevent freezing during aggressive cleanup
                    import threading
                    import signal
                    
                    cleanup_completed = threading.Event()
                    cleanup_error = [None]
                    
                    def aggressive_cleanup():
                        try:
                            # Force synchronization with timeout protection
                            try:
                                torch.cuda.synchronize()
                            except Exception as sync_error:
                                print(f"[GPU_MANAGER] Synchronization warning: {sync_error}")
                            
                            # Multiple cache clears with shorter delays
                            for i in range(2):  # Reduced from 3 to 2
                                torch.cuda.empty_cache()
                                if i < 1:  # Only sleep between iterations, not after last
                                    time.sleep(0.05)  # Reduced from 0.1 to 0.05
                            
                            cleanup_completed.set()
                        except Exception as e:
                            cleanup_error[0] = e
                            cleanup_completed.set()
                    
                    # Run aggressive cleanup in a separate thread with timeout
                    cleanup_thread = threading.Thread(target=aggressive_cleanup, daemon=True)
                    cleanup_thread.start()
                    
                    # Wait for completion with timeout
                    if cleanup_completed.wait(timeout=5.0):  # 5 second timeout
                        if cleanup_error[0]:
                            print(f"[GPU_MANAGER] Aggressive cleanup error: {cleanup_error[0]}")
                    else:
                        print("[GPU_MANAGER] Aggressive cleanup timed out - continuing with basic cleanup")
                
                # Get memory after cleanup (always safe)
                try:
                    after_allocated = torch.cuda.memory_allocated(0) / (1024**2)
                    after_reserved = torch.cuda.memory_reserved(0) / (1024**2)
                    
                    freed_allocated = before_allocated - after_allocated
                    freed_reserved = before_reserved - after_reserved
                    
                    print(f"[GPU_MANAGER] GPU cache cleared:")
                    print(f"  Allocated: {before_allocated:.1f}MB → {after_allocated:.1f}MB (freed {freed_allocated:.1f}MB)")
                    print(f"  Reserved: {before_reserved:.1f}MB → {after_reserved:.1f}MB (freed {freed_reserved:.1f}MB)")
                except Exception as memory_check_error:
                    print(f"[GPU_MANAGER] Memory status check error: {memory_check_error}")
                    print(f"[GPU_MANAGER] GPU cache clearing completed (status check failed)")
                    
            except Exception as e:
                print(f"[GPU_MANAGER] Error clearing GPU cache: {e}")
                # Fallback: basic cleanup only
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("[GPU_MANAGER] Fallback GPU cleanup completed")
                except:
                    print("[GPU_MANAGER] Even fallback cleanup failed")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status including GPU and system"""
        gpu_status = {}
        system_status = {}
        
        # Get GPU memory status
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                reserved = torch.cuda.memory_reserved(0) / (1024**2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                free = total - reserved
                
                # Fix utilization calculation - use allocated vs total (not reserved vs total)
                utilization_percent = (allocated / total) * 100
                
                # Add memory pressure warnings
                memory_pressure = "low"
                if utilization_percent > 90:
                    memory_pressure = "critical"
                elif utilization_percent > 80:
                    memory_pressure = "high"
                elif utilization_percent > 60:
                    memory_pressure = "medium"
                
                gpu_status = {
                    "device": torch.cuda.get_device_name(0),
                    "allocated_mb": allocated,
                    "reserved_mb": reserved,
                    "total_mb": total,
                    "free_mb": free,
                    "utilization_percent": utilization_percent,
                    "memory_pressure": memory_pressure,
                    "tracked_models": len(self.allocated_models),
                    "can_allocate_more": (total - allocated) > 1000  # Can allocate 1GB more
                }
            except Exception as e:
                gpu_status = {"error": str(e), "device": self.device}
        else:
            gpu_status = {"device": self.device, "gpu_available": False}
        
        # Get system memory status
        if self.system_manager:
            try:
                system_status = self.system_manager.get_system_status()
            except Exception as e:
                print(f"[GPU_MANAGER] Warning: Could not get system status: {e}")
                system_status = {"error": str(e)}
        
        # Return unified status
        return {
            "gpu": gpu_status,
            "system": system_status,
            "timestamp": time.time(),
            "unified_status": self._get_unified_resource_status(gpu_status, system_status)
        }
    
    def _get_unified_resource_status(self, gpu_status, system_status):
        """Get unified resource status combining GPU and system metrics"""
        try:
            status = "healthy"
            
            # Check GPU pressure
            if gpu_status.get("memory_pressure") in ["high", "critical"]:
                status = "gpu_stressed"
            
            # Check system pressure
            if system_status.get("memory", {}).get("pressure_level") in ["high", "critical"]:
                if status == "gpu_stressed":
                    status = "critically_stressed"
                else:
                    status = "system_stressed"
            
            return {
                "overall_status": status,
                "needs_cleanup": status != "healthy",
                "recommended_action": self._get_recommended_action(status)
            }
            
        except Exception as e:
            return {"overall_status": "unknown", "error": str(e)}
    
    def _get_recommended_action(self, status):
        """Get recommended action based on resource status"""
        actions = {
            "healthy": "No action needed",
            "gpu_stressed": "Consider GPU memory cleanup",
            "system_stressed": "Consider system memory cleanup",
            "critically_stressed": "Immediate comprehensive cleanup recommended",
            "unknown": "Check system status"
        }
        return actions.get(status, "Monitor resource usage")
    
    def start_memory_monitoring(self, interval: float = 30.0):
        """
        Start background memory monitoring
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active or self.device != "cuda":
            return
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    status = self.get_memory_status()
                    
                    # Store snapshot
                    snapshot = {
                        "timestamp": time.time(),
                        "status": status
                    }
                    
                    with self.lock:
                        self.memory_snapshots.append(snapshot)
                        # Keep only last 100 snapshots
                        if len(self.memory_snapshots) > 100:
                            self.memory_snapshots.pop(0)
                    
                    # Log if memory usage is high
                    # Handle both old and new status format
                    gpu_status = status.get("gpu", status)  # Fallback to old format if needed
                    
                    if gpu_status.get("utilization_percent", 0) > 80:
                        pressure = gpu_status.get("memory_pressure", "unknown")
                        print(f"[GPU_MANAGER] High GPU memory usage: {gpu_status['utilization_percent']:.1f}% ({pressure} pressure)")
                        
                        # Trigger cleanup if memory is critically high
                        if gpu_status.get("utilization_percent", 0) > 95:
                            print(f"[GPU_MANAGER] CRITICAL memory usage - triggering emergency cleanup...")
                            try:
                                self.clear_gpu_cache(aggressive=True)
                            except Exception as cleanup_error:
                                print(f"[GPU_MANAGER] Emergency cleanup failed: {cleanup_error}")
                    
                    time.sleep(interval)
                
                except Exception as e:
                    print(f"[GPU_MANAGER] Monitoring error: {e}")
                    time.sleep(interval)
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"[GPU_MANAGER] Memory monitoring started (interval: {interval}s)")
    
    def stop_memory_monitoring(self):
        """Stop background memory monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            print("[GPU_MANAGER] Memory monitoring stopped")
    
    def cleanup_models(self):
        """Clean up tracked models"""
        with self.lock:
            print(f"[GPU_MANAGER] Cleaning up {len(self.allocated_models)} tracked models...")
            
            for model_info in self.allocated_models:
                try:
                    model = model_info['ref']()
                    if model is not None:
                        # Move to CPU to free GPU memory
                        if hasattr(model, 'cpu'):
                            model.cpu()
                        # Delete reference
                        del model
                    print(f"[GPU_MANAGER] Cleaned up model: {model_info['name']}")
                except Exception as e:
                    print(f"[GPU_MANAGER] Error cleaning model {model_info['name']}: {e}")
            
            self.allocated_models.clear()
    
    def cleanup_all(self):
        """Fast, non-blocking comprehensive cleanup of GPU and system resources"""
        print("[GPU_MANAGER] Starting fast comprehensive cleanup...")
        
        try:
            # 1. Stop monitoring (fast)
            self.stop_memory_monitoring()
            
            # 2. Clean up tracked models (fast)
            self.cleanup_models()
            
            # 3. Basic GPU cache clearing only (avoid aggressive to prevent hanging)
            self.clear_gpu_cache(aggressive=False)
            
            # 4. Quick system resource cleanup (non-blocking)
            if self.system_manager:
                print("[GPU_MANAGER] Skipping comprehensive system cleanup to prevent blocking")
                print("[GPU_MANAGER] (Use /manual-system-cleanup endpoint for full system cleanup)")
                
                # Just do basic garbage collection
                try:
                    import gc
                    collected = gc.collect()
                    print(f"[GPU_MANAGER] ✅ Quick cleanup: {collected} objects collected")
                except Exception as e:
                    print(f"[GPU_MANAGER] ⚠️ Quick cleanup failed: {e}")
            else:
                # Final garbage collection if no system manager
                gc.collect()
            
            print("[GPU_MANAGER] ✅ Fast comprehensive cleanup completed")
            
        except Exception as e:
            print(f"[GPU_MANAGER] Error during fast cleanup: {e}")
            # Emergency fallback - just garbage collection
            try:
                gc.collect()
                print("[GPU_MANAGER] ✅ Emergency fallback cleanup completed")
            except:
                print("[GPU_MANAGER] ❌ Even emergency cleanup failed")
    
    def comprehensive_system_cleanup_background(self):
        """Run comprehensive system cleanup in background (for manual use)"""
        if not self.system_manager:
            print("[GPU_MANAGER] No system manager available")
            return
        
        def background_system_cleanup():
            try:
                print("[GPU_MANAGER] Starting background comprehensive system cleanup...")
                result = self.system_manager.comprehensive_cleanup()
                if result.get("overall_success"):
                    print("[GPU_MANAGER] ✅ Background system cleanup completed successfully")
                else:
                    print(f"[GPU_MANAGER] ⚠️ Background system cleanup had issues: {result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"[GPU_MANAGER] ❌ Background system cleanup failed: {e}")
        
        import threading
        background_thread = threading.Thread(target=background_system_cleanup, daemon=True)
        background_thread.start()
        print("[GPU_MANAGER] Comprehensive system cleanup started in background")
        return background_thread
    def check_memory_available(self, required_mb: float = 2000) -> bool:
        """
        Check if enough GPU memory is available for an operation
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            True if enough memory is available
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return True  # CPU operations don't need GPU memory
        
        try:
            status = self.get_memory_status()
            # Handle both old and new status format
            gpu_status = status.get("gpu", status)  # Fallback to old format if needed
            
            available_mb = gpu_status.get("free_mb", 0)
            current_util = gpu_status.get("utilization_percent", 0)
            
            # Check both absolute memory and utilization percentage
            memory_ok = available_mb >= required_mb
            util_ok = current_util < 85  # Don't start if already high usage
            
            if not memory_ok:
                print(f"[GPU_MANAGER] Insufficient memory: need {required_mb}MB, have {available_mb}MB")
            if not util_ok:
                print(f"[GPU_MANAGER] High memory utilization: {current_util:.1f}%")
            
            return memory_ok and util_ok
            
        except Exception as e:
            print(f"[GPU_MANAGER] Error checking memory: {e}")
            return False
    
    def force_cleanup_before_synthesis(self):
        """Force cleanup before synthesis to ensure memory is available"""
        try:
            print("[GPU_MANAGER] Pre-synthesis cleanup...")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Check memory status after cleanup
            status = self.get_memory_status()
            print(f"[GPU_MANAGER] Post-cleanup status: {status.get('utilization_percent', 0):.1f}% utilization")
            
        except Exception as e:
            print(f"[GPU_MANAGER] Pre-synthesis cleanup failed: {e}")
    
    @contextmanager
    def gpu_context(self, tts_service=None, required_memory_mb: float = 2000):
        """
        Context manager for GPU operations with automatic cleanup and memory checks
        
        Args:
            tts_service: TTS service to manage
            required_memory_mb: Required memory for the operation
        
        Usage:
            with gpu_manager.gpu_context(tts_service, required_memory_mb=3000) as device:
                # GPU operations here
                pass
            # Automatic cleanup happens here
        """
        # Pre-operation checks
        if not self.check_memory_available(required_memory_mb):
            # Try cleanup and check again
            self.force_cleanup_before_synthesis()
            if not self.check_memory_available(required_memory_mb):
                raise RuntimeError(f"Insufficient GPU memory for operation (need {required_memory_mb}MB)")
        
        try:
            if tts_service and self.device == "cuda":
                self.force_models_to_device(tts_service, self.device)
            
            yield self.device
            
        finally:
            # Non-blocking cleanup after context
            try:
                # Use a separate thread for cleanup to prevent blocking
                def async_cleanup():
                    try:
                        self.clear_gpu_cache()
                        print("[GPU_MANAGER] Context cleanup completed")
                    except Exception as cleanup_error:
                        print(f"[GPU_MANAGER] Context cleanup error: {cleanup_error}")
                
                cleanup_thread = threading.Thread(target=async_cleanup, daemon=True)
                cleanup_thread.start()
                
                # Wait briefly for cleanup, but don't block indefinitely
                cleanup_thread.join(timeout=2.0)
                
                if cleanup_thread.is_alive():
                    print("[GPU_MANAGER] Context cleanup continuing in background")
                
            except Exception as final_error:
                print(f"[GPU_MANAGER] Context cleanup setup error: {final_error}")


class TortoiseProcessManager:
    """
    Process management for Tortoise TTS to prevent lingering processes
    """
    
    @staticmethod
    def kill_python_processes(exclude_current=True):
        """
        Kill all Python processes (optionally excluding current process)
        
        Args:
            exclude_current: If True, don't kill the current process
        """
        current_pid = os.getpid()
        killed_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'].lower().startswith('python'):
                        if exclude_current and proc.info['pid'] == current_pid:
                            continue
                        
                        # Check if it's a Tortoise-related process
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if any(keyword in cmdline.lower() for keyword in ['tortoise', 'tts', 'aws_microservices']):
                            proc.terminate()
                            killed_processes.append(proc.info['pid'])
                            print(f"[PROCESS_MANAGER] Terminated Python process: {proc.info['pid']}")
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        
        except Exception as e:
            print(f"[PROCESS_MANAGER] Error killing processes: {e}")
        
        return killed_processes
    
    @staticmethod
    def force_kill_gpu_processes():
        """Force kill processes using GPU memory"""
        if not torch.cuda.is_available():
            return []
        
        try:
            import subprocess
            # Get processes using GPU
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                gpu_pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
                
                for pid in gpu_pids:
                    try:
                        proc = psutil.Process(pid)
                        if proc.name().lower().startswith('python'):
                            proc.terminate()
                            print(f"[PROCESS_MANAGER] Terminated GPU process: {pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                return gpu_pids
        
        except Exception as e:
            print(f"[PROCESS_MANAGER] Error killing GPU processes: {e}")
        
        return []


# Global GPU manager instance
gpu_manager = TortoiseGPUManager()


def get_gpu_manager() -> TortoiseGPUManager:
    """Get the global GPU manager instance"""
    return gpu_manager


def cleanup_tortoise_gpu():
    """Quick cleanup function for external use"""
    gpu_manager.cleanup_all()


def emergency_gpu_cleanup():
    """Emergency cleanup - kills processes and clears GPU + system resources"""
    print("[EMERGENCY] Starting emergency GPU cleanup...")
    
    try:
        # Kill GPU processes
        TortoiseProcessManager.force_kill_gpu_processes()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Emergency system cleanup if manager is available
        if hasattr(gpu_manager, 'system_manager') and gpu_manager.system_manager:
            print("[EMERGENCY] Performing system resource cleanup...")
            gpu_manager.system_manager.clear_system_memory(aggressive=True)
            gpu_manager.system_manager.kill_resource_intensive_processes()
        
        # Force garbage collection
        gc.collect()
        
        print("[EMERGENCY] Emergency cleanup completed")
        
    except Exception as e:
        print(f"[EMERGENCY] Emergency cleanup failed: {e}")


if __name__ == "__main__":
    # Test GPU manager
    print("Testing Tortoise GPU Manager...")
    
    manager = TortoiseGPUManager()
    device = manager.initialize_device()
    
    print(f"Device: {device}")
    print(f"Memory status: {manager.get_memory_status()}")
    
    # Test cleanup
    manager.cleanup_all()
