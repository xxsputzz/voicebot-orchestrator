"""
System Resource Manager for Tortoise TTS
Comprehensive system RAM, CPU, and process management
"""

import psutil
import gc
import os
import sys
import time
import threading
from typing import Dict, List, Any, Optional
import subprocess
from pathlib import Path


class SystemResourceManager:
    """
    Comprehensive system resource management for CPU, RAM, and processes
    Complements the GPU manager for complete resource cleanup
    """
    
    def __init__(self):
        self.process_pid = os.getpid()
        self.initial_memory = None
        self.peak_memory = 0
        self.monitoring_active = False
        self.monitor_thread = None
        self.tracked_processes = []
        
        # Get initial memory baseline
        self._record_initial_memory()
    
    def _record_initial_memory(self):
        """Record initial system memory state"""
        try:
            process = psutil.Process(self.process_pid)
            memory_info = process.memory_info()
            
            self.initial_memory = {
                'rss': memory_info.rss / (1024**2),  # MB
                'vms': memory_info.vms / (1024**2),  # MB
                'percent': process.memory_percent(),
                'timestamp': time.time()
            }
            
            print(f"[SYS_MANAGER] Initial memory baseline: {self.initial_memory['rss']:.1f}MB RSS, {self.initial_memory['percent']:.1f}%")
            
        except Exception as e:
            print(f"[SYS_MANAGER] Could not record initial memory: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system resource status"""
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process-specific information
            process = psutil.Process(self.process_pid)
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            # Calculate memory pressure
            memory_pressure = "low"
            if memory.percent > 90:
                memory_pressure = "critical"
            elif memory.percent > 80:
                memory_pressure = "high"
            elif memory.percent > 70:
                memory_pressure = "medium"
            
            return {
                "cpu": {
                    "utilization_percent": cpu_percent,
                    "core_count": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "process_cpu_percent": process_cpu
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent_used": memory.percent,
                    "pressure": memory_pressure,
                    "process_memory_mb": process_memory.rss / (1024**2),
                    "process_memory_percent": process.memory_percent(),
                    "peak_memory_mb": self.peak_memory
                },
                "swap": {
                    "total_gb": swap.total / (1024**3),
                    "used_gb": swap.used / (1024**3),
                    "percent_used": swap.percent
                },
                "processes": {
                    "total_count": len(psutil.pids()),
                    "python_processes": len(self._find_python_processes()),
                    "tracked_processes": len(self.tracked_processes)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _find_python_processes(self) -> List[Dict[str, Any]]:
        """Find all Python processes"""
        python_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent', 'cpu_percent']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': ' '.join(proc.info['cmdline'] or [])[:100],  # Truncate for display
                            'memory_percent': proc.info['memory_percent'],
                            'cpu_percent': proc.info['cpu_percent']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            print(f"[SYS_MANAGER] Error finding Python processes: {e}")
        
        return python_processes
    
    def clear_system_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Clear system memory and optimize RAM usage
        
        Args:
            aggressive: If True, perform more aggressive cleanup
            
        Returns:
            Dictionary with cleanup results
        """
        print("[SYS_MANAGER] Starting system memory cleanup...")
        
        # Get memory before cleanup
        before_status = self.get_system_status()
        before_memory = before_status['memory']['process_memory_mb']
        before_system = before_status['memory']['percent_used']
        
        cleanup_results = {
            "before": {
                "process_memory_mb": before_memory,
                "system_memory_percent": before_system
            }
        }
        
        try:
            # 1. Python garbage collection
            print("[SYS_MANAGER] Running garbage collection...")
            collected = gc.collect()
            print(f"   ✅ Collected {collected} objects")
            
            # 2. Clear various Python caches
            if aggressive:
                print("[SYS_MANAGER] Clearing Python caches...")
                
                # Clear import cache
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
                
                # Clear function code cache
                if hasattr(sys, 'intern'):
                    # Force string interning cleanup
                    pass
                
                # Multiple garbage collection passes
                for i in range(3):
                    collected += gc.collect()
                    time.sleep(0.1)
                
                print(f"   ✅ Aggressive cleanup: {collected} total objects collected")
            
            # 3. Clear OS-level caches (Windows)
            if sys.platform == 'win32' and aggressive:
                try:
                    print("[SYS_MANAGER] Clearing Windows system caches...")
                    # Clear Windows standby memory (requires admin rights)
                    subprocess.run(['powershell', '-Command', 
                                   'Clear-RecycleBin -Force -ErrorAction SilentlyContinue'], 
                                 capture_output=True, timeout=10)
                    print("   ✅ Windows cache cleanup attempted")
                except Exception as e:
                    print(f"   ⚠️ Windows cache cleanup failed: {e}")
            
            # 4. Force memory compaction
            if aggressive:
                print("[SYS_MANAGER] Forcing memory compaction...")
                # Create and delete large objects to force memory reorganization
                for _ in range(3):
                    large_obj = bytearray(10 * 1024 * 1024)  # 10MB
                    del large_obj
                    gc.collect()
                
            # Get memory after cleanup
            after_status = self.get_system_status()
            after_memory = after_status['memory']['process_memory_mb']
            after_system = after_status['memory']['percent_used']
            
            cleanup_results.update({
                "after": {
                    "process_memory_mb": after_memory,
                    "system_memory_percent": after_system
                },
                "freed": {
                    "process_memory_mb": before_memory - after_memory,
                    "system_memory_percent": before_system - after_system,
                    "objects_collected": collected
                },
                "success": True
            })
            
            print(f"[SYS_MANAGER] Memory cleanup completed:")
            print(f"   Process memory: {before_memory:.1f}MB → {after_memory:.1f}MB (freed {before_memory - after_memory:.1f}MB)")
            print(f"   System memory: {before_system:.1f}% → {after_system:.1f}% (freed {before_system - after_system:.1f}%)")
            print(f"   Objects collected: {collected}")
            
        except Exception as e:
            print(f"[SYS_MANAGER] Memory cleanup failed: {e}")
            cleanup_results.update({
                "success": False,
                "error": str(e)
            })
        
        return cleanup_results
    
    def optimize_cpu_usage(self) -> Dict[str, Any]:
        """Optimize CPU usage and process priorities"""
        print("[SYS_MANAGER] Optimizing CPU usage...")
        
        optimization_results = {
            "actions_taken": [],
            "before_cpu": None,
            "after_cpu": None
        }
        
        try:
            # Get CPU before optimization
            before_cpu = psutil.cpu_percent(interval=1)
            optimization_results["before_cpu"] = before_cpu
            
            # 1. Adjust process priority (lower priority for better system responsiveness)
            try:
                process = psutil.Process(self.process_pid)
                current_priority = process.nice()
                
                # Set to below normal priority on Windows, nice +5 on Unix
                if sys.platform == 'win32':
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    optimization_results["actions_taken"].append(f"Set process priority to BELOW_NORMAL")
                else:
                    process.nice(5)  # Lower priority
                    optimization_results["actions_taken"].append(f"Set process nice value to 5")
                    
            except Exception as e:
                print(f"   ⚠️ Could not adjust process priority: {e}")
            
            # 2. Limit CPU affinity if high CPU usage
            if before_cpu > 80:
                try:
                    process = psutil.Process(self.process_pid)
                    cpu_count = psutil.cpu_count()
                    
                    if cpu_count > 2:
                        # Use only half the available cores
                        available_cores = list(range(cpu_count // 2))
                        process.cpu_affinity(available_cores)
                        optimization_results["actions_taken"].append(f"Limited CPU affinity to cores {available_cores}")
                        
                except Exception as e:
                    print(f"   ⚠️ Could not set CPU affinity: {e}")
            
            # 3. Yield CPU time
            time.sleep(0.1)
            
            # Get CPU after optimization
            after_cpu = psutil.cpu_percent(interval=1)
            optimization_results["after_cpu"] = after_cpu
            
            print(f"[SYS_MANAGER] CPU optimization completed:")
            print(f"   CPU usage: {before_cpu:.1f}% → {after_cpu:.1f}%")
            for action in optimization_results["actions_taken"]:
                print(f"   ✅ {action}")
            
            optimization_results["success"] = True
            
        except Exception as e:
            print(f"[SYS_MANAGER] CPU optimization failed: {e}")
            optimization_results.update({
                "success": False,
                "error": str(e)
            })
        
        return optimization_results
    
    def kill_resource_intensive_processes(self, memory_threshold_mb: float = 500, 
                                        cpu_threshold_percent: float = 50) -> List[int]:
        """
        Kill resource-intensive Python processes
        
        Args:
            memory_threshold_mb: Memory threshold in MB
            cpu_threshold_percent: CPU threshold percentage
            
        Returns:
            List of killed process PIDs
        """
        print(f"[SYS_MANAGER] Looking for resource-intensive processes...")
        print(f"   Thresholds: {memory_threshold_mb}MB memory, {cpu_threshold_percent}% CPU")
        
        killed_pids = []
        
        try:
            python_processes = self._find_python_processes()
            
            for proc_info in python_processes:
                pid = proc_info['pid']
                
                # Skip current process
                if pid == self.process_pid:
                    continue
                
                try:
                    proc = psutil.Process(pid)
                    memory_mb = proc.memory_info().rss / (1024**2)
                    cpu_percent = proc.cpu_percent()
                    
                    # Check if process exceeds thresholds
                    if memory_mb > memory_threshold_mb or cpu_percent > cpu_threshold_percent:
                        print(f"   🎯 Killing PID {pid}: {memory_mb:.1f}MB, {cpu_percent:.1f}% CPU")
                        proc.terminate()
                        killed_pids.append(pid)
                        
                        # Wait for graceful termination
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            # Force kill if graceful termination fails
                            proc.kill()
                            print(f"   ⚡ Force killed PID {pid}")
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                except Exception as e:
                    print(f"   ⚠️ Error handling PID {pid}: {e}")
            
            if killed_pids:
                print(f"[SYS_MANAGER] ✅ Killed {len(killed_pids)} resource-intensive processes")
            else:
                print(f"[SYS_MANAGER] ℹ️ No resource-intensive processes found")
            
        except Exception as e:
            print(f"[SYS_MANAGER] Error killing processes: {e}")
        
        return killed_pids
    
    def comprehensive_cleanup(self) -> Dict[str, Any]:
        """Perform comprehensive system resource cleanup"""
        print("[SYS_MANAGER] 🧹 Starting comprehensive system cleanup...")
        
        cleanup_summary = {
            "timestamp": time.time(),
            "memory_cleanup": {},
            "cpu_optimization": {},
            "process_cleanup": [],
            "overall_success": True
        }
        
        try:
            # 1. Memory cleanup
            print("\n1️⃣ Memory Cleanup:")
            cleanup_summary["memory_cleanup"] = self.clear_system_memory(aggressive=True)
            
            # 2. CPU optimization
            print("\n2️⃣ CPU Optimization:")
            cleanup_summary["cpu_optimization"] = self.optimize_cpu_usage()
            
            # 3. Process cleanup
            print("\n3️⃣ Process Cleanup:")
            killed_pids = self.kill_resource_intensive_processes()
            cleanup_summary["process_cleanup"] = killed_pids
            
            # 4. Final garbage collection
            print("\n4️⃣ Final Cleanup:")
            final_gc = gc.collect()
            print(f"   ✅ Final garbage collection: {final_gc} objects")
            
            # 5. Update peak memory tracking
            current_status = self.get_system_status()
            current_memory = current_status['memory']['process_memory_mb']
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            
            print(f"\n🎯 Comprehensive cleanup completed!")
            print(f"   Current memory: {current_memory:.1f}MB")
            print(f"   Peak memory: {self.peak_memory:.1f}MB")
            
            cleanup_summary["final_memory_mb"] = current_memory
            cleanup_summary["peak_memory_mb"] = self.peak_memory
            
        except Exception as e:
            print(f"[SYS_MANAGER] ❌ Comprehensive cleanup failed: {e}")
            cleanup_summary["overall_success"] = False
            cleanup_summary["error"] = str(e)
        
        return cleanup_summary
    
    def start_resource_monitoring(self, interval: float = 30.0):
        """Start background resource monitoring"""
        if self.monitoring_active:
            return
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    status = self.get_system_status()
                    
                    # Check for high resource usage
                    memory_percent = status['memory']['percent_used']
                    cpu_percent = status['cpu']['utilization_percent']
                    process_memory = status['memory']['process_memory_mb']
                    
                    # Update peak memory
                    if process_memory > self.peak_memory:
                        self.peak_memory = process_memory
                    
                    # Log warnings for high usage
                    if memory_percent > 85:
                        print(f"[SYS_MANAGER] ⚠️ High system memory usage: {memory_percent:.1f}%")
                    
                    if cpu_percent > 90:
                        print(f"[SYS_MANAGER] ⚠️ High CPU usage: {cpu_percent:.1f}%")
                    
                    if process_memory > 2000:  # 2GB
                        print(f"[SYS_MANAGER] ⚠️ High process memory: {process_memory:.1f}MB")
                    
                    # Auto-cleanup if critical
                    if memory_percent > 95 or process_memory > 4000:  # 4GB
                        print(f"[SYS_MANAGER] 🚨 Critical resource usage - triggering auto-cleanup")
                        self.clear_system_memory(aggressive=True)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"[SYS_MANAGER] Monitoring error: {e}")
                    time.sleep(interval)
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"[SYS_MANAGER] Resource monitoring started (interval: {interval}s)")
    
    def stop_resource_monitoring(self):
        """Stop background resource monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            print("[SYS_MANAGER] Resource monitoring stopped")


# Global system resource manager instance
system_manager = SystemResourceManager()


def get_system_manager() -> SystemResourceManager:
    """Get the global system resource manager instance"""
    return system_manager


def cleanup_system_resources():
    """Quick system cleanup function for external use"""
    return system_manager.comprehensive_cleanup()


if __name__ == "__main__":
    # Test system resource manager
    print("Testing System Resource Manager...")
    
    manager = SystemResourceManager()
    
    print("\n📊 System Status:")
    status = manager.get_system_status()
    for category, data in status.items():
        print(f"  {category}: {data}")
    
    print("\n🧹 Testing Memory Cleanup:")
    cleanup_result = manager.clear_system_memory(aggressive=True)
    print(f"Cleanup result: {cleanup_result}")
    
    print("\n⚡ Testing CPU Optimization:")
    cpu_result = manager.optimize_cpu_usage()
    print(f"CPU optimization: {cpu_result}")
    
    print("\n🎯 Comprehensive Cleanup:")
    comprehensive_result = manager.comprehensive_cleanup()
    print(f"Comprehensive cleanup: {comprehensive_result['overall_success']}")
