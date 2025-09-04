#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'aws_microservices'))

from enhanced_service_manager import EnhancedServiceManager
import time

def test_cache():
    """Test the cache mechanism"""
    manager = EnhancedServiceManager()
    
    print("=== Cache Debug Test ===")
    print(f"Initial cache state:")
    print(f"  Cache data: {manager._status_cache}")
    print(f"  Cache times: {manager._status_cache_time}")
    print(f"  Cache duration: {manager._cache_duration}")
    
    print("\n1. First status check...")
    status1 = manager.get_service_status(fast_mode=True)
    print(f"Cache after first check:")
    print(f"  Cache data: {manager._status_cache}")
    print(f"  Cache times: {list(manager._status_cache_time.keys())}")
    
    print("\n2. Second status check (should use cache)...")
    start_time = time.time()
    status2 = manager.get_service_status(fast_mode=True)
    end_time = time.time()
    print(f"Second check took: {end_time - start_time:.3f} seconds")
    
    print("\n3. Check cache age for all services:")
    current_time = time.time()
    for service_name in manager.service_configs.keys():
        if service_name in manager._status_cache_time:
            cache_age = current_time - manager._status_cache_time[service_name]
            print(f"  {service_name}: {cache_age:.2f} seconds old (valid: {cache_age < manager._cache_duration})")
        else:
            print(f"  {service_name}: Not in cache")
    
    print("\n4. Test the cache checking logic like in manage_individual_services:")
    current_time = time.time()
    cached_services = []
    for service_name in manager.service_configs.keys():
        if service_name in manager._status_cache_time:
            cache_age = current_time - manager._status_cache_time[service_name]
            if cache_age < manager._cache_duration:
                cached_services.append(service_name)
    
    print(f"Total services: {len(manager.service_configs)}")
    print(f"Cached services: {len(cached_services)} - {cached_services}")
    print(f"Should show loading: {len(cached_services) < len(manager.service_configs)}")
    
    print("\n5. Test individual service health check with cache:")
    for service_name in ['stt', 'kokoro_tts']:
        start_time = time.time()
        result = manager.check_service_health(service_name, timeout=1, use_cache=True)
        end_time = time.time()
        print(f"  {service_name}: {result} (took {end_time - start_time:.3f}s)")
        

if __name__ == "__main__":
    test_cache()
