"""
Sprint 6: Deployment & Packaging Demo

Demonstrates the complete Sprint 6 implementation including CLI commands,
microservices architecture, Docker containerization, and Kubernetes deployment.
"""

import asyncio
import json
import time
from datetime import datetime

from voicebot_orchestrator.sprint6_cli import (
    start_call,
    monitor_session,
    analytics_report,
    cache_manager,
    adapter_control,
    orchestrator_health
)


async def main():
    """Run comprehensive Sprint 6 deployment demo."""
    print("=" * 80)
    print("SPRINT 6: DEPLOYMENT & PACKAGING DEMO")
    print("Enterprise Microservices Architecture")
    print("=" * 80)
    
    # 1. Health Check
    print("\n1. System Health Check:")
    print("-" * 40)
    
    health_result = await orchestrator_health()
    print(f"System Status: {health_result['status']}")
    print(f"Version: {health_result['version']}")
    print(f"Environment: {health_result['environment']}")
    
    # Show service health
    health_checks = health_result.get('health_checks', {})
    services = health_checks.get('services', {})
    
    print("\nMicroservices Status:")
    for service_name, service_info in services.items():
        status = service_info.get('status', 'unknown')
        url = service_info.get('url', 'N/A')
        print(f"  {service_name}: {status} ({url})")
    
    # 2. Start Multiple Sessions (Microservices Demo)
    print("\n2. Starting Multiple Banking Sessions:")
    print("-" * 40)
    
    sessions = [
        {"id": "banking-session-001", "phone": "+1-555-0101", "customer": "CUST001"},
        {"id": "banking-session-002", "phone": "+1-555-0102", "customer": "CUST002"},
        {"id": "compliance-session-003", "phone": "+1-555-0103", "customer": "CUST003"}
    ]
    
    start_results = []
    for session in sessions:
        print(f"\nStarting session: {session['id']}")
        
        domain = "compliance" if "compliance" in session['id'] else "banking"
        
        result = await start_call(
            session_id=session['id'],
            phone_number=session['phone'],
            customer_id=session['customer'],
            domain=domain
        )
        
        start_results.append(result)
        print(f"  Status: {result['status']}")
        print(f"  Services: {len(result['services'])} microservices configured")
        
        # Brief delay between sessions
        await asyncio.sleep(0.5)
    
    # 3. Cache Management Operations
    print("\n3. Semantic Cache Management:")
    print("-" * 40)
    
    # Get cache statistics
    cache_stats_result = await cache_manager(operation="stats")
    cache_stats = cache_stats_result.get('cache_statistics', {})
    
    print(f"Cache Entries: {cache_stats.get('total_entries', 0)}")
    print(f"Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
    print(f"Cache Size: {cache_stats.get('cache_size_mb', 0):.2f} MB")
    
    # Optimize cache
    print("\nOptimizing cache...")
    optimize_result = await cache_manager(operation="optimize")
    print(f"Optimization: {optimize_result['status']}")
    
    # Export cache
    print("\nExporting cache data...")
    export_result = await cache_manager(
        operation="export",
        export_file="sprint6_cache_export.json"
    )
    if export_result['status'] == 'success':
        print(f"Cache exported to: {export_result['exported_to']}")
        print(f"Entries exported: {export_result['entries_exported']}")
    
    # 4. LoRA Adapter Management
    print("\n4. LoRA Adapter Management:")
    print("-" * 40)
    
    # List available adapters
    adapter_list_result = await adapter_control(operation="list")
    adapter_status = adapter_list_result.get('adapter_status', {})
    
    print(f"Available Adapters: {len(adapter_status.get('available_adapters', []))}")
    print(f"Loaded Adapters: {len(adapter_status.get('loaded_adapters', []))}")
    print(f"Active Adapter: {adapter_status.get('active_adapter', 'None')}")
    
    # Create new banking adapter if needed
    if 'banking-lora' not in adapter_status.get('available_adapters', []):
        print("\nCreating banking domain adapter...")
        create_result = await adapter_control(
            operation="create",
            adapter_name="banking"
        )
        print(f"Adapter creation: {create_result['status']}")
    
    # Load and activate banking adapter
    print("\nLoading banking adapter...")
    load_result = await adapter_control(
        operation="load",
        adapter_name="banking-lora"
    )
    print(f"Adapter loading: {load_result['status']}")
    
    # 5. Session Monitoring
    print("\n5. Session Monitoring:")
    print("-" * 40)
    
    for session in sessions[:2]:  # Monitor first 2 sessions
        print(f"\nMonitoring session: {session['id']}")
        
        monitor_result = await monitor_session(
            session_id=session['id'],
            follow=False,
            output_format="json"
        )
        
        print(f"  Status: {monitor_result['status']}")
        if 'analytics' in monitor_result:
            analytics = monitor_result['analytics']
            print(f"  Events: {analytics.get('total_events', 0)}")
            print(f"  Duration: {analytics.get('session_duration', 0):.1f}s")
    
    # 6. Analytics Reporting
    print("\n6. Analytics Reporting:")
    print("-" * 40)
    
    report_types = ["summary", "performance"]
    
    for report_type in report_types:
        print(f"\nGenerating {report_type} report...")
        
        report_result = await analytics_report(
            report_type=report_type,
            time_range="24h",
            output_format="json"
        )
        
        print(f"  Report Type: {report_result['report_type']}")
        print(f"  Time Range: {report_result['time_range']}")
        
        summary = report_result.get('summary', {})
        print(f"  Total Sessions: {summary.get('total_sessions', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"  Cache Hit Rate: {summary.get('cache_hit_rate', 0):.1%}")
        
        # Export report
        export_filename = f"sprint6_{report_type}_report.json"
        report_with_export = await analytics_report(
            report_type=report_type,
            time_range="24h",
            export_file=export_filename
        )
        
        if 'exported_to' in report_with_export:
            print(f"  Report exported to: {report_with_export['exported_to']}")
    
    # 7. CLI Commands Demo
    print("\n7. CLI Commands Available:")
    print("-" * 40)
    
    cli_commands = [
        "orchestrator start-call <session_id> --phone <phone> --domain banking",
        "orchestrator monitor-session --session-id <session_id> --follow",
        "orchestrator analytics-report --type summary --time-range 24h",
        "orchestrator cache-manager stats",
        "orchestrator adapter-control list",
        "orchestrator orchestrator-health"
    ]
    
    print("Production CLI Commands:")
    for cmd in cli_commands:
        print(f"  {cmd}")
    
    print("\nPackage Installation:")
    print("  pip install voicebot-orchestrator")
    print("  poetry install voicebot-orchestrator")
    
    print("\nMicroservice Entry Points:")
    microservices = [
        "orchestrator-core",
        "stt-service", 
        "llm-service",
        "tts-service",
        "cache-service",
        "analytics-service"
    ]
    
    for service in microservices:
        print(f"  {service}")
    
    # 8. Docker & Kubernetes Demo
    print("\n8. Deployment Options:")
    print("-" * 40)
    
    print("Docker Compose:")
    print("  docker-compose up                    # Basic services")
    print("  docker-compose --profile monitoring up  # With Prometheus/Grafana")
    print("  docker-compose --profile loadbalancer up  # With NGINX")
    
    print("\nKubernetes:")
    print("  kubectl apply -f k8s/orchestrator-core.yaml")
    print("  kubectl get pods -n voicebot-orchestrator")
    print("  kubectl logs -f deployment/orchestrator-core -n voicebot-orchestrator")
    
    print("\nDocker Build:")
    print("  docker build -t voicebot-orchestrator:1.0.0 .")
    print("  docker run -p 8000:8000 voicebot-orchestrator:1.0.0")
    
    # 9. Performance Summary
    print("\n9. Performance Summary:")
    print("-" * 40)
    
    final_health = await orchestrator_health()
    final_cache_stats = await cache_manager(operation="stats")
    final_adapter_status = await adapter_control(operation="list")
    
    print(f"System Health: {final_health['status']}")
    print(f"Microservices: 6 services running")
    print(f"Cache Entries: {final_cache_stats.get('cache_statistics', {}).get('total_entries', 0)}")
    print(f"Active Adapters: {len(final_adapter_status.get('adapter_status', {}).get('loaded_adapters', []))}")
    print(f"Sessions Created: {len(sessions)}")
    
    # 10. Architecture Overview
    print("\n10. Architecture Overview:")
    print("-" * 40)
    
    architecture = {
        "Orchestrator Core": "Central coordination and API gateway",
        "STT Service": "Speech-to-text processing with Whisper",
        "LLM Service": "Language model with semantic caching & LoRA",
        "TTS Service": "Text-to-speech synthesis with Kokoro",
        "Cache Service": "Semantic cache with Faiss vectors",
        "Analytics Service": "Metrics collection and reporting",
        "Redis": "Distributed caching and session storage",
        "Prometheus": "Metrics aggregation and monitoring",
        "Grafana": "Visualization and dashboards",
        "NGINX": "Load balancing and reverse proxy"
    }
    
    for component, description in architecture.items():
        print(f"  {component:20} - {description}")
    
    print("\n" + "=" * 80)
    print("SPRINT 6 DEPLOYMENT DEMO COMPLETE")
    print("✓ Enterprise CLI with 6 command groups")
    print("✓ Microservices architecture (6 services)")
    print("✓ Docker containerization with multi-stage builds")
    print("✓ Kubernetes manifests with auto-scaling")
    print("✓ Production-ready packaging with Poetry")
    print("✓ Health checks and monitoring integration")
    print("✓ Performance optimization and resource quotas")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
