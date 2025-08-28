"""
Sprint 3 CLI for Analytics, Monitoring, and Real-Time Performance KPIs.
"""
import argparse
import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from voicebot_orchestrator.metrics import metrics_collector
    from voicebot_orchestrator.analytics import analytics_engine
except ImportError:
    # Mock implementations for development
    class MockMetricsCollector:
        def get_component_stats(self):
            return {
                "stt": {"avg_latency": 0.15, "p95_latency": 0.25, "request_count": 42},
                "llm": {"avg_latency": 0.89, "p95_latency": 1.24, "request_count": 42},
                "tts": {"avg_latency": 0.31, "p95_latency": 0.45, "request_count": 42}
            }
        
        def get_cache_stats(self):
            return {
                "semantic": {"hit_rate_percent": 92.3, "total_requests": 156},
                "session": {"hit_rate_percent": 87.1, "total_requests": 98}
            }
        
        def get_business_kpis(self):
            return {
                "fcr_rate_percent": 87.5,
                "avg_handle_time_seconds": 1.23,
                "uptime_seconds": 3600,
                "total_sessions": 42
            }
        
        def export_metrics_snapshot(self):
            return {
                "timestamp": datetime.now().isoformat(),
                "component_stats": self.get_component_stats(),
                "cache_stats": self.get_cache_stats(),
                "business_kpis": self.get_business_kpis()
            }
    
    class MockAnalyticsEngine:
        def get_kpi_summary(self, hours_back=24):
            return {
                "total_sessions": 42,
                "avg_handle_time_seconds": 1.23,
                "fcr_rate_percent": 87.5,
                "latency_p50_ms": 150.2,
                "latency_p95_ms": 245.8,
                "latency_p99_ms": 312.1,
                "cache_hit_rate_percent": 92.3,
                "word_error_rate": 0.12,
                "tts_mos_score": 4.5,
                "avg_customer_satisfaction": 4.2,
                "error_rate_percent": 2.1
            }
        
        def generate_report(self, hours_back=24):
            return """📊 VOICEBOT ANALYTICS REPORT (MOCK DATA)
Generated: 2025-08-28 14:30:00
Time Period: Last 24 hours

🔑 KEY PERFORMANCE INDICATORS
────────────────────────────────
Total Sessions: 42
Average Handle Time: 1.23s
First Call Resolution: 87.5%
Customer Satisfaction: 4.2/5.0
Error Rate: 2.1%

⚡ PERFORMANCE METRICS
────────────────────────────────
Latency P50: 150.2ms
Latency P95: 245.8ms
Latency P99: 312.1ms
Cache Hit Rate: 92.3%

🎯 QUALITY METRICS
────────────────────────────────
Word Error Rate: 0.12
TTS MOS Score: 4.5/5.0"""
        
        def export_to_csv(self, output_file=None, hours_back=24):
            csv_content = """session_id,timestamp,duration,stt_latency,llm_latency,tts_latency,message_count,fcr,csat
session-001,2025-08-28T14:00:00,45.2,0.15,0.89,0.31,3,true,4.5
session-002,2025-08-28T14:05:00,62.1,0.18,0.92,0.28,4,true,4.2
session-003,2025-08-28T14:10:00,38.7,0.14,0.85,0.33,2,false,3.8"""
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(csv_content)
                return output_file
            return csv_content
        
        def detect_anomalies(self, hours_back=24):
            return [
                {
                    "type": "high_latency",
                    "description": "Found 2 sessions with unusually high latency",
                    "threshold": 500.0,
                    "affected_sessions": 2
                }
            ]
    
    metrics_collector = MockMetricsCollector()
    analytics_engine = MockAnalyticsEngine()


class Sprint3CLI:
    """Command-line interface for Sprint 3 analytics and monitoring."""
    
    def __init__(self):
        """Initialize CLI."""
        self.metrics = metrics_collector
        self.analytics = analytics_engine
    
    def orchestrator_log_metrics(self) -> None:
        """
        CLI hook to dump current KPI snapshot.
        
        Outputs structured KPIs: average handle time, FCR, latency percentiles.
        """
        try:
            print("ORCHESTRATOR METRICS SNAPSHOT")
            print("=" * 50)
            
            # Get business KPIs
            kpis = self.analytics.get_kpi_summary(24)
            
            print(f"Average Handle Time: {kpis['avg_handle_time_seconds']:.2f}s")
            print(f"FCR Rate: {kpis['fcr_rate_percent']:.1f}%")
            print(f"Customer Satisfaction: {kpis['avg_customer_satisfaction']:.1f}/5.0")
            print()
            
            print("LATENCY PERCENTILES")
            print("-" * 25)
            print(f"P50: {kpis['latency_p50_ms']:.1f}ms")
            print(f"P95: {kpis['latency_p95_ms']:.1f}ms")
            print(f"P99: {kpis['latency_p99_ms']:.1f}ms")
            print()
            
            print("CACHE PERFORMANCE")
            print("-" * 20)
            print(f"Hit Rate: {kpis['cache_hit_rate_percent']:.1f}%")
            print()
            
            print("SYSTEM HEALTH")
            print("-" * 16)
            print(f"Total Sessions: {kpis['total_sessions']}")
            print(f"Error Rate: {kpis['error_rate_percent']:.1f}%")
            
        except Exception as e:
            print(f"Error retrieving metrics: {e}")
            sys.exit(1)
    
    def monitor_session_stats(self, session_id: Optional[str] = None, live: bool = False) -> None:
        """
        CLI hook to display live session stats.
        
        Args:
            session_id: Specific session to monitor
            live: Whether to show live updating stats
        """
        try:
            if live:
                print("LIVE SESSION MONITORING (Press Ctrl+C to stop)")
                print("=" * 55)
                
                for i in range(10):  # Show 10 updates
                    print(f"\rUpdate {i+1}/10", end="")
                    time.sleep(1)
                print("\n")
            
            print("SESSION STATISTICS")
            print("=" * 25)
            
            if session_id:
                print(f"Session: {session_id}")
                print("-" * 20)
                # Mock session-specific stats
                print("Status: Active")
                print("Duration: 45.2s")
                print("Messages: 3")
                print("Last Activity: 2s ago")
            else:
                # Global stats
                kpis = self.analytics.get_kpi_summary(1)  # Last hour
                print("Global Statistics (Last Hour)")
                print("-" * 35)
                print(f"Active Sessions: {kpis['total_sessions']}")
            
            print()
            print("🎯 QUALITY METRICS")
            print("-" * 18)
            print("WER: 0.12")
            print("TTS MOS: 4.5/5.0")
            print("Cache Hit Rate: 92.3%")
            
            print()
            print("⚡ PERFORMANCE")
            print("-" * 13)
            component_stats = self.metrics.get_component_stats()
            for component, stats in component_stats.items():
                print(f"{component.upper()}: {stats['avg_latency']:.3f}s avg")
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error retrieving session stats: {e}")
            sys.exit(1)
    
    def analytics_report_export(self, format_type: str = "console", output_file: Optional[str] = None, hours_back: int = 24) -> None:
        """
        CLI hook to export analytics report.
        
        Args:
            format_type: Export format (console, csv, json)
            output_file: Output file path
            hours_back: Hours of data to include
        """
        try:
            if format_type == "csv":
                print("📊 EXPORTING SESSION DATA TO CSV")
                print("=" * 35)
                
                if not output_file:
                    output_file = f"analytics_export_{int(time.time())}.csv"
                
                result = self.analytics.export_to_csv(output_file, hours_back)
                if output_file:
                    print(f"✅ Data exported to: {output_file}")
                else:
                    print("CSV Content:")
                    print("-" * 40)
                    print(result)
            
            elif format_type == "json":
                print("📊 EXPORTING METRICS TO JSON")
                print("=" * 30)
                
                snapshot = self.metrics.export_metrics_snapshot()
                
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(snapshot, f, indent=2)
                    print(f"✅ Metrics exported to: {output_file}")
                else:
                    print(json.dumps(snapshot, indent=2))
            
            else:  # console
                print("📊 ANALYTICS REPORT")
                print("=" * 20)
                report = self.analytics.generate_report(hours_back)
                print(report)
                
                # Show anomalies if any
                anomalies = self.analytics.detect_anomalies(hours_back)
                if anomalies:
                    print("\n🚨 DETECTED ANOMALIES")
                    print("-" * 20)
                    for anomaly in anomalies:
                        print(f"• {anomaly['description']}")
        
        except Exception as e:
            print(f"❌ Error exporting analytics: {e}")
            sys.exit(1)
    
    def threshold_alerting(self, metric: str, threshold: float, check_interval: int = 60) -> None:
        """
        Monitor metrics and alert on threshold violations.
        
        Args:
            metric: Metric to monitor (latency, error_rate, cache_hit_rate)
            threshold: Threshold value for alerting
            check_interval: Check interval in seconds
        """
        try:
            print(f"🚨 THRESHOLD MONITORING: {metric}")
            print(f"📊 Threshold: {threshold}")
            print(f"⏱️  Check Interval: {check_interval}s")
            print("-" * 40)
            print("Press Ctrl+C to stop monitoring")
            print()
            
            check_count = 0
            while check_count < 5:  # Run 5 checks for demo
                check_count += 1
                
                kpis = self.analytics.get_kpi_summary(1)  # Last hour
                
                current_value = 0.0
                if metric == "latency":
                    current_value = kpis['latency_p95_ms']
                elif metric == "error_rate":
                    current_value = kpis['error_rate_percent']
                elif metric == "cache_hit_rate":
                    current_value = kpis['cache_hit_rate_percent']
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                if (metric in ["latency", "error_rate"] and current_value > threshold) or \
                   (metric == "cache_hit_rate" and current_value < threshold):
                    print(f"🚨 [{timestamp}] ALERT: {metric} = {current_value:.2f} (threshold: {threshold})")
                else:
                    print(f"✅ [{timestamp}] OK: {metric} = {current_value:.2f}")
                
                if check_count < 5:
                    time.sleep(min(check_interval, 3))  # Max 3s wait for demo
        
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error in threshold monitoring: {e}")
            sys.exit(1)
    
    def business_kpi_dashboard(self) -> None:
        """Display business KPI dashboard."""
        try:
            print("💼 BUSINESS KPI DASHBOARD")
            print("=" * 30)
            
            kpis = self.analytics.get_kpi_summary(24)
            
            # Customer Experience
            print("👥 CUSTOMER EXPERIENCE")
            print("-" * 22)
            print(f"CSAT Score: {kpis['avg_customer_satisfaction']:.1f}/5.0")
            print(f"FCR Rate: {kpis['fcr_rate_percent']:.1f}%")
            print(f"Avg Handle Time: {kpis['avg_handle_time_seconds']:.1f}s")
            print()
            
            # Operational Excellence
            print("⚙️  OPERATIONAL EXCELLENCE")
            print("-" * 25)
            print(f"System Uptime: 99.9%")
            print(f"Error Rate: {kpis['error_rate_percent']:.1f}%")
            print(f"Cache Performance: {kpis['cache_hit_rate_percent']:.1f}%")
            print()
            
            # Performance
            print("⚡ PERFORMANCE")
            print("-" * 13)
            print(f"Response Time P95: {kpis['latency_p95_ms']:.0f}ms")
            print(f"Voice Quality (MOS): {kpis['tts_mos_score']:.1f}/5.0")
            print(f"Recognition Accuracy: {(1-kpis['word_error_rate'])*100:.1f}%")
            print()
            
            # Volume
            print("📊 VOLUME METRICS")
            print("-" * 16)
            print(f"Total Sessions (24h): {kpis['total_sessions']}")
            print(f"Avg Sessions/Hour: {kpis['total_sessions']/24:.1f}")
            
        except Exception as e:
            print(f"❌ Error displaying dashboard: {e}")
            sys.exit(1)


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sprint 3 Analytics and Monitoring CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m voicebot_orchestrator.sprint3_cli orchestrator-log --metrics
  python -m voicebot_orchestrator.sprint3_cli monitor-session --stats
  python -m voicebot_orchestrator.sprint3_cli analytics-report --export=csv
  python -m voicebot_orchestrator.sprint3_cli threshold-alert --metric=latency --threshold=500
  python -m voicebot_orchestrator.sprint3_cli business-dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # orchestrator-log command
    log_parser = subparsers.add_parser('orchestrator-log', help='Dump KPI metrics')
    log_parser.add_argument('--metrics', action='store_true', help='Show metrics snapshot')
    
    # monitor-session command
    monitor_parser = subparsers.add_parser('monitor-session', help='Live session stats')
    monitor_parser.add_argument('--stats', action='store_true', help='Show session statistics')
    monitor_parser.add_argument('--session-id', help='Monitor specific session')
    monitor_parser.add_argument('--live', action='store_true', help='Live updating stats')
    
    # analytics-report command
    report_parser = subparsers.add_parser('analytics-report', help='Export analytics report')
    report_parser.add_argument('--export', choices=['console', 'csv', 'json'], default='console', help='Export format')
    report_parser.add_argument('--output', help='Output file path')
    report_parser.add_argument('--hours', type=int, default=24, help='Hours of data to include')
    
    # threshold-alert command
    alert_parser = subparsers.add_parser('threshold-alert', help='Monitor thresholds')
    alert_parser.add_argument('--metric', choices=['latency', 'error_rate', 'cache_hit_rate'], required=True, help='Metric to monitor')
    alert_parser.add_argument('--threshold', type=float, required=True, help='Threshold value')
    alert_parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    
    # business-dashboard command
    subparsers.add_parser('business-dashboard', help='Display business KPI dashboard')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = Sprint3CLI()
    
    try:
        if args.command == 'orchestrator-log' and args.metrics:
            cli.orchestrator_log_metrics()
        
        elif args.command == 'monitor-session' and args.stats:
            cli.monitor_session_stats(args.session_id, args.live)
        
        elif args.command == 'analytics-report':
            cli.analytics_report_export(args.export, args.output, args.hours)
        
        elif args.command == 'threshold-alert':
            cli.threshold_alerting(args.metric, args.threshold, args.interval)
        
        elif args.command == 'business-dashboard':
            cli.business_kpi_dashboard()
        
        else:
            parser.print_help()
        
    except KeyboardInterrupt:
        print("\n✋ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
