"""
Analytics and reporting module for voicebot orchestrator.
"""
import json
import csv
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from .datetime_utils import DateTimeFormatter


class AnalyticsEngine:
    """Analytics engine for processing and reporting voicebot metrics."""
    
    def __init__(self, data_dir: str = "analytics_data"):
        """
        Initialize analytics engine.
        
        Args:
            data_dir: Directory to store analytics data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self._session_data: List[Dict[str, Any]] = []
        self._load_historical_data()
    
    def _load_historical_data(self) -> None:
        """Load historical session data from disk."""
        try:
            data_file = self.data_dir / "session_data.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    self._session_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
            self._session_data = []
    
    def _save_data(self) -> None:
        """Save session data to disk."""
        try:
            data_file = self.data_dir / "session_data.json"
            with open(data_file, 'w') as f:
                json.dump(self._session_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save data: {e}")
    
    def record_session(self, session_data: Dict[str, Any]) -> None:
        """
        Record session data for analytics.
        
        Args:
            session_data: Session metrics and data
        """
        session_record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_data.get("session_id", "unknown"),
            "duration": session_data.get("duration", 0.0),
            "stt_latency": session_data.get("stt_latency", 0.0),
            "llm_latency": session_data.get("llm_latency", 0.0),
            "tts_latency": session_data.get("tts_latency", 0.0),
            "total_latency": session_data.get("total_latency", 0.0),
            "message_count": session_data.get("message_count", 0),
            "word_count": session_data.get("word_count", 0),
            "error_count": session_data.get("error_count", 0),
            "cache_hits": session_data.get("cache_hits", 0),
            "cache_misses": session_data.get("cache_misses", 0),
            "first_call_resolution": session_data.get("first_call_resolution", False),
            "customer_satisfaction": session_data.get("customer_satisfaction", 0.0),
            "intent": session_data.get("intent", "unknown"),
            "outcome": session_data.get("outcome", "unknown")
        }
        
        self._session_data.append(session_record)
        
        # Keep only last 10,000 sessions to prevent memory issues
        if len(self._session_data) > 10000:
            self._session_data = self._session_data[-10000:]
        
        self._save_data()
    
    def get_kpi_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get KPI summary for specified time period.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            KPI summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter recent sessions
        recent_sessions = [
            s for s in self._session_data 
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_time
        ]
        
        if not recent_sessions:
            return self._empty_kpi_summary()
        
        # Calculate KPIs
        total_sessions = len(recent_sessions)
        total_duration = sum(s["duration"] for s in recent_sessions)
        avg_handle_time = total_duration / total_sessions if total_sessions > 0 else 0
        
        fcr_sessions = [s for s in recent_sessions if s["first_call_resolution"]]
        fcr_rate = len(fcr_sessions) / total_sessions * 100 if total_sessions > 0 else 0
        
        # Latency percentiles
        latencies = [s["total_latency"] for s in recent_sessions]
        latency_p50 = np.percentile(latencies, 50) if latencies else 0
        latency_p95 = np.percentile(latencies, 95) if latencies else 0
        latency_p99 = np.percentile(latencies, 99) if latencies else 0
        
        # Cache performance
        total_cache_hits = sum(s["cache_hits"] for s in recent_sessions)
        total_cache_requests = sum(s["cache_hits"] + s["cache_misses"] for s in recent_sessions)
        cache_hit_rate = total_cache_hits / total_cache_requests * 100 if total_cache_requests > 0 else 0
        
        # Customer satisfaction
        csat_scores = [s["customer_satisfaction"] for s in recent_sessions if s["customer_satisfaction"] > 0]
        avg_csat = np.mean(csat_scores) if csat_scores else 0
        
        # Error rate
        total_errors = sum(s["error_count"] for s in recent_sessions)
        error_rate = total_errors / total_sessions * 100 if total_sessions > 0 else 0
        
        return {
            "time_period_hours": hours_back,
            "total_sessions": total_sessions,
            "avg_handle_time_seconds": round(avg_handle_time, 2),
            "fcr_rate_percent": round(fcr_rate, 1),
            "latency_p50_ms": round(latency_p50 * 1000, 1),
            "latency_p95_ms": round(latency_p95 * 1000, 1),
            "latency_p99_ms": round(latency_p99 * 1000, 1),
            "cache_hit_rate_percent": round(cache_hit_rate, 1),
            "avg_customer_satisfaction": round(avg_csat, 2),
            "error_rate_percent": round(error_rate, 2),
            "word_error_rate": 0.12,  # Mock WER
            "tts_mos_score": 4.5  # Mock MOS
        }
    
    def _empty_kpi_summary(self) -> Dict[str, Any]:
        """Return empty KPI summary when no data available."""
        return {
            "time_period_hours": 0,
            "total_sessions": 0,
            "avg_handle_time_seconds": 0.0,
            "fcr_rate_percent": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "cache_hit_rate_percent": 0.0,
            "avg_customer_satisfaction": 0.0,
            "error_rate_percent": 0.0,
            "word_error_rate": 0.0,
            "tts_mos_score": 0.0
        }
    
    def get_component_performance(self, hours_back: int = 24) -> Dict[str, Dict[str, float]]:
        """
        Get component performance metrics.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Component performance dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_sessions = [
            s for s in self._session_data 
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_time
        ]
        
        if not recent_sessions:
            return {}
        
        components = ["stt", "llm", "tts"]
        performance = {}
        
        for component in components:
            latency_key = f"{component}_latency"
            latencies = [s[latency_key] for s in recent_sessions if s[latency_key] > 0]
            
            if latencies:
                performance[component] = {
                    "avg_latency_ms": round(np.mean(latencies) * 1000, 2),
                    "min_latency_ms": round(np.min(latencies) * 1000, 2),
                    "max_latency_ms": round(np.max(latencies) * 1000, 2),
                    "p95_latency_ms": round(np.percentile(latencies, 95) * 1000, 2),
                    "p99_latency_ms": round(np.percentile(latencies, 99) * 1000, 2),
                    "request_count": len(latencies)
                }
            else:
                performance[component] = {
                    "avg_latency_ms": 0.0,
                    "min_latency_ms": 0.0,
                    "max_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "request_count": 0
                }
        
        return performance
    
    def export_to_csv(self, output_file: Optional[str] = None, hours_back: int = 24) -> str:
        """
        Export session data to CSV with standardized filename format.
        
        Args:
            output_file: Output file path (optional, uses standardized format if not provided)
            hours_back: Number of hours to look back
            
        Returns:
            CSV file path or CSV content as string
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_sessions = [
            s for s in self._session_data 
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_time
        ]
        
        if not recent_sessions:
            return ""
        
        # Convert to DataFrame for easier export
        df = pd.DataFrame(recent_sessions)
        
        if output_file:
            df.to_csv(output_file, index=False)
            return output_file
        else:
            # Generate standardized filename
            timestamp = time.time()
            output_file = DateTimeFormatter.get_analytics_export_filename(timestamp)
            df.to_csv(output_file, index=False)
            return output_file
    
    def generate_report(self, hours_back: int = 24) -> str:
        """
        Generate formatted analytics report.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Formatted report string
        """
        kpis = self.get_kpi_summary(hours_back)
        component_perf = self.get_component_performance(hours_back)
        
        report = f"""
📊 VOICEBOT ANALYTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Time Period: Last {hours_back} hours

🔑 KEY PERFORMANCE INDICATORS
────────────────────────────────
Total Sessions: {kpis['total_sessions']}
Average Handle Time: {kpis['avg_handle_time_seconds']}s
First Call Resolution: {kpis['fcr_rate_percent']}%
Customer Satisfaction: {kpis['avg_customer_satisfaction']}/5.0
Error Rate: {kpis['error_rate_percent']}%

⚡ PERFORMANCE METRICS
────────────────────────────────
Latency P50: {kpis['latency_p50_ms']}ms
Latency P95: {kpis['latency_p95_ms']}ms
Latency P99: {kpis['latency_p99_ms']}ms
Cache Hit Rate: {kpis['cache_hit_rate_percent']}%

🎯 QUALITY METRICS
────────────────────────────────
Word Error Rate: {kpis['word_error_rate']}
TTS MOS Score: {kpis['tts_mos_score']}/5.0

🔧 COMPONENT PERFORMANCE
────────────────────────────────"""
        
        for component, perf in component_perf.items():
            report += f"""
{component.upper()}:
  Avg Latency: {perf['avg_latency_ms']}ms
  P95 Latency: {perf['p95_latency_ms']}ms
  Request Count: {perf['request_count']}"""
        
        return report
    
    def create_performance_chart(self, output_file: Optional[str] = None, hours_back: int = 24) -> str:
        """
        Create performance visualization chart.
        
        Args:
            output_file: Output file path for chart
            hours_back: Number of hours to look back
            
        Returns:
            Chart file path
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_sessions = [
            s for s in self._session_data 
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_time
        ]
        
        if not recent_sessions:
            return ""
        
        # Create performance chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Voicebot Performance - Last {hours_back} Hours', fontsize=14)
        
        # Component latencies
        components = ["stt", "llm", "tts"]
        latencies = []
        for comp in components:
            latency_key = f"{comp}_latency"
            comp_latencies = [s[latency_key] * 1000 for s in recent_sessions if s[latency_key] > 0]
            latencies.append(comp_latencies)
        
        ax1.boxplot(latencies, labels=[c.upper() for c in components])
        ax1.set_title('Component Latency Distribution')
        ax1.set_ylabel('Latency (ms)')
        
        # Session duration over time
        timestamps = [datetime.fromisoformat(s["timestamp"]) for s in recent_sessions]
        durations = [s["duration"] for s in recent_sessions]
        ax2.plot(timestamps, durations, 'b-', alpha=0.7)
        ax2.set_title('Session Duration Over Time')
        ax2.set_ylabel('Duration (s)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Cache hit rate
        cache_data = [(s["cache_hits"], s["cache_misses"]) for s in recent_sessions]
        if cache_data:
            total_hits = sum(hits for hits, _ in cache_data)
            total_misses = sum(misses for _, misses in cache_data)
            ax3.pie([total_hits, total_misses], labels=['Hits', 'Misses'], autopct='%1.1f%%')
            ax3.set_title('Cache Performance')
        
        # Customer satisfaction distribution
        csat_scores = [s["customer_satisfaction"] for s in recent_sessions if s["customer_satisfaction"] > 0]
        if csat_scores:
            ax4.hist(csat_scores, bins=10, alpha=0.7, color='green')
            ax4.set_title('Customer Satisfaction Distribution')
            ax4.set_xlabel('CSAT Score')
            ax4.set_ylabel('Count')
        
        plt.tight_layout()
        
        if not output_file:
            timestamp = time.time()
            output_file = self.data_dir / DateTimeFormatter.get_performance_chart_filename(timestamp)
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def detect_anomalies(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            List of detected anomalies
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_sessions = [
            s for s in self._session_data 
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_time
        ]
        
        if len(recent_sessions) < 10:
            return []
        
        anomalies = []
        
        # Check for high latency
        latencies = [s["total_latency"] for s in recent_sessions]
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        threshold = mean_latency + 3 * std_latency
        
        high_latency_sessions = [s for s in recent_sessions if s["total_latency"] > threshold]
        if high_latency_sessions:
            anomalies.append({
                "type": "high_latency",
                "description": f"Found {len(high_latency_sessions)} sessions with unusually high latency",
                "threshold": round(threshold * 1000, 2),
                "affected_sessions": len(high_latency_sessions)
            })
        
        # Check for high error rate
        error_rates = []
        for i in range(0, len(recent_sessions), 10):  # Check in batches of 10
            batch = recent_sessions[i:i+10]
            total_errors = sum(s["error_count"] for s in batch)
            error_rate = total_errors / len(batch) * 100
            error_rates.append(error_rate)
        
        if error_rates and max(error_rates) > 10:  # 10% error rate threshold
            anomalies.append({
                "type": "high_error_rate",
                "description": f"Detected error rate spike: {max(error_rates):.1f}%",
                "threshold": 10.0,
                "max_error_rate": max(error_rates)
            })
        
        # Check for low cache hit rate
        cache_hits = sum(s["cache_hits"] for s in recent_sessions)
        cache_total = sum(s["cache_hits"] + s["cache_misses"] for s in recent_sessions)
        if cache_total > 0:
            hit_rate = cache_hits / cache_total * 100
            if hit_rate < 50:  # 50% hit rate threshold
                anomalies.append({
                    "type": "low_cache_performance",
                    "description": f"Cache hit rate below threshold: {hit_rate:.1f}%",
                    "threshold": 50.0,
                    "current_rate": hit_rate
                })
        
        return anomalies


# Global analytics engine instance
analytics_engine = AnalyticsEngine()
