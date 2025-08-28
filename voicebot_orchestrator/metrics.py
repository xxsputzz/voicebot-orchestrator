"""
Metrics and monitoring module using Prometheus and OpenTelemetry.
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json

from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode


# Constants
_METRICS_PORT: int = 8000
_DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]

# Prometheus metrics registry
metrics_registry = CollectorRegistry()

# Define Prometheus metrics
REQUEST_COUNT: Counter = Counter(
    "orchestrator_requests_total", 
    "Total number of orchestration requests",
    ["session_id", "endpoint"],
    registry=metrics_registry
)

REQUEST_LATENCY: Histogram = Histogram(
    "orchestrator_request_latency_seconds", 
    "Latency of orchestration requests",
    ["component", "session_id"], 
    buckets=_DEFAULT_BUCKETS,
    registry=metrics_registry
)

ACTIVE_SESSIONS: Gauge = Gauge(
    "orchestrator_active_sessions",
    "Number of active sessions",
    registry=metrics_registry
)

PIPELINE_ERRORS: Counter = Counter(
    "orchestrator_pipeline_errors_total",
    "Total pipeline errors",
    ["component", "error_type"],
    registry=metrics_registry
)

CACHE_OPERATIONS: Counter = Counter(
    "orchestrator_cache_operations_total",
    "Cache operations",
    ["operation", "cache_type"],
    registry=metrics_registry
)

CACHE_HIT_RATE: Gauge = Gauge(
    "orchestrator_cache_hit_rate",
    "Cache hit rate percentage",
    ["cache_type"],
    registry=metrics_registry
)

# Business KPIs
CUSTOMER_SATISFACTION: Gauge = Gauge(
    "orchestrator_customer_satisfaction_score",
    "Customer satisfaction score (CSAT)",
    ["session_id"],
    registry=metrics_registry
)

FIRST_CALL_RESOLUTION: Counter = Counter(
    "orchestrator_first_call_resolution_total",
    "First call resolution events",
    ["resolved"],
    registry=metrics_registry
)

AVERAGE_HANDLE_TIME: Histogram = Histogram(
    "orchestrator_average_handle_time_seconds",
    "Average handle time per session",
    buckets=_DEFAULT_BUCKETS,
    registry=metrics_registry
)


class MetricsCollector:
    """Centralized metrics collection and reporting."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self._session_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._component_timings: Dict[str, List[float]] = defaultdict(list)
        self._cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})
        self._start_time = time.time()
        self._lock = threading.Lock()
        
        # Initialize OpenTelemetry
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(__name__)
    
    def start_metrics_server(self, port: int = _METRICS_PORT) -> None:
        """
        Start the Prometheus HTTP metrics endpoint.
        
        Args:
            port: Port to serve metrics on
        """
        try:
            start_http_server(port, registry=metrics_registry)
            print(f"📊 Metrics server started on port {port}")
        except Exception as e:
            print(f"❌ Failed to start metrics server: {e}")
    
    def record_request(self, session_id: str, endpoint: str, latency: float) -> None:
        """
        Record a request with latency.
        
        Args:
            session_id: Session identifier
            endpoint: API endpoint
            latency: Request latency in seconds
        """
        REQUEST_COUNT.labels(session_id=session_id, endpoint=endpoint).inc()
        REQUEST_LATENCY.labels(component="api", session_id=session_id).observe(latency)
    
    def record_component_latency(self, component: str, session_id: str, latency: float) -> None:
        """
        Record component processing latency.
        
        Args:
            component: Component name (stt, llm, tts)
            session_id: Session identifier
            latency: Processing latency in seconds
        """
        REQUEST_LATENCY.labels(component=component, session_id=session_id).observe(latency)
        
        with self._lock:
            self._component_timings[component].append(latency)
            # Keep only last 1000 measurements
            if len(self._component_timings[component]) > 1000:
                self._component_timings[component] = self._component_timings[component][-1000:]
    
    def record_pipeline_error(self, component: str, error_type: str) -> None:
        """
        Record a pipeline error.
        
        Args:
            component: Component where error occurred
            error_type: Type of error
        """
        PIPELINE_ERRORS.labels(component=component, error_type=error_type).inc()
    
    def record_cache_operation(self, operation: str, cache_type: str, hit: bool = False) -> None:
        """
        Record cache operation.
        
        Args:
            operation: Cache operation (get, set, delete)
            cache_type: Type of cache (semantic, session)
            hit: Whether operation was a cache hit
        """
        CACHE_OPERATIONS.labels(operation=operation, cache_type=cache_type).inc()
        
        with self._lock:
            if operation == "get":
                if hit:
                    self._cache_stats[cache_type]["hits"] += 1
                else:
                    self._cache_stats[cache_type]["misses"] += 1
                
                # Update hit rate
                total = self._cache_stats[cache_type]["hits"] + self._cache_stats[cache_type]["misses"]
                if total > 0:
                    hit_rate = (self._cache_stats[cache_type]["hits"] / total) * 100
                    CACHE_HIT_RATE.labels(cache_type=cache_type).set(hit_rate)
    
    def update_active_sessions(self, count: int) -> None:
        """
        Update active session count.
        
        Args:
            count: Current number of active sessions
        """
        ACTIVE_SESSIONS.set(count)
    
    def record_customer_satisfaction(self, session_id: str, score: float) -> None:
        """
        Record customer satisfaction score.
        
        Args:
            session_id: Session identifier
            score: CSAT score (1-5)
        """
        CUSTOMER_SATISFACTION.labels(session_id=session_id).set(score)
    
    def record_first_call_resolution(self, resolved: bool) -> None:
        """
        Record first call resolution.
        
        Args:
            resolved: Whether call was resolved on first attempt
        """
        FIRST_CALL_RESOLUTION.labels(resolved=str(resolved).lower()).inc()
    
    def record_handle_time(self, duration: float) -> None:
        """
        Record session handle time.
        
        Args:
            duration: Session duration in seconds
        """
        AVERAGE_HANDLE_TIME.observe(duration)
    
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session metrics dictionary
        """
        with self._lock:
            return self._session_metrics.get(session_id, {})
    
    def update_session_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update metrics for a specific session.
        
        Args:
            session_id: Session identifier
            metrics: Metrics to update
        """
        with self._lock:
            self._session_metrics[session_id].update(metrics)
    
    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get component performance statistics.
        
        Returns:
            Component statistics dictionary
        """
        with self._lock:
            stats = {}
            for component, timings in self._component_timings.items():
                if timings:
                    stats[component] = {
                        "avg_latency": sum(timings) / len(timings),
                        "min_latency": min(timings),
                        "max_latency": max(timings),
                        "p95_latency": sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings),
                        "p99_latency": sorted(timings)[int(len(timings) * 0.99)] if len(timings) > 100 else max(timings),
                        "request_count": len(timings)
                    }
                else:
                    stats[component] = {
                        "avg_latency": 0.0,
                        "min_latency": 0.0,
                        "max_latency": 0.0,
                        "p95_latency": 0.0,
                        "p99_latency": 0.0,
                        "request_count": 0
                    }
            return stats
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        with self._lock:
            stats = {}
            for cache_type, cache_data in self._cache_stats.items():
                total = cache_data["hits"] + cache_data["misses"]
                hit_rate = (cache_data["hits"] / total * 100) if total > 0 else 0
                stats[cache_type] = {
                    "hits": cache_data["hits"],
                    "misses": cache_data["misses"],
                    "total_requests": total,
                    "hit_rate_percent": hit_rate
                }
            return stats
    
    def get_business_kpis(self) -> Dict[str, Any]:
        """
        Get business KPI summary.
        
        Returns:
            Business KPIs dictionary
        """
        # Calculate FCR rate
        fcr_resolved = 0
        fcr_total = 0
        try:
            for sample in FIRST_CALL_RESOLUTION.collect():
                for metric_family in sample.samples:
                    if metric_family.labels.get("resolved") == "true":
                        fcr_resolved += metric_family.value
                    fcr_total += metric_family.value
        except:
            pass
        
        fcr_rate = (fcr_resolved / fcr_total * 100) if fcr_total > 0 else 0
        
        # Calculate average handle time
        avg_handle_time = 0.0
        try:
            for sample in AVERAGE_HANDLE_TIME.collect():
                for metric_family in sample.samples:
                    if metric_family.name.endswith("_sum") and metric_family.value > 0:
                        count_sample = next((s for s in sample.samples if s.name.endswith("_count")), None)
                        if count_sample and count_sample.value > 0:
                            avg_handle_time = metric_family.value / count_sample.value
                        break
        except:
            pass
        
        return {
            "fcr_rate_percent": fcr_rate,
            "avg_handle_time_seconds": avg_handle_time,
            "uptime_seconds": time.time() - self._start_time,
            "total_sessions": fcr_total
        }
    
    def create_span(self, operation_name: str) -> Any:
        """
        Create OpenTelemetry span for tracing.
        
        Args:
            operation_name: Name of the operation being traced
            
        Returns:
            Trace span
        """
        return self.tracer.start_span(operation_name)
    
    def export_metrics_snapshot(self) -> Dict[str, Any]:
        """
        Export complete metrics snapshot.
        
        Returns:
            Complete metrics snapshot
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "component_stats": self.get_component_stats(),
            "cache_stats": self.get_cache_stats(),
            "business_kpis": self.get_business_kpis(),
            "session_metrics": dict(self._session_metrics)
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Decorator for automatic latency measurement
def measure_latency(component: str):
    """
    Decorator to automatically measure function latency.
    
    Args:
        component: Component name for metrics
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                session_id = kwargs.get('session_id', 'unknown')
                metrics_collector.record_component_latency(component, session_id, latency)
                return result
            except Exception as e:
                metrics_collector.record_pipeline_error(component, type(e).__name__)
                raise
        return wrapper
    return decorator


# Async version of the decorator
def measure_async_latency(component: str):
    """
    Decorator to automatically measure async function latency.
    
    Args:
        component: Component name for metrics
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                session_id = kwargs.get('session_id', 'unknown')
                metrics_collector.record_component_latency(component, session_id, latency)
                return result
            except Exception as e:
                metrics_collector.record_pipeline_error(component, type(e).__name__)
                raise
        return wrapper
    return decorator
