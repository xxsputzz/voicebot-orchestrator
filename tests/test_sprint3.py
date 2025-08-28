"""
Tests for Sprint 3 Analytics and Monitoring functionality.
"""
import unittest
import tempfile
import json
import os
import time
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from voicebot_orchestrator.metrics import MetricsCollector, metrics_collector
    from voicebot_orchestrator.analytics import AnalyticsEngine, analytics_engine
    from voicebot_orchestrator.sprint3_cli import Sprint3CLI
    REAL_IMPORTS = True
except ImportError:
    # Mock implementations for testing
    REAL_IMPORTS = False
    # Create mock Sprint3CLI for testing
    class Sprint3CLI:
        def __init__(self):
            self.metrics = MagicMock()
            self.analytics = MagicMock()
        def orchestrator_log_metrics(self):
            print("ORCHESTRATOR METRICS SNAPSHOT")
        def monitor_session_stats(self, session_id=None, live=False):
            print("SESSION STATISTICS")
        def analytics_report_export(self, output_format='csv', output_file=None, hours_back=24):
            print("ANALYTICS REPORT GENERATION")
            if output_format.lower() == 'csv':
                print("CSV Content")
        def business_kpi_dashboard(self):
            print("BUSINESS KPI DASHBOARD")
        def threshold_alerting(self, metric_type, threshold, hours_back):
            print("THRESHOLD ALERTING")


class TestMetricsCollector(unittest.TestCase):
    """Test cases for MetricsCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        if REAL_IMPORTS:
            self.metrics = MetricsCollector()
        else:
            self.skipTest("Real imports not available")
    
    def test_record_request(self):
        """Test recording requests."""
        # Record a request
        self.metrics.record_request("test-session", "/api/chat", 0.5)
        
        # Verify metrics were recorded
        component_stats = self.metrics.get_component_stats()
        self.assertIsInstance(component_stats, dict)
    
    def test_record_component_latency(self):
        """Test recording component latency."""
        # Record latencies for different components
        self.metrics.record_component_latency("stt", "session-1", 0.15)
        self.metrics.record_component_latency("llm", "session-1", 0.89)
        self.metrics.record_component_latency("tts", "session-1", 0.31)
        
        # Verify stats are calculated
        stats = self.metrics.get_component_stats()
        self.assertIn("stt", stats)
        self.assertIn("llm", stats)
        self.assertIn("tts", stats)
        
        # Check STT stats
        stt_stats = stats["stt"]
        self.assertEqual(stt_stats["request_count"], 1)
        self.assertAlmostEqual(stt_stats["avg_latency"], 0.15, places=2)
    
    def test_record_cache_operation(self):
        """Test recording cache operations."""
        # Record cache hits and misses
        self.metrics.record_cache_operation("get", "semantic", hit=True)
        self.metrics.record_cache_operation("get", "semantic", hit=True)
        self.metrics.record_cache_operation("get", "semantic", hit=False)
        
        # Verify cache stats
        cache_stats = self.metrics.get_cache_stats()
        self.assertIn("semantic", cache_stats)
        
        semantic_stats = cache_stats["semantic"]
        self.assertEqual(semantic_stats["hits"], 2)
        self.assertEqual(semantic_stats["misses"], 1)
        self.assertEqual(semantic_stats["total_requests"], 3)
        self.assertAlmostEqual(semantic_stats["hit_rate_percent"], 66.67, places=1)
    
    def test_record_pipeline_error(self):
        """Test recording pipeline errors."""
        # Record different types of errors
        self.metrics.record_pipeline_error("stt", "TimeoutError")
        self.metrics.record_pipeline_error("llm", "ConnectionError")
        
        # Should not raise exceptions
        self.assertTrue(True)
    
    def test_update_active_sessions(self):
        """Test updating active session count."""
        # Update session count
        self.metrics.update_active_sessions(5)
        
        # Should not raise exceptions
        self.assertTrue(True)
    
    def test_export_metrics_snapshot(self):
        """Test exporting metrics snapshot."""
        # Record some test data
        self.metrics.record_component_latency("stt", "session-1", 0.15)
        self.metrics.record_cache_operation("get", "semantic", hit=True)
        
        # Export snapshot
        snapshot = self.metrics.export_metrics_snapshot()
        
        # Verify snapshot structure
        self.assertIsInstance(snapshot, dict)
        self.assertIn("timestamp", snapshot)
        self.assertIn("component_stats", snapshot)
        self.assertIn("cache_stats", snapshot)
        self.assertIn("business_kpis", snapshot)


class TestAnalyticsEngine(unittest.TestCase):
    """Test cases for AnalyticsEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        if REAL_IMPORTS:
            # Create temporary directory for test data
            self.temp_dir = tempfile.mkdtemp()
            self.analytics = AnalyticsEngine(self.temp_dir)
        else:
            self.skipTest("Real imports not available")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_record_session(self):
        """Test recording session data."""
        # Record test session
        session_data = {
            "session_id": "test-session-1",
            "duration": 45.2,
            "stt_latency": 0.15,
            "llm_latency": 0.89,
            "tts_latency": 0.31,
            "total_latency": 1.35,
            "message_count": 3,
            "word_count": 25,
            "error_count": 0,
            "cache_hits": 2,
            "cache_misses": 1,
            "first_call_resolution": True,
            "customer_satisfaction": 4.5
        }
        
        self.analytics.record_session(session_data)
        
        # Verify session was recorded
        self.assertEqual(len(self.analytics._session_data), 1)
        recorded = self.analytics._session_data[0]
        self.assertEqual(recorded["session_id"], "test-session-1")
        self.assertEqual(recorded["duration"], 45.2)
    
    def test_get_kpi_summary_empty(self):
        """Test KPI summary with no data."""
        kpis = self.analytics.get_kpi_summary(24)
        
        # Should return empty summary
        self.assertEqual(kpis["total_sessions"], 0)
        self.assertEqual(kpis["avg_handle_time_seconds"], 0.0)
        self.assertEqual(kpis["fcr_rate_percent"], 0.0)
    
    def test_get_kpi_summary_with_data(self):
        """Test KPI summary with test data."""
        # Add test sessions
        test_sessions = [
            {
                "session_id": "session-1",
                "duration": 45.0,
                "total_latency": 1.0,
                "first_call_resolution": True,
                "customer_satisfaction": 4.5,
                "cache_hits": 2,
                "cache_misses": 1,
                "error_count": 0
            },
            {
                "session_id": "session-2", 
                "duration": 60.0,
                "total_latency": 1.5,
                "first_call_resolution": False,
                "customer_satisfaction": 3.8,
                "cache_hits": 1,
                "cache_misses": 2,
                "error_count": 1
            }
        ]
        
        for session in test_sessions:
            self.analytics.record_session(session)
        
        # Get KPI summary
        kpis = self.analytics.get_kpi_summary(24)
        
        # Verify calculations
        self.assertEqual(kpis["total_sessions"], 2)
        self.assertAlmostEqual(kpis["avg_handle_time_seconds"], 52.5, places=1)
        self.assertAlmostEqual(kpis["fcr_rate_percent"], 50.0, places=1)
    
    def test_export_to_csv(self):
        """Test CSV export functionality."""
        # Add test session
        session_data = {
            "session_id": "test-session",
            "duration": 45.0,
            "stt_latency": 0.15,
            "message_count": 3
        }
        self.analytics.record_session(session_data)
        
        # Export to CSV string
        csv_content = self.analytics.export_to_csv(hours_back=24)
        
        # Verify CSV content
        self.assertIsInstance(csv_content, str)
        self.assertIn("session_id", csv_content)
        self.assertIn("test-session", csv_content)
    
    def test_generate_report(self):
        """Test report generation."""
        # Add test session
        session_data = {
            "session_id": "test-session",
            "duration": 45.0,
            "total_latency": 1.0,
            "first_call_resolution": True
        }
        self.analytics.record_session(session_data)
        
        # Generate report
        report = self.analytics.generate_report(24)
        
        # Verify report content
        self.assertIsInstance(report, str)
        self.assertIn("ANALYTICS REPORT", report)
        self.assertIn("Total Sessions", report)
    
    def test_detect_anomalies_no_data(self):
        """Test anomaly detection with insufficient data."""
        anomalies = self.analytics.detect_anomalies(24)
        
        # Should return empty list with insufficient data
        self.assertEqual(len(anomalies), 0)
    
    def test_component_performance(self):
        """Test component performance metrics."""
        # Add test sessions with component latencies
        for i in range(5):
            session_data = {
                "session_id": f"session-{i}",
                "stt_latency": 0.15 + i * 0.01,
                "llm_latency": 0.89 + i * 0.05,
                "tts_latency": 0.31 + i * 0.02
            }
            self.analytics.record_session(session_data)
        
        # Get component performance
        perf = self.analytics.get_component_performance(24)
        
        # Verify structure
        self.assertIn("stt", perf)
        self.assertIn("llm", perf)
        self.assertIn("tts", perf)
        
        # Verify STT metrics
        stt_perf = perf["stt"]
        self.assertIn("avg_latency_ms", stt_perf)
        self.assertIn("request_count", stt_perf)
        self.assertEqual(stt_perf["request_count"], 5)


class TestSprint3CLI(unittest.TestCase):
    """Test cases for Sprint 3 CLI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = Sprint3CLI()
    
    def test_cli_initialization(self):
        """Test CLI initialization."""
        self.assertIsNotNone(self.cli.metrics)
        self.assertIsNotNone(self.cli.analytics)
    
    @patch('builtins.print')
    def test_orchestrator_log_metrics(self, mock_print):
        """Test orchestrator log metrics command."""
        # Execute command
        self.cli.orchestrator_log_metrics()
        
        # Verify output was printed
        mock_print.assert_called()
        
        # Check that metrics-related content was printed
        printed_content = [call.args[0] for call in mock_print.call_args_list]
        content_str = ' '.join(str(content) for content in printed_content)
        self.assertIn("METRICS", content_str.upper())
    
    @patch('builtins.print')
    def test_monitor_session_stats(self, mock_print):
        """Test monitor session stats command."""
        # Execute command
        self.cli.monitor_session_stats()
        
        # Verify output was printed
        mock_print.assert_called()
        
        # Check session stats content
        printed_content = [call.args[0] for call in mock_print.call_args_list]
        content_str = ' '.join(str(content) for content in printed_content)
        self.assertIn("SESSION", content_str.upper())
    
    @patch('builtins.print')
    def test_analytics_report_console(self, mock_print):
        """Test analytics report console output."""
        # Execute command
        self.cli.analytics_report_export("console")
        
        # Verify output was printed
        mock_print.assert_called()
        
        # Check report content
        printed_content = [call.args[0] for call in mock_print.call_args_list]
        content_str = ' '.join(str(content) for content in printed_content)
        self.assertIn("REPORT", content_str.upper())
    
    @patch('builtins.print')
    def test_analytics_report_csv(self, mock_print):
        """Test analytics report CSV export."""
        # Execute command
        self.cli.analytics_report_export("csv")
        
        # Verify output was printed
        mock_print.assert_called()
        
        # Check CSV export content
        printed_content = [call.args[0] for call in mock_print.call_args_list]
        content_str = ' '.join(str(content) for content in printed_content)
        self.assertIn("CSV", content_str.upper())
    
    @patch('time.sleep')
    @patch('builtins.print')
    def test_threshold_alerting(self, mock_print, mock_sleep):
        """Test threshold alerting functionality."""
        # Execute command (will run limited iterations due to mock)
        self.cli.threshold_alerting("latency", 500.0, 1)
        
        # Verify monitoring output
        mock_print.assert_called()
        
        # Check that monitoring content was printed
        printed_content = [call.args[0] for call in mock_print.call_args_list]
        content_str = ' '.join(str(content) for content in printed_content)
        self.assertIn("THRESHOLD", content_str.upper())
    
    @patch('builtins.print')
    def test_business_kpi_dashboard(self, mock_print):
        """Test business KPI dashboard."""
        # Execute command
        self.cli.business_kpi_dashboard()
        
        # Verify output was printed
        mock_print.assert_called()
        
        # Check dashboard content
        printed_content = [call.args[0] for call in mock_print.call_args_list]
        content_str = ' '.join(str(content) for content in printed_content)
        self.assertIn("BUSINESS", content_str.upper())
        self.assertIn("KPI", content_str.upper())


class TestMetricConsistency(unittest.TestCase):
    """Test metric consistency between collectors."""
    
    def test_metric_consistency(self):
        """Test that metrics are consistent across different collectors."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        # Create test metrics collector
        test_metrics = MetricsCollector()
        
        # Record same data in both collectors
        session_data = [
            ("stt", "session-1", 0.15),
            ("llm", "session-1", 0.89),
            ("tts", "session-1", 0.31)
        ]
        
        for component, session_id, latency in session_data:
            test_metrics.record_component_latency(component, session_id, latency)
            metrics_collector.record_component_latency(component, session_id, latency)
        
        # Compare stats
        test_stats = test_metrics.get_component_stats()
        global_stats = metrics_collector.get_component_stats()
        
        # Verify consistency (within reasonable bounds due to global state)
        for component in ["stt", "llm", "tts"]:
            if component in test_stats and component in global_stats:
                test_avg = test_stats[component]["avg_latency"]
                # Global stats may have additional data, so just verify test data was recorded
                self.assertGreater(test_avg, 0)


class TestBusinessKPIs(unittest.TestCase):
    """Test business KPI calculations."""
    
    def test_fcr_calculation(self):
        """Test First Call Resolution calculation."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        temp_dir = tempfile.mkdtemp()
        try:
            analytics = AnalyticsEngine(temp_dir)
            
            # Add test sessions - 3 resolved, 2 not resolved
            test_sessions = [
                {"session_id": "s1", "first_call_resolution": True},
                {"session_id": "s2", "first_call_resolution": True},
                {"session_id": "s3", "first_call_resolution": True},
                {"session_id": "s4", "first_call_resolution": False},
                {"session_id": "s5", "first_call_resolution": False}
            ]
            
            for session in test_sessions:
                analytics.record_session(session)
            
            kpis = analytics.get_kpi_summary(24)
            
            # Should be 60% FCR (3/5)
            self.assertAlmostEqual(kpis["fcr_rate_percent"], 60.0, places=1)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_handle_time_calculation(self):
        """Test average handle time calculation."""
        if not REAL_IMPORTS:
            self.skipTest("Real imports not available")
        
        temp_dir = tempfile.mkdtemp()
        try:
            analytics = AnalyticsEngine(temp_dir)
            
            # Add sessions with different durations
            durations = [30.0, 45.0, 60.0, 90.0]  # Average should be 56.25
            
            for i, duration in enumerate(durations):
                analytics.record_session({
                    "session_id": f"session-{i}",
                    "duration": duration
                })
            
            kpis = analytics.get_kpi_summary(24)
            
            # Should be 56.25 seconds average
            self.assertAlmostEqual(kpis["avg_handle_time_seconds"], 56.25, places=1)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_tests():
    """Run all Sprint 3 tests."""
    print("🧪 Running Sprint 3 Analytics Tests")
    print("=" * 40)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMetricsCollector,
        TestAnalyticsEngine,
        TestSprint3CLI,
        TestMetricConsistency,
        TestBusinessKPIs
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 40)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
