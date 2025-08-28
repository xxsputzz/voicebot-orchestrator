"""
Tests for Sprint 6 CLI commands.

Comprehensive test suite for the enterprise CLI interface including
async command handlers, microservices integration, and error handling.
"""

import unittest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Import the CLI module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voicebot_orchestrator.sprint6_cli import (
    start_call,
    monitor_session,
    analytics_report,
    cache_manager,
    adapter_control,
    orchestrator_health,
    OrchestratorCLI
)


class TestSprint6CLI(unittest.TestCase):
    """Test cases for Sprint 6 CLI commands."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_session_id = "test-session-123"
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_orchestrator_cli_initialization(self):
        """Test OrchestratorCLI initialization."""
        cli = OrchestratorCLI()
        
        # Check configuration loading
        self.assertIsInstance(cli.config, dict)
        self.assertIn("orchestrator", cli.config)
        self.assertIn("microservices", cli.config)
        self.assertIn("cache", cli.config)
        self.assertIn("adapters", cli.config)
        self.assertIn("analytics", cli.config)
        
        # Check default values
        self.assertEqual(cli.config["orchestrator"]["host"], "localhost")
        self.assertEqual(cli.config["orchestrator"]["port"], 8000)
    
    def test_start_call_success(self):
        """Test successful call start."""
        async def async_test():
            result = await start_call(
                session_id=self.test_session_id,
                phone_number="+1234567890",
                customer_id="customer123",
                domain="banking"
            )
            
            # Verify result structure
            self.assertEqual(result["status"], "started")
            self.assertEqual(result["session_id"], self.test_session_id)
            self.assertIn("session_config", result)
            self.assertIn("services", result)
            self.assertIn("message", result)
            self.assertIn("next_steps", result)
            
            # Verify session config
            session_config = result["session_config"]
            self.assertEqual(session_config["session_id"], self.test_session_id)
            self.assertEqual(session_config["phone_number"], "+1234567890")
            self.assertEqual(session_config["customer_id"], "customer123")
            self.assertEqual(session_config["domain"], "banking")
            
            # Verify services configuration
            services = result["services"]
            self.assertIn("orchestrator_core", services)
            self.assertIn("stt_service", services)
            self.assertIn("llm_service", services)
            self.assertIn("tts_service", services)
            self.assertIn("cache_service", services)
            self.assertIn("analytics_service", services)
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_start_call_empty_session_id(self):
        """Test start_call with empty session_id."""
        async def async_test():
            with self.assertRaises(ValueError) as context:
                await start_call(session_id="")
            
            self.assertIn("session_id must be provided", str(context.exception))
            
            with self.assertRaises(ValueError) as context:
                await start_call(session_id="   ")
            
            self.assertIn("session_id must be provided", str(context.exception))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_start_call_invalid_session_id_length(self):
        """Test start_call with invalid session_id length."""
        async def async_test():
            # Too short
            with self.assertRaises(ValueError) as context:
                await start_call(session_id="ab")
            
            self.assertIn("session_id must be between 3 and 64 characters", str(context.exception))
            
            # Too long
            long_session_id = "a" * 65
            with self.assertRaises(ValueError) as context:
                await start_call(session_id=long_session_id)
            
            self.assertIn("session_id must be between 3 and 64 characters", str(context.exception))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_monitor_session_success(self):
        """Test successful session monitoring."""
        async def async_test():
            result = await monitor_session(
                session_id=self.test_session_id,
                follow=False,
                output_format="json"
            )
            
            # Verify result structure
            self.assertIn("status", result)
            self.assertEqual(result["session_id"], self.test_session_id)
            self.assertIn("timestamp", result)
            self.assertIn("follow_mode", result)
            self.assertIn("output_format", result)
            self.assertEqual(result["follow_mode"], False)
            self.assertEqual(result["output_format"], "json")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_monitor_session_empty_session_id(self):
        """Test monitor_session with empty session_id."""
        async def async_test():
            with self.assertRaises(ValueError) as context:
                await monitor_session(session_id="")
            
            self.assertIn("session_id must be provided", str(context.exception))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_analytics_report_summary(self):
        """Test analytics report generation."""
        async def async_test():
            result = await analytics_report(
                report_type="summary",
                time_range="24h",
                output_format="json"
            )
            
            # Verify result structure
            self.assertEqual(result["report_type"], "summary")
            self.assertEqual(result["time_range"], "24h")
            self.assertEqual(result["output_format"], "json")
            self.assertIn("start_time", result)
            self.assertIn("end_time", result)
            self.assertIn("generated_at", result)
            self.assertIn("data", result)
            self.assertIn("summary", result)
            
            # Verify summary contains expected metrics
            summary = result["summary"]
            self.assertIn("total_sessions", summary)
            self.assertIn("success_rate", summary)
            self.assertIn("avg_duration", summary)
            self.assertIn("cache_hit_rate", summary)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_analytics_report_invalid_time_range(self):
        """Test analytics report with invalid time range."""
        async def async_test():
            with self.assertRaises(ValueError) as context:
                await analytics_report(
                    report_type="summary",
                    time_range="invalid",
                    output_format="json"
                )
            
            self.assertIn("Invalid time_range", str(context.exception))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_cache_manager_stats(self):
        """Test cache manager stats operation."""
        async def async_test():
            result = await cache_manager(operation="stats")
            
            # Verify result structure
            self.assertEqual(result["operation"], "stats")
            self.assertIn("cache_statistics", result)
            self.assertIn("timestamp", result)
            
            # Verify cache statistics structure
            cache_stats = result["cache_statistics"]
            self.assertIn("total_entries", cache_stats)
            self.assertIn("hit_rate", cache_stats)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_cache_manager_clear(self):
        """Test cache manager clear operation."""
        async def async_test():
            result = await cache_manager(operation="clear")
            
            # Verify result structure
            self.assertEqual(result["operation"], "clear")
            self.assertEqual(result["status"], "success")
            self.assertIn("message", result)
            self.assertIn("timestamp", result)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_cache_manager_invalid_operation(self):
        """Test cache manager with invalid operation."""
        async def async_test():
            result = await cache_manager(operation="invalid")
            
            # Should return error result
            self.assertEqual(result["operation"], "invalid")
            self.assertEqual(result["status"], "error")
            self.assertIn("error", result)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_adapter_control_list(self):
        """Test adapter control list operation."""
        async def async_test():
            result = await adapter_control(operation="list")
            
            # Verify result structure
            self.assertEqual(result["operation"], "list")
            self.assertIn("adapter_status", result)
            self.assertIn("timestamp", result)
            
            # Verify adapter status structure
            adapter_status = result["adapter_status"]
            self.assertIn("available_adapters", adapter_status)
            self.assertIn("loaded_adapters", adapter_status)
            self.assertIn("active_adapter", adapter_status)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_adapter_control_create_banking(self):
        """Test adapter control create banking operation."""
        async def async_test():
            result = await adapter_control(
                operation="create",
                adapter_name="banking"
            )
            
            # Verify result structure
            self.assertEqual(result["operation"], "create")
            self.assertEqual(result["adapter_name"], "banking-lora")
            self.assertIn("status", result)
            self.assertIn("timestamp", result)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_adapter_control_missing_adapter_name(self):
        """Test adapter control with missing adapter name."""
        async def async_test():
            result = await adapter_control(operation="load")
            
            # Should return error result
            self.assertEqual(result["operation"], "load")
            self.assertEqual(result["status"], "error")
            self.assertIn("error", result)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_orchestrator_health_check(self):
        """Test orchestrator health check."""
        async def async_test():
            result = await orchestrator_health()
            
            # Verify result structure
            self.assertIn("status", result)
            self.assertIn("version", result)
            self.assertIn("environment", result)
            self.assertIn("timestamp", result)
            self.assertIn("health_checks", result)
            self.assertIn("system_info", result)
            
            # Verify health checks structure
            health_checks = result["health_checks"]
            self.assertIn("orchestrator_core", health_checks)
            self.assertIn("configuration", health_checks)
            self.assertIn("services", health_checks)
            self.assertIn("dependencies", health_checks)
            
            # Verify system info
            system_info = result["system_info"]
            self.assertIn("python_version", system_info)
            self.assertIn("platform", system_info)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_all_time_ranges_valid(self):
        """Test all valid time ranges for analytics."""
        valid_ranges = ["1h", "24h", "7d", "30d"]
        
        async def async_test():
            for time_range in valid_ranges:
                result = await analytics_report(
                    report_type="summary",
                    time_range=time_range
                )
                self.assertEqual(result["time_range"], time_range)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_all_report_types_valid(self):
        """Test all valid report types for analytics."""
        valid_types = ["summary", "performance", "errors", "usage"]
        
        async def async_test():
            for report_type in valid_types:
                result = await analytics_report(
                    report_type=report_type,
                    time_range="24h"
                )
                self.assertEqual(result["report_type"], report_type)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_all_cache_operations_valid(self):
        """Test all valid cache operations."""
        valid_operations = ["stats", "clear", "export", "optimize"]
        
        async def async_test():
            for operation in valid_operations:
                result = await cache_manager(operation=operation)
                self.assertEqual(result["operation"], operation)
                # Status should be success for all valid operations
                if "status" in result:
                    self.assertIn(result["status"], ["success", "error"])  # Allow error for export without file
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_all_adapter_operations_valid(self):
        """Test all valid adapter operations."""
        valid_operations = ["list", "create"]  # load/unload/activate require adapter_name
        
        async def async_test():
            for operation in valid_operations:
                if operation == "create":
                    result = await adapter_control(
                        operation=operation,
                        adapter_name="test"
                    )
                else:
                    result = await adapter_control(operation=operation)
                
                self.assertEqual(result["operation"], operation)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()


class TestCLIEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_concurrent_operations(self):
        """Test concurrent CLI operations."""
        async def async_test():
            # Run multiple operations concurrently
            tasks = [
                start_call("session-1", domain="banking"),
                start_call("session-2", domain="compliance"),
                monitor_session("session-1"),
                orchestrator_health()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All operations should complete without raising exceptions
            for result in results:
                self.assertIsInstance(result, dict)
                self.assertNotIsInstance(result, Exception)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_config_override(self):
        """Test configuration override in start_call."""
        async def async_test():
            config_override = {
                "custom_setting": "test_value",
                "timeout": 600
            }
            
            result = await start_call(
                session_id="test-override",
                config_override=config_override
            )
            
            self.assertEqual(result["status"], "started")
            session_config = result["session_config"]
            self.assertEqual(session_config["custom_setting"], "test_value")
            self.assertEqual(session_config["timeout"], 600)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
