"""
Sprint 5: Simple CLI Tests

Basic tests for Sprint 5 CLI commands to verify functionality.
"""

import pytest
import subprocess
import json
import sys
from pathlib import Path


class TestSprint5CLI:
    """Test Sprint 5 CLI commands."""
    
    def test_cache_manager_help(self):
        """Test cache-manager help command."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "cache-manager", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert "cache-manager" in result.stdout.lower()
        assert "semantic cache" in result.stdout.lower()
    
    def test_adapter_control_help(self):
        """Test adapter-control help command."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "adapter-control", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert "adapter-control" in result.stdout.lower()
        assert "lora" in result.stdout.lower()
    
    def test_orchestrator_log_help(self):
        """Test orchestrator-log help command."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "orchestrator-log", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert "orchestrator-log" in result.stdout.lower()
        assert "analytics" in result.stdout.lower()
    
    def test_cache_manager_stats_json(self):
        """Test cache manager stats with JSON output."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "--json", "cache-manager", "--stats"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        try:
            data = json.loads(result.stdout)
            assert "total_entries" in data
            assert "hit_rate" in data
            assert isinstance(data["total_entries"], int)
            assert isinstance(data["hit_rate"], (int, float))
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    def test_cache_manager_stats_human(self):
        """Test cache manager stats with human-readable output."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "cache-manager", "--stats"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert "Cache Statistics" in result.stdout
        assert "Total Entries" in result.stdout
        assert "Hit Rate" in result.stdout
    
    def test_cache_manager_analyze(self):
        """Test cache manager analysis."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "cache-manager", "--analyze"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert "Performance Analysis" in result.stdout or "cache_performance" in result.stdout
    
    def test_adapter_control_list_json(self):
        """Test adapter control list with JSON output."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "--json", "adapter-control", "--list"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        try:
            data = json.loads(result.stdout)
            assert "available_adapters" in data
            assert "loaded_adapters" in data
            assert "active_adapter" in data
            assert isinstance(data["available_adapters"], list)
            assert isinstance(data["loaded_adapters"], list)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    def test_adapter_control_list_human(self):
        """Test adapter control list with human-readable output."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "adapter-control", "--list"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert "LoRA Adapter Status" in result.stdout
        assert "Available Adapters" in result.stdout
    
    def test_adapter_control_create_banking(self):
        """Test creating banking adapter."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "--json", "adapter-control", "--create-banking", "test-banking"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        try:
            data = json.loads(result.stdout)
            assert "operation" in data
            assert "adapter" in data
            assert "success" in data
            assert data["operation"] == "create_banking"
            assert data["adapter"] == "test-banking"
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    def test_orchestrator_log_all_json(self):
        """Test orchestrator log with all metrics in JSON."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "--json", "orchestrator-log", "--all"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        try:
            data = json.loads(result.stdout)
            # Should have at least some analytics data
            assert len(data) > 0
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    def test_orchestrator_log_cache_hits(self):
        """Test orchestrator log cache hits."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "orchestrator-log", "--cache-hits"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert "Orchestrator Analytics" in result.stdout or "semantic_cache" in result.stdout
    
    def test_orchestrator_log_adapter_metrics(self):
        """Test orchestrator log adapter metrics."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "orchestrator-log", "--adapter-metrics"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        assert "Orchestrator Analytics" in result.stdout or "lora_adapters" in result.stdout
    
    def test_invalid_command(self):
        """Test invalid command handling."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "invalid-command"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode != 0
        assert "Unknown command" in result.stderr or "help" in result.stdout.lower()
    
    def test_no_command(self):
        """Test running without command shows help."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Should show help (exit code 0) or usage (might be non-zero)
        assert "usage:" in result.stdout.lower() or "help" in result.stdout.lower()
    
    def test_cache_manager_evict(self):
        """Test cache manager eviction."""
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "--json", "cache-manager", "--evict"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Should get JSON response about eviction
        try:
            data = json.loads(result.stdout)
            assert "operation" in data
            assert data["operation"] in ["evict", "clear"]
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    def test_adapter_load_unload_sequence(self):
        """Test adapter load/unload sequence."""
        # Test loading adapter (should work with mock)
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "--json", "adapter-control", "--load", "test-adapter"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Parse JSON output
        try:
            data = json.loads(result.stdout)
            assert "operation" in data
            assert "adapter" in data
            assert data["operation"] == "load"
            assert data["adapter"] == "test-adapter"
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
        
        # Test unloading adapter
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "--json", "adapter-control", "--unload", "test-adapter"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        try:
            data = json.loads(result.stdout)
            assert data["operation"] == "unload"
            assert data["adapter"] == "test-adapter"
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    def test_adapter_activate_deactivate(self):
        """Test adapter activation/deactivation."""
        # Test activation
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "--json", "adapter-control", "--activate", "banking-lora"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        try:
            data = json.loads(result.stdout)
            assert data["operation"] == "activate"
            assert data["adapter"] == "banking-lora"
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
        
        # Test deactivation
        result = subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint5_cli",
            "--json", "adapter-control", "--deactivate"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        try:
            data = json.loads(result.stdout)
            assert data["operation"] == "deactivate"
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
