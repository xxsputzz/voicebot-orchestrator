"""
Sprint 5: LoRA Adapter Training & Management

Low-Rank Adaptation (LoRA) implementation for efficient fine-tuning of banking domain models.
Enables domain-specific adaptation without full model retraining.
"""

from __future__ import annotations
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Mock implementations for restricted environment
class MockLoraConfig:
    """Mock LoRA configuration for environments without peft."""
    
    def __init__(self, task_type: str = "CAUSAL_LM", r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1, target_modules: Optional[List[str]] = None):
        self.task_type = task_type
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules
        }


class MockAutoModel:
    """Mock model for environments without transformers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = {"hidden_size": 4096, "num_attention_heads": 32}
    
    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)
    
    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "config.json", "w") as f:
            json.dump({"model_name": self.model_name}, f)


class MockPeftModel:
    """Mock PEFT model for environments without peft."""
    
    def __init__(self, base_model, lora_config):
        self.base_model = base_model
        self.lora_config = lora_config
        self.is_loaded = True
    
    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "adapter_config.json", "w") as f:
            json.dump(self.lora_config.to_dict(), f)
    
    def load_adapter(self, path: str):
        self.is_loaded = True
        return True
    
    def unload_adapter(self):
        self.is_loaded = False


# Try to import real packages, fall back to mocks
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    REAL_LIBS = True
except ImportError:
    LoraConfig = MockLoraConfig
    AutoModelForCausalLM = MockAutoModel
    AutoTokenizer = MockAutoModel
    
    def get_peft_model(base_model, config):
        return MockPeftModel(base_model, config)
    
    PeftModel = MockPeftModel
    REAL_LIBS = False


class LoraAdapter:
    """
    LoRA (Low-Rank Adaptation) adapter for efficient fine-tuning.
    
    Enables domain-specific fine-tuning of large language models without
    updating the full parameter set, reducing memory and compute requirements.
    """
    
    def __init__(
        self,
        adapter_name: str,
        base_model_name: str,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        """
        Initialize LoRA adapter.
        
        Args:
            adapter_name: unique name for the adapter
            base_model_name: name of the base model to adapt
            r: rank of adaptation matrices
            lora_alpha: LoRA scaling parameter
            lora_dropout: dropout rate for LoRA layers
            target_modules: list of modules to apply LoRA to
        """
        if not adapter_name:
            raise ValueError("adapter_name cannot be empty")
        
        if not base_model_name:
            raise ValueError("base_model_name cannot be empty")
        
        if r <= 0:
            raise ValueError("r (rank) must be positive")
        
        if not (0 <= lora_dropout <= 1):
            raise ValueError("lora_dropout must be between 0 and 1")
        
        self.adapter_name = adapter_name
        self.base_model_name = base_model_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # State
        self.is_loaded = False
        self.model = None
        self.base_model = None
        self.training_data = []
        self.training_metrics = {}
        
        self.logger = logging.getLogger(__name__)
    
    def create_lora_config(self) -> Any:
        """
        Create LoRA configuration.
        
        Returns:
            LoRA configuration object
        """
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules
        )
    
    def load_base_model(self) -> bool:
        """
        Load base model for adaptation.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading base model: {self.base_model_name}")
            self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            return False
    
    def create_adapter(self) -> bool:
        """
        Create LoRA adapter from base model.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.base_model:
            if not self.load_base_model():
                return False
        
        try:
            lora_config = self.create_lora_config()
            self.model = get_peft_model(self.base_model, lora_config)
            self.is_loaded = True
            
            self.logger.info(f"Created LoRA adapter: {self.adapter_name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to create adapter: {e}")
            return False
    
    def save_adapter(self, save_path: str) -> bool:
        """
        Save LoRA adapter to disk.
        
        Args:
            save_path: directory to save adapter
            
        Returns:
            True if successful, False otherwise
        """
        if not self.model:
            raise ValueError("No adapter model to save")
        
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save adapter weights
            self.model.save_pretrained(str(save_dir))
            
            # Save metadata
            metadata = {
                "adapter_name": self.adapter_name,
                "base_model_name": self.base_model_name,
                "r": self.r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.target_modules,
                "training_metrics": self.training_metrics
            }
            
            with open(save_dir / "adapter_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved adapter to {save_dir}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save adapter: {e}")
            return False
    
    def load_adapter(self, load_path: str) -> bool:
        """
        Load LoRA adapter from disk.
        
        Args:
            load_path: directory containing adapter
            
        Returns:
            True if successful, False otherwise
        """
        try:
            load_dir = Path(load_path)
            
            if not load_dir.exists():
                raise ValueError(f"Adapter path does not exist: {load_path}")
            
            # Load metadata
            metadata_file = load_dir / "adapter_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                self.adapter_name = metadata["adapter_name"]
                self.base_model_name = metadata["base_model_name"]
                self.r = metadata["r"]
                self.lora_alpha = metadata["lora_alpha"]
                self.lora_dropout = metadata["lora_dropout"]
                self.target_modules = metadata["target_modules"]
                self.training_metrics = metadata.get("training_metrics", {})
            
            # Load base model if needed
            if not self.base_model:
                if not self.load_base_model():
                    return False
            
            # Create adapter config and load weights
            lora_config = self.create_lora_config()
            self.model = get_peft_model(self.base_model, lora_config)
            
            # In real implementation, you'd load the actual weights
            # For mock, we just set loaded state
            self.is_loaded = True
            
            self.logger.info(f"Loaded adapter from {load_dir}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load adapter: {e}")
            return False
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get adapter information.
        
        Returns:
            dictionary with adapter details
        """
        return {
            "adapter_name": self.adapter_name,
            "base_model_name": self.base_model_name,
            "is_loaded": self.is_loaded,
            "config": {
                "r": self.r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.target_modules
            },
            "training_metrics": self.training_metrics,
            "parameter_count": self._estimate_parameter_count()
        }
    
    def add_training_data(self, input_text: str, target_text: str, metadata: Optional[Dict] = None) -> None:
        """
        Add training data for adapter fine-tuning.
        
        Args:
            input_text: input prompt
            target_text: expected output
            metadata: optional metadata
        """
        if not input_text or not target_text:
            raise ValueError("input_text and target_text cannot be empty")
        
        training_example = {
            "input": input_text.strip(),
            "target": target_text.strip(),
            "metadata": metadata or {}
        }
        
        self.training_data.append(training_example)
    
    def simulate_training(self, epochs: int = 3, learning_rate: float = 1e-4) -> Dict[str, Any]:
        """
        Simulate adapter training (mock implementation).
        
        Args:
            epochs: number of training epochs
            learning_rate: learning rate for training
            
        Returns:
            training metrics
        """
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if not self.training_data:
            raise ValueError("No training data available")
        
        # Simulate training metrics
        import random
        
        metrics = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "training_samples": len(self.training_data),
            "final_loss": round(random.uniform(0.1, 0.5), 4),
            "perplexity": round(random.uniform(1.5, 3.0), 2),
            "convergence_epoch": min(epochs, random.randint(1, epochs))
        }
        
        self.training_metrics = metrics
        
        self.logger.info(f"Simulated training completed: {metrics}")
        return metrics
    
    def _estimate_parameter_count(self) -> int:
        """Estimate number of trainable parameters in LoRA adapter."""
        # Rough estimate based on LoRA configuration
        # In practice, this would be calculated from actual model layers
        base_params = 7_000_000_000  # Assuming 7B parameter base model
        
        # LoRA adds 2 * r * hidden_dim parameters per target module
        hidden_dim = 4096  # Typical for 7B models
        lora_params_per_module = 2 * self.r * hidden_dim
        total_lora_params = lora_params_per_module * len(self.target_modules)
        
        return total_lora_params


class LoraAdapterManager:
    """
    Manager for multiple LoRA adapters.
    
    Handles loading, unloading, and switching between different domain-specific adapters.
    """
    
    def __init__(self, adapter_dir: str = "./adapters"):
        """
        Initialize adapter manager.
        
        Args:
            adapter_dir: directory containing adapters
        """
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        
        self.adapters: Dict[str, LoraAdapter] = {}
        self.active_adapter: Optional[str] = None
        
        self.logger = logging.getLogger(__name__)
    
    def create_adapter(
        self,
        adapter_name: str,
        base_model_name: str,
        **kwargs
    ) -> bool:
        """
        Create new LoRA adapter.
        
        Args:
            adapter_name: unique adapter name
            base_model_name: base model to adapt
            **kwargs: additional LoRA configuration
            
        Returns:
            True if successful, False otherwise
        """
        if adapter_name in self.adapters:
            self.logger.warning(f"Adapter {adapter_name} already exists")
            return False
        
        try:
            adapter = LoraAdapter(adapter_name, base_model_name, **kwargs)
            
            if adapter.create_adapter():
                self.adapters[adapter_name] = adapter
                self.logger.info(f"Created adapter: {adapter_name}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Failed to create adapter {adapter_name}: {e}")
            return False
    
    def load_adapter(self, adapter_name: str, adapter_path: Optional[str] = None) -> bool:
        """
        Load adapter from disk.
        
        Args:
            adapter_name: adapter name
            adapter_path: path to adapter (if None, uses default location)
            
        Returns:
            True if successful, False otherwise
        """
        if adapter_path is None:
            adapter_path = str(self.adapter_dir / adapter_name)
        
        try:
            adapter = LoraAdapter(adapter_name, "")  # Will be loaded from metadata
            
            if adapter.load_adapter(adapter_path):
                self.adapters[adapter_name] = adapter
                self.logger.info(f"Loaded adapter: {adapter_name}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Failed to load adapter {adapter_name}: {e}")
            return False
    
    def unload_adapter(self, adapter_name: str) -> bool:
        """
        Unload adapter from memory.
        
        Args:
            adapter_name: adapter to unload
            
        Returns:
            True if successful, False otherwise
        """
        if adapter_name not in self.adapters:
            self.logger.warning(f"Adapter {adapter_name} not loaded")
            return False
        
        # Deactivate if currently active
        if self.active_adapter == adapter_name:
            self.active_adapter = None
        
        # Remove from memory
        del self.adapters[adapter_name]
        
        self.logger.info(f"Unloaded adapter: {adapter_name}")
        return True
    
    def activate_adapter(self, adapter_name: str) -> bool:
        """
        Activate an adapter for inference.
        
        Args:
            adapter_name: adapter to activate
            
        Returns:
            True if successful, False otherwise
        """
        if adapter_name not in self.adapters:
            self.logger.error(f"Adapter {adapter_name} not loaded")
            return False
        
        self.active_adapter = adapter_name
        self.logger.info(f"Activated adapter: {adapter_name}")
        return True
    
    def deactivate_adapter(self) -> bool:
        """
        Deactivate current adapter.
        
        Returns:
            True if successful, False otherwise
        """
        if self.active_adapter:
            self.logger.info(f"Deactivated adapter: {self.active_adapter}")
            self.active_adapter = None
            return True
        
        return False
    
    def list_adapters(self) -> List[str]:
        """
        List available adapters in adapter directory.
        
        Returns:
            list of adapter names
        """
        adapter_dirs = [
            d.name for d in self.adapter_dir.iterdir() 
            if d.is_dir() and (d / "adapter_metadata.json").exists()
        ]
        
        return sorted(adapter_dirs)
    
    def list_loaded_adapters(self) -> List[str]:
        """
        List currently loaded adapters.
        
        Returns:
            list of loaded adapter names
        """
        return list(self.adapters.keys())
    
    def get_adapter_status(self) -> Dict[str, Any]:
        """
        Get status of all adapters.
        
        Returns:
            adapter status information
        """
        return {
            "available_adapters": self.list_adapters(),
            "loaded_adapters": self.list_loaded_adapters(),
            "active_adapter": self.active_adapter,
            "adapter_directory": str(self.adapter_dir)
        }
    
    def get_adapter_info(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an adapter.
        
        Args:
            adapter_name: adapter name
            
        Returns:
            adapter information or None if not found
        """
        if adapter_name in self.adapters:
            return self.adapters[adapter_name].get_adapter_info()
        
        return None
    
    def create_banking_adapter(self, adapter_name: str = "banking-lora") -> bool:
        """
        Create pre-configured adapter for banking domain.
        
        Args:
            adapter_name: name for banking adapter
            
        Returns:
            True if successful, False otherwise
        """
        banking_config = {
            "r": 16,  # Higher rank for complex banking domain
            "lora_alpha": 32,
            "lora_dropout": 0.05,  # Lower dropout for stability
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
        
        success = self.create_adapter(
            adapter_name,
            "mistralai/Mistral-7B-v0.1",  # Default base model
            **banking_config
        )
        
        if success:
            # Add sample banking training data
            adapter = self.adapters[adapter_name]
            self._add_banking_training_data(adapter)
            
            # Simulate training
            adapter.simulate_training(epochs=5, learning_rate=2e-4)
            
            # Save adapter
            adapter_path = self.adapter_dir / adapter_name
            adapter.save_adapter(str(adapter_path))
        
        return success
    
    def _add_banking_training_data(self, adapter: LoraAdapter) -> None:
        """Add sample banking domain training data."""
        banking_examples = [
            {
                "input": "What is APR?",
                "target": "APR stands for Annual Percentage Rate. It represents the yearly cost of borrowing money, including both the interest rate and additional fees, expressed as a percentage."
            },
            {
                "input": "How do I calculate loan payments?",
                "target": "Loan payments can be calculated using the formula: M = P[r(1+r)^n]/[(1+r)^n-1], where M is monthly payment, P is principal, r is monthly interest rate, and n is number of payments."
            },
            {
                "input": "What is KYC compliance?",
                "target": "KYC (Know Your Customer) compliance involves verifying customer identity and assessing risk to prevent money laundering and fraud. Banks must collect and verify customer information before providing services."
            },
            {
                "input": "Explain mortgage amortization",
                "target": "Mortgage amortization is the process of paying off a mortgage through regular payments. Early payments go mostly toward interest, while later payments pay more toward principal."
            },
            {
                "input": "What are wire transfer limits?",
                "target": "Wire transfer limits vary by account type and bank policy. Typically, domestic wires may have daily limits of $100,000-$250,000, while international wires may have lower limits and require additional verification."
            }
        ]
        
        for example in banking_examples:
            adapter.add_training_data(
                example["input"],
                example["target"],
                {"domain": "banking", "category": "faq"}
            )


# Analytics integration
_adapter_analytics = {
    "adapters_created": 0,
    "adapters_loaded": 0,
    "adapter_switches": 0,
    "training_sessions": 0
}


def get_lora_analytics() -> Dict[str, Any]:
    """Get LoRA adapter analytics."""
    return {
        "adapters_created": _adapter_analytics["adapters_created"],
        "adapters_loaded": _adapter_analytics["adapters_loaded"],
        "adapter_switches": _adapter_analytics["adapter_switches"],
        "training_sessions": _adapter_analytics["training_sessions"],
        "service_name": "lora_adapters"
    }


def _update_adapter_analytics(operation: str) -> None:
    """Update adapter analytics counters."""
    if operation in _adapter_analytics:
        _adapter_analytics[operation] += 1
