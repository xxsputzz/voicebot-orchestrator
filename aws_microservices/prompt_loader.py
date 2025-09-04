"""
Prompt Loader Utility
Loads and manages prompts from the docs/prompts folder for LLM services
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PromptLoader:
    """Loads and manages prompts from the docs/prompts folder"""
    
    def __init__(self):
        # Get project root (go up from aws_microservices to project root)
        self.project_root = Path(__file__).parent.parent
        self.prompts_dir = self.project_root / "docs" / "prompts"
        self._cached_prompts: Optional[Dict[str, str]] = None
        
    def get_prompts_directory(self) -> Path:
        """Get the prompts directory path"""
        return self.prompts_dir
    
    def load_all_prompts(self, force_reload: bool = False) -> Dict[str, str]:
        """
        Load all prompt files from the docs/prompts folder
        
        Args:
            force_reload: If True, reload from disk even if cached
            
        Returns:
            Dictionary mapping filename (without extension) to prompt content
        """
        if not force_reload and self._cached_prompts is not None:
            return self._cached_prompts
            
        prompts = {}
        
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory does not exist: {self.prompts_dir}")
            self._cached_prompts = prompts
            return prompts
            
        # Load all .txt files from the prompts directory
        for prompt_file in self.prompts_dir.glob("*.txt"):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only include non-empty files
                        # Use filename without extension as key
                        key = prompt_file.stem
                        prompts[key] = content
                        logger.info(f"Loaded prompt: {key} ({len(content)} characters)")
                    else:
                        logger.warning(f"Empty prompt file skipped: {prompt_file.name}")
            except Exception as e:
                logger.error(f"Error loading prompt file {prompt_file.name}: {e}")
        
        self._cached_prompts = prompts
        logger.info(f"Loaded {len(prompts)} prompt files from {self.prompts_dir}")
        return prompts
    
    def get_system_prompt(self, include_files: Optional[List[str]] = None, call_type: Optional[str] = None) -> str:
        """
        Generate a comprehensive system prompt by combining all loaded prompts
        
        Args:
            include_files: List of specific prompt files to include (without .txt extension)
                          If None, includes all available prompts
            call_type: Type of call ("inbound", "outbound", or None for general)
                          
        Returns:
            Combined system prompt string
        """
        prompts = self.load_all_prompts()
        
        if not prompts:
            return ""
        
        # Handle call type specific prompts
        if call_type:
            call_type_lower = call_type.lower()
            # Priority order for call-specific prompts
            if call_type_lower == "outbound" and "outbound-call" in prompts:
                include_files = ["outbound-call", "prompt-main"]
            elif call_type_lower == "inbound" and "inbound-call" in prompts:
                include_files = ["inbound-call", "prompt-main"]
            # If specific call type prompt doesn't exist, fall back to main prompt
            
        # Filter prompts if specific files requested
        if include_files is not None:
            filtered_prompts = {k: v for k, v in prompts.items() if k in include_files}
            if not filtered_prompts:
                logger.warning(f"None of the requested prompt files found: {include_files}")
            prompts = filtered_prompts
        
        # Combine all prompts into a system message
        combined_sections = []
        
        for filename, content in prompts.items():
            section = f"=== {filename.upper()} PROMPT ===\n{content}\n"
            combined_sections.append(section)
        
        system_prompt = "\n".join(combined_sections)
        
        if system_prompt:
            # Add instruction header
            header = ("You are an AI assistant with the following specialized instructions and context. "
                     "Follow these guidelines carefully in all your responses:\n\n")
            system_prompt = header + system_prompt
            
        logger.info(f"Generated system prompt: {len(system_prompt)} characters from {len(prompts)} files")
        return system_prompt
    
    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt files (without .txt extension)"""
        prompts = self.load_all_prompts()
        return list(prompts.keys())
    
    def reload_prompts(self):
        """Force reload all prompts from disk"""
        self._cached_prompts = None
        return self.load_all_prompts(force_reload=True)

# Global instance for use across services
prompt_loader = PromptLoader()
