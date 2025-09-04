# Scripts Directory

This directory contains all the development scripts, tests, and utilities that were moved from the root directory to keep it clean.

## Directory Structure

### 📁 `tests/`
Contains all test scripts and validation utilities:
- `test_*.py` - Test scripts for various services and components
- `validate_*.py` - Validation scripts for system verification  
- `verify_*.py` - Voice and service verification scripts
- `check_*.py` - Health check and diagnostic scripts
- `diagnose_*.py` - Diagnostic and analysis scripts
- `analyze_*.py` - Analysis and monitoring scripts

### 📁 `debug/`
Contains debugging and diagnostic utilities:
- `debug_*.py` - Debug scripts for troubleshooting
- `comprehensive_*.py` - Comprehensive diagnostic tools
- `monitor_*.py` - System monitoring scripts
- `cuda_diagnostics.py` - GPU/CUDA diagnostic tools

### 📁 `utilities/`
Contains utility scripts and helper tools:
- `download_*.py` - Download and setup scripts
- `emergency_*.py` - Emergency cleanup and recovery tools
- `system_*.py` - System management utilities
- `tortoise_*.py` - Tortoise TTS related utilities
- `simple_*.py` - Simple test and demo scripts
- `quick_*.py` - Quick test utilities
- `emoji_*.py` - Text processing utilities
- `fix_*.py` - Fix and repair scripts
- `enhanced_*.py` - Enhanced service implementations
- `interactive_*.py` - Interactive testing tools
- `final_*.py` - Final verification scripts
- JSON configuration files and implementations

### 📁 `batch/`
Contains batch scripts and automation files:
- `*.bat` - Windows batch files
- Setup and installation scripts
- Service startup scripts
- Test automation scripts

## Usage

To run any script from the root directory:

```bash
# Run a test script
python scripts/tests/test_tortoise_direct.py

# Run a debug utility
python scripts/debug/debug_gpu_tortoise.py

# Run a utility script
python scripts/utilities/download_official_voices.py

# Run a batch file
scripts/batch/setup_voicebot.bat
```

## Organization Benefits

1. **Clean Root**: Root directory now only contains essential files
2. **Easy Navigation**: Scripts are organized by purpose
3. **Better Maintenance**: Related scripts are grouped together
4. **Clear Structure**: Easy to find the right tool for the task

## Essential Files Kept in Root

- `launcher.py` - Main application launcher
- `README.md` - Project documentation
- `LICENSE` - License information
- `pyproject.toml` - Project configuration
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker configuration
- `Dockerfile` - Container definition
- Essential binaries and configuration files
