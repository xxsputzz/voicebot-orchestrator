#!/usr/bin/env python3
"""
Root Directory Cleanup Script
Moves all test, debug, and utility files to organized subdirectories
Keeps only essential files in the root directory
"""

import os
import shutil
from pathlib import Path

# Define the root directory
root_dir = Path(__file__).parent
scripts_dir = root_dir / "scripts"

# Essential files that should stay in root
KEEP_IN_ROOT = {
    # Core files
    'launcher.py',
    'README.md',
    'LICENSE',
    'pyproject.toml',
    'requirements.txt',
    
    # Docker files
    'docker-compose.yml',
    'Dockerfile',
    
    # Environment files
    '.gitignore',
    '.env.example',
    
    # Core binaries
    'kokoro-v1.0.onnx',
    'voices-v1.0.bin',
}

# File patterns to move to specific directories
MOVE_PATTERNS = {
    'scripts/tests/': [
        'test_*.py',
        'validate_*.py',
        'verify_*.py',
        'check_*.py',
        'diagnose_*.py',
        'analyze_*.py',
    ],
    'scripts/debug/': [
        'debug_*.py',
        'comprehensive_*.py',
        'monitor_*.py',
        'cuda_diagnostics.py',
    ],
    'scripts/utilities/': [
        'download_*.py',
        'emergency_*.py',
        'emergency_*.bat',
        'system_*.py',
        'tortoise_*.py',
        'simple_*.py',
        'quick_*.py',
        'emoji_*.py',
        'fix_*.py',
        'enhanced_*.py',
        'interactive_*.py',
        'final_*.py',
        'TTS_*.py',
        'tts_*.py',
        'demo_*.py',
        'run_*.py',
        'start_*.py',
    ],
    'scripts/batch/': [
        '*.bat',
        'voicebot_*.bat',
        'setup_*.bat',
        'install_*.bat',
        'start_*.bat',
        'fix_*.bat',
        'test_*.bat',
    ],
    'docs/': [
        '*.md',
    ]
}

def should_keep_in_root(filename):
    """Check if file should stay in root directory"""
    return filename in KEEP_IN_ROOT

def get_destination_dir(filename):
    """Get the destination directory for a file"""
    import fnmatch
    
    # Keep essential files in root
    if should_keep_in_root(filename):
        return None
    
    # Check patterns for destination
    for dest_dir, patterns in MOVE_PATTERNS.items():
        for pattern in patterns:
            if fnmatch.fnmatch(filename, pattern):
                return dest_dir
    
    # Special cases
    if filename.endswith('.json') and any(x in filename for x in ['config', 'timeout', 'cli']):
        return 'scripts/utilities/'
    
    if filename.endswith('.log'):
        return 'logs/'
    
    # Default: move to scripts/misc if it's a script-like file
    if filename.endswith(('.py', '.bat', '.sh')):
        return 'scripts/utilities/'
    
    return None

def cleanup_root():
    """Main cleanup function"""
    print("🧹 Starting Root Directory Cleanup")
    print("=" * 50)
    
    moved_count = 0
    kept_count = 0
    
    # Get all files in root directory (not directories)
    root_files = [f for f in os.listdir(root_dir) if os.path.isfile(root_dir / f)]
    
    for filename in root_files:
        source_path = root_dir / filename
        
        # Skip this cleanup script itself
        if filename == 'cleanup_root.py':
            continue
            
        dest_dir = get_destination_dir(filename)
        
        if dest_dir is None:
            print(f"✅ Keeping: {filename}")
            kept_count += 1
            continue
        
        # Create destination directory if it doesn't exist
        dest_path = root_dir / dest_dir
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Move the file
        dest_file = dest_path / filename
        try:
            shutil.move(str(source_path), str(dest_file))
            print(f"📁 Moved: {filename} → {dest_dir}")
            moved_count += 1
        except Exception as e:
            print(f"❌ Error moving {filename}: {e}")
    
    print("\n" + "=" * 50)
    print(f"✅ Cleanup Complete!")
    print(f"📁 Files moved: {moved_count}")
    print(f"🏠 Files kept in root: {kept_count}")
    print(f"🎯 Root directory is now clean!")
    
    # Show final root directory contents
    print("\n🏠 Final Root Directory Contents:")
    remaining_files = [f for f in os.listdir(root_dir) if os.path.isfile(root_dir / f)]
    for filename in sorted(remaining_files):
        print(f"   • {filename}")

if __name__ == "__main__":
    cleanup_root()
