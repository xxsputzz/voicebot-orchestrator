"""
SERVICE STARTUP METHODS GUIDE
============================

This guide shows all available methods to start up voicebot services.

## 🚀 AVAILABLE STARTUP METHODS

### Method 1: Interactive CLI Menu (Recommended)
```bash
# Start the modular CLI
cd C:\Users\miken\Desktop\Orkestra
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py

# Then navigate:
# 2. 🔧 Service Management
```

### Method 2: Voice Pipeline Auto-Start
```bash
# Start the modular CLI
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py

# Then navigate:
# 1. 🎙️ Voice Pipeline → 1. 🗣️ Start Voice Conversation
# This auto-initializes required services
```

### Method 3: Direct Commands
```bash
# Check current status
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py --status

# Run health check
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py --health-check
```

### Method 4: Convenience Launcher
```bash
# Double-click or run:
voicebot_cli.bat
```

## 🎯 SERVICE INITIALIZATION OPTIONS

### Individual Services:
1. **🎤 Initialize STT** - Speech-to-Text (Whisper)
2. **🧠 Initialize LLM** - Language Model (Mistral) 
3. **🔊 Initialize TTS Kokoro** - Fast TTS (~0.8s)
4. **🎭 Initialize TTS Nari Dia** - Quality TTS (~3min)

### Bulk Options:
5. **🚀 Initialize All Services** - Everything (with memory warning)
6. **📊 Service Status** - Show current state
7. **🧹 Cleanup Services** - Free all memory

## ⚠️  MEMORY MANAGEMENT

### GPU Memory Usage:
- **No Services**: 0GB (clean start)
- **STT + LLM**: ~1-2GB
- **+ Kokoro TTS**: ~2-3GB
- **+ Nari Dia TTS**: ~6-8GB total

### Recommendations:
- 🎯 **For Conversation**: STT + LLM + Kokoro only
- 🎭 **For Quality**: Use Nari Dia separately 
- ⚠️ **Avoid Both TTS**: Don't load Kokoro + Nari Dia together

## 🔄 SERVICE STATES

Services can be in these states:
- **❌ Not loaded** - Service not initialized (no memory usage)
- **✅ Ready** - Service loaded and ready (using memory)

## 📊 STARTUP VERIFICATION

After starting services, check status:
```
📊 Service Status:
   🎤 STT: ✅ Ready
   🧠 LLM: ✅ Ready
   🔊 TTS Kokoro: ✅ Ready
   🔊 TTS Nari Dia: ❌ Not loaded
   🎯 Active TTS: kokoro
```

## 🎯 RECOMMENDED WORKFLOWS

### For Voice Conversation:
1. Start CLI: `voicebot_cli.bat`
2. Choose: `1. Voice Pipeline`
3. Choose: `1. Start Voice Conversation`
4. Select: `1. Kokoro (Fast)`
5. Ready for conversation!

### For High-Quality Audio:
1. Start CLI: `voicebot_cli.bat`
2. Choose: `2. Service Management`
3. Choose: `4. Initialize TTS Nari Dia`
4. Choose: `4. Audio Generation`
5. Generate high-quality speech

### For Testing:
1. Start CLI: `voicebot_cli.bat`
2. Choose: `3. Testing & Demos`
3. Choose desired test
4. Services auto-initialize as needed

## 🚀 QUICK REFERENCE

**Fastest Startup for Conversation:**
```bash
voicebot_cli.bat
# → 1 → 1 → 1 (Voice Pipeline → Start Conversation → Kokoro)
```

**Memory-Efficient Startup:**
```bash
voicebot_cli.bat
# → 2 → 1,2,3 (Service Management → Initialize STT, LLM, Kokoro)
```

**Status Check:**
```bash
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py --status
```
"""
