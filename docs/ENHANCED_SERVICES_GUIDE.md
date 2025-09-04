# Enhanced Independent Microservices - User Guide

## 🎭 Enhanced Service Manager

The Enhanced Service Manager provides a **numbered menu interface** that follows your existing test patterns and fixes the menu system issues you mentioned.

### Quick Start

1. **Interactive Mode (Default):**
   ```bash
   cd aws_microservices
   python enhanced_service_manager.py
   ```

2. **Command Line Options:**
   ```bash
   python enhanced_service_manager.py --status          # Show service status
   python enhanced_service_manager.py --start fast      # Start fast combination
   python enhanced_service_manager.py --test            # Test running services
   ```

### 📋 Main Menu (Numbers 1-9 Work Correctly!)

When you run in interactive mode, you'll see:

```
🎭 Enhanced Independent Microservices Manager
============================================================
🔴 No services currently running

📋 Service Combinations (Following existing patterns):
  1. Show detailed service status
  2. Start Fast Combo (Kokoro TTS + Mistral LLM)
  3. Start Quality Combo (Hira Dia TTS + GPT LLM) 
  4. Start Balanced Combo (Kokoro TTS + GPT LLM)
  5. Start Efficient Combo (Hira Dia TTS + Mistral LLM)

⚙️ Service Management:
  6. Manage individual services
  7. Stop all services
  8. Test running services
  9. Launch comprehensive test suite

  0. Exit

Enter your choice (0-9):
```

### 🔧 Fixed Menu Issues

**The numbered menu (1-5) now works correctly:**

- **Choice 1:** Shows detailed status of all services
- **Choice 2:** Starts fast combination (real-time processing)
- **Choice 3:** Starts quality combination (maximum quality)
- **Choice 4:** Starts balanced combination (fast TTS + advanced LLM)
- **Choice 5:** Starts efficient combination (quality TTS + efficient LLM)

### 🎯 Service Combinations

Each combination automatically starts the required services:

#### 2. Fast Combo (Real-time)
- **STT Service** (Port 8001)
- **Kokoro TTS** (Port 8011) - Fast generation
- **Mistral LLM** (Port 8021) - Quick responses
- **Use case:** Real-time conversation, live demos

#### 3. Quality Combo (Maximum quality)
- **STT Service** (Port 8001)
- **Hira Dia TTS** (Port 8012) - High quality voice
- **GPT LLM** (Port 8022) - Advanced reasoning
- **Use case:** Professional presentations, content creation

#### 4. Balanced Combo
- **STT Service** (Port 8001)
- **Kokoro TTS** (Port 8011) - Fast TTS
- **GPT LLM** (Port 8022) - Advanced LLM
- **Use case:** Quick TTS with smart responses

#### 5. Efficient Combo
- **STT Service** (Port 8001)
- **Hira Dia TTS** (Port 8012) - Quality TTS
- **Mistral LLM** (Port 8021) - Efficient LLM
- **Use case:** Quality output with reasonable processing time

### 🔧 Individual Service Management (Choice 6)

When you select option 6, you get another numbered menu:

```
🔧 Individual Service Management
----------------------------------------
  1. Start Speech-to-Text Service (⏹️ stopped)
  2. Start Kokoro TTS (Fast) (⏹️ stopped)
  3. Start Hira Dia TTS (High Quality) (⏹️ stopped)
  4. Start Mistral LLM (⏹️ stopped)
  5. Start GPT LLM (⏹️ stopped)
  0. Back to main menu

Select service (0-5):
```

### 🧪 Comprehensive Testing (Choice 9)

The comprehensive test suite follows patterns from your existing `tests/` folder:

- **Audio Testing:** Like `test_real_kokoro.py` - generates actual audio files
- **STT Testing:** Tests transcription with realistic audio data
- **LLM Testing:** Tests text generation with banking context
- **Combination Testing:** Tests service combinations end-to-end
- **Audio Output:** Saves test audio to `tests/audio_samples/`

### 🔊 Audio Testing Integration

Following your existing `tests/` patterns:

```bash
# Run comprehensive tests (includes audio generation)
python test_independent_services.py

# Test specific TTS service
python test_independent_services.py --service kokoro_tts

# Test service combinations
python test_independent_services.py --combinations
```

**Audio files are saved to:** `tests/audio_samples/`
- `kokoro_tts_test_1.wav`
- `hira_dia_tts_test_1.wav`
- etc.

### 🛠️ Troubleshooting

**If numbers 1-5 don't work:**
1. Make sure you're using the `enhanced_service_manager.py` (not the old one)
2. Check that you're in the correct directory: `cd aws_microservices`
3. Try: `python enhanced_service_manager.py --status` first

**Service won't start:**
1. Check if the individual service scripts exist
2. Verify Python dependencies are installed
3. Check port availability (no conflicts)

### 📊 Example Session

```bash
cd aws_microservices
python enhanced_service_manager.py

# Choose 2 for Fast Combo
Enter your choice (0-9): 2

🚀 Starting Fast Combo
Description: Kokoro TTS + Mistral LLM (Real-time)
Use case: Real-time conversation, quick responses
--------------------------------------------------
🚀 Starting Speech-to-Text Service on port 8001...
✅ Speech-to-Text Service started successfully
🚀 Starting Kokoro TTS (Fast) on port 8011...
✅ Kokoro TTS (Fast) started successfully
🚀 Starting Mistral LLM on port 8021...
✅ Mistral LLM started successfully

📊 Results: 3/3 services started
🎉 Fast Combo is ready!

🔗 Service URLs:
  Speech-to-Text Service: http://localhost:8001
  Kokoro TTS (Fast): http://localhost:8011
  Mistral LLM: http://localhost:8021

Press Enter to continue...

# Choose 8 to test the running services
Enter your choice (0-9): 8

🧪 Testing 3 running services...
--------------------------------------------------
  ✅ Speech-to-Text Service (Port 8001)
  ✅ Kokoro TTS (Fast) (Port 8011)
  ✅ Mistral LLM (Port 8021)

🔬 Testing Service Functionality:
------------------------------
  ✅ STT Service: Basic functionality test passed
  ✅ Kokoro TTS (Fast): Audio generation test passed
  ✅ Mistral LLM: Text generation test passed
```

### 🎉 Key Improvements

1. **Fixed Numbered Menu:** Choices 1-5 now work correctly
2. **Following Existing Patterns:** Based on your `tests/` folder structure
3. **Audio Testing:** Real audio generation like `test_real_kokoro.py`
4. **Smart Service Management:** Automatic dependency handling
5. **Comprehensive Testing:** Integration with existing test infrastructure
6. **Better Error Handling:** Clear status messages and troubleshooting
7. **Service Combinations:** Pre-configured setups for different use cases

The system now provides the reliable numbered menu interface you requested while integrating seamlessly with your existing comprehensive test suite!
