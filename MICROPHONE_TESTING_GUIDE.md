# Microphone Testing Suite - Complete Setup Guide

## Overview
The Microphone Testing Suite provides comprehensive testing of the STT→LLM→TTS pipeline with real microphone input. This allows you to test voice conversation functionality end-to-end.

## Available Test Modes

### 1. 🎤 STT Only Test
- **Purpose**: Test speech-to-text conversion
- **Flow**: Record audio → Send to STT → Display transcript
- **Services Required**: STT (Whisper)
- **Use Case**: Validate microphone input and speech recognition accuracy

### 2. 🎤➡️🧠 STT → LLM Test  
- **Purpose**: Test speech to AI response generation
- **Flow**: Record audio → STT → LLM → Display text response
- **Services Required**: STT (Whisper) + LLM (GPT/Mistral)
- **Use Case**: Test conversational understanding and response generation

### 3. 🧠➡️🗣️ LLM → TTS Test
- **Purpose**: Test text input to speech synthesis
- **Flow**: Text input → LLM → TTS → Play audio
- **Services Required**: LLM (GPT/Mistral) + TTS (Kokoro/Zonos/Tortoise)
- **Use Case**: Test AI response quality and speech synthesis

### 4. 🗣️ TTS Only Test
- **Purpose**: Test direct text-to-speech conversion
- **Flow**: Text input → TTS → Play audio
- **Services Required**: TTS (Kokoro/Zonos/Tortoise)
- **Use Case**: Test speech synthesis quality and voice selection

### 5. 🎤➡️🧠➡️🗣️ Full Pipeline Test
- **Purpose**: Complete voice conversation test
- **Flow**: Record audio → STT → LLM → TTS → Play audio
- **Services Required**: STT + LLM + TTS
- **Use Case**: End-to-end voice conversation validation

### 6. 🔄 Continuous Conversation Mode
- **Purpose**: Ongoing voice chat with AI
- **Flow**: Repeated full pipeline cycles
- **Services Required**: STT + LLM + TTS  
- **Use Case**: Multi-turn conversation testing

## Setup Instructions

### Prerequisites
1. **Orchestrator**: WebSocket orchestrator must be running
2. **Services**: Required services must be started (STT, LLM, TTS combinations)
3. **Audio Hardware**: Working microphone and speakers/headphones
4. **Dependencies**: Python audio libraries installed

### Installation Steps

#### 1. Install Audio Dependencies
```bash
# Run the dependency installer
python install_microphone_deps.py

# Or install manually:
pip install pyaudio websockets wave
```

#### 2. Start Required Services
Use the comprehensive launcher to start service combinations:

```bash
# Launch the comprehensive service manager
python comprehensive_ws_launcher.py

# Choose option 2 (Service Combos) and select:
# - Fast Combo (Whisper + Mistral + Kokoro)
# - Balanced Combo (Whisper + GPT + Kokoro) 
# - Quality Combo (Whisper + GPT + Zonos)
# - Premium Combo (Whisper + GPT + Tortoise)
```

#### 3. Launch Microphone Testing

**Option A: From Comprehensive Launcher**
```bash
python comprehensive_ws_launcher.py
# Choose option 8: 🎤 Microphone Testing Suite
```

**Option B: From Main Launcher**
```bash
python launcher.py
# Choose option 4: Other Tests
# Choose option 7: 🎤 Microphone Testing Suite  
```

**Option C: Direct Launch**
```bash
python microphone_test_suite.py
```

## Usage Guide

### Audio Configuration
- **Sample Rate**: 16kHz (optimized for Whisper STT)
- **Channels**: Mono (single channel)
- **Format**: 16-bit PCM
- **Recording Duration**: 5-7 seconds per test

### Service Requirements by Test

| Test Mode | STT | LLM | TTS | Description |
|-----------|-----|-----|-----|-------------|
| STT Only | ✅ | ❌ | ❌ | Speech recognition only |
| STT → LLM | ✅ | ✅ | ❌ | Speech to text response |
| LLM → TTS | ❌ | ✅ | ✅ | Text to speech response |
| TTS Only | ❌ | ❌ | ✅ | Direct speech synthesis |
| Full Pipeline | ✅ | ✅ | ✅ | Complete voice conversation |
| Conversation Loop | ✅ | ✅ | ✅ | Multi-turn chat |

### Recommended Service Combinations

#### Fast Testing (Low Latency)
- **STT**: Whisper 
- **LLM**: Mistral (local)
- **TTS**: Kokoro (fastest synthesis)
- **Total Latency**: ~800ms

#### Balanced Testing (Quality/Speed)
- **STT**: Whisper
- **LLM**: GPT (cloud)
- **TTS**: Zonos (balanced quality)
- **Total Latency**: ~900ms

#### Premium Testing (Highest Quality)
- **STT**: Whisper
- **LLM**: GPT (cloud)  
- **TTS**: Tortoise (29 voices, premium quality)
- **Total Latency**: ~2-3s

## Troubleshooting

### Common Issues

#### Audio System Problems
```
❌ Audio system not available
```
**Solution**: Install PyAudio dependencies
- Windows: May need Visual Studio Build Tools
- Linux: `sudo apt-get install portaudio19-dev`
- macOS: `brew install portaudio`

#### Service Connection Issues
```
❌ Missing services: stt, llm, tts
```
**Solution**: Start required services first
1. Use comprehensive launcher option 2 (Service Combos)
2. Wait for services to fully register (green status)
3. Retry microphone test

#### WebSocket Connection Failed
```
❌ WebSocket connection failed
```
**Solution**: Verify orchestrator is running
1. Check orchestrator status: `curl http://localhost:8080/health`
2. Restart orchestrator if needed
3. Ensure no port conflicts (9000, 8080)

#### No Audio Input Devices
```
❌ No audio input devices found
```
**Solution**: Check system audio settings
1. Verify microphone is connected and enabled
2. Check Windows/system audio permissions
3. Test microphone in other applications first

### Performance Optimization

#### For Best Latency
- Use Fast Combo (Mistral + Kokoro)
- Local processing only
- Close unnecessary applications

#### For Best Quality  
- Use Premium Combo (GPT + Tortoise)
- Allow longer processing time
- Use high-quality microphone

#### For Balanced Testing
- Use Quality Combo (GPT + Zonos)
- Good balance of speed and quality
- Most reliable for general testing

## Integration with Existing Launchers

### Comprehensive Launcher Integration
The microphone testing suite is fully integrated into the comprehensive launcher as option 8. This provides:
- Automatic service dependency checking
- Seamless workflow from service startup to testing
- Centralized management interface

### Main Launcher Integration
Also available through the main launcher's "Other Tests" menu as option 7, providing:
- Easy access from the main application entry point
- Consistent user experience
- Familiar menu navigation

## Technical Details

### Audio Processing Pipeline
1. **Recording**: PyAudio captures 16kHz mono audio
2. **Encoding**: Raw audio encoded as base64 for WebSocket transmission  
3. **Processing**: Services process audio/text through WebSocket protocol
4. **Playback**: Response audio played through system audio output

### WebSocket Protocol
- **Connection**: Client connects to `ws://localhost:9000`
- **Messages**: JSON-formatted StreamingMessage protocol
- **Types**: `audio_chunk`, `text_input`, `tts_request`, responses
- **Session**: Unique session ID for message routing

### Service Communication
- **STT**: Receives `audio_chunk`, returns `transcript_partial`/`transcript_final`
- **LLM**: Receives `text_input`, returns `llm_token`/`llm_stream_complete`
- **TTS**: Receives `tts_request`, returns `audio_output`

This comprehensive testing suite enables full validation of the voice conversation pipeline with real-world microphone input and audio output.
