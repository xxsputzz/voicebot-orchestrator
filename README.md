# 🤖 Enterprise Voicebot Orchestration Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-blue.svg)](https://kubernetes.io/)

> **Enterprise-grade voicebot orchestration platform with semantic caching, LoRA adapters, and microservices architecture for banking and financial services.**

## 🎯 Overview

This platform provides a complete solution for deploying production-ready voicebots with advanced AI capabilities:

- **🎙️ Speech Processing**: OpenAI Whisper STT + Kokoro TTS
- **🧠 Intelligence**: Mistral LLM with semantic caching & LoRA adapters  
- **🏗️ Architecture**: 6 microservices with Docker & Kubernetes
- **📊 Analytics**: Real-time performance monitoring & reporting
- **🚀 Deployment**: Production-ready with auto-scaling

## ✨ Key Features

### 🔧 **6 CLI Commands**
```bash
orchestrator start-call <session> --phone <phone> --domain banking
orchestrator monitor-session --session-id <session>
orchestrator analytics-report --type summary --time-range 24h
orchestrator cache-manager stats
orchestrator adapter-control list
orchestrator orchestrator-health
```

### 🏗️ **Microservices Architecture**
- **orchestrator-core** - Central coordination (port 8000)
- **stt-service** - Speech-to-text processing (port 8001)
- **llm-service** - Language model with caching (port 8002)
- **tts-service** - Text-to-speech synthesis (port 8003)
- **cache-service** - Semantic caching with Faiss (port 8004)
- **analytics-service** - Metrics & reporting (port 8005)

### 🐳 **Production Deployment**
- **Docker Compose** with monitoring stack
- **Kubernetes** manifests with HPA auto-scaling
- **NGINX** load balancing & reverse proxy
- **Prometheus + Grafana** monitoring

## 🚀 Quick Start

### Option 1: Enhanced CLI (Recommended)
```bash
git clone https://github.com/your-username/voicebot-orchestrator.git
cd voicebot-orchestrator

# Install dependencies
pip install -r requirements.txt

# Start Enhanced CLI (Interactive Mode)
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py

# Or use specific commands
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py --help
```

### Option 2: Docker Compose (Production)
```bash
# All services with monitoring
docker-compose --profile monitoring up

# Access dashboards:
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
# - API docs: http://localhost:8000/docs
```

### Option 3: Kubernetes (Enterprise)
```bash
kubectl apply -f k8s/orchestrator-core.yaml
kubectl get pods -n voicebot-orchestrator
```

## 🎭 Modular Voicebot CLI

The new Modular CLI provides efficient service management with on-demand initialization, preventing GPU memory conflicts.

### 🚀 **Quick Access**

```bash
# Navigate to project directory
cd C:\Users\miken\Desktop\Orkestra

# Start Modular CLI (Interactive Mode)
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py

# Quick commands
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py --status
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py --health-check

# Or use convenience launcher
voicebot_cli.bat
```

### 🎯 **Main Features**
- **🚀 No Auto-Loading**: Services initialize only when needed
- **💾 GPU Memory Efficient**: Load only required TTS engines  
- **🎯 Clean Interface**: 6-8 main options with organized submenus
- **🎙️ Voice Pipeline**: Dedicated STT→LLM→TTS conversation flow

### 📋 **Main Menu Structure**

1. **🎙️ Voice Pipeline** - Full conversation (STT→LLM→TTS)
2. **🔧 Service Management** - Initialize/manage microservices
3. **🧪 Testing & Demos** - Run tests and demonstrations  
4. **🎵 Audio Generation** - Direct TTS text-to-speech
5. **🏥 Health & Diagnostics** - System health and benchmarks
6. **📚 Documentation** - Help and guides

### ⚡ **Service Initialization**
```bash
# Services start unloaded (no GPU memory usage)
# Initialize only what you need:

# Option 1: Through CLI menus
# 2. Service Management → 1. Initialize STT
# 2. Service Management → 3. Initialize TTS Kokoro

# Option 2: Voice Pipeline auto-initializes required services
# 1. Voice Pipeline → 1. Start Voice Conversation
```
# Run dual TTS demonstration
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py demo

# Run specific tests
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py test nari-proper
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py test nari-quick
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py test tts-comparison
```

#### **System Health & Diagnostics**
```bash
# Comprehensive system health check
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py health-check

# Performance benchmarks
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py benchmark --engines all
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py benchmark --engines kokoro
```

### 🎤 **Interactive Mode Commands**

Start interactive mode and use these commands:

- `help` - Show all available commands with descriptions
- `status` - Display TTS engine status and GPU information
- `speak <text>` - Generate speech with current engine
- `auto <text>` - Auto-select best engine and generate speech
- `kokoro` / `nari` - Switch between TTS engines
- `switch` - Interactive engine selection menu
- `test` - Run engine comparison test
- `conversation` - Start voice conversation mode
- `demo` - Run dual TTS demonstration
- `nari-test <type>` - Run specific Nari Dia tests
- `health` - Comprehensive system health check
- `benchmark` - Performance benchmark tests
- `quit` / `exit` - Exit the CLI

### 🎯 **TTS Engine Options**

The CLI supports dual TTS engines with different strengths:

#### **🚀 Kokoro TTS (Fast)**
- **Speed**: ~0.8s generation (real-time capable)
- **Voice**: af_bella (professional African female)
- **Best for**: Real-time conversation, interactive responses
- **Usage**: `--engine kokoro` or `kokoro` in interactive mode

#### **🎭 Nari Dia-1.6B (Quality)**
- **Speed**: ~3+ minutes generation (high quality)
- **Voice**: Adaptive dialogue-focused
- **Best for**: Pre-recorded messages, maximum quality
- **Requirements**: CUDA-enabled GPU
- **Usage**: `--engine nari_dia` or `nari` in interactive mode

### 💡 **Usage Examples**

#### **Quick Testing**
```bash
# Check if everything is working
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py health-check

# Test Kokoro (fast)
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py --text "Testing Kokoro TTS"

# Test performance
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py benchmark --engines kokoro
```

#### **Voice Conversation**
```bash
# Start real-time conversation
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py conversation --engine kokoro

# High-quality conversation (slower)
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py conversation --engine nari_dia
```

#### **Development & Research**
```bash
# Run proven Nari Dia test
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py test nari-proper

# Compare engine performance
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py test tts-comparison

# Full system demonstration
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py demo
```

### 🛠️ **Setup Requirements**

#### **Prerequisites**
- Python 3.11+
- Virtual environment activated (`.venv`)
- CUDA-enabled GPU (for Nari Dia TTS)
- Ollama service running (for LLM)

#### **Quick Setup**
```bash
# Ensure virtual environment is activated
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify setup
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py health-check
```

### 🔧 **Troubleshooting**

#### **Common Issues**
```bash
# If CLI won't start
.venv\Scripts\python.exe voicebot_orchestrator\enhanced_cli.py health-check

# If Nari Dia fails
# Check CUDA: Should show "CUDA available: True"

# If voice conversation has issues
# Try Kokoro first: --engine kokoro
```

#### **Performance Notes**
- **Kokoro**: Real-time capable (~0.8s), perfect for conversation
- **Nari Dia**: High quality (~3+ min), best for non-real-time use
- **Auto mode**: Automatically selects best engine based on context

### Option 2: CLI Demo (Original)
```bash
git clone https://github.com/your-username/voicebot-orchestrator.git
cd voicebot-orchestrator
python run_app.py cli
```

### Option 2: Docker Compose (Production)
```bash
# All services with monitoring
docker-compose --profile monitoring up

# Access dashboards:
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
# - API docs: http://localhost:8000/docs
```

### Option 3: Kubernetes (Enterprise)
```bash
kubectl apply -f k8s/orchestrator-core.yaml
kubectl get pods -n voicebot-orchestrator
```

A modular, real-time voicebot orchestration system built with Python 3.11.

## Sprint 1 - Foundation

This is the initial implementation of the voicebot orchestrator, providing:

- **FastAPI microservice** with WebSocket support for real-time voice conversations
- **Speech-to-Text (STT)** using Whisper for audio transcription
- **Large Language Model (LLM)** using Mistral for conversational AI
- **Text-to-Speech (TTS)** using Kokoro for speech synthesis
- **Session management** for persistent conversation state
- **CLI tooling** for testing and management
- **Comprehensive testing** suite with unit and integration tests

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WebSocket     │    │   Session       │    │   FastAPI       │
│   Client        │◄──►│   Manager       │◄──►│   Server        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Pipeline      │
                       │                 │
                       │  STT ──► LLM    │
                       │   ▲      │      │
                       │   │      ▼      │
                       │  TTS ◄── AI     │
                       └─────────────────┘
```

## Quick Start

### 1. Environment Setup

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Whisper STT Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# Mistral LLM Configuration
MISTRAL_MODEL_PATH=./models/mistral-7b-instruct
MISTRAL_MAX_TOKENS=512
MISTRAL_TEMPERATURE=0.7

# Kokoro TTS Configuration  
KOKORO_VOICE=default
KOKORO_LANGUAGE=en
KOKORO_SPEED=1.0

# Server Configuration
HOST=localhost
PORT=8000
LOG_LEVEL=INFO
```

### 2. Install Dependencies

```bash
pip install fastapi websockets uvicorn numpy
```

### 3. Run the Server

```bash
python -m voicebot_orchestrator.main
```

The server will start at `http://localhost:8000`

### 4. Use the CLI

```bash
# Start a call session
python -m voicebot_orchestrator.cli start-call --session-id test-123

# Test STT functionality
python -m voicebot_orchestrator.cli stt-test --input audio.wav

# Test TTS functionality
python -m voicebot_orchestrator.cli tts-test --text "Hello world" --output test.wav

# List active sessions
python -m voicebot_orchestrator.cli list-sessions

# Show configuration
python -m voicebot_orchestrator.cli config
```

## API Endpoints

### REST API

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /sessions` - List active sessions
- `POST /sessions/{session_id}` - Create a new session
- `DELETE /sessions/{session_id}` - End a session
- `POST /stt/test` - Test STT functionality
- `POST /tts/test` - Test TTS functionality

### WebSocket

- `WS /ws/{session_id}` - Real-time voice conversation endpoint

## WebSocket Protocol

1. **Connect**: Establish WebSocket connection to `/ws/{session_id}`
2. **Send Audio**: Send binary audio data (WAV format recommended)
3. **Receive Audio**: Receive binary audio response
4. **Repeat**: Continue the conversation loop

Example using Python WebSocket client:

```python
import asyncio
import websockets

async def voice_chat():
    uri = "ws://localhost:8000/ws/test-session"
    
    async with websockets.connect(uri) as websocket:
        # Read audio file
        with open("input.wav", "rb") as f:
            audio_data = f.read()
        
        # Send audio
        await websocket.send(audio_data)
        
        # Receive response
        response_audio = await websocket.recv()
        
        # Save response
        with open("response.wav", "wb") as f:
            f.write(response_audio)

asyncio.run(voice_chat())
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test suites
python tests/run_tests.py session stt llm tts
python tests/run_tests.py integration
```

Individual test modules:

```bash
python tests/test_session_manager.py
python tests/test_stt.py
python tests/test_llm.py
python tests/test_tts.py
python tests/test_integration.py
```

## Project Structure

```
voicebot_orchestrator/
├── voicebot_orchestrator/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── session_manager.py   # Session state management
│   ├── stt.py              # Speech-to-Text (Whisper)
│   ├── llm.py              # Large Language Model (Mistral)
│   ├── tts.py              # Text-to-Speech (Kokoro)
│   └── cli.py              # Command-line interface
├── tests/
│   ├── conftest.py         # Test configuration
│   ├── test_session_manager.py
│   ├── test_stt.py
│   ├── test_llm.py
│   ├── test_tts.py
│   ├── test_integration.py
│   └── run_tests.py        # Test runner
├── .env.example            # Environment configuration template
└── README.md               # This file
```

## Configuration

The system uses environment variables for configuration. Key settings:

### STT (Speech-to-Text)
- `WHISPER_MODEL`: Whisper model size (tiny, base, small, medium, large)
- `WHISPER_DEVICE`: Device for inference (cpu, cuda)

### LLM (Language Model)
- `MISTRAL_MODEL_PATH`: Path to Mistral model files
- `MISTRAL_MAX_TOKENS`: Maximum tokens to generate
- `MISTRAL_TEMPERATURE`: Sampling temperature (0.0-2.0)

### TTS (Text-to-Speech)
- `KOKORO_VOICE`: Voice profile (default, male, female, neutral)
- `KOKORO_LANGUAGE`: Language code (en, es, fr, de, etc.)
- `KOKORO_SPEED`: Speech speed multiplier (0.5-2.0)

### Server
- `HOST`: Server host address
- `PORT`: Server port number
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Sessions
- `SESSION_TIMEOUT`: Session timeout in seconds
- `MAX_CONCURRENT_SESSIONS`: Maximum concurrent sessions

## Banking Domain Features

The LLM is configured for banking domain conversations with:

- **Account balance inquiries**
- **Transaction history requests**
- **General banking information**
- **Customer service interactions**
- **Input validation** for security (filters sensitive information)

## Error Handling

The system includes comprehensive error handling:

- **Audio validation** for STT input
- **Text validation** for LLM and TTS input
- **Session management** with timeout and cleanup
- **WebSocket connection** error recovery
- **Concurrent request** limiting with semaphores

## Security Features

- **Input sanitization** to prevent injection attacks
- **Content filtering** to reject sensitive information (passwords, SSN, etc.)
- **Session isolation** with unique session IDs
- **Rate limiting** with concurrent request semaphores
- **No hardcoded secrets** in the codebase

## Performance Optimizations

- **Lazy model loading** to reduce startup time
- **Async processing** for non-blocking operations
- **Connection pooling** for WebSocket management
- **Memory management** with conversation history limits
- **Efficient audio processing** with numpy operations

## Future Enhancements (Next Sprints)

Sprint 1 provides the foundation. Upcoming sprints will add:

- **Analytics and monitoring** (Prometheus/Grafana)
- **Semantic caching** for improved performance
- **LoRA adapter training** for domain customization
- **Advanced compliance** features
- **Microservices architecture** expansion
- **Production deployment** configurations

## 📚 Quick Reference

### **Most Common Commands**
```bash
# Start Modular CLI
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py

# Check service status
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py --status

# Health check
.venv\Scripts\python.exe voicebot_orchestrator\modular_cli.py --health-check

# Quick launcher
voicebot_cli.bat
```

### **Service Management Strategy**
- **🚀 Start Clean**: No services loaded = no GPU memory usage
- **🎯 Load on Demand**: Initialize only what you need
- **💾 Memory Efficient**: Avoid loading both TTS engines simultaneously
- **🔄 Smart Pipeline**: Voice conversation auto-initializes required services

### **Main Workflow**
1. **Start CLI**: `voicebot_cli.bat` or `modular_cli.py`
2. **Choose Pipeline**: Option 1 (Voice Pipeline) for conversation
3. **Auto-Initialize**: System loads STT, LLM, and preferred TTS
4. **Clean Exit**: Services automatically cleaned up

### **TTS Engine Selection**
- **🚀 Kokoro**: Fast (~0.8s) - Recommended for conversation
- **🎭 Nari Dia**: Quality (~3min) - Use for high-quality recordings only
- **⚠️ Memory Warning**: Loading both engines uses ~8GB GPU memory

## License

This project is part of the Orkestra voicebot orchestration system.

## Support

For issues and questions, please refer to the test suite and documentation.
