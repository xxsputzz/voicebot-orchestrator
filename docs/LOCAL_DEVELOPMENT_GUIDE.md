# Local Microservices Development Guide

This guide helps you run and test the microservices architecture locally on a single machine before deploying to AWS.

## Quick Start

### Option 1: Windows Batch Script (Easiest)
```bash
# Run the test script (installs dependencies and runs tests)
test_local.bat
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install requests fastapi uvicorn aiohttp aiofiles

# Run test suite
python test_local_microservices.py
```

### Option 3: Docker Compose (Recommended for isolation)
```bash
# Build and start all services
docker-compose -f aws_microservices/docker-compose.local.yml up -d

# Check service health
curl http://localhost:8001/health  # STT Service
curl http://localhost:8002/health  # LLM Service  
curl http://localhost:8003/health  # TTS Service

# Stop services
docker-compose -f aws_microservices/docker-compose.local.yml down
```

## Services Overview

### Local Service Ports
- **STT Service**: http://localhost:8001 (Speech-to-Text)
- **LLM Service**: http://localhost:8002 (Language Model)
- **TTS Service**: http://localhost:8003 (Text-to-Speech)

### Service Endpoints
Each service provides:
- `GET /health` - Health check
- `POST /[service-specific]` - Main functionality
  - STT: `/transcribe` (upload audio file)
  - LLM: `/generate` (send text, get response)
  - TTS: `/synthesize` (send text, get audio)

## Testing Methods

### 1. Interactive Test Suite
```bash
python test_local_microservices.py
```
Choose from:
1. Docker Compose testing (recommended)
2. Python runner testing
3. Orchestrator client testing
4. Exit

### 2. Manual Service Testing

#### Test STT Service
```python
import requests

# Upload audio file
with open("audio.wav", "rb") as f:
    files = {"audio": f}
    response = requests.post("http://localhost:8001/transcribe", files=files)
    print(response.json())
```

#### Test LLM Service
```python
import requests

payload = {"text": "Hello, how are you?", "model": "mistral"}
response = requests.post("http://localhost:8002/generate", json=payload)
print(response.json())
```

#### Test TTS Service
```python
import requests

payload = {"text": "Hello world", "engine": "pyttsx3"}
response = requests.post("http://localhost:8003/synthesize", json=payload)
print("Audio generated:", len(response.json()["audio"]))
```

### 3. End-to-End Pipeline Test
```python
# Test complete voice pipeline
from aws_microservices.orchestrator_client import example_voice_conversation
import asyncio

asyncio.run(example_voice_conversation())
```

## Development Workflow

### 1. Start Services Locally
```bash
# Option A: Docker Compose (isolated environment)
docker-compose -f aws_microservices/docker-compose.local.yml up -d

# Option B: Python runner (development mode)
python aws_microservices/local_runner.py
```

### 2. Develop and Test
- Edit service code in `aws_microservices/`
- Use test suite to validate changes
- Check logs for debugging

### 3. Prepare for AWS Deployment
- Test all services work together locally
- Verify Docker builds succeed
- Review AWS configuration in `terraform/`

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
netstat -ano | findstr :8001

# Kill process if needed
taskkill /PID [process_id] /F
```

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt
pip install fastapi uvicorn aiohttp aiofiles requests
```

#### Docker Issues
```bash
# Rebuild containers
docker-compose -f aws_microservices/docker-compose.local.yml build --no-cache

# Check container logs
docker-compose -f aws_microservices/docker-compose.local.yml logs [service_name]
```

#### Service Not Starting
1. Check logs in `aws_microservices/local_runner.py` output
2. Verify all dependencies are installed
3. Ensure model files exist (kokoro-v1.0.onnx, voices-v1.0.bin)
4. Check firewall/antivirus blocking ports

### Debug Commands

```bash
# Check service health
curl http://localhost:8001/health
curl http://localhost:8002/health  
curl http://localhost:8003/health

# View service logs
docker-compose -f aws_microservices/docker-compose.local.yml logs -f

# Test individual services
python -c "import requests; print(requests.get('http://localhost:8001/health').json())"
```

## Performance Notes

### Local vs AWS Performance
- **Local**: Good for development, limited by single machine resources
- **AWS**: Better scaling, GPU acceleration, geographic distribution

### Resource Usage
- **STT**: CPU-intensive (Whisper model)
- **LLM**: GPU-preferred (Mistral model) 
- **TTS**: Mixed CPU/GPU (Kokoro + pyttsx3)

### Optimization Tips
1. Use lighter models for local development
2. Enable GPU acceleration if available
3. Adjust batch sizes for your hardware
4. Monitor memory usage with multiple services

## Next Steps

Once local testing is successful:

1. **Deploy to AWS**: Use the Terraform configuration in `aws_microservices/terraform/`
2. **Scale Services**: Configure auto-scaling groups for production load
3. **Monitor Performance**: Set up CloudWatch metrics and alerts
4. **Optimize Costs**: Use spot instances and auto-scaling to manage expenses

## Files Created for Local Development

- `aws_microservices/docker-compose.local.yml` - Docker Compose configuration
- `aws_microservices/local_runner.py` - Python process manager
- `test_local_microservices.py` - Comprehensive test suite
- `test_local.bat` - Windows batch script for easy testing
- `orchestrator_client.py` - Updated with local configuration support

This local setup provides a complete development environment that mirrors the AWS architecture while running on a single machine for testing and development.
