# Independent Microservices Architecture

This setup provides completely independent TTS and LLM services that can be run individually or in any combination. Each service is self-contained and can operate without the others.

## 🎯 **Architecture Overview**

### **Service Independence**
- **Kokoro TTS Service** (Port 8011) - Fast, real-time speech synthesis
- **Hira Dia TTS Service** (Port 8012) - High-quality, GPU-accelerated synthesis  
- **Mistral LLM Service** (Port 8021) - Efficient language model
- **GPT LLM Service** (Port 8022) - Advanced reasoning capabilities
- **STT Service** (Port 8001) - Speech-to-text (shared across all combinations)

### **Benefits of Independence**
✅ **Selective Resource Usage** - Run only what you need  
✅ **Independent Scaling** - Scale services based on demand  
✅ **Fault Isolation** - One service failure doesn't affect others  
✅ **Technology Flexibility** - Different engines can use optimal configurations  
✅ **Cost Optimization** - Pay only for active services  

## 🚀 **Quick Start**

### **Option 1: Interactive Service Manager**
```bash
# Start the interactive manager
python aws_microservices/independent_local_runner.py --interactive

# Or start specific services
python aws_microservices/independent_local_runner.py --start kokoro_tts mistral_llm
```

### **Option 2: Docker Compose Profiles**
```bash
# Start Kokoro TTS + Mistral LLM combination
docker-compose -f aws_microservices/docker-compose.independent.yml --profile kokoro --profile mistral up -d

# Start Hira Dia TTS + GPT LLM combination  
docker-compose -f aws_microservices/docker-compose.independent.yml --profile hira-dia --profile gpt up -d

# Start all services
docker-compose -f aws_microservices/docker-compose.independent.yml --profile kokoro --profile hira-dia --profile mistral --profile gpt up -d
```

### **Option 3: Manual Service Startup**
```bash
# Start individual services manually
python aws_microservices/tts_kokoro_service.py      # Port 8011
python aws_microservices/tts_hira_dia_service.py    # Port 8012
python aws_microservices/llm_mistral_service.py     # Port 8021
python aws_microservices/llm_gpt_service.py         # Port 8022
```

## 🎭 **Service Details**

### **TTS Services**

#### **Kokoro TTS Service (Port 8011)**
- **Speed**: ~0.8s generation time
- **Quality**: Good
- **Voice**: af_bella (professional female)
- **Best For**: Real-time conversation, quick responses
- **Resource**: CPU-optimized
- **Use Cases**: Live chat, demonstrations, interactive apps

```python
# Test Kokoro TTS
import requests

response = requests.post("http://localhost:8011/synthesize", json={
    "text": "Hello, this is Kokoro speaking!",
    "voice": "af_bella"
})
```

#### **Unified Hira Dia TTS Service (Port 8012)** 🆕
- **Engines**: Dual-mode support (Full Dia + 4-bit Dia)
- **Speed**: 
  - Full Dia: ~3+ minutes generation time
  - 4-bit Dia: ~30-60 seconds generation time
- **Quality**: 
  - Full Dia: Maximum quality
  - 4-bit Dia: Optimized quality
- **Voice**: Adaptive dialogue-focused
- **Features**: Auto engine selection, runtime switching, quality/speed optimization
- **Best For**: Flexible use cases requiring both quality and speed
- **Resource**: GPU required
- **Use Cases**: Content creation, professional presentations, speed-optimized applications

```python
# Test Unified Hira Dia TTS with engine preferences
import requests

# High quality synthesis (Full Dia)
response = requests.post("http://localhost:8012/synthesize", json={
    "text": "This is Full Dia with maximum quality synthesis.",
    "engine_preference": "full",
    "high_quality": True
})

# Speed-optimized synthesis (4-bit Dia)
response = requests.post("http://localhost:8012/synthesize", json={
    "text": "This is 4-bit Dia with speed optimization.",
    "engine_preference": "4bit",
    "high_quality": False
})

# Auto selection based on text length and quality preference
response = requests.post("http://localhost:8012/synthesize", json={
    "text": "Auto-selected engine based on text and preferences.",
    "engine_preference": "auto",
    "high_quality": True  # Bias towards quality
})
```

### **LLM Services**

#### **Mistral LLM Service (Port 8021)**
- **Model**: Mistral Latest
- **Speed**: 2-5 seconds response time
- **Features**: Semantic cache, LoRA adapters, banking domain
- **Best For**: General conversation, customer service
- **Resource**: GPU-preferred
- **Use Cases**: Chat bots, Q&A, domain-specific tasks

```python
# Test Mistral LLM
import requests

response = requests.post("http://localhost:8021/generate", json={
    "text": "What are the benefits of microservices?",
    "use_cache": True
})
```

#### **GPT LLM Service (Port 8022)**
- **Model**: GPT Open Source 20B
- **Speed**: 5-15 seconds response time
- **Features**: Advanced reasoning, chat completion API
- **Best For**: Complex problem solving, creative tasks
- **Resource**: High GPU memory (16GB+ recommended)
- **Use Cases**: Technical analysis, creative writing, complex queries

```python
# Test GPT LLM
import requests

response = requests.post("http://localhost:8022/generate", json={
    "text": "Explain the architectural benefits of independent microservices.",
    "max_tokens": 200
})
```

## 🎯 **Service Combinations**

### **Fast Combination (Real-time)**
- **Kokoro TTS** + **Mistral LLM**
- **Total Time**: ~3-6 seconds
- **Use Case**: Live conversations, demos
- **Resource**: Moderate GPU/CPU

### **Quality Combination (Professional)**
- **Unified Hira Dia TTS (Full mode)** + **GPT LLM**  
- **Total Time**: ~8+ minutes
- **Use Case**: High-quality content, presentations
- **Resource**: High GPU memory

### **Speed-Optimized Quality (NEW)**
- **Unified Hira Dia TTS (4-bit mode)** + **Mistral LLM**
- **Total Time**: ~1-2 minutes
- **Use Case**: Professional quality with reasonable speed
- **Resource**: Moderate GPU memory

### **Adaptive Combinations (NEW)**
- **Unified Hira Dia TTS (Auto mode)** + **Any LLM**: Smart quality/speed selection
- **Kokoro TTS** + **GPT LLM**: Fast TTS, advanced reasoning
- **Unified Hira Dia TTS** + **Mistral LLM**: Flexible quality with efficient reasoning

## 🛠 **Service Management**

### **Independent Service Manager**
```bash
# Interactive mode
python aws_microservices/independent_local_runner.py -i

# Commands in interactive mode:
> start kokoro_tts      # Start Kokoro TTS
> start mistral_llm     # Start Mistral LLM  
> status               # Show all service status
> stop hira_dia_tts    # Stop Hira Dia TTS
> start all            # Start all services
> stop all             # Stop all services
> quit                 # Exit
```

### **Service Status Monitoring**
```bash
# Check service health
curl http://localhost:8011/health  # Kokoro TTS
curl http://localhost:8012/health  # Hira Dia TTS
curl http://localhost:8021/health  # Mistral LLM
curl http://localhost:8022/health  # GPT LLM

# Get service information
curl http://localhost:8011/info    # Kokoro details
curl http://localhost:8021/info    # Mistral details
```

## 🧪 **Testing**

### **Comprehensive Test Suite**
```bash
# Run the independent services test
python test_independent_services.py

# Options:
# 1. Check running services
# 2. Test all running services  
# 3. Test service combinations
# 4. Test independent orchestrator
# 5. Start service manager
# 6. Run comprehensive test
```

### **Individual Service Testing**
```python
# Test specific TTS engine
import requests

# Kokoro TTS
kokoro_response = requests.post("http://localhost:8011/synthesize", json={
    "text": "Testing Kokoro engine"
})

# Hira Dia TTS
hira_response = requests.post("http://localhost:8012/synthesize", json={
    "text": "Testing Hira Dia engine"
})

# Test specific LLM model
# Mistral LLM
mistral_response = requests.post("http://localhost:8021/generate", json={
    "text": "Hello Mistral"
})

# GPT LLM
gpt_response = requests.post("http://localhost:8022/generate", json={
    "text": "Hello GPT"
})
```

## 🎬 **Using the Independent Orchestrator**

```python
from aws_microservices.independent_orchestrator import IndependentServicesOrchestrator, get_local_service_config
import asyncio

async def example_usage():
    # Configure which services to use
    config = get_local_service_config()
    
    async with IndependentServicesOrchestrator(config) as orchestrator:
        # Check which services are available
        health = await orchestrator.health_check_all()
        print("Available services:", [s for s, h in health.items() if h['status'] == 'healthy'])
        
        # Process voice with specific engines
        result = await orchestrator.process_voice_pipeline(
            audio_data=audio_bytes,
            tts_engine="kokoro",    # or "hira_dia"
            llm_model="mistral"     # or "gpt"
        )
        
        if result["success"]:
            print(f"Response: {result['response_text']}")
            # Audio data available in result['audio_data']

# Run the example
asyncio.run(example_usage())
```

## 📊 **Resource Requirements**

### **Minimum Requirements**
- **Kokoro TTS**: 2GB RAM, CPU
- **Mistral LLM**: 4GB RAM, CPU (GPU recommended)
- **STT Service**: 2GB RAM, CPU

### **Recommended Requirements**
- **Hira Dia TTS**: 8GB GPU memory, CUDA
- **GPT LLM**: 16GB GPU memory, CUDA
- **Full Setup**: 24GB+ GPU memory for all services

### **Production Scaling**
- **Start Small**: Kokoro + Mistral (fast, efficient)
- **Scale Quality**: Add Hira Dia for premium features
- **Scale Intelligence**: Add GPT for complex tasks
- **Independent Scaling**: Scale each service based on demand

## 🚀 **Deployment Options**

### **Local Development**
```bash
# Start development services
python aws_microservices/independent_local_runner.py --start kokoro_tts mistral_llm
```

### **Docker Local**
```bash
# Docker with specific profiles
docker-compose -f aws_microservices/docker-compose.independent.yml --profile kokoro --profile mistral up -d
```

### **AWS Deployment**
- Each service can be deployed independently
- Use separate EC2 instances or ECS tasks
- Auto-scaling groups per service type
- Load balancers for high availability

## 🎯 **Use Case Examples**

### **Customer Service Bot**
- **Services**: STT + Mistral LLM + Kokoro TTS
- **Benefit**: Fast response, good quality, cost-effective

### **Premium Content Creation**
- **Services**: STT + GPT LLM + Hira Dia TTS  
- **Benefit**: Maximum quality output, advanced reasoning

### **Live Demo System**
- **Services**: STT + Mistral LLM + Kokoro TTS
- **Benefit**: Real-time performance, reliable operation

### **Multi-Tenant Platform**
- **Services**: All services available
- **Benefit**: Users choose quality/speed trade-offs per request

## 🔧 **Troubleshooting**

### **Service Won't Start**
```bash
# Check logs
python aws_microservices/tts_kokoro_service.py  # See startup errors

# Check dependencies
pip install fastapi uvicorn aiohttp aiofiles requests

# Check ports
netstat -ano | findstr :8011  # Windows
lsof -i :8011                 # Linux/Mac
```

### **GPU Services Failing**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Use CPU fallback for development
# (Hira Dia and GPT will be slower but functional)
```

### **Import Errors**
```bash
# Install missing packages
pip install aiohttp  # For orchestrator
pip install requests  # For testing
pip install torch torchvision torchaudio  # For GPU services
```

This independent architecture gives you complete flexibility to run exactly the services you need, when you need them, with optimal resource utilization and fault isolation.
