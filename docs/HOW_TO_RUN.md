# 🚀 How to Run the Voicebot Orchestration Platform

## Quick Start Guide

Your **Sprint 6 Enterprise Voicebot Orchestration Platform** is complete and ready to run! Here are all the ways you can deploy and use it:

## 🎯 **Option 1: Quick CLI Demo (Recommended for Testing)**

Use the built-in application runner for immediate testing:

```bash
# Run CLI demonstration
python run_app.py cli

# Run comprehensive demo
python run_app.py demo

# Show deployment options
python run_app.py deploy

# Install missing dependencies
python run_app.py install
```

## 🎮 **Option 2: Direct CLI Commands**

Use the CLI directly for specific operations:

```bash
# System health check
python -m voicebot_orchestrator.sprint6_cli orchestrator-health

# Start a banking session
python -m voicebot_orchestrator.sprint6_cli start-call banking-session-001 --phone +1234567890 --domain banking

# Monitor a session
python -m voicebot_orchestrator.sprint6_cli monitor-session --session-id banking-session-001

# Generate analytics report
python -m voicebot_orchestrator.sprint6_cli analytics-report --type summary --time-range 24h

# Check cache statistics
python -m voicebot_orchestrator.sprint6_cli cache-manager stats

# List LoRA adapters
python -m voicebot_orchestrator.sprint6_cli adapter-control list
```

## 🐳 **Option 3: Docker Compose (Production Ready)**

For production deployment with all services:

```bash
# Basic services (6 microservices + Redis)
docker-compose up

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up

# With load balancer (NGINX + monitoring)
docker-compose --profile loadbalancer up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f orchestrator-core
```

## ☸️ **Option 4: Kubernetes Deployment**

For enterprise auto-scaling deployment:

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/orchestrator-core.yaml

# Check deployment status
kubectl get pods -n voicebot-orchestrator

# View service status
kubectl get services -n voicebot-orchestrator

# Check logs
kubectl logs -f deployment/orchestrator-core -n voicebot-orchestrator

# Scale services
kubectl scale deployment orchestrator-core --replicas=5 -n voicebot-orchestrator
```

## 🔧 **Option 5: Individual Microservices**

Run services individually for development:

```bash
# Core orchestrator (port 8000)
python -m voicebot_orchestrator.microservices.orchestrator_core

# Speech-to-Text service (port 8001)
python -m voicebot_orchestrator.microservices.stt_service

# Language Model service (port 8002)
python -m voicebot_orchestrator.microservices.llm_service

# Text-to-Speech service (port 8003)
python -m voicebot_orchestrator.microservices.tts_service

# Cache service (port 8004)
python -m voicebot_orchestrator.microservices.cache_service

# Analytics service (port 8005)
python -m voicebot_orchestrator.microservices.analytics_service
```

## 📦 **Option 6: Package Installation**

Install as a Python package:

```bash
# With pip
pip install voicebot-orchestrator

# With poetry
poetry install voicebot-orchestrator

# Then use CLI commands directly
orchestrator orchestrator-health
orchestrator start-call session-001 --phone +1234567890 --domain banking
```

## 🎯 **Recommended Workflow**

### For Development & Testing:
1. **Start with CLI demo**: `python run_app.py cli`
2. **Run comprehensive demo**: `python run_app.py demo`
3. **Test individual commands**: Use direct CLI commands

### For Production Deployment:
1. **Local production-like**: `docker-compose --profile monitoring up`
2. **Enterprise deployment**: Use Kubernetes manifests
3. **Monitoring**: Access Grafana at `http://localhost:3000`

## 🏥 **Health Checks**

Always start by checking system health:

```bash
# CLI health check
python -m voicebot_orchestrator.sprint6_cli orchestrator-health

# Docker health check
docker-compose ps

# Kubernetes health check
kubectl get pods -n voicebot-orchestrator
```

## 📊 **Monitoring & Analytics**

Once running, you can:

- **View metrics**: Prometheus at `http://localhost:9090`
- **Dashboards**: Grafana at `http://localhost:3000`
- **API docs**: FastAPI docs at `http://localhost:8000/docs`
- **System status**: CLI health commands

## 🔧 **Configuration**

Customize deployment with environment variables:

```bash
# Set environment
export ENVIRONMENT=production
export ORCHESTRATOR_PORT=8000
export LOG_LEVEL=INFO

# Redis configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Service URLs
export STT_SERVICE_URL=http://localhost:8001
export LLM_SERVICE_URL=http://localhost:8002
export TTS_SERVICE_URL=http://localhost:8003
```

## 🎉 **You're Ready!**

Your **Sprint 6 Enterprise Voicebot Orchestration Platform** includes:

✅ **6 Microservices** - Scalable architecture  
✅ **Enterprise CLI** - 6 command groups  
✅ **Docker & Kubernetes** - Production deployment  
✅ **Monitoring** - Prometheus + Grafana  
✅ **Load Balancing** - NGINX reverse proxy  
✅ **Auto-scaling** - Kubernetes HPA  
✅ **Health Checks** - Comprehensive monitoring  
✅ **Analytics** - Performance reporting  

**Start with**: `python run_app.py cli` to see everything in action!
