# AWS Microservices Deployment Guide

This guide walks you through deploying your voicebot orchestrator as microservices on AWS.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     AWS Multi-Machine Setup                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  STT Service    │  │  LLM Service    │  │  TTS Service    │ │
│  │  c5.xlarge      │  │  g4dn.xlarge    │  │  g4dn.xlarge    │ │
│  │  CPU-optimized  │  │  GPU-optimized  │  │  GPU-optimized  │ │
│  │  Port 8001      │  │  Port 8002      │  │  Port 8003      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                     │                     │        │
│           └─────────────────────┼─────────────────────┘        │
│                                │                               │
│                   ┌─────────────────┐                         │
│                   │ Load Balancer   │                         │
│                   │ Internal ALB    │                         │
│                   └─────────────────┘                         │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## 💰 Cost Analysis

### Monthly Cost Estimates (us-east-1):

**Option 1: Dedicated Instances**
- STT (c5.xlarge): $122.40/month
- LLM (g4dn.xlarge): $113.88/month  
- TTS (g4dn.xlarge): $113.88/month
- **Total: ~$350/month**

**Option 2: Spot Instances (60-90% savings)**
- STT (c5.xlarge spot): $30-50/month
- LLM (g4dn.xlarge spot): $30-50/month
- TTS (g4dn.xlarge spot): $30-50/month
- **Total: ~$90-150/month**

**Option 3: Auto-scaling (Recommended)**
- Scale down to 1 instance each during low usage
- Scale up during peak usage
- **Estimated: $200-400/month** depending on usage

## 🚀 Deployment Steps

### Prerequisites

1. **AWS Account with appropriate permissions**
2. **Terraform installed** (>= 1.0)
3. **AWS CLI configured**
4. **EC2 Key Pair created**
5. **Docker images built** (optional - can build on instances)

### Step 1: Prepare the Code

```bash
# Clone your repository
git clone https://github.com/xxsputzz/voicebot-orchestrator.git
cd voicebot-orchestrator

# Build Docker images locally (optional)
cd aws_microservices
docker build -f Dockerfile.stt -t voicebot-stt .
docker build -f Dockerfile.llm -t voicebot-llm .
docker build -f Dockerfile.tts -t voicebot-tts .

# Push to ECR or Docker Hub (if building locally)
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
# docker tag voicebot-stt:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/voicebot-stt:latest
# docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/voicebot-stt:latest
```

### Step 2: Configure Terraform

```bash
cd terraform

# Create terraform.tfvars file
cat > terraform.tfvars << EOF
aws_region     = "us-east-1"
environment    = "production"
key_pair_name  = "your-ec2-keypair"  # Replace with your key pair name
EOF

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply the infrastructure
terraform apply
```

### Step 3: Configure Service Endpoints

After Terraform deployment, update your orchestrator client:

```python
# Update orchestrator_client.py with your actual endpoints
def get_aws_service_config() -> Dict[str, ServiceEndpoint]:
    return {
        "stt": ServiceEndpoint(
            name="stt",
            host="voicebot-production-lb-123456789.us-east-1.elb.amazonaws.com",  # From Terraform output
            port=8001
        ),
        "llm": ServiceEndpoint(
            name="llm", 
            host="voicebot-production-lb-123456789.us-east-1.elb.amazonaws.com",  # From Terraform output
            port=8002
        ),
        "tts": ServiceEndpoint(
            name="tts",
            host="voicebot-production-lb-123456789.us-east-1.elb.amazonaws.com",  # From Terraform output
            port=8003
        )
    }
```

### Step 4: Test the Deployment

```bash
# Install orchestrator dependencies
pip install -r requirements-orchestrator.txt

# Test the microservices
python orchestrator_client.py
```

## 🔧 Configuration Options

### Environment Variables

Each service supports these environment variables:

**Common:**
- `LOG_LEVEL`: INFO, DEBUG, WARNING, ERROR
- `SERVICE_PORT`: Port number (8001/8002/8003)

**LLM Service:**
- `MODEL_TYPE`: mistral, gpt-oss
- `ENABLE_CACHE`: true, false
- `CACHE_SIZE`: 50000 (default)

**TTS Service:**
- `DEFAULT_ENGINE`: kokoro, nari_dia, auto
- `ENABLE_NARI`: true, false (GPU required)

### Auto Scaling Configuration

```bash
# Scale up STT service
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name voicebot-stt-production-asg \
  --desired-capacity 2

# Scale down LLM service  
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name voicebot-llm-production-asg \
  --desired-capacity 1
```

## 📊 Monitoring & Management

### Health Checks

```bash
# Check service health
curl http://your-load-balancer:8001/health  # STT
curl http://your-load-balancer:8002/health  # LLM  
curl http://your-load-balancer:8003/health  # TTS
```

### Performance Monitoring

```bash
# LLM performance metrics
curl http://your-load-balancer:8002/performance

# TTS engine status
curl http://your-load-balancer:8003/engines

# Service info
curl http://your-load-balancer:8001/info
```

### Logs

```bash
# SSH into instances to check logs
ssh -i your-key.pem ubuntu@instance-ip

# Check service logs
docker-compose logs -f

# Check monitoring logs
tail -f /var/log/voicebot-*.log
```

## 🔄 CI/CD Pipeline (Optional)

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Deploy Voicebot Microservices

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Build and push Docker images
      run: |
        # Build and push to ECR
        aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
        docker build -f aws_microservices/Dockerfile.stt -t $ECR_REGISTRY/voicebot-stt:latest .
        docker push $ECR_REGISTRY/voicebot-stt:latest
    
    - name: Deploy with Terraform
      run: |
        cd terraform
        terraform init
        terraform apply -auto-approve
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Restart NVIDIA Docker
   sudo systemctl restart docker
   ```

2. **Service not starting**
   ```bash
   # Check Docker logs
   docker-compose logs service-name
   
   # Check system resources
   htop
   nvidia-smi
   ```

3. **High costs**
   ```bash
   # Use spot instances
   # Scale down during off-hours
   # Implement auto-shutdown for dev environments
   ```

### Performance Optimization

1. **STT Service**: Use faster Whisper models (base vs large)
2. **LLM Service**: Enable caching, use LoRA adapters
3. **TTS Service**: Use Kokoro for real-time, Nari Dia for quality

## 🔐 Security Considerations

1. **Network Security**: Services communicate internally via load balancer
2. **Instance Security**: Regular security updates, minimal packages
3. **Secrets Management**: Use AWS Secrets Manager for API keys
4. **Access Control**: IAM roles with minimal permissions

## 📈 Scaling Strategies

### Horizontal Scaling
- Add more instances to auto-scaling groups
- Use Application Load Balancer for distribution

### Vertical Scaling  
- Upgrade instance types (c5.xlarge → c5.2xlarge)
- More GPU memory for TTS (g4dn.xlarge → g4dn.2xlarge)

### Geographic Scaling
- Deploy in multiple AWS regions
- Use Route 53 for geographic routing

## 🎯 Next Steps

1. **Deploy and test** the basic setup
2. **Monitor performance** and costs
3. **Optimize** based on real usage patterns
4. **Add monitoring** (CloudWatch, Grafana)
5. **Implement CI/CD** for automated deployments
6. **Consider** managed services (EKS, Fargate) for easier management

This microservices approach will give you:
- **Independent scaling** of each service
- **Fault isolation** (one service failure doesn't break others)
- **Technology flexibility** (different optimizations per service)
- **Cost optimization** (right-size each service type)
- **Easier maintenance** (update services independently)
