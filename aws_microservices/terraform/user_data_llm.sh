#!/bin/bash
# User data script for LLM service (GPU-optimized)

# Update system
apt-get update
apt-get upgrade -y

# Install NVIDIA drivers for GPU instances
ubuntu-drivers autoinstall

# Install Docker
apt-get install -y docker.io
systemctl start docker
systemctl enable docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /app/voicebot-llm
cd /app/voicebot-llm

# Download application code (replace with your actual repository)
# git clone https://github.com/xxsputzz/voicebot-orchestrator.git .

# Create Docker Compose file for LLM service
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  llm-service:
    build:
      context: .
      dockerfile: aws_microservices/Dockerfile.llm
    ports:
      - "${service_port}:${service_port}"
    environment:
      - SERVICE_PORT=${service_port}
      - LOG_LEVEL=INFO
      - CUDA_VISIBLE_DEVICES=0
    runtime: nvidia
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:${service_port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s  # LLM takes longer to start
EOF

# Start the service
docker-compose up -d

# Setup log rotation
cat > /etc/logrotate.d/voicebot-llm << 'EOF'
/var/log/voicebot-llm.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF

# Setup monitoring script
cat > /usr/local/bin/monitor-llm.sh << 'EOF'
#!/bin/bash
# Monitoring script for LLM service with GPU checks

SERVICE_URL="http://localhost:${service_port}/health"
LOG_FILE="/var/log/voicebot-llm.log"

while true; do
    # Check service health
    if curl -f $SERVICE_URL > /dev/null 2>&1; then
        # Check GPU utilization
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        echo "$(date): LLM service healthy - GPU: ${GPU_UTIL}% util, ${GPU_MEM}MB used" >> $LOG_FILE
    else
        echo "$(date): LLM service unhealthy - restarting" >> $LOG_FILE
        cd /app/voicebot-llm && docker-compose restart
    fi
    sleep 60
done
EOF

chmod +x /usr/local/bin/monitor-llm.sh

# Create systemd service for monitoring
cat > /etc/systemd/system/voicebot-llm-monitor.service << 'EOF'
[Unit]
Description=Voicebot LLM Service Monitor
After=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/monitor-llm.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl enable voicebot-llm-monitor.service
systemctl start voicebot-llm-monitor.service

echo "LLM service setup completed" >> /var/log/user-data.log
