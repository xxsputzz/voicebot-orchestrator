#!/bin/bash
# User data script for TTS service (High-GPU)

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
mkdir -p /app/voicebot-tts
cd /app/voicebot-tts

# Download application code (replace with your actual repository)
# git clone https://github.com/xxsputzz/voicebot-orchestrator.git .

# Create Docker Compose file for TTS service
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  tts-service:
    build:
      context: .
      dockerfile: aws_microservices/Dockerfile.tts
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
      start_period: 180s  # TTS takes longest to start (model loading)
    volumes:
      - tts_models:/app/models
    
volumes:
  tts_models:
EOF

# Start the service
docker-compose up -d

# Setup log rotation
cat > /etc/logrotate.d/voicebot-tts << 'EOF'
/var/log/voicebot-tts.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF

# Setup monitoring script with TTS-specific checks
cat > /usr/local/bin/monitor-tts.sh << 'EOF'
#!/bin/bash
# Monitoring script for TTS service with GPU and generation checks

SERVICE_URL="http://localhost:${service_port}/health"
LOG_FILE="/var/log/voicebot-tts.log"

while true; do
    # Check service health
    if curl -f $SERVICE_URL > /dev/null 2>&1; then
        # Check GPU utilization and memory
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
        
        # Check available engines
        ENGINES=$(curl -s http://localhost:${service_port}/engines | jq -r '.available_engines[] | .name' | tr '\n' ',' | sed 's/,$//')
        
        echo "$(date): TTS service healthy - GPU: ${GPU_UTIL}% util, ${GPU_MEM}MB used, ${GPU_TEMP}°C - Engines: $ENGINES" >> $LOG_FILE
        
        # Alert if GPU temp is too high
        if [ "$GPU_TEMP" -gt 85 ]; then
            echo "$(date): WARNING - High GPU temperature: ${GPU_TEMP}°C" >> $LOG_FILE
        fi
    else
        echo "$(date): TTS service unhealthy - restarting" >> $LOG_FILE
        cd /app/voicebot-tts && docker-compose restart
    fi
    sleep 60
done
EOF

chmod +x /usr/local/bin/monitor-tts.sh

# Create systemd service for monitoring
cat > /etc/systemd/system/voicebot-tts-monitor.service << 'EOF'
[Unit]
Description=Voicebot TTS Service Monitor
After=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/monitor-tts.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl enable voicebot-tts-monitor.service
systemctl start voicebot-tts-monitor.service

# Install jq for JSON parsing in monitoring
apt-get install -y jq

echo "TTS service setup completed" >> /var/log/user-data.log
