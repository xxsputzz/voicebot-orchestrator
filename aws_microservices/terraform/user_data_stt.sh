#!/bin/bash
# User data script for STT service (CPU-optimized)

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
apt-get install -y docker.io
systemctl start docker
systemctl enable docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /app/voicebot-stt
cd /app/voicebot-stt

# Download application code (replace with your actual repository)
# git clone https://github.com/xxsputzz/voicebot-orchestrator.git .
# For now, create placeholder structure
mkdir -p voicebot_orchestrator aws_microservices

# Create Docker Compose file for STT service
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  stt-service:
    build:
      context: .
      dockerfile: aws_microservices/Dockerfile.stt
    ports:
      - "${service_port}:${service_port}"
    environment:
      - SERVICE_PORT=${service_port}
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:${service_port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
EOF

# Start the service
docker-compose up -d

# Setup log rotation
cat > /etc/logrotate.d/voicebot-stt << 'EOF'
/var/log/voicebot-stt.log {
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
cat > /usr/local/bin/monitor-stt.sh << 'EOF'
#!/bin/bash
# Simple monitoring script for STT service

SERVICE_URL="http://localhost:${service_port}/health"
LOG_FILE="/var/log/voicebot-stt.log"

while true; do
    if curl -f $SERVICE_URL > /dev/null 2>&1; then
        echo "$(date): STT service healthy" >> $LOG_FILE
    else
        echo "$(date): STT service unhealthy - restarting" >> $LOG_FILE
        cd /app/voicebot-stt && docker-compose restart
    fi
    sleep 60
done
EOF

chmod +x /usr/local/bin/monitor-stt.sh

# Create systemd service for monitoring
cat > /etc/systemd/system/voicebot-stt-monitor.service << 'EOF'
[Unit]
Description=Voicebot STT Service Monitor
After=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/monitor-stt.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl enable voicebot-stt-monitor.service
systemctl start voicebot-stt-monitor.service

echo "STT service setup completed" >> /var/log/user-data.log
