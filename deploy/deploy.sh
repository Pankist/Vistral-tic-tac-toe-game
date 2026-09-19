#!/bin/bash
# Production deployment script - ALWAYS fetches API key from SSM

set -e

cd /home/ubuntu/app

# Pull latest code
sudo -u ubuntu git pull

# Fetch API key from SSM and write .env
sudo aws ssm get-parameter \
  --region us-east-1 \
  --name /vistral-ttt/anthropic-api-key \
  --with-decryption \
  --query Parameter.Value \
  --output text > /tmp/ttt-key

# Write .env with proper ownership
cat > /tmp/ttt-env <<EOF
ANTHROPIC_API_KEY=$(cat /tmp/ttt-key)
PORT=8000
EOF

sudo mv /tmp/ttt-env /home/ubuntu/app/.env
sudo chown ubuntu:ubuntu /home/ubuntu/app/.env
rm /tmp/ttt-key

# Restart service
sudo systemctl restart ttt-agent

# Wait and check health
sleep 3
curl -s http://localhost:8000/health
