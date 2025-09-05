#!/bin/bash

# deploy.sh
# Automates the build, test, and run process for the Docker container.
# This script is a single source of truth for deploying the agent.

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}🔧 Building agent-001 for deployment...${NC}"

# Build the Docker image
echo -e "${GREEN}🐳 Building Docker container...${NC}"
docker build -t agent-001:latest .

# Run a smoke test
echo -e "${GREEN}🧪 Running container smoke test...${NC}"
docker run -d -p 9000:9000 --name agent-test -e TARGETS="http://localhost:8000/health" agent-001:latest

# Wait for container to start and run health check
sleep 5
echo -e "${GREEN}🔍 Running health check...${NC}"
if curl -f http://localhost:9000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed! The agent is healthy.${NC}"
else
    echo -e "${RED}❌ Health check failed. The agent is unhealthy.${NC}"
    docker logs agent-test
    docker stop agent-test && docker rm agent-test
    exit 1
fi

# Test the monitoring endpoint
echo -e "${GREEN}🔍 Testing /check endpoint...${NC}"
if curl -f http://localhost:9000/check > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Check endpoint functional!${NC}"
else
    echo -e "${RED}❌ Check endpoint failed.${NC}"
fi

# Clean up test container
docker stop agent-test >/dev/null
docker rm agent-test >/dev/null

echo -e "${CYAN}🚀 agent-001 deployment ready!${NC}"
echo -e "${CYAN}To run: docker run -d -p 9000:9000 -e TARGETS=\"http://localhost:8000/health\" agent-001:latest${NC}"
echo -e "${CYAN}To run with docker-compose: docker-compose up -d${NC}"