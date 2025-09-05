#!/bin/bash

# ecosystem-deploy.sh
# Complete MUSE Platform + Agent-001 ecosystem deployment
# This demonstrates the evolution from single agent to orchestrated AI empire

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}🏗️  Deploying MUSE Platform AI Agent Ecosystem${NC}"
echo -e "${CYAN}=====================================================${NC}"

# Export git sha for container tagging
export GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "dev")
echo -e "${YELLOW}📦 Version: ${GIT_SHA}${NC}"

# Build all services
echo -e "${GREEN}🔨 Building ecosystem services...${NC}"
docker-compose -f ecosystem-compose.yml build --parallel

# Deploy the ecosystem
echo -e "${GREEN}🚀 Launching ecosystem...${NC}"
docker-compose -f ecosystem-compose.yml up -d

# Wait for services to start
echo -e "${GREEN}⏳ Waiting for services to initialize...${NC}"
sleep 15

echo -e "${GREEN}🔍 Running ecosystem health checks...${NC}"

# Check PostgreSQL
echo -n "  📊 PostgreSQL: "
if docker exec muse_postgres pg_isready -U muse_user > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${RED}❌ Unhealthy${NC}"
fi

# Check Redis
echo -n "  🚀 Redis: "
if docker exec muse_redis redis-cli ping | grep -q PONG; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${RED}❌ Unhealthy${NC}"
fi

# Check MUSE Backend
echo -n "  🎭 MUSE Backend: "
if curl -fs http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${RED}❌ Unhealthy${NC}"
fi

# Check Agent-001
echo -n "  🤖 Agent-001 Monitor: "
if curl -fs http://localhost:9000/health > /dev/null; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${RED}❌ Unhealthy${NC}"
fi

echo -e "${CYAN}🌐 Ecosystem Access Points:${NC}"
echo -e "  🎭 MUSE Platform:      http://localhost:8000"
echo -e "  🎭 MUSE API Docs:      http://localhost:8000/docs"
echo -e "  🤖 Agent-001 Monitor: http://localhost:9000"
echo -e "  🤖 Monitor Check:     http://localhost:9000/check"
echo -e "  📊 Grafana Dashboard: http://localhost:3001"
echo -e "  📊 PostgreSQL:        localhost:5432 (muse_user/muse_password)"

echo -e "${GREEN}🎯 Sample monitoring query:${NC}"
echo -e "  curl -s http://localhost:9000/check | jq '.summary'"

echo -e "${CYAN}🚀 MUSE Platform AI Agent Ecosystem is live!${NC}"

# Show resource usage
echo -e "${YELLOW}📈 Resource Usage:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -6