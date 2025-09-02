#!/bin/bash

# MUSE Platform - Graceful Shutdown Script

set -e

# Colors
PURPLE='\033[0;35m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'
BOLD='\033[1m'

echo -e "${PURPLE}${BOLD}"
echo "═══════════════════════════════════════════════"
echo "    Gracefully stopping MUSE Platform..."
echo "═══════════════════════════════════════════════"
echo -e "${NC}"

# Function to stop a service
stop_service() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${YELLOW}⏹  Stopping ${service_name} (PID: $PID)...${NC}"
            kill $PID 2>/dev/null || true
            
            # Wait for process to stop
            for i in {1..10}; do
                if ! ps -p $PID > /dev/null 2>&1; then
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if ps -p $PID > /dev/null 2>&1; then
                echo -e "${RED}⚠️  Force stopping ${service_name}...${NC}"
                kill -9 $PID 2>/dev/null || true
            fi
            
            rm -f "$pid_file"
            echo -e "${GREEN}✅ ${service_name} stopped${NC}"
        else
            echo -e "${YELLOW}ℹ️  ${service_name} not running (stale PID file)${NC}"
            rm -f "$pid_file"
        fi
    else
        echo -e "${YELLOW}ℹ️  ${service_name} not running${NC}"
    fi
}

# Stop services
stop_service "pids/frontend.pid" "Frontend"
stop_service "pids/backend.pid" "Backend"

# Optional: Stop Docker containers
if docker ps | grep -q muse-postgres; then
    echo -e "${YELLOW}⏹  Stopping PostgreSQL...${NC}"
    docker stop muse-postgres > /dev/null 2>&1
    docker rm muse-postgres > /dev/null 2>&1
    echo -e "${GREEN}✅ PostgreSQL stopped${NC}"
fi

if docker ps | grep -q muse-redis; then
    echo -e "${YELLOW}⏹  Stopping Redis...${NC}"
    docker stop muse-redis > /dev/null 2>&1
    docker rm muse-redis > /dev/null 2>&1
    echo -e "${GREEN}✅ Redis stopped${NC}"
fi

echo -e "\n${PURPLE}${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}✨ MUSE Platform stopped gracefully ✨${NC}"
echo -e "${PURPLE}${BOLD}═══════════════════════════════════════════════${NC}\n"