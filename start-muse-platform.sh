#!/bin/bash

# MUSE Platform Quick Start Script
# Rapid development environment startup

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${PURPLE}🎭 Starting MUSE Platform Development Environment${NC}"
echo ""

# Check if services are already running
if docker-compose ps | grep -q "Up"; then
    echo -e "${YELLOW}⚠️  Some services are already running. Stopping them first...${NC}"
    docker-compose down
fi

# Start core services (database, cache)
echo -e "${BLUE}🗄️  Starting core services (PostgreSQL, Redis)...${NC}"
docker-compose up -d postgres redis

# Wait for database
echo -e "${BLUE}⏳ Waiting for database to be ready...${NC}"
timeout 30 bash -c 'until docker-compose exec -T postgres pg_isready -U muse_user -d muse_db &>/dev/null; do sleep 2; done'

# Initialize database if needed
echo -e "${BLUE}🔧 Initializing database schema...${NC}"
cd "$SCRIPT_DIR/backend"
if poetry run python -c "
from database import engine, init_db
from sqlalchemy import text
try:
    with engine.connect() as conn:
        result = conn.execute(text('SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = \'public\''))
        table_count = result.scalar()
        if table_count == 0:
            print('Initializing database...')
            init_db()
        else:
            print('Database already initialized')
except Exception as e:
    print('Initializing database...')
    init_db()
"; then
    echo -e "${GREEN}✅ Database ready${NC}"
else
    echo -e "${YELLOW}⚠️  Database initialization completed with warnings${NC}"
fi

# Start backend
echo -e "${BLUE}🚀 Starting MUSE backend API...${NC}"
cd "$SCRIPT_DIR/backend"

# Run backend in background
poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo -e "${BLUE}⏳ Waiting for backend API...${NC}"
timeout 60 bash -c 'until curl -f http://localhost:8000/health &>/dev/null; do sleep 2; done'
echo -e "${GREEN}✅ Backend API ready at http://localhost:8000${NC}"

# Start frontend
echo -e "${BLUE}🎨 Starting MUSE frontend...${NC}"
cd "$SCRIPT_DIR/frontend"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
    npm install
fi

# Start frontend in background
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to be ready
echo -e "${BLUE}⏳ Waiting for frontend...${NC}"
timeout 60 bash -c 'until curl -f http://localhost:3000 &>/dev/null; do sleep 2; done'
echo -e "${GREEN}✅ Frontend ready at http://localhost:3000${NC}"

echo ""
echo -e "${PURPLE}🎭 MUSE Platform is now running!${NC}"
echo ""
echo -e "${GREEN}📊 Access Points:${NC}"
echo -e "   • MUSE App:        ${BLUE}http://localhost:3000${NC}"
echo -e "   • API Documentation: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "   • Health Check:    ${BLUE}http://localhost:8000/health${NC}"
echo -e "   • Validation:      ${BLUE}http://localhost:8000/api/muse/validation/summary${NC}"
echo ""
echo -e "${GREEN}🔧 Development:${NC}"
echo -e "   • Backend logs:    ${BLUE}tail -f logs/backend.log${NC}"
echo -e "   • Frontend logs:   ${BLUE}Check terminal output${NC}"
echo -e "   • Database:        ${BLUE}docker-compose exec postgres psql -U muse_user -d muse_db${NC}"
echo ""
echo -e "${GREEN}🎯 Test MUSE:${NC}"
echo -e "   1. Open ${BLUE}http://localhost:3000${NC}"
echo -e "   2. Complete personality assessment"
echo -e "   3. Discover mathematical poetry"
echo -e "   4. Explore frequency-based community"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping MUSE Platform...${NC}"
    
    # Kill background processes
    if [ -n "${BACKEND_PID:-}" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Stop Docker services
    docker-compose down
    
    echo -e "${GREEN}✅ MUSE Platform stopped${NC}"
    echo -e "${PURPLE}Thank you for exploring mathematical creativity! 🎭✨${NC}"
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Keep script running
wait