#!/bin/bash

# Master Orchestrator Launch Script
# Starts both backend API and frontend development servers

echo "🚀 Launching Master Orchestrator Platform..."
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is required but not installed${NC}"
    exit 1
fi

# Install Python dependencies if needed
echo -e "${BLUE}📦 Checking Python dependencies...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
else
    echo -e "${YELLOW}⚠️  No requirements.txt found, installing core dependencies...${NC}"
    pip install -q fastapi uvicorn websockets pydantic python-multipart
fi

# Install frontend dependencies if needed
if [ -d "frontend" ]; then
    echo -e "${BLUE}📦 Checking frontend dependencies...${NC}"
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    fi
    cd ..
fi

# Kill any existing processes on our ports
echo -e "${YELLOW}🔍 Checking for existing processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Start backend API server
echo -e "${GREEN}🖥️  Starting Backend API Server...${NC}"
echo "   URL: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
python3 orchestrator_api.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000 > /dev/null; then
    echo -e "${RED}❌ Backend failed to start${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✅ Backend API is running${NC}"

# Start frontend development server
echo -e "${GREEN}🎨 Starting Frontend Development Server...${NC}"
if [ -d "frontend" ]; then
    cd frontend
    echo "   URL: http://localhost:5173"
    npm run dev &
    FRONTEND_PID=$!
    cd ..
else
    echo -e "${YELLOW}⚠️  Frontend directory not found, creating React app...${NC}"
    npm create vite@latest frontend -- --template react-ts
    cd frontend
    npm install
    npm install lucide-react framer-motion recharts
    npm run dev &
    FRONTEND_PID=$!
    cd ..
fi

# Function to handle shutdown
cleanup() {
    echo -e "\n${YELLOW}👋 Shutting down Master Orchestrator...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Set up trap for clean shutdown
trap cleanup INT TERM

# Wait for services and show status
echo ""
echo -e "${GREEN}🎯 Master Orchestrator is running!${NC}"
echo "============================================"
echo -e "${BLUE}📊 Dashboard:${NC} http://localhost:5173"
echo -e "${BLUE}🔌 API:${NC} http://localhost:8000"
echo -e "${BLUE}📚 API Docs:${NC} http://localhost:8000/docs"
echo -e "${BLUE}🔄 WebSocket:${NC} ws://localhost:8000/ws"
echo "============================================"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running
while true; do
    sleep 1
done