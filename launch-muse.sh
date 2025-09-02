#!/bin/bash

# MUSE Platform - Elegant Launch Script
# Mathematical Universal Sacred Expression

set -e  # Exit on error

# Colors for beautiful terminal output
PURPLE='\033[0;35m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ASCII Art Logo
print_logo() {
    echo -e "${PURPLE}"
    cat << "EOF"
    
    ███╗   ███╗██╗   ██╗███████╗███████╗
    ████╗ ████║██║   ██║██╔════╝██╔════╝
    ██╔████╔██║██║   ██║███████╗█████╗  
    ██║╚██╔╝██║██║   ██║╚════██║██╔══╝  
    ██║ ╚═╝ ██║╚██████╔╝███████║███████╗
    ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝
    
    Mathematical Universal Sacred Expression
    🎭 Computational Platonism Creative Discovery Platform
    
EOF
    echo -e "${NC}"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}⏳ Waiting for ${service_name}...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|302"; then
            echo -e "${GREEN}✅ ${service_name} is ready!${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo -e "${RED}❌ ${service_name} failed to start${NC}"
    return 1
}

# Function to start backend
start_backend() {
    echo -e "${BLUE}${BOLD}🚀 Starting MUSE Backend...${NC}"
    
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies if needed
    echo -e "${YELLOW}📦 Installing backend dependencies...${NC}"
    pip install -q -r requirements.txt 2>/dev/null || {
        # If requirements.txt doesn't exist, install core packages
        pip install -q fastapi uvicorn sqlalchemy psycopg2-binary redis pydantic numpy
        pip install -q pronouncing syllapy
    }
    
    # Check if backend is already running
    if check_port 8000; then
        echo -e "${YELLOW}⚠️  Backend already running on port 8000${NC}"
    else
        # Start backend in background
        echo -e "${GREEN}▶️  Starting FastAPI server...${NC}"
        nohup python main.py > ../logs/backend.log 2>&1 &
        BACKEND_PID=$!
        echo $BACKEND_PID > ../pids/backend.pid
        
        # Wait for backend to be ready
        wait_for_service "http://localhost:8000/health" "Backend API"
    fi
    
    cd ..
}

# Function to start frontend
start_frontend() {
    echo -e "${BLUE}${BOLD}🎨 Starting MUSE Frontend...${NC}"
    
    cd frontend
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
        npm install --silent
    fi
    
    # Check if frontend is already running
    if check_port 3000; then
        echo -e "${YELLOW}⚠️  Frontend already running on port 3000${NC}"
    else
        # Start frontend in background
        echo -e "${GREEN}▶️  Starting React development server...${NC}"
        nohup npm run dev > ../logs/frontend.log 2>&1 &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > ../pids/frontend.pid
        
        # Wait for frontend to be ready
        wait_for_service "http://localhost:3000" "Frontend"
    fi
    
    cd ..
}

# Function to start database
start_database() {
    echo -e "${BLUE}${BOLD}🗄️  Starting Database Services...${NC}"
    
    # Check if PostgreSQL is running (Docker or local)
    if docker ps 2>/dev/null | grep -q postgres; then
        echo -e "${GREEN}✅ PostgreSQL is running (Docker)${NC}"
    elif pg_isready -h localhost -p 5432 2>/dev/null; then
        echo -e "${GREEN}✅ PostgreSQL is running (local)${NC}"
    else
        echo -e "${YELLOW}📦 Starting PostgreSQL with Docker...${NC}"
        docker run -d \
            --name muse-postgres \
            -e POSTGRES_USER=muse \
            -e POSTGRES_PASSWORD=muse_sacred_geometry \
            -e POSTGRES_DB=muse_db \
            -p 5432:5432 \
            postgres:14 > /dev/null 2>&1
        
        echo -e "${GREEN}✅ PostgreSQL started${NC}"
    fi
    
    # Check if Redis is running
    if docker ps 2>/dev/null | grep -q redis; then
        echo -e "${GREEN}✅ Redis is running (Docker)${NC}"
    elif redis-cli ping 2>/dev/null | grep -q PONG; then
        echo -e "${GREEN}✅ Redis is running (local)${NC}"
    else
        echo -e "${YELLOW}📦 Starting Redis with Docker...${NC}"
        docker run -d \
            --name muse-redis \
            -p 6379:6379 \
            redis:7 > /dev/null 2>&1
        
        echo -e "${GREEN}✅ Redis started${NC}"
    fi
}

# Function to open browser
open_browser() {
    echo -e "${BLUE}${BOLD}🌐 Opening MUSE in browser...${NC}"
    
    # Wait a moment for everything to stabilize
    sleep 3
    
    # Detect OS and open browser
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:3000"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "http://localhost:3000" 2>/dev/null || echo -e "${YELLOW}Please open http://localhost:3000 in your browser${NC}"
    else
        echo -e "${YELLOW}Please open http://localhost:3000 in your browser${NC}"
    fi
}

# Function to show status
show_status() {
    echo -e "\n${PURPLE}${BOLD}═══════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}🎭 MUSE Platform Status${NC}"
    echo -e "${PURPLE}${BOLD}═══════════════════════════════════════════════${NC}\n"
    
    echo -e "${BOLD}Services:${NC}"
    
    # Check Backend
    if check_port 8000; then
        echo -e "  ${GREEN}●${NC} Backend API:    ${GREEN}http://localhost:8000${NC}"
        echo -e "                   ${BLUE}Docs: http://localhost:8000/docs${NC}"
    else
        echo -e "  ${RED}●${NC} Backend API:    ${RED}Not running${NC}"
    fi
    
    # Check Frontend
    if check_port 3000; then
        echo -e "  ${GREEN}●${NC} Frontend:       ${GREEN}http://localhost:3000${NC}"
    else
        echo -e "  ${RED}●${NC} Frontend:       ${RED}Not running${NC}"
    fi
    
    # Check Database
    if pg_isready -h localhost -p 5432 2>/dev/null; then
        echo -e "  ${GREEN}●${NC} PostgreSQL:     ${GREEN}Running on port 5432${NC}"
    else
        echo -e "  ${RED}●${NC} PostgreSQL:     ${RED}Not running${NC}"
    fi
    
    # Check Redis
    if redis-cli ping 2>/dev/null | grep -q PONG; then
        echo -e "  ${GREEN}●${NC} Redis:          ${GREEN}Running on port 6379${NC}"
    else
        echo -e "  ${RED}●${NC} Redis:          ${RED}Not running${NC}"
    fi
    
    echo -e "\n${BOLD}Quick Actions:${NC}"
    echo -e "  ${BLUE}◆${NC} View API docs:     ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "  ${BLUE}◆${NC} Stop all:          ${BLUE}./stop-muse.sh${NC}"
    echo -e "  ${BLUE}◆${NC} View logs:         ${BLUE}tail -f logs/*.log${NC}"
    echo -e "  ${BLUE}◆${NC} Restart:           ${BLUE}./restart-muse.sh${NC}"
    
    echo -e "\n${PURPLE}${BOLD}═══════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}✨ Discover the eternal mathematical forms ✨${NC}"
    echo -e "${PURPLE}${BOLD}═══════════════════════════════════════════════${NC}\n"
}

# Function to tail logs
tail_logs() {
    echo -e "\n${YELLOW}📜 Showing live logs (Ctrl+C to exit)...${NC}\n"
    tail -f logs/*.log
}

# Main execution
main() {
    clear
    print_logo
    
    # Create necessary directories
    mkdir -p logs pids
    
    # Parse command line arguments
    case "${1:-}" in
        status)
            show_status
            ;;
        logs)
            tail_logs
            ;;
        stop)
            ./stop-muse.sh
            ;;
        restart)
            ./stop-muse.sh
            sleep 2
            main
            ;;
        *)
            # Start all services
            echo -e "${PURPLE}${BOLD}═══════════════════════════════════════════════${NC}"
            echo -e "${BLUE}${BOLD}Initializing MUSE Platform...${NC}"
            echo -e "${PURPLE}${BOLD}═══════════════════════════════════════════════${NC}\n"
            
            # Start services in order
            start_database
            echo ""
            start_backend
            echo ""
            start_frontend
            echo ""
            
            # Open browser and show status
            open_browser
            show_status
            
            echo -e "${GREEN}${BOLD}✨ MUSE Platform is ready! ✨${NC}"
            echo -e "${BLUE}Press Ctrl+C to stop all services${NC}\n"
            
            # Keep script running and show logs on interrupt
            trap 'echo -e "\n${YELLOW}Stopping MUSE...${NC}"; ./stop-muse.sh; exit' INT
            
            # Optional: tail logs
            if [ "${2:-}" == "--logs" ]; then
                tail_logs
            else
                echo -e "${YELLOW}Tip: Run './launch-muse.sh status' to check service status${NC}"
                echo -e "${YELLOW}     Run './launch-muse.sh logs' to view logs${NC}\n"
                
                # Keep script running
                while true; do
                    sleep 60
                done
            fi
            ;;
    esac
}

# Run main function
main "$@"