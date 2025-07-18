#!/bin/bash

# MUSE Platform Complete Deployment Script
# Deploys the entire MUSE platform with comprehensive validation

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/deployment.log"
ENVIRONMENT="${ENVIRONMENT:-development}"

# Logging function
log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}ERROR: ${1}${NC}"
    exit 1
}

# Success message
success() {
    log "${GREEN}✅ ${1}${NC}"
}

# Info message
info() {
    log "${BLUE}ℹ️  ${1}${NC}"
}

# Warning message
warn() {
    log "${YELLOW}⚠️  ${1}${NC}"
}

# Header
header() {
    log "${PURPLE}🎭 ${1}${NC}"
}

# Check prerequisites
check_prerequisites() {
    header "Checking Prerequisites"
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "node" "npm" "python3" "poetry" "curl")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "$cmd is required but not installed"
        fi
        success "$cmd is available"
    done
    
    # Check Python version
    local python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
        error_exit "Python 3.9+ is required, found $python_version"
    fi
    success "Python version: $python_version"
    
    # Check Node version
    local node_version=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$node_version" -lt 16 ]; then
        error_exit "Node.js 16+ is required, found v$node_version"
    fi
    success "Node.js version: v$(node --version | cut -d'v' -f2)"
    
    # Check Docker
    if ! docker info &> /dev/null; then
        error_exit "Docker is not running"
    fi
    success "Docker is running"
    
    # Check available disk space (require at least 5GB)
    local available_space=$(df . | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 5000000 ]; then
        warn "Less than 5GB disk space available"
    fi
    
    info "Prerequisites check completed"
}

# Setup environment
setup_environment() {
    header "Setting Up Environment"
    
    # Create necessary directories
    mkdir -p logs monitoring/grafana/{dashboards,provisioning} nginx/ssl
    
    # Set environment variables
    export DATABASE_URL="${DATABASE_URL:-postgresql://muse_user:muse_password@localhost:5432/muse_db}"
    export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
    export HARDCARD_ENTROPY_PATH="${HARDCARD_ENTROPY_PATH:-/dev/urandom}"
    
    # Create .env files if they don't exist
    if [ ! -f backend/.env ]; then
        cat > backend/.env << EOF
DATABASE_URL=${DATABASE_URL}
REDIS_URL=${REDIS_URL}
ENVIRONMENT=${ENVIRONMENT}
HARDCARD_ENTROPY_PATH=${HARDCARD_ENTROPY_PATH}
SECRET_KEY=$(openssl rand -hex 32)
EOF
        success "Created backend/.env"
    fi
    
    if [ ! -f frontend/.env ]; then
        cat > frontend/.env << EOF
REACT_APP_API_URL=http://localhost:8000/api
REACT_APP_WS_URL=ws://localhost:8000/ws
NODE_ENV=${ENVIRONMENT}
EOF
        success "Created frontend/.env"
    fi
    
    info "Environment setup completed"
}

# Install backend dependencies
install_backend_deps() {
    header "Installing Backend Dependencies"
    
    cd "$SCRIPT_DIR/backend"
    
    # Install Poetry if not available
    if ! command -v poetry &> /dev/null; then
        info "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Install dependencies
    info "Installing Python dependencies..."
    poetry install
    
    # Verify installation
    poetry run python -c "import muse.core.sacred_geometry_calculator; print('Core engines imported successfully')"
    success "Backend dependencies installed and verified"
    
    cd "$SCRIPT_DIR"
}

# Install frontend dependencies
install_frontend_deps() {
    header "Installing Frontend Dependencies"
    
    cd "$SCRIPT_DIR/frontend"
    
    # Install Node dependencies
    info "Installing Node.js dependencies..."
    npm ci
    
    # Build frontend
    info "Building frontend..."
    npm run build
    
    success "Frontend dependencies installed and built"
    
    cd "$SCRIPT_DIR"
}

# Initialize database
init_database() {
    header "Initializing Database"
    
    if [ "$ENVIRONMENT" = "development" ]; then
        # Start PostgreSQL with Docker Compose
        info "Starting PostgreSQL..."
        docker-compose up -d postgres redis
        
        # Wait for database to be ready
        info "Waiting for database to be ready..."
        timeout 60 bash -c 'until docker-compose exec -T postgres pg_isready -U muse_user -d muse_db; do sleep 2; done'
        
        # Initialize database schema
        cd "$SCRIPT_DIR/backend"
        info "Creating database schema..."
        poetry run python -c "
from database import init_db
init_db()
print('Database initialized successfully')
"
        success "Database initialized"
        cd "$SCRIPT_DIR"
    else
        info "Skipping database initialization in $ENVIRONMENT mode"
    fi
}

# Run tests
run_tests() {
    header "Running Test Suite"
    
    cd "$SCRIPT_DIR/backend"
    
    # Run unit tests
    info "Running unit tests..."
    poetry run pytest tests/ -m "unit" --tb=short -v
    
    # Run integration tests if database is available
    if [ "$ENVIRONMENT" = "development" ]; then
        info "Running integration tests..."
        poetry run pytest tests/ -m "integration" --tb=short -v
    fi
    
    # Check test coverage
    info "Generating coverage report..."
    poetry run pytest tests/ --cov=muse --cov-report=html --cov-report=term
    
    success "Test suite completed"
    
    cd "$SCRIPT_DIR"
}

# Start services
start_services() {
    header "Starting MUSE Platform Services"
    
    if [ "$ENVIRONMENT" = "development" ]; then
        # Start all services with Docker Compose
        info "Starting all services..."
        docker-compose up -d
        
        # Wait for services to be healthy
        info "Waiting for services to be ready..."
        
        # Check backend health
        timeout 120 bash -c 'until curl -f http://localhost:8000/health &>/dev/null; do sleep 5; done'
        success "Backend API is ready"
        
        # Check frontend health
        timeout 60 bash -c 'until curl -f http://localhost:3000 &>/dev/null; do sleep 5; done'
        success "Frontend is ready"
        
    else
        info "Starting services in $ENVIRONMENT mode..."
        # Production deployment would go here
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    fi
}

# Validate deployment
validate_deployment() {
    header "Validating Deployment"
    
    # Test API endpoints
    info "Testing API endpoints..."
    
    # Health check
    local health_response=$(curl -s http://localhost:8000/health)
    if echo "$health_response" | grep -q "healthy"; then
        success "Health endpoint is working"
    else
        error_exit "Health endpoint failed: $health_response"
    fi
    
    # Test MUSE endpoints
    local endpoints=(
        "/api/muse/assessment/start"
        "/api/muse/validation/summary"
        "/docs"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s "http://localhost:8000$endpoint" &>/dev/null; then
            success "Endpoint $endpoint is accessible"
        else
            warn "Endpoint $endpoint may not be working"
        fi
    done
    
    # Test frontend
    info "Testing frontend..."
    if curl -f -s http://localhost:3000 &>/dev/null; then
        success "Frontend is accessible"
    else
        error_exit "Frontend is not accessible"
    fi
    
    # Test database connection
    if [ "$ENVIRONMENT" = "development" ]; then
        info "Testing database connection..."
        cd "$SCRIPT_DIR/backend"
        poetry run python -c "
from database import engine
from sqlalchemy import text
try:
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1'))
        print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
"
        success "Database connection verified"
        cd "$SCRIPT_DIR"
    fi
}

# Performance benchmarks
run_benchmarks() {
    header "Running Performance Benchmarks"
    
    cd "$SCRIPT_DIR/backend"
    
    info "Running core engine benchmarks..."
    poetry run pytest tests/ -m "performance" --tb=short -v
    
    info "Testing API performance..."
    # Simple load test
    if command -v ab &> /dev/null; then
        ab -n 100 -c 10 http://localhost:8000/health || warn "Apache Bench not available for load testing"
    fi
    
    success "Performance benchmarks completed"
    
    cd "$SCRIPT_DIR"
}

# Display status
show_status() {
    header "MUSE Platform Status"
    
    echo ""
    log "🎭 ${GREEN}MUSE Platform Successfully Deployed!${NC}"
    echo ""
    log "📊 ${BLUE}Service URLs:${NC}"
    log "   • Frontend:        http://localhost:3000"
    log "   • Backend API:     http://localhost:8000"
    log "   • API Docs:        http://localhost:8000/docs"
    log "   • Validation:      http://localhost:8000/api/muse/validation/summary"
    log "   • Grafana:         http://localhost:3001 (admin/muse_admin)"
    log "   • Prometheus:      http://localhost:9090"
    echo ""
    log "🔧 ${BLUE}Management Commands:${NC}"
    log "   • View logs:       docker-compose logs -f"
    log "   • Stop services:   docker-compose down"
    log "   • Restart:         docker-compose restart"
    log "   • Run tests:       cd backend && poetry run pytest"
    echo ""
    log "🎯 ${BLUE}MUSE Features:${NC}"
    log "   • Sacred Geometry Calculator"
    log "   • 12 Archetypal Frequencies" 
    log "   • Frequency-Based Community"
    log "   • Real-time Discovery Interface"
    log "   • Empirical Validation Framework"
    echo ""
    log "📝 ${BLUE}Next Steps:${NC}"
    log "   1. Visit http://localhost:3000 to access MUSE"
    log "   2. Complete the personality assessment"
    log "   3. Explore mathematical poetry discovery"
    log "   4. Connect with kindred spirits in the community"
    echo ""
    log "${PURPLE}Let's awaken the mathematical muses! 🎭✨👁️‍🗨️${NC}"
}

# Cleanup on exit
cleanup() {
    if [ $? -ne 0 ]; then
        warn "Deployment failed. Check logs at $LOG_FILE"
        log "To debug:"
        log "  docker-compose logs backend"
        log "  docker-compose logs frontend"
    fi
}

trap cleanup EXIT

# Main deployment flow
main() {
    # Start logging
    echo "MUSE Platform Deployment - $(date)" > "$LOG_FILE"
    
    header "🎭 MUSE Platform Deployment Starting"
    log "Environment: $ENVIRONMENT"
    log "Script directory: $SCRIPT_DIR"
    echo ""
    
    # Check if forced
    if [ "${1:-}" = "--force" ]; then
        warn "Force deployment requested, skipping some checks"
    fi
    
    # Run deployment steps
    check_prerequisites
    setup_environment
    install_backend_deps
    install_frontend_deps
    
    if [ "$ENVIRONMENT" = "development" ]; then
        init_database
        run_tests
    fi
    
    start_services
    validate_deployment
    
    if [ "$ENVIRONMENT" = "development" ]; then
        run_benchmarks
    fi
    
    show_status
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "MUSE Platform Deployment Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --force     Skip some checks and force deployment"
        echo "  --help      Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  ENVIRONMENT       Deployment environment (development|staging|production)"
        echo "  DATABASE_URL      PostgreSQL connection string"
        echo "  REDIS_URL         Redis connection string"
        echo ""
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac