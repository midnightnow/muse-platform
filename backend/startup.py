#!/usr/bin/env python3
"""
MUSE Platform Startup Script

This script initializes the MUSE Platform backend with all necessary
components, performs health checks, and starts the FastAPI server.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import init_database, health_check
from muse.dependencies import test_all_dependencies, get_dependency_info


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print the MUSE Platform banner"""
    banner = """
    ███╗   ███╗██╗   ██╗███████╗███████╗
    ████╗ ████║██║   ██║██╔════╝██╔════╝
    ██╔████╔██║██║   ██║███████╗█████╗  
    ██║╚██╔╝██║██║   ██║╚════██║██╔══╝  
    ██║ ╚═╝ ██║╚██████╔╝███████║███████╗
    ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝
    
    🎭 MUSE Platform - Computational Platonism Creative Discovery
    
    Mathematical discovery of pre-existing creative forms through
    archetypal frequency signatures and sacred geometry.
    """
    print(banner)


def initialize_database(reset: bool = False) -> bool:
    """Initialize the database and perform health checks"""
    try:
        logger.info("🗄️  Initializing database...")
        
        # Initialize database
        init_database(reset=reset)
        
        # Perform health check
        health_data = health_check()
        
        if health_data["status"] == "healthy":
            logger.info("✅ Database initialization successful")
            logger.info(f"   Engine: {health_data['engine_info'].get('dialect', 'unknown')}")
            logger.info(f"   Tables: {len(health_data['table_info'])}")
            return True
        else:
            logger.error("❌ Database health check failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


def test_dependencies() -> bool:
    """Test all system dependencies"""
    try:
        logger.info("🔧 Testing system dependencies...")
        
        # Test all dependencies
        test_results = test_all_dependencies()
        
        if test_results["overall_status"] == "pass":
            logger.info("✅ All dependencies initialized successfully")
            
            # Log dependency info
            dependency_info = get_dependency_info()
            logger.info("📊 Dependency Information:")
            
            # Core engines
            core_engines = dependency_info["components"]["core_engines"]
            for engine_name, engine_info in core_engines.items():
                logger.info(f"   🔧 {engine_name}: {engine_info['class']}")
            
            # Services
            services = dependency_info["components"]["services"]
            for service_name, service_info in services.items():
                logger.info(f"   🛠️  {service_name}: {service_info['class']}")
            
            return True
        else:
            logger.error("❌ Dependency tests failed")
            if "error" in test_results:
                logger.error(f"   Error: {test_results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Dependency testing failed: {e}")
        return False


def validate_environment() -> bool:
    """Validate environment configuration"""
    try:
        logger.info("🌍 Validating environment...")
        
        # Check for required environment variables
        required_vars = []
        optional_vars = {
            "DATABASE_URL": "sqlite:///./muse_platform.db",
            "HOST": "0.0.0.0",
            "PORT": "8000",
            "RELOAD": "false",
            "WORKERS": "1"
        }
        
        # Check required variables
        for var in required_vars:
            if not os.getenv(var):
                logger.error(f"❌ Required environment variable {var} is not set")
                return False
        
        # Set default values for optional variables
        for var, default_value in optional_vars.items():
            if not os.getenv(var):
                os.environ[var] = default_value
                logger.info(f"🔧 Set {var} to default value: {default_value}")
        
        logger.info("✅ Environment validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return False


def perform_startup_checks() -> bool:
    """Perform all startup checks"""
    checks = [
        ("Environment", validate_environment),
        ("Database", lambda: initialize_database()),
        ("Dependencies", test_dependencies)
    ]
    
    for check_name, check_func in checks:
        logger.info(f"⏳ Running {check_name} check...")
        if not check_func():
            logger.error(f"❌ {check_name} check failed")
            return False
    
    logger.info("✅ All startup checks passed")
    return True


def start_server():
    """Start the FastAPI server"""
    try:
        import uvicorn
        
        # Get configuration from environment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        reload = os.getenv("RELOAD", "false").lower() == "true"
        workers = int(os.getenv("WORKERS", "1"))
        
        logger.info("🚀 Starting MUSE Platform server...")
        logger.info(f"   Host: {host}")
        logger.info(f"   Port: {port}")
        logger.info(f"   Reload: {reload}")
        logger.info(f"   Workers: {workers}")
        
        # Start server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)


def main():
    """Main startup function"""
    print_banner()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="MUSE Platform Startup")
    parser.add_argument("--reset-db", action="store_true", help="Reset database on startup")
    parser.add_argument("--check-only", action="store_true", help="Only perform checks, don't start server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Set environment variables from command line
    os.environ["PORT"] = str(args.port)
    os.environ["HOST"] = args.host
    os.environ["RELOAD"] = "true" if args.reload else "false"
    os.environ["WORKERS"] = str(args.workers)
    
    # Perform startup checks
    if not perform_startup_checks():
        logger.error("❌ Startup checks failed. Exiting.")
        sys.exit(1)
    
    # Reset database if requested
    if args.reset_db:
        logger.warning("🔄 Resetting database...")
        if not initialize_database(reset=True):
            logger.error("❌ Database reset failed. Exiting.")
            sys.exit(1)
    
    # If check-only mode, exit here
    if args.check_only:
        logger.info("✅ All checks passed. Exiting (check-only mode).")
        sys.exit(0)
    
    # Start the server
    start_server()


if __name__ == "__main__":
    main()