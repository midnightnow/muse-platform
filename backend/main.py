"""
Main FastAPI Application for MUSE Platform

This is the primary application entry point that brings together all the
components of the MUSE Computational Platonism creative discovery system.
It integrates the three core engines, API routers, and services into a
unified platform for mathematical-based creative discovery.
"""

import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from datetime import datetime

from database import db_manager, init_database
from muse.api import main as main_api
from muse.api import integration as integration_api
from muse.api import community as community_api
from muse.api import music as music_api
from muse.validation.validation_dashboard import router as validation_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    
    Handles startup and shutdown events for the MUSE platform,
    including database initialization and cleanup.
    """
    # Startup
    logger.info("🎭 Starting MUSE Platform - Computational Platonism Creative Discovery")
    
    try:
        # Initialize database
        logger.info("📊 Initializing database...")
        init_database(reset=False)
        
        # Test database connection
        if db_manager.test_connection():
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection failed")
            raise Exception("Database connection failed")
        
        # Log system information
        engine_info = db_manager.get_engine_info()
        table_info = db_manager.get_table_info()
        
        logger.info(f"📈 Database Engine: {engine_info.get('dialect', 'unknown')}")
        logger.info(f"📋 Database Tables: {len(table_info)}")
        
        # Initialize core engines
        logger.info("🔧 Initializing core engines...")
        logger.info("   ⚡ Frequency Engine - Archetypal mathematics")
        logger.info("   📐 Sacred Geometry Calculator - Mathematical constants")
        logger.info("   🧠 Semantic Projection Engine - Meaning mathematics")
        
        # Initialize validation framework
        logger.info("🔬 Initializing validation framework...")
        logger.info("   📊 Mathematical Validation Framework - Statistical analysis")
        logger.info("   🎯 Validation Dashboard - Experiment management")
        logger.info("   👥 Participant Recruitment - Automated recruitment")
        logger.info("   🔄 Real-time Analysis - Live statistical monitoring")
        logger.info("   📈 Metrics Calculator - Core engine metrics")
        
        logger.info("🌟 MUSE Platform startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MUSE Platform...")
    logger.info("👋 MUSE Platform shutdown complete")


# Create FastAPI application with lifespan
app = FastAPI(
    title="MUSE Platform API",
    description="""
    # MUSE Platform - Computational Platonism Creative Discovery

    The MUSE Platform implements a revolutionary approach to creative discovery
    based on **Computational Platonism** - the idea that mathematical structures
    exist in a realm of eternal forms, and creativity is the process of discovering
    (not generating) these pre-existing patterns.

    ## Core Philosophy

    Unlike traditional AI systems that generate content, MUSE **discovers** creative
    works that already exist in the mathematical realm. Our three-engine system
    coordinates to map users' archetypal frequencies and guide them toward
    their optimal creative expressions.

    ## Three Core Engines

    ### 1. 🎭 Frequency Engine
    Maps personality traits to archetypal frequencies using the 12 Muses of classical
    mythology. Each user gets a unique frequency signature that represents their
    creative essence in mathematical form.

    ### 2. 📐 Sacred Geometry Calculator
    Applies sacred mathematical constants (golden ratio, π, fibonacci sequence) to
    structure creative works according to universal proportions found in nature
    and great works of art.

    ### 3. 🧠 Semantic Projection Engine
    Bridges meaning and mathematics by creating semantic-mathematical mappings that
    ensure discovered works are both mathematically elegant and emotionally resonant.

    ## Key Features

    - **Personality Assessment**: Transform traits into frequency signatures
    - **Live Discovery**: Real-time creative discovery using all three engines
    - **Constraint Optimization**: Mathematical optimization of creative constraints
    - **Community Resonance**: Find kindred spirits through archetypal compatibility
    - **Collaborative Sessions**: Multi-user creative discovery sessions
    - **Sacred Geometry Gallery**: Explore works optimized for mathematical beauty
    - **Validation Framework**: Empirical testing of Computational Platonism claims

    ## Discovery Process

    1. **Initialization**: User provides theme and creative preferences
    2. **Frequency Alignment**: Map user's archetypal signature to creative constraints
    3. **Geometric Optimization**: Apply sacred geometry for structural harmony
    4. **Semantic Projection**: Ensure meaning-mathematics alignment
    5. **Creative Synthesis**: Coordinate all engines for unified discovery
    6. **Refinement**: Iterative optimization based on fitness scores

    ## Mathematical Foundation

    The platform is built on the principle that creativity is **mathematical discovery**
    rather than generation. Each creative work exists in a mathematical space defined
    by archetypal frequencies, sacred proportions, and semantic vectors.

    Users don't create poetry - they discover the poems that already exist in the
    mathematical realm and are perfectly suited to their unique frequency signature.
    
    ## Validation Framework
    
    The MUSE platform includes a comprehensive validation framework for empirical
    testing of Computational Platonism claims:
    
    - **Mathematical Validation**: Statistical analysis of creative discovery effectiveness
    - **Participant Recruitment**: Automated recruitment system for validation studies
    - **Data Collection Pipeline**: Real-time data collection with quality control
    - **Statistical Analysis**: Live monitoring and adaptive sampling
    - **Metrics Calculator**: Comprehensive metrics using all three core engines
    - **Validation Dashboard**: Web interface and CLI for experiment management
    
    The validation framework tests key hypotheses:
    - Sacred geometry constraints improve creative output quality
    - Hardware entropy produces more unique outputs than software randomness
    - Archetypal frequencies accurately predict user preferences
    - Mathematical discovery produces higher quality than AI generation
    """,
    version="1.0.0",
    contact={
        "name": "MUSE Platform Team",
        "url": "https://muse.platform",
        "email": "contact@muse.platform"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="MUSE Platform API",
        version="1.0.0",
        description="Computational Platonism Creative Discovery System",
        routes=app.routes,
    )
    
    # Add custom schema properties
    openapi_schema["info"]["x-logo"] = {
        "url": "https://muse.platform/logo.png",
        "altText": "MUSE Platform Logo"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.muse.platform", "description": "Production server"}
    ]
    
    # Add tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "Core API",
            "description": "Core discovery and frequency signature endpoints"
        },
        {
            "name": "Integration",
            "description": "Advanced integration endpoints for live discovery"
        },
        {
            "name": "Community",
            "description": "Community features and social interactions"
        },
        {
            "name": "Music",
            "description": "Musical phrase generation and real-time audio synthesis"
        },
        {
            "name": "Validation",
            "description": "Validation framework for empirical testing of Computational Platonism"
        },
        {
            "name": "Health",
            "description": "System health and monitoring endpoints"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Global exception: {exc}")
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": "server_error",
            "timestamp": str(datetime.utcnow())
        }
    )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "type": "http_error",
            "timestamp": str(datetime.utcnow())
        }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    import time
    
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url}")
    
    # Process request
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"📤 {response.status_code} {request.method} {request.url} ({process_time:.3f}s)")
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ ERROR {request.method} {request.url} ({process_time:.3f}s): {e}")
        raise


# Health check endpoints
@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint with platform information
    
    Returns basic information about the MUSE Platform and its
    Computational Platonism approach to creative discovery.
    """
    return {
        "platform": "MUSE - Computational Platonism Creative Discovery",
        "version": "1.0.0",
        "philosophy": "Mathematical discovery of pre-existing creative forms",
        "engines": {
            "frequency_engine": "Archetypal frequency mapping",
            "sacred_geometry": "Universal mathematical proportions",
            "semantic_projection": "Meaning-mathematics bridge"
        },
        "features": [
            "Personality-based frequency signatures",
            "Real-time creative discovery",
            "Sacred geometry optimization",
            "Community resonance matching",
            "Collaborative discovery sessions",
            "Comprehensive validation framework",
            "Empirical testing of Computational Platonism",
            "Real-time statistical analysis",
            "Automated participant recruitment"
        ],
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint
    
    Returns detailed health information about all system components
    including database connection, engine status, and performance metrics.
    """
    try:
        # Get database health
        db_health = db_manager.health_check()
        
        # Calculate overall health
        overall_health = "healthy" if db_health["status"] == "healthy" else "unhealthy"
        
        health_data = {
            "status": overall_health,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": {
                    "status": db_health["status"],
                    "connection": db_health["connection_test"],
                    "tables": len(db_health["table_info"]),
                    "engine": db_health["engine_info"].get("dialect", "unknown")
                },
                "core_engines": {
                    "frequency_engine": "operational",
                    "sacred_geometry_calculator": "operational",
                    "semantic_projection_engine": "operational"
                },
                "services": {
                    "discovery_orchestrator": "operational",
                    "resonance_matcher": "operational",
                    "community_curator": "operational"
                },
                "validation_framework": {
                    "mathematical_validation": "operational",
                    "validation_dashboard": "operational",
                    "participant_recruitment": "operational",
                    "automated_data_collection": "operational",
                    "real_time_analysis": "operational",
                    "metrics_calculator": "operational"
                }
            },
            "api_endpoints": {
                "core_api": "operational",
                "integration_api": "operational",
                "community_api": "operational",
                "validation_api": "operational"
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@app.get("/metrics", tags=["Health"])
async def get_metrics():
    """
    Get platform metrics and analytics
    
    Returns performance metrics, usage statistics, and system information
    for monitoring and optimization purposes.
    """
    try:
        # Get database metrics
        db_health = db_manager.health_check()
        
        # Calculate basic metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": {
                "status": db_health["status"],
                "connection_test": db_health["connection_test"],
                "tables": len(db_health["table_info"]),
                "engine_info": db_health["engine_info"]
            },
            "api": {
                "endpoints_count": len(app.routes),
                "middleware_count": len(app.user_middleware),
                "documentation_url": "/docs"
            },
            "system": {
                "platform": "MUSE Computational Platonism",
                "version": "1.0.0",
                "engines": ["frequency", "sacred_geometry", "semantic_projection"],
                "services": ["discovery_orchestrator", "resonance_matcher", "community_curator"],
                "validation_framework": ["mathematical_validation", "validation_dashboard", "participant_recruitment", "automated_data_collection", "real_time_analysis", "metrics_calculator"]
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect metrics")


# Include API routers
app.include_router(
    main_api.router,
    prefix="/api/v1",
    tags=["Core API"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

app.include_router(
    integration_api.router,
    prefix="/api/v1/integration",
    tags=["Integration"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

app.include_router(
    community_api.router,
    prefix="/api/v1/community",
    tags=["Community"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

app.include_router(
    music_api.router,
    tags=["Music"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

app.include_router(
    validation_router,
    prefix="/api/v1",
    tags=["Validation"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)


# Custom docs endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with MUSE branding"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="MUSE Platform API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://muse.platform/favicon.ico"
    )


if __name__ == "__main__":
    """
    Application entry point
    
    Runs the MUSE Platform with production-ready configuration.
    """
    import os
    from datetime import datetime
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    
    # Log startup configuration
    logger.info("🚀 Starting MUSE Platform Server")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   Reload: {reload}")
    logger.info(f"   Workers: {workers}")
    logger.info(f"   Time: {datetime.utcnow().isoformat()}")
    
    # Run application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
        access_log=True
    )