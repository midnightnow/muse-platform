"""
MUSE API Module

This module contains all FastAPI routers and endpoints for the MUSE platform.
It provides web interfaces for:

- Core MUSE functionality (assessment, signatures, sessions)
- Live discovery endpoints (real-time poetry discovery)
- Community features (sharing, following, resonance matching)
- Mathematical validation (research and empirical testing)

The API is designed to expose MUSE's Computational Platonism capabilities
through RESTful endpoints with proper authentication, validation, and
error handling.
"""

from typing import Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

# API Configuration
API_VERSION = "v1"
API_PREFIX = "/api/muse"
API_TITLE = "MUSE Platform API"
API_DESCRIPTION = "Mathematical Universal Sacred Expression - Computational Platonism API"

# Import routers (these will be imported when the modules are created)
try:
    from .main import muse_router
    MAIN_ROUTER_AVAILABLE = True
except ImportError:
    logger.warning("Main router not available")
    MAIN_ROUTER_AVAILABLE = False

try:
    from .integration import live_muse_router
    INTEGRATION_ROUTER_AVAILABLE = True
except ImportError:
    logger.warning("Integration router not available")
    INTEGRATION_ROUTER_AVAILABLE = False

try:
    from .community import community_router
    COMMUNITY_ROUTER_AVAILABLE = True
except ImportError:
    logger.warning("Community router not available")
    COMMUNITY_ROUTER_AVAILABLE = False

# Router registry
ROUTER_REGISTRY = {}

if MAIN_ROUTER_AVAILABLE:
    ROUTER_REGISTRY["main"] = muse_router

if INTEGRATION_ROUTER_AVAILABLE:
    ROUTER_REGISTRY["integration"] = live_muse_router

if COMMUNITY_ROUTER_AVAILABLE:
    ROUTER_REGISTRY["community"] = community_router

# API metadata
API_METADATA = {
    "version": API_VERSION,
    "prefix": API_PREFIX,
    "title": API_TITLE,
    "description": API_DESCRIPTION,
    "philosophy": "Computational Platonism",
    "approach": "Creative Discovery",
    "routers_available": list(ROUTER_REGISTRY.keys()),
    "total_routers": len(ROUTER_REGISTRY)
}

def get_api_info() -> Dict[str, Any]:
    """Get API information and status"""
    return {
        "metadata": API_METADATA,
        "routers": {
            "main": {
                "available": MAIN_ROUTER_AVAILABLE,
                "description": "Core MUSE functionality - assessment, signatures, sessions",
                "endpoints": ["assessment", "signatures", "sessions"]
            },
            "integration": {
                "available": INTEGRATION_ROUTER_AVAILABLE,
                "description": "Live discovery endpoints for real-time poetry discovery",
                "endpoints": ["discover-poem", "optimize-constraints", "stream-discovery"]
            },
            "community": {
                "available": COMMUNITY_ROUTER_AVAILABLE,
                "description": "Community features - sharing, following, resonance matching",
                "endpoints": ["profiles", "creations", "feed", "kindred", "collaborative"]
            }
        },
        "status": "operational" if len(ROUTER_REGISTRY) > 0 else "degraded"
    }

def get_available_routers():
    """Get all available routers"""
    return ROUTER_REGISTRY

def validate_api_setup():
    """Validate API setup and configuration"""
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check router availability
    expected_routers = ["main", "integration", "community"]
    for router_name in expected_routers:
        if router_name not in ROUTER_REGISTRY:
            validation_results["warnings"].append(f"Router '{router_name}' not available")
    
    # Check if we have at least one router
    if len(ROUTER_REGISTRY) == 0:
        validation_results["valid"] = False
        validation_results["errors"].append("No routers available")
    
    return validation_results

# Export available routers
__all__ = []

if MAIN_ROUTER_AVAILABLE:
    __all__.append("muse_router")

if INTEGRATION_ROUTER_AVAILABLE:
    __all__.append("live_muse_router")

if COMMUNITY_ROUTER_AVAILABLE:
    __all__.append("community_router")

# Add utility functions
__all__.extend([
    "get_api_info",
    "get_available_routers", 
    "validate_api_setup",
    "API_METADATA",
    "ROUTER_REGISTRY"
])

# Log API initialization
logger.info(f"MUSE API module initialized with {len(ROUTER_REGISTRY)} routers")
if ROUTER_REGISTRY:
    logger.info(f"Available routers: {list(ROUTER_REGISTRY.keys())}")
else:
    logger.warning("No routers available - API functionality limited")