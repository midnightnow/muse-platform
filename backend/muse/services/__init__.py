"""
MUSE Services Module

This module contains business logic and service classes for the MUSE platform.
Services handle complex operations that bridge the core mathematical engines
with the data models and API endpoints.

Services:
- ResonanceMatcher: Sophisticated archetypal similarity calculations
- DiscoveryOrchestrator: Coordinates the discovery process across engines
- CommunityManager: Manages community interactions and social features
- ValidationService: Handles empirical validation of mathematical claims
- CollaborationService: Manages multi-user creative sessions
- AnalyticsService: Provides insights and metrics

The services implement the business logic for:
- Frequency-based user matching
- Creative discovery orchestration
- Community engagement
- Real-time collaboration
- Performance analytics
"""

from typing import Dict, Any, List, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Service registry
SERVICE_REGISTRY = {}

# Import services as they become available
try:
    from .resonance_matcher import ResonanceMatcher
    SERVICE_REGISTRY["resonance_matcher"] = ResonanceMatcher
    RESONANCE_MATCHER_AVAILABLE = True
except ImportError:
    logger.warning("ResonanceMatcher service not available")
    RESONANCE_MATCHER_AVAILABLE = False

# Service metadata
SERVICE_METADATA = {
    "total_services": len(SERVICE_REGISTRY),
    "available_services": list(SERVICE_REGISTRY.keys()),
    "core_services": ["resonance_matcher"],
    "planned_services": [
        "discovery_orchestrator",
        "community_manager", 
        "validation_service",
        "collaboration_service",
        "analytics_service"
    ]
}

def get_service_info() -> Dict[str, Any]:
    """Get information about all services"""
    return {
        "metadata": SERVICE_METADATA,
        "services": {
            "resonance_matcher": {
                "available": RESONANCE_MATCHER_AVAILABLE,
                "description": "Sophisticated archetypal similarity calculations for user matching",
                "methods": ["calculate_archetypal_similarity", "find_kindred_spirits", "find_resonant_creations"],
                "requires": ["database_session", "frequency_signatures"]
            },
            "discovery_orchestrator": {
                "available": False,
                "description": "Coordinates the discovery process across all mathematical engines",
                "methods": ["orchestrate_discovery", "optimize_constraints", "validate_results"],
                "requires": ["sacred_geometry", "semantic_projection", "frequency_engine"]
            },
            "community_manager": {
                "available": False,
                "description": "Manages community interactions and social features",
                "methods": ["manage_feed", "handle_interactions", "moderate_content"],
                "requires": ["database_session", "user_profiles", "community_creations"]
            },
            "validation_service": {
                "available": False,
                "description": "Handles empirical validation of mathematical claims",
                "methods": ["setup_experiments", "collect_data", "analyze_results"],
                "requires": ["statistical_analysis", "participant_recruitment"]
            },
            "collaboration_service": {
                "available": False,
                "description": "Manages multi-user creative sessions",
                "methods": ["create_session", "manage_participants", "coordinate_discovery"],
                "requires": ["websocket_connection", "entropy_mixing", "real_time_sync"]
            },
            "analytics_service": {
                "available": False,
                "description": "Provides insights and metrics for platform optimization",
                "methods": ["generate_reports", "track_metrics", "provide_insights"],
                "requires": ["database_session", "statistical_analysis", "visualization"]
            }
        }
    }

def get_available_services() -> Dict[str, Any]:
    """Get all available services"""
    return SERVICE_REGISTRY

def validate_services() -> Dict[str, Any]:
    """Validate service setup and dependencies"""
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "services_available": len(SERVICE_REGISTRY),
        "services_expected": len(SERVICE_METADATA["core_services"])
    }
    
    # Check core services
    for service_name in SERVICE_METADATA["core_services"]:
        if service_name not in SERVICE_REGISTRY:
            validation_results["warnings"].append(f"Core service '{service_name}' not available")
    
    # Check if we have at least one service
    if len(SERVICE_REGISTRY) == 0:
        validation_results["valid"] = False
        validation_results["errors"].append("No services available")
    
    return validation_results

def initialize_services(database_session=None) -> Dict[str, Any]:
    """Initialize all available services with dependencies"""
    initialized_services = {}
    
    try:
        # Initialize ResonanceMatcher if available
        if RESONANCE_MATCHER_AVAILABLE and database_session:
            initialized_services["resonance_matcher"] = ResonanceMatcher(database_session)
            logger.info("ResonanceMatcher initialized")
        
        # Add other services as they become available
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        raise
    
    return initialized_services

class ServiceManager:
    """
    Central service manager for MUSE platform
    """
    
    def __init__(self, database_session=None):
        """
        Initialize service manager
        
        Args:
            database_session: Database session for services that require it
        """
        self.database_session = database_session
        self.services = {}
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all available services"""
        try:
            self.services = initialize_services(self.database_session)
            logger.info(f"ServiceManager initialized with {len(self.services)} services")
        except Exception as e:
            logger.error(f"ServiceManager initialization failed: {e}")
            raise
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get a service by name
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance or None if not available
        """
        return self.services.get(service_name)
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all initialized services"""
        return self.services.copy()
    
    def reinitialize_service(self, service_name: str) -> bool:
        """
        Reinitialize a specific service
        
        Args:
            service_name: Name of the service to reinitialize
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if service_name == "resonance_matcher" and RESONANCE_MATCHER_AVAILABLE:
                self.services["resonance_matcher"] = ResonanceMatcher(self.database_session)
                logger.info(f"Service '{service_name}' reinitialized")
                return True
            
            logger.warning(f"Service '{service_name}' not available for reinitialization")
            return False
            
        except Exception as e:
            logger.error(f"Failed to reinitialize service '{service_name}': {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all services
        
        Returns:
            Health check results
        """
        health_results = {
            "status": "healthy",
            "services": {},
            "total_services": len(self.services),
            "healthy_services": 0,
            "unhealthy_services": 0
        }
        
        for service_name, service_instance in self.services.items():
            try:
                # Basic health check - see if service is callable
                if hasattr(service_instance, '__class__'):
                    health_results["services"][service_name] = {
                        "status": "healthy",
                        "class": service_instance.__class__.__name__,
                        "methods": [method for method in dir(service_instance) if not method.startswith('_')]
                    }
                    health_results["healthy_services"] += 1
                else:
                    health_results["services"][service_name] = {
                        "status": "unhealthy",
                        "error": "Service instance not properly initialized"
                    }
                    health_results["unhealthy_services"] += 1
                    
            except Exception as e:
                health_results["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_results["unhealthy_services"] += 1
        
        # Overall status
        if health_results["unhealthy_services"] > 0:
            health_results["status"] = "degraded" if health_results["healthy_services"] > 0 else "unhealthy"
        
        return health_results

# Export available services and utilities
__all__ = []

if RESONANCE_MATCHER_AVAILABLE:
    __all__.append("ResonanceMatcher")

# Add utility functions and classes
__all__.extend([
    "ServiceManager",
    "get_service_info",
    "get_available_services",
    "validate_services",
    "initialize_services",
    "SERVICE_REGISTRY",
    "SERVICE_METADATA"
])

# Log service initialization
logger.info(f"MUSE services module initialized with {len(SERVICE_REGISTRY)} services")
validation = validate_services()
if validation["valid"]:
    logger.info("Services validated successfully")
else:
    logger.warning(f"Service validation issues: {validation['errors']}")
    if validation["warnings"]:
        logger.info(f"Service warnings: {validation['warnings']}")