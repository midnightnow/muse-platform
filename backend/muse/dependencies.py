"""
Dependency Injection System for MUSE Platform

This module provides a centralized dependency injection system for all
core engines, services, and database connections. It ensures proper
initialization, singleton patterns, and clean resource management.
"""

import logging
from typing import Generator, Optional
from functools import lru_cache
from sqlalchemy.orm import Session

from database import db_manager
from muse.core.frequency_engine import MuseFrequencyEngine
from muse.core.sacred_geometry_calculator import SacredGeometryCalculator
from muse.core.semantic_projection_engine import SemanticProjectionEngine
from muse.services.discovery_orchestrator import DiscoveryOrchestrator
from muse.services.resonance_matcher import ResonanceMatcher
from muse.services.community_curator import CommunityCurator


logger = logging.getLogger(__name__)


# Database dependencies
def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI
    
    Provides a database session with proper error handling
    and automatic cleanup.
    """
    db = db_manager.SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# Core engine dependencies (singletons)
@lru_cache()
def get_frequency_engine() -> MuseFrequencyEngine:
    """
    Get frequency engine singleton instance
    
    Returns the archetypal frequency engine used for generating
    and managing user frequency signatures.
    """
    logger.debug("Creating MuseFrequencyEngine instance")
    return MuseFrequencyEngine()


@lru_cache()
def get_sacred_geometry_calculator() -> SacredGeometryCalculator:
    """
    Get sacred geometry calculator singleton instance
    
    Returns the sacred geometry calculator used for applying
    mathematical constants and proportions to creative works.
    """
    logger.debug("Creating SacredGeometryCalculator instance")
    return SacredGeometryCalculator()


@lru_cache()
def get_semantic_projection_engine() -> SemanticProjectionEngine:
    """
    Get semantic projection engine singleton instance
    
    Returns the semantic projection engine used for bridging
    meaning and mathematics in creative discovery.
    """
    logger.debug("Creating SemanticProjectionEngine instance")
    return SemanticProjectionEngine()


# Service dependencies
def get_discovery_orchestrator(
    frequency_engine: MuseFrequencyEngine = None,
    geometry_calculator: SacredGeometryCalculator = None,
    semantic_engine: SemanticProjectionEngine = None
) -> DiscoveryOrchestrator:
    """
    Get discovery orchestrator instance
    
    Creates a discovery orchestrator that coordinates all three
    core engines for unified creative discovery sessions.
    """
    if frequency_engine is None:
        frequency_engine = get_frequency_engine()
    if geometry_calculator is None:
        geometry_calculator = get_sacred_geometry_calculator()
    if semantic_engine is None:
        semantic_engine = get_semantic_projection_engine()
    
    logger.debug("Creating DiscoveryOrchestrator instance")
    return DiscoveryOrchestrator(frequency_engine, geometry_calculator, semantic_engine)


def get_resonance_matcher(
    frequency_engine: MuseFrequencyEngine = None,
    geometry_calculator: SacredGeometryCalculator = None
) -> ResonanceMatcher:
    """
    Get resonance matcher instance
    
    Creates a resonance matcher for calculating archetypal
    similarity between users.
    """
    if frequency_engine is None:
        frequency_engine = get_frequency_engine()
    if geometry_calculator is None:
        geometry_calculator = get_sacred_geometry_calculator()
    
    logger.debug("Creating ResonanceMatcher instance")
    return ResonanceMatcher(frequency_engine, geometry_calculator)


def get_community_curator(
    resonance_matcher: ResonanceMatcher = None
) -> CommunityCurator:
    """
    Get community curator instance
    
    Creates a community curator for managing content curation
    and social interactions based on archetypal resonance.
    """
    if resonance_matcher is None:
        resonance_matcher = get_resonance_matcher()
    
    logger.debug("Creating CommunityCurator instance")
    return CommunityCurator(resonance_matcher)


# Dependency factory for complete system
class DependencyFactory:
    """
    Factory class for creating and managing all system dependencies
    
    Provides a centralized way to access all core engines and services
    with proper initialization and lifecycle management.
    """
    
    def __init__(self):
        """Initialize the dependency factory"""
        self._frequency_engine: Optional[MuseFrequencyEngine] = None
        self._geometry_calculator: Optional[SacredGeometryCalculator] = None
        self._semantic_engine: Optional[SemanticProjectionEngine] = None
        self._discovery_orchestrator: Optional[DiscoveryOrchestrator] = None
        self._resonance_matcher: Optional[ResonanceMatcher] = None
        self._community_curator: Optional[CommunityCurator] = None
        
        logger.info("DependencyFactory initialized")
    
    @property
    def frequency_engine(self) -> MuseFrequencyEngine:
        """Get frequency engine instance"""
        if self._frequency_engine is None:
            self._frequency_engine = get_frequency_engine()
        return self._frequency_engine
    
    @property
    def geometry_calculator(self) -> SacredGeometryCalculator:
        """Get sacred geometry calculator instance"""
        if self._geometry_calculator is None:
            self._geometry_calculator = get_sacred_geometry_calculator()
        return self._geometry_calculator
    
    @property
    def semantic_engine(self) -> SemanticProjectionEngine:
        """Get semantic projection engine instance"""
        if self._semantic_engine is None:
            self._semantic_engine = get_semantic_projection_engine()
        return self._semantic_engine
    
    @property
    def discovery_orchestrator(self) -> DiscoveryOrchestrator:
        """Get discovery orchestrator instance"""
        if self._discovery_orchestrator is None:
            self._discovery_orchestrator = get_discovery_orchestrator(
                self.frequency_engine,
                self.geometry_calculator,
                self.semantic_engine
            )
        return self._discovery_orchestrator
    
    @property
    def resonance_matcher(self) -> ResonanceMatcher:
        """Get resonance matcher instance"""
        if self._resonance_matcher is None:
            self._resonance_matcher = get_resonance_matcher(
                self.frequency_engine,
                self.geometry_calculator
            )
        return self._resonance_matcher
    
    @property
    def community_curator(self) -> CommunityCurator:
        """Get community curator instance"""
        if self._community_curator is None:
            self._community_curator = get_community_curator(
                self.resonance_matcher
            )
        return self._community_curator
    
    def get_all_engines(self) -> dict:
        """Get all core engines as a dictionary"""
        return {
            'frequency_engine': self.frequency_engine,
            'geometry_calculator': self.geometry_calculator,
            'semantic_engine': self.semantic_engine
        }
    
    def get_all_services(self) -> dict:
        """Get all services as a dictionary"""
        return {
            'discovery_orchestrator': self.discovery_orchestrator,
            'resonance_matcher': self.resonance_matcher,
            'community_curator': self.community_curator
        }
    
    def health_check(self) -> dict:
        """Perform health check on all components"""
        health_data = {
            'factory_status': 'healthy',
            'components': {}
        }
        
        try:
            # Test core engines
            health_data['components']['frequency_engine'] = {
                'status': 'healthy',
                'class': str(type(self.frequency_engine).__name__)
            }
            
            health_data['components']['geometry_calculator'] = {
                'status': 'healthy',
                'class': str(type(self.geometry_calculator).__name__)
            }
            
            health_data['components']['semantic_engine'] = {
                'status': 'healthy',
                'class': str(type(self.semantic_engine).__name__)
            }
            
            # Test services
            health_data['components']['discovery_orchestrator'] = {
                'status': 'healthy',
                'class': str(type(self.discovery_orchestrator).__name__)
            }
            
            health_data['components']['resonance_matcher'] = {
                'status': 'healthy',
                'class': str(type(self.resonance_matcher).__name__)
            }
            
            health_data['components']['community_curator'] = {
                'status': 'healthy',
                'class': str(type(self.community_curator).__name__)
            }
            
        except Exception as e:
            health_data['factory_status'] = 'unhealthy'
            health_data['error'] = str(e)
            logger.error(f"Dependency factory health check failed: {e}")
        
        return health_data
    
    def cleanup(self):
        """Clean up all resources"""
        logger.info("Cleaning up dependency factory")
        
        # Clear all cached instances
        self._frequency_engine = None
        self._geometry_calculator = None
        self._semantic_engine = None
        self._discovery_orchestrator = None
        self._resonance_matcher = None
        self._community_curator = None
        
        # Clear LRU cache
        get_frequency_engine.cache_clear()
        get_sacred_geometry_calculator.cache_clear()
        get_semantic_projection_engine.cache_clear()
        
        logger.info("Dependency factory cleanup complete")


# Global dependency factory instance
dependency_factory = DependencyFactory()


# FastAPI dependency functions using the factory
def get_frequency_engine_dependency() -> MuseFrequencyEngine:
    """FastAPI dependency for frequency engine"""
    return dependency_factory.frequency_engine


def get_geometry_calculator_dependency() -> SacredGeometryCalculator:
    """FastAPI dependency for geometry calculator"""
    return dependency_factory.geometry_calculator


def get_semantic_engine_dependency() -> SemanticProjectionEngine:
    """FastAPI dependency for semantic engine"""
    return dependency_factory.semantic_engine


def get_discovery_orchestrator_dependency() -> DiscoveryOrchestrator:
    """FastAPI dependency for discovery orchestrator"""
    return dependency_factory.discovery_orchestrator


def get_resonance_matcher_dependency() -> ResonanceMatcher:
    """FastAPI dependency for resonance matcher"""
    return dependency_factory.resonance_matcher


def get_community_curator_dependency() -> CommunityCurator:
    """FastAPI dependency for community curator"""
    return dependency_factory.community_curator


# System-wide dependency context manager
class DependencyContext:
    """
    Context manager for dependency lifecycle management
    
    Provides a clean way to manage dependencies with proper
    initialization and cleanup.
    """
    
    def __init__(self):
        self.factory = DependencyFactory()
    
    def __enter__(self):
        logger.info("Entering dependency context")
        return self.factory
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Exiting dependency context")
        self.factory.cleanup()
        
        if exc_type is not None:
            logger.error(f"Exception in dependency context: {exc_type.__name__}: {exc_val}")
        
        return False  # Don't suppress exceptions


# Utility functions for dependency testing
def test_all_dependencies() -> dict:
    """
    Test all dependencies for proper initialization
    
    Returns a comprehensive test result showing the status
    of all core engines and services.
    """
    test_results = {
        'timestamp': str(datetime.utcnow()),
        'overall_status': 'unknown',
        'tests': {}
    }
    
    try:
        with DependencyContext() as factory:
            # Test core engines
            test_results['tests']['frequency_engine'] = {
                'status': 'pass',
                'instance': str(type(factory.frequency_engine).__name__)
            }
            
            test_results['tests']['geometry_calculator'] = {
                'status': 'pass',
                'instance': str(type(factory.geometry_calculator).__name__)
            }
            
            test_results['tests']['semantic_engine'] = {
                'status': 'pass',
                'instance': str(type(factory.semantic_engine).__name__)
            }
            
            # Test services
            test_results['tests']['discovery_orchestrator'] = {
                'status': 'pass',
                'instance': str(type(factory.discovery_orchestrator).__name__)
            }
            
            test_results['tests']['resonance_matcher'] = {
                'status': 'pass',
                'instance': str(type(factory.resonance_matcher).__name__)
            }
            
            test_results['tests']['community_curator'] = {
                'status': 'pass',
                'instance': str(type(factory.community_curator).__name__)
            }
            
            # Test database dependency
            with get_db() as db:
                test_results['tests']['database'] = {
                    'status': 'pass',
                    'connection': True
                }
        
        test_results['overall_status'] = 'pass'
        
    except Exception as e:
        test_results['overall_status'] = 'fail'
        test_results['error'] = str(e)
        logger.error(f"Dependency test failed: {e}")
    
    return test_results


def get_dependency_info() -> dict:
    """
    Get information about all dependencies
    
    Returns detailed information about the current state
    of all dependencies and their configurations.
    """
    info = {
        'timestamp': str(datetime.utcnow()),
        'factory_status': 'active',
        'components': {}
    }
    
    try:
        # Core engines info
        info['components']['core_engines'] = {
            'frequency_engine': {
                'class': 'MuseFrequencyEngine',
                'description': 'Archetypal frequency mapping and signature generation'
            },
            'geometry_calculator': {
                'class': 'SacredGeometryCalculator',
                'description': 'Sacred mathematical constants and proportions'
            },
            'semantic_engine': {
                'class': 'SemanticProjectionEngine',
                'description': 'Meaning-mathematics bridging and semantic analysis'
            }
        }
        
        # Services info
        info['components']['services'] = {
            'discovery_orchestrator': {
                'class': 'DiscoveryOrchestrator',
                'description': 'Coordinates all engines for unified discovery'
            },
            'resonance_matcher': {
                'class': 'ResonanceMatcher',
                'description': 'Calculates archetypal similarity between users'
            },
            'community_curator': {
                'class': 'CommunityCurator',
                'description': 'Manages content curation and social interactions'
            }
        }
        
        # Database info
        info['components']['database'] = {
            'class': 'SQLAlchemy Session',
            'description': 'Database session management and connection pooling'
        }
        
    except Exception as e:
        info['factory_status'] = 'error'
        info['error'] = str(e)
        logger.error(f"Failed to get dependency info: {e}")
    
    return info