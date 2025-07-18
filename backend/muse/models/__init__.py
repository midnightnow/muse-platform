"""
MUSE Data Models

This module contains all SQLAlchemy models for the MUSE platform's
frequency-based social network and creative discovery system.

Models:
- UserProfile: User accounts with frequency signatures and social metrics
- FrequencySignature: Archetypal frequency signatures for creative resonance
- CommunityCreation: Shared creative discoveries in the community
- Comment: Comments on community creations
- Like: Likes on creations and comments
- Follow: User following relationships
- CollaborativeSession: Multi-user creative collaboration sessions
- SessionParticipant: Participants in collaborative sessions
- ResonanceCache: Cache for expensive resonance calculations
- CommunityAnalytics: Community-wide analytics and insights

The models are designed to support:
- Frequency-based social matching
- Mathematical creativity discovery
- Community engagement and interaction
- Real-time collaboration
- Performance optimization through caching
"""

from typing import Dict, Any, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import the declarative base
from .community import Base

# Import all models
from .community import (
    UserProfile,
    FrequencySignature,
    CommunityCreation,
    Comment,
    Like,
    Follow,
    CollaborativeSession,
    SessionParticipant,
    ResonanceCache,
    CommunityAnalytics
)

# Model registry for introspection
MODEL_REGISTRY = {
    "UserProfile": UserProfile,
    "FrequencySignature": FrequencySignature,
    "CommunityCreation": CommunityCreation,
    "Comment": Comment,
    "Like": Like,
    "Follow": Follow,
    "CollaborativeSession": CollaborativeSession,
    "SessionParticipant": SessionParticipant,
    "ResonanceCache": ResonanceCache,
    "CommunityAnalytics": CommunityAnalytics
}

# Model metadata
MODEL_METADATA = {
    "total_models": len(MODEL_REGISTRY),
    "core_models": ["UserProfile", "FrequencySignature", "CommunityCreation"],
    "social_models": ["Comment", "Like", "Follow"],
    "collaboration_models": ["CollaborativeSession", "SessionParticipant"],
    "optimization_models": ["ResonanceCache", "CommunityAnalytics"],
    "database_engine": "SQLAlchemy",
    "supports_postgresql": True,
    "supports_sqlite": True
}

def get_model_info() -> Dict[str, Any]:
    """Get information about all models"""
    model_info = {}
    
    for model_name, model_class in MODEL_REGISTRY.items():
        table = model_class.__table__
        
        model_info[model_name] = {
            "table_name": table.name,
            "columns": len(table.columns),
            "indexes": len(table.indexes),
            "foreign_keys": len(table.foreign_keys),
            "primary_key": [col.name for col in table.primary_key.columns],
            "description": getattr(model_class, "__doc__", "").strip().split('\n')[0] if hasattr(model_class, "__doc__") else ""
        }
    
    return model_info

def get_model_relationships() -> Dict[str, List[str]]:
    """Get relationship information between models"""
    relationships = {}
    
    for model_name, model_class in MODEL_REGISTRY.items():
        model_relationships = []
        
        # Get relationships from the model's __mapper__
        if hasattr(model_class, '__mapper__'):
            for rel in model_class.__mapper__.relationships:
                target_model = rel.mapper.class_.__name__
                model_relationships.append(f"{rel.key} -> {target_model}")
        
        relationships[model_name] = model_relationships
    
    return relationships

def validate_models() -> Dict[str, Any]:
    """Validate model definitions and relationships"""
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "model_count": len(MODEL_REGISTRY),
        "tables_defined": 0
    }
    
    try:
        # Check each model
        for model_name, model_class in MODEL_REGISTRY.items():
            # Check if model has a table
            if hasattr(model_class, '__table__'):
                validation_results["tables_defined"] += 1
            else:
                validation_results["errors"].append(f"Model {model_name} has no table definition")
                validation_results["valid"] = False
            
            # Check if model has required methods
            if not hasattr(model_class, 'to_dict'):
                validation_results["warnings"].append(f"Model {model_name} missing to_dict method")
        
        # Check for circular imports
        if validation_results["tables_defined"] != len(MODEL_REGISTRY):
            validation_results["errors"].append("Not all models have table definitions")
            validation_results["valid"] = False
        
    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Model validation failed: {str(e)}")
    
    return validation_results

def get_database_schema() -> Dict[str, Any]:
    """Get complete database schema information"""
    schema = {
        "metadata": MODEL_METADATA,
        "models": get_model_info(),
        "relationships": get_model_relationships(),
        "validation": validate_models()
    }
    
    return schema

# Export all models
__all__ = [
    # Base
    "Base",
    
    # Core models
    "UserProfile",
    "FrequencySignature", 
    "CommunityCreation",
    
    # Social models
    "Comment",
    "Like",
    "Follow",
    
    # Collaboration models
    "CollaborativeSession",
    "SessionParticipant",
    
    # Optimization models
    "ResonanceCache",
    "CommunityAnalytics",
    
    # Utility functions
    "get_model_info",
    "get_model_relationships",
    "validate_models",
    "get_database_schema",
    "MODEL_REGISTRY",
    "MODEL_METADATA"
]

# Log model initialization
logger.info(f"MUSE models initialized: {len(MODEL_REGISTRY)} models loaded")
validation = validate_models()
if validation["valid"]:
    logger.info("All models validated successfully")
else:
    logger.warning(f"Model validation issues: {validation['errors']}")
    if validation["warnings"]:
        logger.info(f"Model warnings: {validation['warnings']}")

# Check database schema on import
try:
    schema_info = get_database_schema()
    logger.info(f"Database schema ready: {schema_info['metadata']['total_models']} models, {schema_info['validation']['tables_defined']} tables")
except Exception as e:
    logger.error(f"Database schema check failed: {e}")