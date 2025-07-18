"""
MUSE Validation Framework

This module provides comprehensive empirical validation for MUSE's
Computational Platonism claims through rigorous scientific methodology.

Components:
- MathematicalValidationFramework: Core validation system for testing hypotheses
- ValidationDashboard: Dashboard for managing experiments and viewing results
- ParticipantRecruitmentSystem: Manages participant recruitment for studies
- AutomatedDataCollectionPipeline: Automated data collection during experiments
- RealTimeStatisticalAnalysis: Real-time statistical analysis of ongoing experiments
- MetricsCalculator: Calculates comprehensive quality metrics for creative outputs

The validation framework tests key claims:
1. Sacred geometry constraints improve creative output quality
2. Hardware entropy produces more unique creative discoveries
3. Archetypal frequency signatures predict user creative preferences
4. Mathematical discovery approach outperforms AI generation

This scientific approach transforms MUSE from a philosophical concept
into an empirically validated platform for mathematical creativity.
"""

from typing import Dict, Any, List, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Validation framework registry
VALIDATION_REGISTRY = {}

# Import validation components as they become available
try:
    from .mathematical_validation_framework import (
        MUSEValidationFramework,
        ValidationExperiment,
        CreativeOutput,
        UserProfile as ValidationUserProfile,
        ExperimentStatus
    )
    VALIDATION_REGISTRY["mathematical_framework"] = MUSEValidationFramework
    MATHEMATICAL_FRAMEWORK_AVAILABLE = True
except ImportError:
    logger.warning("Mathematical validation framework not available")
    MATHEMATICAL_FRAMEWORK_AVAILABLE = False

try:
    from .validation_dashboard import ValidationDashboard, validation_router
    VALIDATION_REGISTRY["dashboard"] = ValidationDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    logger.warning("Validation dashboard not available")
    DASHBOARD_AVAILABLE = False

try:
    from .participant_recruitment_system import (
        ParticipantRecruitmentSystem,
        RecruitmentStrategy,
        Participant
    )
    VALIDATION_REGISTRY["recruitment"] = ParticipantRecruitmentSystem
    RECRUITMENT_AVAILABLE = True
except ImportError:
    logger.warning("Participant recruitment system not available")
    RECRUITMENT_AVAILABLE = False

try:
    from .automated_data_collection_pipeline import (
        AutomatedDataCollectionPipeline,
        DataCollectionSession
    )
    VALIDATION_REGISTRY["data_collection"] = AutomatedDataCollectionPipeline
    DATA_COLLECTION_AVAILABLE = True
except ImportError:
    logger.warning("Automated data collection pipeline not available")
    DATA_COLLECTION_AVAILABLE = False

try:
    from .real_time_statistical_analysis import RealTimeStatisticalAnalysis
    VALIDATION_REGISTRY["statistical_analysis"] = RealTimeStatisticalAnalysis
    STATISTICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    logger.warning("Real-time statistical analysis not available")
    STATISTICAL_ANALYSIS_AVAILABLE = False

try:
    from .metrics_calculator import MetricsCalculator
    VALIDATION_REGISTRY["metrics"] = MetricsCalculator
    METRICS_CALCULATOR_AVAILABLE = True
except ImportError:
    logger.warning("Metrics calculator not available")
    METRICS_CALCULATOR_AVAILABLE = False

# Validation metadata
VALIDATION_METADATA = {
    "total_components": len(VALIDATION_REGISTRY),
    "available_components": list(VALIDATION_REGISTRY.keys()),
    "core_hypotheses": [
        "Sacred geometry constraints improve creative output quality",
        "Hardware entropy produces more unique creative discoveries", 
        "Archetypal frequency signatures predict user creative preferences",
        "Mathematical discovery approach outperforms AI generation"
    ],
    "validation_methods": [
        "A/B testing",
        "Statistical significance testing",
        "Effect size calculation",
        "Confidence interval analysis",
        "Power analysis",
        "Longitudinal studies"
    ],
    "supported_metrics": [
        "mathematical_fitness",
        "semantic_coherence",
        "user_satisfaction",
        "discovery_uniqueness",
        "preference_accuracy",
        "archetypal_alignment"
    ]
}

def get_validation_info() -> Dict[str, Any]:
    """Get information about validation framework"""
    return {
        "metadata": VALIDATION_METADATA,
        "components": {
            "mathematical_framework": {
                "available": MATHEMATICAL_FRAMEWORK_AVAILABLE,
                "description": "Core validation system for testing MUSE hypotheses",
                "methods": ["create_experiment", "run_experiment", "analyze_results"],
                "supports": ["sacred_geometry", "entropy_uniqueness", "archetypal_prediction", "discovery_vs_generation"]
            },
            "dashboard": {
                "available": DASHBOARD_AVAILABLE,
                "description": "Management interface for experiments and results",
                "features": ["experiment_tracking", "real_time_metrics", "report_generation"],
                "interfaces": ["CLI", "FastAPI"]
            },
            "recruitment": {
                "available": RECRUITMENT_AVAILABLE,
                "description": "Participant recruitment system for validation studies",
                "strategies": ["random", "balanced_demographics", "targeted_archetype"],
                "supports": ["1000+ simulated participants"]
            },
            "data_collection": {
                "available": DATA_COLLECTION_AVAILABLE,
                "description": "Automated data collection during experiments",
                "features": ["real_time_collection", "session_simulation", "quality_metrics"],
                "collection_rate": "continuous"
            },
            "statistical_analysis": {
                "available": STATISTICAL_ANALYSIS_AVAILABLE,
                "description": "Real-time statistical analysis of ongoing experiments",
                "methods": ["t_tests", "effect_size", "confidence_intervals", "power_analysis"],
                "update_frequency": "5_minutes"
            },
            "metrics_calculator": {
                "available": METRICS_CALCULATOR_AVAILABLE,
                "description": "Comprehensive quality metrics for creative outputs",
                "categories": ["mathematical", "semantic", "archetypal", "form_specific"],
                "metrics_count": "20+"
            }
        }
    }

def get_available_components() -> Dict[str, Any]:
    """Get all available validation components"""
    return VALIDATION_REGISTRY

def validate_framework() -> Dict[str, Any]:
    """Validate the validation framework setup"""
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "components_available": len(VALIDATION_REGISTRY),
        "components_expected": 6,
        "ready_for_validation": False
    }
    
    # Check core components
    core_components = ["mathematical_framework", "dashboard", "recruitment", "data_collection"]
    missing_core = []
    
    for component in core_components:
        if component not in VALIDATION_REGISTRY:
            missing_core.append(component)
    
    if missing_core:
        validation_results["warnings"].extend([f"Core component '{comp}' not available" for comp in missing_core])
        validation_results["ready_for_validation"] = False
    else:
        validation_results["ready_for_validation"] = True
    
    # Check if we have minimum viable setup
    if len(VALIDATION_REGISTRY) < 2:
        validation_results["valid"] = False
        validation_results["errors"].append("Insufficient components for validation")
    
    return validation_results

def create_validation_suite(data_directory: str = "./validation_data") -> Dict[str, Any]:
    """
    Create a complete validation suite with all available components
    
    Args:
        data_directory: Directory for validation data storage
        
    Returns:
        Dictionary with initialized validation components
    """
    validation_suite = {}
    
    try:
        # Initialize mathematical framework
        if MATHEMATICAL_FRAMEWORK_AVAILABLE:
            validation_suite["framework"] = MUSEValidationFramework(data_directory=data_directory)
            logger.info("Mathematical validation framework initialized")
        
        # Initialize dashboard
        if DASHBOARD_AVAILABLE:
            validation_suite["dashboard"] = ValidationDashboard(data_directory=data_directory)
            logger.info("Validation dashboard initialized")
        
        # Initialize recruitment system
        if RECRUITMENT_AVAILABLE:
            validation_suite["recruitment"] = ParticipantRecruitmentSystem(data_directory=data_directory)
            logger.info("Participant recruitment system initialized")
        
        # Initialize data collection
        if DATA_COLLECTION_AVAILABLE:
            validation_suite["data_collection"] = AutomatedDataCollectionPipeline(data_directory=data_directory)
            logger.info("Automated data collection pipeline initialized")
        
        # Initialize statistical analysis
        if STATISTICAL_ANALYSIS_AVAILABLE:
            validation_suite["statistics"] = RealTimeStatisticalAnalysis(data_directory=data_directory)
            logger.info("Real-time statistical analysis initialized")
        
        # Initialize metrics calculator
        if METRICS_CALCULATOR_AVAILABLE:
            # Note: MetricsCalculator requires core engines, so we'll initialize it when needed
            validation_suite["metrics"] = "available"
            logger.info("Metrics calculator available")
        
        logger.info(f"Validation suite created with {len(validation_suite)} components")
        
    except Exception as e:
        logger.error(f"Validation suite creation failed: {e}")
        raise
    
    return validation_suite

class ValidationManager:
    """
    Central validation manager for MUSE platform
    """
    
    def __init__(self, data_directory: str = "./validation_data"):
        """
        Initialize validation manager
        
        Args:
            data_directory: Directory for validation data storage
        """
        self.data_directory = data_directory
        self.validation_suite = {}
        self.active_experiments = {}
        self._initialize_validation_suite()
    
    def _initialize_validation_suite(self):
        """Initialize the complete validation suite"""
        try:
            self.validation_suite = create_validation_suite(self.data_directory)
            logger.info(f"ValidationManager initialized with {len(self.validation_suite)} components")
        except Exception as e:
            logger.error(f"ValidationManager initialization failed: {e}")
            raise
    
    def get_component(self, component_name: str) -> Optional[Any]:
        """
        Get a validation component by name
        
        Args:
            component_name: Name of the component to retrieve
            
        Returns:
            Component instance or None if not available
        """
        return self.validation_suite.get(component_name)
    
    def start_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """
        Start a new validation experiment
        
        Args:
            experiment_config: Configuration for the experiment
            
        Returns:
            Experiment ID
        """
        if "framework" not in self.validation_suite:
            raise ValueError("Mathematical validation framework not available")
        
        framework = self.validation_suite["framework"]
        
        # Create experiment
        experiment_id = framework.create_experiment(
            hypothesis=experiment_config["hypothesis"],
            control_description=experiment_config.get("control_description", "Standard approach"),
            experimental_description=experiment_config.get("experimental_description", "MUSE approach"),
            variables=experiment_config.get("variables", ["user_satisfaction"]),
            sample_size=experiment_config.get("sample_size", 100),
            duration_days=experiment_config.get("duration_days", 30)
        )
        
        # Start experiment
        framework.start_experiment(experiment_id)
        
        # Track active experiment
        self.active_experiments[experiment_id] = {
            "config": experiment_config,
            "started_at": "now",  # In real implementation, use datetime
            "status": "running"
        }
        
        logger.info(f"Validation experiment started: {experiment_id}")
        return experiment_id
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get status of a validation experiment
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment status and results
        """
        if "framework" not in self.validation_suite:
            raise ValueError("Mathematical validation framework not available")
        
        framework = self.validation_suite["framework"]
        
        # Get experiment details
        experiment = framework.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Get real-time statistics if available
        live_stats = {}
        if "statistics" in self.validation_suite:
            try:
                live_stats = self.validation_suite["statistics"].get_latest_stats_for_experiment(experiment_id)
            except Exception as e:
                logger.warning(f"Failed to get live stats for experiment {experiment_id}: {e}")
        
        return {
            "experiment_id": experiment_id,
            "hypothesis": experiment.hypothesis,
            "status": experiment.status.value,
            "progress": experiment.get_progress_percentage(),
            "live_statistics": live_stats,
            "results": experiment.results
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on validation system
        
        Returns:
            Health check results
        """
        health_results = {
            "status": "healthy",
            "components": {},
            "total_components": len(self.validation_suite),
            "healthy_components": 0,
            "unhealthy_components": 0,
            "active_experiments": len(self.active_experiments)
        }
        
        for component_name, component_instance in self.validation_suite.items():
            try:
                if component_instance == "available":
                    health_results["components"][component_name] = {
                        "status": "available",
                        "description": "Component available for initialization"
                    }
                    health_results["healthy_components"] += 1
                elif hasattr(component_instance, '__class__'):
                    health_results["components"][component_name] = {
                        "status": "healthy",
                        "class": component_instance.__class__.__name__
                    }
                    health_results["healthy_components"] += 1
                else:
                    health_results["components"][component_name] = {
                        "status": "unhealthy",
                        "error": "Component not properly initialized"
                    }
                    health_results["unhealthy_components"] += 1
                    
            except Exception as e:
                health_results["components"][component_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_results["unhealthy_components"] += 1
        
        # Overall status
        if health_results["unhealthy_components"] > 0:
            health_results["status"] = "degraded" if health_results["healthy_components"] > 0 else "unhealthy"
        
        return health_results

# Export available components
__all__ = []

# Add components as they become available
if MATHEMATICAL_FRAMEWORK_AVAILABLE:
    __all__.extend([
        "MUSEValidationFramework",
        "ValidationExperiment", 
        "CreativeOutput",
        "ValidationUserProfile",
        "ExperimentStatus"
    ])

if DASHBOARD_AVAILABLE:
    __all__.extend(["ValidationDashboard", "validation_router"])

if RECRUITMENT_AVAILABLE:
    __all__.extend(["ParticipantRecruitmentSystem", "RecruitmentStrategy", "Participant"])

if DATA_COLLECTION_AVAILABLE:
    __all__.extend(["AutomatedDataCollectionPipeline", "DataCollectionSession"])

if STATISTICAL_ANALYSIS_AVAILABLE:
    __all__.append("RealTimeStatisticalAnalysis")

if METRICS_CALCULATOR_AVAILABLE:
    __all__.append("MetricsCalculator")

# Add utility functions and classes
__all__.extend([
    "ValidationManager",
    "get_validation_info",
    "get_available_components",
    "validate_framework",
    "create_validation_suite",
    "VALIDATION_REGISTRY",
    "VALIDATION_METADATA"
])

# Log validation framework initialization
logger.info(f"MUSE validation framework initialized with {len(VALIDATION_REGISTRY)} components")
framework_validation = validate_framework()
if framework_validation["valid"]:
    logger.info("Validation framework setup is valid")
    if framework_validation["ready_for_validation"]:
        logger.info("✅ Ready for empirical validation of MUSE claims")
    else:
        logger.warning("⚠️  Some components missing - validation capabilities limited")
else:
    logger.error("❌ Validation framework setup has errors")
    if framework_validation["errors"]:
        logger.error(f"Errors: {framework_validation['errors']}")
    if framework_validation["warnings"]:
        logger.warning(f"Warnings: {framework_validation['warnings']}")