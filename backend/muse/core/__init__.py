"""
MUSE Core Mathematical Engines

This module contains the core mathematical engines that power MUSE's
Computational Platonism approach to creative discovery.

The three engines work together to bridge the gap between mathematical
truth and creative expression:

1. SacredGeometryCalculator: Discovers optimal poetic forms using sacred
   mathematical constants (phi, pi, Fibonacci sequences)

2. SemanticProjectionEngine: Maps meaning and emotion to mathematical 
   coordinates, ensuring semantic coherence in discovered works

3. MuseFrequencyEngine: Generates archetypal frequency signatures that
   connect users to their unique creative resonance patterns

These engines transform creativity from generation to discovery,
revealing pre-existing mathematical Forms in the Platonic realm.
"""

from .sacred_geometry_calculator import SacredGeometryCalculator, SacredGeometryResult, SacredConstant
from .semantic_projection_engine import (
    SemanticProjectionEngine, 
    SemanticVector, 
    WordEmbedding, 
    ThemeProjection,
    SemanticDimension
)
from .frequency_engine import (
    MuseFrequencyEngine, 
    MuseArchetype, 
    FrequencySignature, 
    CreativeConstraint
)

__all__ = [
    # Sacred Geometry
    "SacredGeometryCalculator",
    "SacredGeometryResult", 
    "SacredConstant",
    
    # Semantic Projection
    "SemanticProjectionEngine",
    "SemanticVector",
    "WordEmbedding",
    "ThemeProjection",
    "SemanticDimension",
    
    # Frequency Engine
    "MuseFrequencyEngine",
    "MuseArchetype",
    "FrequencySignature",
    "CreativeConstraint"
]

def get_engine_info():
    """Get information about all core engines"""
    return {
        "sacred_geometry": {
            "description": "Discovers optimal poetic forms using sacred mathematical constants",
            "constants": ["phi", "pi", "fibonacci", "e", "sqrt_2", "sqrt_3", "sqrt_5"],
            "methods": ["calculate_volta_position", "derive_syllable_pattern", "solve_word_positions"]
        },
        "semantic_projection": {
            "description": "Maps meaning and emotion to mathematical coordinates",
            "dimensions": ["emotion", "theme", "imagery", "rhythm", "metaphor", "narrative"],
            "methods": ["project_theme_to_geometry", "calculate_semantic_fitness", "optimize_semantic_flow"]
        },
        "frequency_engine": {
            "description": "Generates archetypal frequency signatures for creative resonance",
            "archetypes": 12,
            "entropy_source": "/dev/hardcard",
            "methods": ["generate_frequency_signature", "tune_signature", "measure_resonance"]
        }
    }

def validate_engines():
    """Validate that all core engines are properly configured"""
    try:
        # Test sacred geometry calculator
        sacred_calc = SacredGeometryCalculator()
        sacred_calc.golden_ratio_sequence(3)
        
        # Test semantic projection engine
        semantic_engine = SemanticProjectionEngine()
        semantic_engine.project_theme_to_geometry("love", "phi")
        
        # Test frequency engine
        frequency_engine = MuseFrequencyEngine()
        test_assessment = {
            'user_id': 'test',
            'creative_preferences': {'poetry_style': 'lyric'},
            'personality_traits': {'analytical': True}
        }
        frequency_engine.generate_frequency_signature(test_assessment)
        
        return True
        
    except Exception as e:
        print(f"Engine validation failed: {e}")
        return False

# Initialize engines for module-level access
sacred_geometry_calculator = SacredGeometryCalculator()
semantic_projection_engine = SemanticProjectionEngine()
frequency_engine = MuseFrequencyEngine()

# Export engines for convenience
__engines__ = {
    "sacred_geometry": sacred_geometry_calculator,
    "semantic_projection": semantic_projection_engine,
    "frequency": frequency_engine
}

def get_engines():
    """Get all initialized engines"""
    return __engines__