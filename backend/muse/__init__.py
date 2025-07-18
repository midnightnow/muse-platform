"""
MUSE Platform - Mathematical Universal Sacred Expression

The world's first Computational Platonism creative platform.
Discovering eternal mathematical Forms through hardware entropy and sacred geometry.

Core Philosophy:
- Creativity is discovery, not generation
- Mathematical Forms exist eternally in Platonic realm
- Hardware entropy (/dev/hardcard) provides divine randomness
- Sacred geometry reveals the structure of creative truth

Modules:
- core: Mathematical engines for sacred geometry, frequency, and semantics
- api: FastAPI routers for web interface
- models: SQLAlchemy models for community and data storage
- services: Business logic and matching algorithms
- validation: Empirical validation framework
- tests: Comprehensive test suite
"""

__version__ = "1.0.0"
__author__ = "MUSE Platform Team"
__license__ = "MIT"

# Core mathematical constants
PHI = 1.618033988749895  # Golden ratio
PI = 3.141592653589793   # Circle constant
E = 2.718281828459045    # Euler's number
SQRT_2 = 1.4142135623730951
SQRT_3 = 1.7320508075688772
SQRT_5 = 2.23606797749979

# MUSE archetypes
MUSE_ARCHETYPES = [
    "CALLIOPE",      # Epic poetry, eloquence
    "CLIO",          # History, documentation
    "ERATO",         # Lyric poetry, love
    "EUTERPE",       # Music, harmony
    "MELPOMENE",     # Tragedy, drama
    "POLYHYMNIA",    # Sacred poetry, hymns
    "TERPSICHORE",   # Dance, movement
    "THALIA",        # Comedy, joy
    "URANIA",        # Astronomy, cosmos
    "SOPHIA",        # Wisdom, philosophy
    "TECHNE",        # Craft, skill
    "PSYCHE",        # Soul, psychology
]

# Platform configuration
PLATFORM_NAME = "MUSE"
PLATFORM_DESCRIPTION = "Mathematical Universal Sacred Expression"
PLATFORM_TAGLINE = "Discovering eternal creative Forms through Computational Platonism"

# Hardware entropy configuration
HARDWARE_ENTROPY_DEVICE = "/dev/hardcard"
FALLBACK_ENTROPY_DEVICE = "/dev/urandom"

# Sacred geometry settings
DEFAULT_SACRED_CONSTANT = "phi"
SUPPORTED_SACRED_CONSTANTS = ["phi", "pi", "e", "fibonacci", "sqrt_2", "sqrt_3", "sqrt_5"]

# Poetic forms
SUPPORTED_FORMS = [
    "sonnet",
    "haiku",
    "villanelle",
    "sestina",
    "pantoum",
    "ghazal",
    "free_verse",
    "blank_verse",
    "heroic_couplet",
    "ballad",
    "limerick",
    "tanka"
]

# Fibonacci sequence for sacred calculations
FIBONACCI_SEQUENCE = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]

def get_platform_info():
    """Get platform information"""
    return {
        "name": PLATFORM_NAME,
        "version": __version__,
        "description": PLATFORM_DESCRIPTION,
        "tagline": PLATFORM_TAGLINE,
        "philosophy": "Computational Platonism",
        "approach": "Creative Discovery",
        "entropy_source": HARDWARE_ENTROPY_DEVICE,
        "supported_archetypes": len(MUSE_ARCHETYPES),
        "supported_forms": len(SUPPORTED_FORMS),
        "sacred_constants": SUPPORTED_SACRED_CONSTANTS
    }

def get_sacred_constants():
    """Get sacred mathematical constants"""
    return {
        "phi": PHI,
        "pi": PI,
        "e": E,
        "sqrt_2": SQRT_2,
        "sqrt_3": SQRT_3,
        "sqrt_5": SQRT_5
    }

def get_fibonacci_sequence(n: int = 20):
    """Get Fibonacci sequence up to n terms"""
    return FIBONACCI_SEQUENCE[:n]

# Import core modules for easier access
from .core import SacredGeometryCalculator, SemanticProjectionEngine, MuseFrequencyEngine

__all__ = [
    "__version__",
    "PHI",
    "PI", 
    "E",
    "SQRT_2",
    "SQRT_3",
    "SQRT_5",
    "MUSE_ARCHETYPES",
    "SUPPORTED_FORMS",
    "FIBONACCI_SEQUENCE",
    "get_platform_info",
    "get_sacred_constants",
    "get_fibonacci_sequence",
    "SacredGeometryCalculator",
    "SemanticProjectionEngine",
    "MuseFrequencyEngine"
]