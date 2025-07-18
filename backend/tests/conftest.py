"""
MUSE Platform Test Configuration
Pytest fixtures and configuration for comprehensive testing
"""

import pytest
import tempfile
import os
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch

# Import MUSE components
from muse.models.community import Base
from muse.core.sacred_geometry_calculator import SacredGeometryCalculator
from muse.core.frequency_engine import MuseFrequencyEngine
from muse.core.semantic_projection_engine import SemanticProjectionEngine
from muse.services.resonance_matcher import ResonanceMatcher
from muse.services.discovery_orchestrator import DiscoveryOrchestrator
from muse.validation.mathematical_validation_framework import MUSEValidationFramework
from main import app

# Test database URL (using SQLite for testing)
TEST_DATABASE_URL = "sqlite:///./test_muse.db"

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_muse.db"):
        os.remove("./test_muse.db")

@pytest.fixture(scope="function")
def test_db_session(test_engine):
    """Create test database session"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="function") 
def client():
    """Create test client"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def sacred_calc():
    """Sacred geometry calculator instance"""
    return SacredGeometryCalculator()

@pytest.fixture
def frequency_engine():
    """Frequency engine instance with mocked hardware entropy"""
    with patch('muse.core.frequency_engine.MuseFrequencyEngine._read_hardware_entropy') as mock_entropy:
        mock_entropy.return_value = 0.618  # Golden ratio for testing
        engine = MuseFrequencyEngine()
        yield engine

@pytest.fixture
def semantic_engine():
    """Semantic projection engine instance"""
    return SemanticProjectionEngine()

@pytest.fixture
def resonance_matcher(sacred_calc, frequency_engine, semantic_engine):
    """Resonance matcher with core engines"""
    return ResonanceMatcher(
        sacred_calc=sacred_calc,
        frequency_engine=frequency_engine,
        semantic_engine=semantic_engine
    )

@pytest.fixture
def discovery_orchestrator(sacred_calc, frequency_engine, semantic_engine):
    """Discovery orchestrator with core engines"""
    return DiscoveryOrchestrator(
        sacred_calc=sacred_calc,
        frequency_engine=frequency_engine,
        semantic_engine=semantic_engine
    )

@pytest.fixture
def validation_framework():
    """Validation framework for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        framework = MUSEValidationFramework(data_directory=temp_dir)
        yield framework

@pytest.fixture
def sample_user_profile():
    """Sample user profile for testing"""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "display_name": "Test User",
        "bio": "A test user exploring mathematical creativity",
        "primary_muse": "CALLIOPE",
        "harmonic_blend": {
            "CALLIOPE": 0.8,
            "ERATO": 0.6,
            "URANIA": 0.4
        },
        "sacred_ratios": {
            "phi": 0.9,
            "pi": 0.7,
            "fibonacci": 0.8
        },
        "spiral_coordinates": {
            "x": 1.2,
            "y": 0.8,
            "z": 0.3
        }
    }

@pytest.fixture
def sample_frequency_signature():
    """Sample frequency signature for testing"""
    return {
        "primary_muse": "ERATO",
        "harmonic_blend": {
            "ERATO": 0.85,
            "CALLIOPE": 0.65,
            "SOPHIA": 0.45,
            "URANIA": 0.25
        },
        "sacred_ratios": {
            "phi": 0.88,
            "pi": 0.72,
            "e": 0.61,
            "sqrt_2": 0.54,
            "sqrt_3": 0.43,
            "sqrt_5": 0.37
        },
        "spiral_coordinates": {
            "x": 0.95,
            "y": 1.15,
            "z": 0.75
        },
        "diversity_index": 0.73,
        "resonance_strength": 0.82,
        "mathematical_coherence": 0.91
    }

@pytest.fixture
def sample_creative_output():
    """Sample creative output for testing"""
    return {
        "content": "Golden spirals dance through time's embrace,\nFibonacci whispers in nature's space.",
        "form_type": "couplet",
        "mathematical_fitness": 0.87,
        "semantic_coherence": 0.82,
        "discovery_coordinates": {
            "entropy_seed": 0.618,
            "sacred_constant": "phi",
            "theme": "nature",
            "form_constraints": {"syllable_count": 10, "line_count": 2}
        },
        "archetypal_resonance": 0.79
    }

@pytest.fixture
def mock_hardware_entropy():
    """Mock hardware entropy for consistent testing"""
    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b'\x9e\x88\x7f\x5a'
        yield mock_open

@pytest.fixture
def assessment_data():
    """Sample assessment data for personality testing"""
    return {
        "creative_preferences": {
            "prefers_structure": 7,
            "embraces_randomness": 8,
            "seeks_meaning": 9,
            "values_beauty": 8,
            "explores_complexity": 7
        },
        "personality_traits": {
            "openness": 8,
            "conscientiousness": 6,
            "extraversion": 5,
            "agreeableness": 7,
            "neuroticism": 3
        },
        "sacred_geometry_affinity": {
            "golden_ratio": 9,
            "fibonacci": 8,
            "pi": 7,
            "sacred_spirals": 8,
            "geometric_patterns": 7
        },
        "thematic_preferences": {
            "nature": 9,
            "cosmos": 8,
            "love": 6,
            "time": 7,
            "mystery": 8
        }
    }

# Pytest configuration
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "validation: mark test as a validation framework test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )