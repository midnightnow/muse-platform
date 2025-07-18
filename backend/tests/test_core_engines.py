"""
MUSE Core Engines Unit Tests
Comprehensive testing for sacred geometry, frequency, and semantic engines
"""

import pytest
import numpy as np
from unittest.mock import patch, mock_open

from muse.core.sacred_geometry_calculator import SacredGeometryCalculator
from muse.core.frequency_engine import MuseFrequencyEngine, MuseArchetype
from muse.core.semantic_projection_engine import SemanticProjectionEngine


class TestSacredGeometryCalculator:
    """Test the sacred geometry calculation engine"""
    
    @pytest.mark.unit
    def test_golden_ratio_calculation(self, sacred_calc):
        """Test golden ratio calculations"""
        phi = sacred_calc._get_sacred_constant("phi")
        assert abs(phi - 1.618033988749) < 1e-10
        
    @pytest.mark.unit
    def test_fibonacci_sequence(self, sacred_calc):
        """Test Fibonacci sequence generation"""
        fib_seq = sacred_calc._generate_fibonacci_sequence(10)
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        assert fib_seq == expected
        
    @pytest.mark.unit
    def test_volta_position_calculation(self, sacred_calc):
        """Test volta position using golden ratio"""
        sonnet_length = 14
        volta_pos = sacred_calc.calculate_volta_position(sonnet_length)
        
        # Should be around line 8-9 for a sonnet (golden ratio division)
        assert 7 <= volta_pos <= 9
        assert isinstance(volta_pos, int)
        
    @pytest.mark.unit
    def test_syllable_pattern_derivation(self, sacred_calc):
        """Test syllable pattern generation"""
        pattern = sacred_calc.derive_syllable_pattern("sonnet", "phi")
        
        assert len(pattern) == 14  # Sonnet has 14 lines
        assert all(isinstance(count, int) for count in pattern)
        assert all(count > 0 for count in pattern)  # All lines should have syllables
        
    @pytest.mark.unit
    def test_form_fitness_calculation(self, sacred_calc):
        """Test form fitness scoring"""
        poem_structure = {
            "syllable_counts": [10, 10, 10, 10],
            "rhyme_scheme": ["A", "B", "A", "B"],
            "line_count": 4
        }
        target_form = "quatrain"
        
        fitness = sacred_calc.calculate_form_fitness(poem_structure, target_form)
        
        assert 0 <= fitness <= 1
        assert isinstance(fitness, float)
        
    @pytest.mark.unit
    def test_word_position_optimization(self, sacred_calc):
        """Test word position solving with constraints"""
        line_constraints = {
            "syllable_count": 10,
            "stress_pattern": "iambic",
            "rhyme_sound": "ay"
        }
        entropy_seed = 0.618
        
        positions = sacred_calc.solve_word_positions(line_constraints, entropy_seed)
        
        assert isinstance(positions, dict)
        assert "stress_positions" in positions
        assert "key_word_positions" in positions
        
    @pytest.mark.unit 
    def test_rhyme_scheme_generation(self, sacred_calc):
        """Test rhyme scheme pattern generation"""
        sonnet_scheme = sacred_calc.generate_rhyme_scheme("sonnet")
        
        assert len(sonnet_scheme) == 14
        assert all(isinstance(rhyme, str) for rhyme in sonnet_scheme)
        
        # Test Shakespearean sonnet pattern
        expected_pattern = ["A", "B", "A", "B", "C", "D", "C", "D", "E", "F", "E", "F", "G", "G"]
        assert sonnet_scheme == expected_pattern


class TestMuseFrequencyEngine:
    """Test the Muse frequency archetypal engine"""
    
    @pytest.mark.unit
    def test_muse_archetype_enum(self):
        """Test Muse archetype enumeration"""
        assert MuseArchetype.CALLIOPE.value == "CALLIOPE"
        assert MuseArchetype.ERATO.value == "ERATO"
        assert len(MuseArchetype) == 12  # All 12 Muses
        
    @pytest.mark.unit
    def test_frequency_signature_generation(self, frequency_engine, assessment_data):
        """Test frequency signature generation from assessment"""
        signature = frequency_engine.generate_frequency_signature(assessment_data)
        
        assert "primary_muse" in signature
        assert "harmonic_blend" in signature
        assert "sacred_ratios" in signature
        assert "spiral_coordinates" in signature
        
        # Check primary muse is valid
        assert signature["primary_muse"] in [muse.value for muse in MuseArchetype]
        
        # Check harmonic blend sums and values
        harmonic_blend = signature["harmonic_blend"]
        assert all(0 <= value <= 1 for value in harmonic_blend.values())
        assert len(harmonic_blend) <= 12
        
        # Check spiral coordinates are 3D
        coords = signature["spiral_coordinates"]
        assert all(key in coords for key in ["x", "y", "z"])
        
    @pytest.mark.unit
    def test_hardware_entropy_reading(self, frequency_engine, mock_hardware_entropy):
        """Test hardware entropy reading with fallback"""
        entropy = frequency_engine._read_hardware_entropy()
        assert isinstance(entropy, float)
        assert 0 <= entropy <= 1
        
    @pytest.mark.unit
    def test_spiral_coordinates_calculation(self, frequency_engine):
        """Test 3D spiral coordinate generation"""
        phi_affinity = 0.8
        pi_affinity = 0.6
        fibonacci_affinity = 0.7
        
        coords = frequency_engine.calculate_spiral_coordinates(
            phi_affinity, pi_affinity, fibonacci_affinity
        )
        
        assert all(key in coords for key in ["x", "y", "z"])
        assert all(isinstance(value, float) for value in coords.values())
        
    @pytest.mark.unit
    def test_signature_tuning(self, frequency_engine, sample_frequency_signature):
        """Test frequency signature tuning"""
        original_signature = sample_frequency_signature.copy()
        target_muses = ["ERATO", "SOPHIA"]
        blend_ratios = [0.7, 0.3]
        
        tuned_signature = frequency_engine.tune_signature(
            original_signature, target_muses, blend_ratios
        )
        
        assert tuned_signature["harmonic_blend"]["ERATO"] == 0.7
        assert tuned_signature["harmonic_blend"]["SOPHIA"] == 0.3
        assert tuned_signature != original_signature
        
    @pytest.mark.unit
    def test_resonance_measurement(self, frequency_engine, sample_frequency_signature):
        """Test resonance calculation between signatures"""
        sig1 = sample_frequency_signature
        sig2 = {
            "primary_muse": "CALLIOPE",
            "harmonic_blend": {"CALLIOPE": 0.9, "ERATO": 0.5},
            "sacred_ratios": {"phi": 0.8, "pi": 0.6},
            "spiral_coordinates": {"x": 1.0, "y": 0.9, "z": 0.8}
        }
        
        resonance = frequency_engine.measure_resonance(sig1, sig2)
        
        assert 0 <= resonance <= 1
        assert isinstance(resonance, float)


class TestSemanticProjectionEngine:
    """Test the semantic projection engine"""
    
    @pytest.mark.unit
    def test_theme_projection(self, semantic_engine):
        """Test theme projection to geometric coordinates"""
        theme = "nature"
        sacred_constant = "phi"
        
        projection = semantic_engine.project_theme_to_geometry(theme, sacred_constant)
        
        assert "coordinates" in projection
        assert "semantic_vector" in projection
        assert "geometric_alignment" in projection
        
        coords = projection["coordinates"]
        assert all(key in coords for key in ["x", "y", "z"])
        
    @pytest.mark.unit
    def test_semantic_fitness_calculation(self, semantic_engine):
        """Test semantic fitness measurement"""
        word_list = ["golden", "spiral", "fibonacci", "nature", "harmony"]
        theme_vector = {"nature": 0.8, "mathematics": 0.9, "beauty": 0.7}
        
        fitness = semantic_engine.calculate_semantic_fitness(word_list, theme_vector)
        
        assert 0 <= fitness <= 1
        assert isinstance(fitness, float)
        
    @pytest.mark.unit
    def test_word_embeddings_generation(self, semantic_engine):
        """Test word embedding generation"""
        vocabulary = ["golden", "ratio", "spiral", "fibonacci", "harmony", "beauty"]
        
        embeddings = semantic_engine.generate_word_embeddings(vocabulary)
        
        assert isinstance(embeddings, dict)
        assert all(word in embeddings for word in vocabulary)
        assert all(isinstance(vector, list) for vector in embeddings.values())
        
    @pytest.mark.unit
    def test_semantic_flow_optimization(self, semantic_engine):
        """Test semantic flow optimization"""
        poem_lines = [
            "Golden spirals dance in morning light",
            "Fibonacci whispers through the trees", 
            "Nature's mathematics pure and bright",
            "Sacred geometry on the breeze"
        ]
        target_emotion = "wonder"
        
        optimization = semantic_engine.optimize_semantic_flow(poem_lines, target_emotion)
        
        assert "coherence_score" in optimization
        assert "emotional_alignment" in optimization
        assert "improvement_suggestions" in optimization
        
        assert 0 <= optimization["coherence_score"] <= 1
        assert 0 <= optimization["emotional_alignment"] <= 1
        
    @pytest.mark.unit
    def test_sacred_resonance_calculation(self, semantic_engine, sample_frequency_signature):
        """Test sacred resonance between text and frequency signature"""
        text = "The golden ratio spirals through cosmic harmony, revealing fibonacci's sacred dance."
        
        resonance = semantic_engine.calculate_sacred_resonance(text, sample_frequency_signature)
        
        assert 0 <= resonance <= 1
        assert isinstance(resonance, float)


class TestEngineIntegration:
    """Test integration between multiple engines"""
    
    @pytest.mark.integration
    def test_cross_engine_resonance(self, sacred_calc, frequency_engine, semantic_engine):
        """Test resonance calculation across all three engines"""
        # Create a sample creative work
        creative_work = {
            "content": "Golden spirals weave through time's embrace,\nFibonacci numbers mark each sacred space.",
            "theme": "nature",
            "form": "couplet"
        }
        
        # Get sacred geometry fitness
        poem_structure = {
            "syllable_counts": [10, 11],
            "rhyme_scheme": ["A", "A"],
            "line_count": 2
        }
        sacred_fitness = sacred_calc.calculate_form_fitness(poem_structure, "couplet")
        
        # Get semantic projection
        semantic_projection = semantic_engine.project_theme_to_geometry("nature", "phi")
        
        # Generate frequency signature for comparison
        assessment_data = {
            "creative_preferences": {"values_beauty": 9, "seeks_meaning": 8},
            "thematic_preferences": {"nature": 9, "cosmos": 7}
        }
        frequency_signature = frequency_engine.generate_frequency_signature(assessment_data)
        
        # Test that all components work together
        assert sacred_fitness > 0
        assert semantic_projection["geometric_alignment"] > 0
        assert frequency_signature["primary_muse"] in [muse.value for muse in MuseArchetype]
        
    @pytest.mark.performance
    def test_engine_performance(self, sacred_calc, frequency_engine, semantic_engine):
        """Test performance of core engines under load"""
        import time
        
        # Test sacred geometry performance
        start_time = time.time()
        for _ in range(100):
            sacred_calc.calculate_volta_position(14)
        sacred_time = time.time() - start_time
        
        # Test frequency engine performance
        start_time = time.time()
        assessment_data = {"creative_preferences": {"values_beauty": 8}}
        for _ in range(100):
            frequency_engine.generate_frequency_signature(assessment_data)
        frequency_time = time.time() - start_time
        
        # Test semantic engine performance
        start_time = time.time()
        for _ in range(100):
            semantic_engine.project_theme_to_geometry("nature", "phi")
        semantic_time = time.time() - start_time
        
        # Assert reasonable performance (adjust thresholds as needed)
        assert sacred_time < 1.0  # Should complete 100 operations in under 1 second
        assert frequency_time < 2.0  # Frequency generation is more complex
        assert semantic_time < 1.5  # Semantic projection is moderately complex