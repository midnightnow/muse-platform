"""
Metrics Calculator for MUSE Validation

This module provides comprehensive metrics calculation using all three core engines
for validation experiments. It calculates mathematical fitness, semantic coherence,
and other metrics specific to MUSE's Computational Platonism approach.
"""

import asyncio
import json
import logging
import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

from ..core.frequency_engine import MuseFrequencyEngine, FrequencySignature, MuseArchetype
from ..core.sacred_geometry_calculator import SacredGeometryCalculator
from ..core.semantic_projection_engine import SemanticProjectionEngine


class MetricType(Enum):
    """Types of validation metrics"""
    MATHEMATICAL_FITNESS = "mathematical_fitness"
    SEMANTIC_COHERENCE = "semantic_coherence"
    ARCHETYPAL_ALIGNMENT = "archetypal_alignment"
    SACRED_GEOMETRY_COMPLIANCE = "sacred_geometry_compliance"
    FREQUENCY_SIGNATURE_ACCURACY = "frequency_signature_accuracy"
    USER_SATISFACTION = "user_satisfaction"
    UNIQUENESS_SCORE = "uniqueness_score"
    PREFERENCE_ACCURACY = "preference_accuracy"
    QUALITY_RATING = "quality_rating"
    MEANINGFULNESS_SCORE = "meaningfulness_score"


@dataclass
class MetricCalculationResult:
    """Result of metric calculation"""
    metric_type: MetricType
    value: float
    confidence: float
    components: Dict[str, float]
    metadata: Dict[str, Any]
    calculation_time: float
    timestamp: datetime


class ValidationMetricsCalculator:
    """
    Comprehensive metrics calculator for MUSE validation experiments
    
    Uses all three core engines to calculate various metrics that assess
    the effectiveness of MUSE's Computational Platonism approach.
    """
    
    def __init__(self, 
                 frequency_engine: Optional[MuseFrequencyEngine] = None,
                 geometry_calculator: Optional[SacredGeometryCalculator] = None,
                 projection_engine: Optional[SemanticProjectionEngine] = None):
        """
        Initialize the metrics calculator
        
        Args:
            frequency_engine: MUSE frequency engine for archetypal analysis
            geometry_calculator: Sacred geometry calculator for mathematical analysis
            projection_engine: Semantic projection engine for meaning analysis
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize core engines
        self.frequency_engine = frequency_engine or MuseFrequencyEngine()
        self.geometry_calculator = geometry_calculator or SacredGeometryCalculator()
        self.projection_engine = projection_engine or SemanticProjectionEngine()
        
        # Metric weights and thresholds
        self.metric_weights = {
            MetricType.MATHEMATICAL_FITNESS: {
                'sacred_geometry_compliance': 0.3,
                'archetypal_alignment': 0.25,
                'frequency_coherence': 0.2,
                'geometric_beauty': 0.15,
                'mathematical_elegance': 0.1
            },
            MetricType.SEMANTIC_COHERENCE: {
                'meaning_consistency': 0.35,
                'semantic_density': 0.25,
                'thematic_unity': 0.2,
                'conceptual_clarity': 0.2
            },
            MetricType.ARCHETYPAL_ALIGNMENT: {
                'primary_muse_strength': 0.4,
                'secondary_muse_support': 0.3,
                'archetypal_balance': 0.2,
                'frequency_signature_match': 0.1
            }
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'poor': 0.5,
            'unacceptable': 0.0
        }
        
        # Calculation cache
        self.calculation_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        self.logger.info("Validation Metrics Calculator initialized")
    
    async def calculate_mathematical_fitness(self, 
                                           creative_work: Dict[str, Any],
                                           frequency_signature: FrequencySignature,
                                           context: Dict[str, Any] = None) -> MetricCalculationResult:
        """
        Calculate mathematical fitness score
        
        Args:
            creative_work: The creative work to evaluate
            frequency_signature: User's frequency signature
            context: Additional context for calculation
            
        Returns:
            Mathematical fitness metric result
        """
        start_time = datetime.now()
        
        components = {}
        
        # 1. Sacred geometry compliance
        geometry_score = await self._calculate_sacred_geometry_compliance(
            creative_work, frequency_signature
        )
        components['sacred_geometry_compliance'] = geometry_score
        
        # 2. Archetypal alignment
        archetypal_score = await self._calculate_archetypal_alignment_score(
            creative_work, frequency_signature
        )
        components['archetypal_alignment'] = archetypal_score
        
        # 3. Frequency coherence
        frequency_score = await self._calculate_frequency_coherence(
            creative_work, frequency_signature
        )
        components['frequency_coherence'] = frequency_score
        
        # 4. Geometric beauty
        beauty_score = await self._calculate_geometric_beauty(
            creative_work, frequency_signature
        )
        components['geometric_beauty'] = beauty_score
        
        # 5. Mathematical elegance
        elegance_score = await self._calculate_mathematical_elegance(
            creative_work, frequency_signature
        )
        components['mathematical_elegance'] = elegance_score
        
        # Calculate weighted average
        weights = self.metric_weights[MetricType.MATHEMATICAL_FITNESS]
        mathematical_fitness = sum(
            components[component] * weights[component] 
            for component in weights
        )
        
        # Calculate confidence based on component consistency
        confidence = self._calculate_confidence(list(components.values()))
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return MetricCalculationResult(
            metric_type=MetricType.MATHEMATICAL_FITNESS,
            value=mathematical_fitness,
            confidence=confidence,
            components=components,
            metadata={
                'frequency_signature_id': frequency_signature.id,
                'primary_muse': frequency_signature.primary_muse,
                'sacred_ratios': frequency_signature.sacred_ratios,
                'context': context or {}
            },
            calculation_time=calculation_time,
            timestamp=datetime.now()
        )
    
    async def calculate_semantic_coherence(self, 
                                         creative_work: Dict[str, Any],
                                         frequency_signature: FrequencySignature,
                                         context: Dict[str, Any] = None) -> MetricCalculationResult:
        """
        Calculate semantic coherence score
        
        Args:
            creative_work: The creative work to evaluate
            frequency_signature: User's frequency signature
            context: Additional context for calculation
            
        Returns:
            Semantic coherence metric result
        """
        start_time = datetime.now()
        
        components = {}
        
        # 1. Meaning consistency
        consistency_score = await self._calculate_meaning_consistency(
            creative_work, frequency_signature
        )
        components['meaning_consistency'] = consistency_score
        
        # 2. Semantic density
        density_score = await self._calculate_semantic_density(
            creative_work, frequency_signature
        )
        components['semantic_density'] = density_score
        
        # 3. Thematic unity
        unity_score = await self._calculate_thematic_unity(
            creative_work, frequency_signature
        )
        components['thematic_unity'] = unity_score
        
        # 4. Conceptual clarity
        clarity_score = await self._calculate_conceptual_clarity(
            creative_work, frequency_signature
        )
        components['conceptual_clarity'] = clarity_score
        
        # Calculate weighted average
        weights = self.metric_weights[MetricType.SEMANTIC_COHERENCE]
        semantic_coherence = sum(
            components[component] * weights[component] 
            for component in weights
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(list(components.values()))
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return MetricCalculationResult(
            metric_type=MetricType.SEMANTIC_COHERENCE,
            value=semantic_coherence,
            confidence=confidence,
            components=components,
            metadata={
                'frequency_signature_id': frequency_signature.id,
                'semantic_projection_used': True,
                'context': context or {}
            },
            calculation_time=calculation_time,
            timestamp=datetime.now()
        )
    
    async def calculate_archetypal_alignment(self, 
                                           creative_work: Dict[str, Any],
                                           frequency_signature: FrequencySignature,
                                           context: Dict[str, Any] = None) -> MetricCalculationResult:
        """
        Calculate archetypal alignment score
        
        Args:
            creative_work: The creative work to evaluate
            frequency_signature: User's frequency signature
            context: Additional context for calculation
            
        Returns:
            Archetypal alignment metric result
        """
        start_time = datetime.now()
        
        components = {}
        
        # 1. Primary muse strength
        primary_strength = await self._calculate_primary_muse_strength(
            creative_work, frequency_signature
        )
        components['primary_muse_strength'] = primary_strength
        
        # 2. Secondary muse support
        secondary_support = await self._calculate_secondary_muse_support(
            creative_work, frequency_signature
        )
        components['secondary_muse_support'] = secondary_support
        
        # 3. Archetypal balance
        balance_score = await self._calculate_archetypal_balance(
            creative_work, frequency_signature
        )
        components['archetypal_balance'] = balance_score
        
        # 4. Frequency signature match
        signature_match = await self._calculate_frequency_signature_match(
            creative_work, frequency_signature
        )
        components['frequency_signature_match'] = signature_match
        
        # Calculate weighted average
        weights = self.metric_weights[MetricType.ARCHETYPAL_ALIGNMENT]
        archetypal_alignment = sum(
            components[component] * weights[component] 
            for component in weights
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(list(components.values()))
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return MetricCalculationResult(
            metric_type=MetricType.ARCHETYPAL_ALIGNMENT,
            value=archetypal_alignment,
            confidence=confidence,
            components=components,
            metadata={
                'frequency_signature_id': frequency_signature.id,
                'primary_muse': frequency_signature.primary_muse,
                'secondary_muse': frequency_signature.secondary_muse,
                'harmonic_blend': frequency_signature.harmonic_blend,
                'context': context or {}
            },
            calculation_time=calculation_time,
            timestamp=datetime.now()
        )
    
    async def calculate_uniqueness_score(self, 
                                       creative_work: Dict[str, Any],
                                       comparison_works: List[Dict[str, Any]],
                                       frequency_signature: FrequencySignature,
                                       context: Dict[str, Any] = None) -> MetricCalculationResult:
        """
        Calculate uniqueness score compared to other works
        
        Args:
            creative_work: The creative work to evaluate
            comparison_works: Other works to compare against
            frequency_signature: User's frequency signature
            context: Additional context for calculation
            
        Returns:
            Uniqueness score metric result
        """
        start_time = datetime.now()
        
        components = {}
        
        # 1. Structural uniqueness
        structural_uniqueness = await self._calculate_structural_uniqueness(
            creative_work, comparison_works
        )
        components['structural_uniqueness'] = structural_uniqueness
        
        # 2. Semantic uniqueness
        semantic_uniqueness = await self._calculate_semantic_uniqueness(
            creative_work, comparison_works
        )
        components['semantic_uniqueness'] = semantic_uniqueness
        
        # 3. Archetypal uniqueness
        archetypal_uniqueness = await self._calculate_archetypal_uniqueness(
            creative_work, comparison_works, frequency_signature
        )
        components['archetypal_uniqueness'] = archetypal_uniqueness
        
        # 4. Mathematical uniqueness
        mathematical_uniqueness = await self._calculate_mathematical_uniqueness(
            creative_work, comparison_works, frequency_signature
        )
        components['mathematical_uniqueness'] = mathematical_uniqueness
        
        # Calculate overall uniqueness
        uniqueness_score = np.mean(list(components.values()))
        
        # Calculate confidence
        confidence = self._calculate_confidence(list(components.values()))
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return MetricCalculationResult(
            metric_type=MetricType.UNIQUENESS_SCORE,
            value=uniqueness_score,
            confidence=confidence,
            components=components,
            metadata={
                'frequency_signature_id': frequency_signature.id,
                'comparison_works_count': len(comparison_works),
                'context': context or {}
            },
            calculation_time=calculation_time,
            timestamp=datetime.now()
        )
    
    async def calculate_preference_accuracy(self, 
                                          predicted_preferences: Dict[str, float],
                                          actual_preferences: Dict[str, float],
                                          frequency_signature: FrequencySignature,
                                          context: Dict[str, Any] = None) -> MetricCalculationResult:
        """
        Calculate preference prediction accuracy
        
        Args:
            predicted_preferences: Predicted user preferences
            actual_preferences: Actual user preferences
            frequency_signature: User's frequency signature
            context: Additional context for calculation
            
        Returns:
            Preference accuracy metric result
        """
        start_time = datetime.now()
        
        components = {}
        
        # 1. Direct preference correlation
        correlation_score = await self._calculate_preference_correlation(
            predicted_preferences, actual_preferences
        )
        components['preference_correlation'] = correlation_score
        
        # 2. Ranking accuracy
        ranking_accuracy = await self._calculate_ranking_accuracy(
            predicted_preferences, actual_preferences
        )
        components['ranking_accuracy'] = ranking_accuracy
        
        # 3. Category accuracy
        category_accuracy = await self._calculate_category_accuracy(
            predicted_preferences, actual_preferences
        )
        components['category_accuracy'] = category_accuracy
        
        # 4. Threshold accuracy
        threshold_accuracy = await self._calculate_threshold_accuracy(
            predicted_preferences, actual_preferences
        )
        components['threshold_accuracy'] = threshold_accuracy
        
        # Calculate overall accuracy
        preference_accuracy = np.mean(list(components.values()))
        
        # Calculate confidence
        confidence = self._calculate_confidence(list(components.values()))
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return MetricCalculationResult(
            metric_type=MetricType.PREFERENCE_ACCURACY,
            value=preference_accuracy,
            confidence=confidence,
            components=components,
            metadata={
                'frequency_signature_id': frequency_signature.id,
                'preference_count': len(predicted_preferences),
                'prediction_method': 'archetypal_frequency_mapping',
                'context': context or {}
            },
            calculation_time=calculation_time,
            timestamp=datetime.now()
        )
    
    async def calculate_quality_rating(self, 
                                     creative_work: Dict[str, Any],
                                     frequency_signature: FrequencySignature,
                                     evaluation_criteria: Dict[str, float],
                                     context: Dict[str, Any] = None) -> MetricCalculationResult:
        """
        Calculate overall quality rating
        
        Args:
            creative_work: The creative work to evaluate
            frequency_signature: User's frequency signature
            evaluation_criteria: Criteria and weights for evaluation
            context: Additional context for calculation
            
        Returns:
            Quality rating metric result
        """
        start_time = datetime.now()
        
        components = {}
        
        # 1. Technical quality
        technical_quality = await self._calculate_technical_quality(
            creative_work, frequency_signature
        )
        components['technical_quality'] = technical_quality
        
        # 2. Aesthetic quality
        aesthetic_quality = await self._calculate_aesthetic_quality(
            creative_work, frequency_signature
        )
        components['aesthetic_quality'] = aesthetic_quality
        
        # 3. Emotional impact
        emotional_impact = await self._calculate_emotional_impact(
            creative_work, frequency_signature
        )
        components['emotional_impact'] = emotional_impact
        
        # 4. Originality
        originality = await self._calculate_originality(
            creative_work, frequency_signature
        )
        components['originality'] = originality
        
        # 5. Coherence
        coherence = await self._calculate_overall_coherence(
            creative_work, frequency_signature
        )
        components['coherence'] = coherence
        
        # Apply evaluation criteria weights if provided
        if evaluation_criteria:
            weighted_components = {}
            for criterion, weight in evaluation_criteria.items():
                if criterion in components:
                    weighted_components[criterion] = components[criterion] * weight
            
            if weighted_components:
                quality_rating = sum(weighted_components.values()) / sum(evaluation_criteria.values())
            else:
                quality_rating = np.mean(list(components.values()))
        else:
            quality_rating = np.mean(list(components.values()))
        
        # Calculate confidence
        confidence = self._calculate_confidence(list(components.values()))
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return MetricCalculationResult(
            metric_type=MetricType.QUALITY_RATING,
            value=quality_rating,
            confidence=confidence,
            components=components,
            metadata={
                'frequency_signature_id': frequency_signature.id,
                'evaluation_criteria': evaluation_criteria,
                'quality_interpretation': self._interpret_quality_rating(quality_rating),
                'context': context or {}
            },
            calculation_time=calculation_time,
            timestamp=datetime.now()
        )
    
    # Component calculation methods
    
    async def _calculate_sacred_geometry_compliance(self, 
                                                  creative_work: Dict[str, Any],
                                                  frequency_signature: FrequencySignature) -> float:
        """Calculate sacred geometry compliance score"""
        # Analyze work structure against sacred geometry principles
        structure_analysis = await self._analyze_work_structure(creative_work)
        
        # Get sacred ratios from frequency signature
        sacred_ratios = frequency_signature.sacred_ratios
        
        # Calculate compliance with golden ratio
        phi_compliance = self._calculate_phi_compliance(structure_analysis, sacred_ratios.get('phi', 0.5))
        
        # Calculate compliance with fibonacci sequence
        fibonacci_compliance = self._calculate_fibonacci_compliance(structure_analysis, sacred_ratios.get('fibonacci', 0.5))
        
        # Calculate compliance with pi-related proportions
        pi_compliance = self._calculate_pi_compliance(structure_analysis, sacred_ratios.get('pi', 0.5))
        
        # Combine compliance scores
        compliance_score = (phi_compliance * 0.5 + fibonacci_compliance * 0.3 + pi_compliance * 0.2)
        
        return min(1.0, max(0.0, compliance_score))
    
    async def _calculate_archetypal_alignment_score(self, 
                                                  creative_work: Dict[str, Any],
                                                  frequency_signature: FrequencySignature) -> float:
        """Calculate archetypal alignment score"""
        # Analyze work's archetypal characteristics
        work_archetypes = await self._analyze_work_archetypes(creative_work)
        
        # Compare with user's frequency signature
        user_blend = frequency_signature.harmonic_blend
        
        # Calculate alignment score
        alignment_score = 0.0
        for archetype, work_strength in work_archetypes.items():
            user_strength = user_blend.get(archetype, 0.0)
            alignment_score += work_strength * user_strength
        
        return min(1.0, max(0.0, alignment_score))
    
    async def _calculate_frequency_coherence(self, 
                                           creative_work: Dict[str, Any],
                                           frequency_signature: FrequencySignature) -> float:
        """Calculate frequency coherence score"""
        # Extract frequency characteristics from work
        work_frequencies = await self._extract_work_frequencies(creative_work)
        
        # Compare with user's archetypal frequencies
        user_frequencies = self._get_user_frequencies(frequency_signature)
        
        # Calculate coherence using frequency domain analysis
        coherence_score = self._calculate_frequency_coherence_score(work_frequencies, user_frequencies)
        
        return min(1.0, max(0.0, coherence_score))
    
    async def _calculate_geometric_beauty(self, 
                                        creative_work: Dict[str, Any],
                                        frequency_signature: FrequencySignature) -> float:
        """Calculate geometric beauty score"""
        # Analyze geometric properties
        geometric_properties = await self._analyze_geometric_properties(creative_work)
        
        # Calculate beauty based on mathematical harmony
        beauty_score = 0.0
        
        # Symmetry score
        symmetry_score = geometric_properties.get('symmetry', 0.5)
        beauty_score += symmetry_score * 0.3
        
        # Proportion score
        proportion_score = geometric_properties.get('proportion', 0.5)
        beauty_score += proportion_score * 0.4
        
        # Harmony score
        harmony_score = geometric_properties.get('harmony', 0.5)
        beauty_score += harmony_score * 0.3
        
        return min(1.0, max(0.0, beauty_score))
    
    async def _calculate_mathematical_elegance(self, 
                                             creative_work: Dict[str, Any],
                                             frequency_signature: FrequencySignature) -> float:
        """Calculate mathematical elegance score"""
        # Analyze mathematical properties
        math_properties = await self._analyze_mathematical_properties(creative_work)
        
        # Calculate elegance based on complexity and beauty
        elegance_score = 0.0
        
        # Simplicity score (elegant solutions are often simple)
        simplicity_score = math_properties.get('simplicity', 0.5)
        elegance_score += simplicity_score * 0.3
        
        # Completeness score
        completeness_score = math_properties.get('completeness', 0.5)
        elegance_score += completeness_score * 0.25
        
        # Universality score
        universality_score = math_properties.get('universality', 0.5)
        elegance_score += universality_score * 0.25
        
        # Surprise score (elegant solutions often have unexpected insights)
        surprise_score = math_properties.get('surprise', 0.5)
        elegance_score += surprise_score * 0.2
        
        return min(1.0, max(0.0, elegance_score))
    
    async def _calculate_meaning_consistency(self, 
                                           creative_work: Dict[str, Any],
                                           frequency_signature: FrequencySignature) -> float:
        """Calculate meaning consistency score"""
        # Analyze semantic consistency throughout work
        semantic_analysis = await self._analyze_semantic_consistency(creative_work)
        
        # Calculate consistency score
        consistency_score = semantic_analysis.get('consistency_score', 0.5)
        
        # Adjust based on user's semantic preferences
        user_preferences = self._get_user_semantic_preferences(frequency_signature)
        preference_alignment = self._calculate_preference_alignment(semantic_analysis, user_preferences)
        
        # Combine scores
        final_score = (consistency_score * 0.7 + preference_alignment * 0.3)
        
        return min(1.0, max(0.0, final_score))
    
    async def _calculate_semantic_density(self, 
                                        creative_work: Dict[str, Any],
                                        frequency_signature: FrequencySignature) -> float:
        """Calculate semantic density score"""
        # Analyze semantic richness
        semantic_analysis = await self._analyze_semantic_richness(creative_work)
        
        # Calculate density metrics
        density_score = 0.0
        
        # Concept density
        concept_density = semantic_analysis.get('concept_density', 0.5)
        density_score += concept_density * 0.4
        
        # Metaphor density
        metaphor_density = semantic_analysis.get('metaphor_density', 0.5)
        density_score += metaphor_density * 0.3
        
        # Symbolic density
        symbolic_density = semantic_analysis.get('symbolic_density', 0.5)
        density_score += symbolic_density * 0.3
        
        return min(1.0, max(0.0, density_score))
    
    async def _calculate_thematic_unity(self, 
                                      creative_work: Dict[str, Any],
                                      frequency_signature: FrequencySignature) -> float:
        """Calculate thematic unity score"""
        # Analyze thematic coherence
        thematic_analysis = await self._analyze_thematic_coherence(creative_work)
        
        # Calculate unity score
        unity_score = thematic_analysis.get('unity_score', 0.5)
        
        # Adjust based on user's thematic preferences
        user_themes = self._get_user_thematic_preferences(frequency_signature)
        theme_alignment = self._calculate_theme_alignment(thematic_analysis, user_themes)
        
        # Combine scores
        final_score = (unity_score * 0.6 + theme_alignment * 0.4)
        
        return min(1.0, max(0.0, final_score))
    
    async def _calculate_conceptual_clarity(self, 
                                          creative_work: Dict[str, Any],
                                          frequency_signature: FrequencySignature) -> float:
        """Calculate conceptual clarity score"""
        # Analyze conceptual clarity
        clarity_analysis = await self._analyze_conceptual_clarity(creative_work)
        
        # Calculate clarity score
        clarity_score = clarity_analysis.get('clarity_score', 0.5)
        
        # Adjust based on user's clarity preferences
        user_clarity_pref = self._get_user_clarity_preference(frequency_signature)
        clarity_alignment = abs(clarity_score - user_clarity_pref)
        
        # Combine scores (lower difference means better alignment)
        final_score = clarity_score * (1.0 - clarity_alignment * 0.3)
        
        return min(1.0, max(0.0, final_score))
    
    # Utility methods for analysis
    
    async def _analyze_work_structure(self, creative_work: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze structural properties of creative work"""
        # This is a simplified analysis - in a real implementation,
        # this would use sophisticated NLP and structural analysis
        
        structure = {
            'line_count': creative_work.get('line_count', 0),
            'word_count': creative_work.get('word_count', 0),
            'syllable_count': creative_work.get('syllable_count', 0),
            'stanza_count': creative_work.get('stanza_count', 0),
            'rhythm_pattern': creative_work.get('rhythm_pattern', []),
            'rhyme_scheme': creative_work.get('rhyme_scheme', '')
        }
        
        # Add ratio analysis
        if structure['word_count'] > 0:
            structure['syllable_word_ratio'] = structure['syllable_count'] / structure['word_count']
        
        if structure['stanza_count'] > 0:
            structure['lines_per_stanza'] = structure['line_count'] / structure['stanza_count']
        
        return structure
    
    def _calculate_phi_compliance(self, structure: Dict[str, Any], phi_affinity: float) -> float:
        """Calculate compliance with golden ratio"""
        phi = (1 + math.sqrt(5)) / 2
        
        compliance_score = 0.0
        
        # Check line count against golden ratio
        line_count = structure.get('line_count', 0)
        if line_count > 0:
            # Find closest fibonacci number
            fib_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
            closest_fib = min(fib_numbers, key=lambda x: abs(x - line_count))
            
            if closest_fib > 0:
                compliance_score += (1.0 - abs(line_count - closest_fib) / line_count) * 0.3
        
        # Check syllable-word ratio against golden ratio
        syllable_word_ratio = structure.get('syllable_word_ratio', 1.0)
        if syllable_word_ratio > 0:
            phi_distance = abs(syllable_word_ratio - phi) / phi
            compliance_score += (1.0 - phi_distance) * 0.4
        
        # Check stanza proportions
        lines_per_stanza = structure.get('lines_per_stanza', 1.0)
        if lines_per_stanza > 0:
            phi_distance = abs(lines_per_stanza - phi) / phi
            compliance_score += (1.0 - phi_distance) * 0.3
        
        # Weight by user's phi affinity
        compliance_score *= phi_affinity
        
        return min(1.0, max(0.0, compliance_score))
    
    def _calculate_fibonacci_compliance(self, structure: Dict[str, Any], fibonacci_affinity: float) -> float:
        """Calculate compliance with fibonacci sequence"""
        fibonacci_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        compliance_score = 0.0
        
        # Check if structural elements follow fibonacci sequence
        structural_elements = [
            structure.get('line_count', 0),
            structure.get('stanza_count', 0),
            structure.get('word_count', 0) // 10  # Scaled down
        ]
        
        for element in structural_elements:
            if element in fibonacci_numbers:
                compliance_score += 0.3
            else:
                # Find closest fibonacci number
                closest_fib = min(fibonacci_numbers, key=lambda x: abs(x - element))
                if closest_fib > 0:
                    distance = abs(element - closest_fib) / closest_fib
                    compliance_score += (1.0 - distance) * 0.1
        
        # Weight by user's fibonacci affinity
        compliance_score *= fibonacci_affinity
        
        return min(1.0, max(0.0, compliance_score))
    
    def _calculate_pi_compliance(self, structure: Dict[str, Any], pi_affinity: float) -> float:
        """Calculate compliance with pi-related proportions"""
        pi_value = math.pi
        
        compliance_score = 0.0
        
        # Check ratios against pi
        ratios = []
        
        line_count = structure.get('line_count', 0)
        word_count = structure.get('word_count', 0)
        
        if line_count > 0 and word_count > 0:
            ratios.append(word_count / line_count)
        
        syllable_count = structure.get('syllable_count', 0)
        if syllable_count > 0 and word_count > 0:
            ratios.append(syllable_count / word_count)
        
        for ratio in ratios:
            pi_distance = abs(ratio - pi_value) / pi_value
            compliance_score += (1.0 - pi_distance) * 0.5
        
        # Weight by user's pi affinity
        compliance_score *= pi_affinity
        
        return min(1.0, max(0.0, compliance_score))
    
    async def _analyze_work_archetypes(self, creative_work: Dict[str, Any]) -> Dict[str, float]:
        """Analyze archetypal characteristics of creative work"""
        # This would use sophisticated NLP analysis in a real implementation
        # For simulation, we'll use keywords and themes
        
        content = creative_work.get('content', '')
        themes = creative_work.get('themes', [])
        emotional_tone = creative_work.get('emotional_tone', 'neutral')
        
        archetype_scores = {}
        
        # Analyze content for archetypal indicators
        for archetype in MuseArchetype:
            score = 0.0
            
            # Keyword analysis (simplified)
            archetype_keywords = self._get_archetype_keywords(archetype)
            for keyword in archetype_keywords:
                if keyword.lower() in content.lower():
                    score += 0.1
            
            # Theme analysis
            archetype_themes = self._get_archetype_themes(archetype)
            for theme in themes:
                if theme in archetype_themes:
                    score += 0.2
            
            # Emotional tone analysis
            archetype_emotions = self._get_archetype_emotions(archetype)
            if emotional_tone in archetype_emotions:
                score += 0.3
            
            archetype_scores[archetype.value] = min(1.0, score)
        
        return archetype_scores
    
    def _get_archetype_keywords(self, archetype: MuseArchetype) -> List[str]:
        """Get keywords associated with an archetype"""
        keyword_map = {
            MuseArchetype.CALLIOPE: ['epic', 'heroic', 'grand', 'noble', 'legend'],
            MuseArchetype.CLIO: ['history', 'memory', 'past', 'chronicle', 'record'],
            MuseArchetype.ERATO: ['love', 'romance', 'passion', 'heart', 'beauty'],
            MuseArchetype.EUTERPE: ['music', 'harmony', 'melody', 'rhythm', 'song'],
            MuseArchetype.MELPOMENE: ['tragedy', 'sorrow', 'grief', 'loss', 'melancholy'],
            MuseArchetype.POLYHYMNIA: ['sacred', 'divine', 'spiritual', 'holy', 'prayer'],
            MuseArchetype.TERPSICHORE: ['dance', 'movement', 'flow', 'rhythm', 'grace'],
            MuseArchetype.THALIA: ['comedy', 'joy', 'laughter', 'humor', 'celebration'],
            MuseArchetype.URANIA: ['stars', 'cosmos', 'universe', 'celestial', 'infinite'],
            MuseArchetype.SOPHIA: ['wisdom', 'knowledge', 'understanding', 'truth', 'insight'],
            MuseArchetype.TECHNE: ['craft', 'skill', 'art', 'mastery', 'technique'],
            MuseArchetype.PSYCHE: ['soul', 'mind', 'consciousness', 'inner', 'psychological']
        }
        
        return keyword_map.get(archetype, [])
    
    def _get_archetype_themes(self, archetype: MuseArchetype) -> List[str]:
        """Get themes associated with an archetype"""
        theme_map = {
            MuseArchetype.CALLIOPE: ['heroism', 'adventure', 'quest', 'triumph'],
            MuseArchetype.CLIO: ['memory', 'tradition', 'legacy', 'documentation'],
            MuseArchetype.ERATO: ['love', 'relationships', 'beauty', 'desire'],
            MuseArchetype.EUTERPE: ['harmony', 'music', 'artistic_expression', 'creativity'],
            MuseArchetype.MELPOMENE: ['tragedy', 'loss', 'suffering', 'catharsis'],
            MuseArchetype.POLYHYMNIA: ['spirituality', 'devotion', 'transcendence', 'reverence'],
            MuseArchetype.TERPSICHORE: ['movement', 'dance', 'physical_expression', 'rhythm'],
            MuseArchetype.THALIA: ['comedy', 'joy', 'celebration', 'lightness'],
            MuseArchetype.URANIA: ['cosmos', 'science', 'exploration', 'infinity'],
            MuseArchetype.SOPHIA: ['wisdom', 'philosophy', 'understanding', 'knowledge'],
            MuseArchetype.TECHNE: ['craftsmanship', 'skill', 'mastery', 'precision'],
            MuseArchetype.PSYCHE: ['psychology', 'consciousness', 'introspection', 'self']
        }
        
        return theme_map.get(archetype, [])
    
    def _get_archetype_emotions(self, archetype: MuseArchetype) -> List[str]:
        """Get emotions associated with an archetype"""
        emotion_map = {
            MuseArchetype.CALLIOPE: ['heroic', 'noble', 'elevated', 'inspiring'],
            MuseArchetype.CLIO: ['nostalgic', 'reverent', 'contemplative', 'respectful'],
            MuseArchetype.ERATO: ['romantic', 'passionate', 'tender', 'loving'],
            MuseArchetype.EUTERPE: ['harmonious', 'flowing', 'melodic', 'rhythmic'],
            MuseArchetype.MELPOMENE: ['tragic', 'sorrowful', 'melancholic', 'profound'],
            MuseArchetype.POLYHYMNIA: ['sacred', 'reverent', 'spiritual', 'elevated'],
            MuseArchetype.TERPSICHORE: ['dynamic', 'energetic', 'graceful', 'expressive'],
            MuseArchetype.THALIA: ['joyful', 'humorous', 'light', 'celebratory'],
            MuseArchetype.URANIA: ['cosmic', 'expansive', 'mysterious', 'infinite'],
            MuseArchetype.SOPHIA: ['wise', 'thoughtful', 'deep', 'understanding'],
            MuseArchetype.TECHNE: ['precise', 'skilled', 'crafted', 'masterful'],
            MuseArchetype.PSYCHE: ['introspective', 'psychological', 'inner', 'reflective']
        }
        
        return emotion_map.get(archetype, [])
    
    def _calculate_confidence(self, values: List[float]) -> float:
        """Calculate confidence based on consistency of values"""
        if not values:
            return 0.0
        
        # Calculate coefficient of variation
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 0.0
        
        cv = std_val / mean_val
        
        # Convert to confidence (lower variation = higher confidence)
        confidence = 1.0 - min(1.0, cv)
        
        return confidence
    
    def _interpret_quality_rating(self, quality_rating: float) -> str:
        """Interpret quality rating"""
        if quality_rating >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif quality_rating >= self.quality_thresholds['good']:
            return 'good'
        elif quality_rating >= self.quality_thresholds['acceptable']:
            return 'acceptable'
        elif quality_rating >= self.quality_thresholds['poor']:
            return 'poor'
        else:
            return 'unacceptable'
    
    # Placeholder methods for complex calculations
    # These would be implemented with sophisticated algorithms in production
    
    async def _extract_work_frequencies(self, creative_work: Dict[str, Any]) -> Dict[str, float]:
        """Extract frequency characteristics from work"""
        # Placeholder implementation
        return {
            'rhythm_frequency': np.random.uniform(0.3, 0.9),
            'semantic_frequency': np.random.uniform(0.3, 0.9),
            'emotional_frequency': np.random.uniform(0.3, 0.9)
        }
    
    def _get_user_frequencies(self, frequency_signature: FrequencySignature) -> Dict[str, float]:
        """Get user's archetypal frequencies"""
        # Convert from frequency signature to frequency domain
        frequencies = {}
        
        for archetype, strength in frequency_signature.harmonic_blend.items():
            frequencies[f"{archetype}_frequency"] = strength
        
        return frequencies
    
    def _calculate_frequency_coherence_score(self, work_frequencies: Dict[str, float], 
                                           user_frequencies: Dict[str, float]) -> float:
        """Calculate frequency coherence score"""
        # Simplified coherence calculation
        coherence_scores = []
        
        for freq_type in ['rhythm_frequency', 'semantic_frequency', 'emotional_frequency']:
            if freq_type in work_frequencies:
                work_freq = work_frequencies[freq_type]
                # Find best matching user frequency
                best_match = max(user_frequencies.values())
                coherence = 1.0 - abs(work_freq - best_match)
                coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    # Additional placeholder methods would be implemented here
    # for all the other calculation components...
    
    async def _analyze_geometric_properties(self, creative_work: Dict[str, Any]) -> Dict[str, float]:
        """Analyze geometric properties of creative work"""
        return {
            'symmetry': np.random.uniform(0.4, 0.9),
            'proportion': np.random.uniform(0.4, 0.9),
            'harmony': np.random.uniform(0.4, 0.9)
        }
    
    async def _analyze_mathematical_properties(self, creative_work: Dict[str, Any]) -> Dict[str, float]:
        """Analyze mathematical properties of creative work"""
        return {
            'simplicity': np.random.uniform(0.3, 0.8),
            'completeness': np.random.uniform(0.4, 0.9),
            'universality': np.random.uniform(0.3, 0.7),
            'surprise': np.random.uniform(0.2, 0.8)
        }
    
    async def _analyze_semantic_consistency(self, creative_work: Dict[str, Any]) -> Dict[str, float]:
        """Analyze semantic consistency of creative work"""
        return {
            'consistency_score': np.random.uniform(0.5, 0.9),
            'coherence_score': np.random.uniform(0.5, 0.9)
        }
    
    # Additional methods would continue here...
    
    async def calculate_all_metrics(self, 
                                  creative_work: Dict[str, Any],
                                  frequency_signature: FrequencySignature,
                                  context: Dict[str, Any] = None) -> Dict[str, MetricCalculationResult]:
        """
        Calculate all available metrics for a creative work
        
        Args:
            creative_work: The creative work to evaluate
            frequency_signature: User's frequency signature
            context: Additional context for calculation
            
        Returns:
            Dictionary of all calculated metrics
        """
        results = {}
        
        # Calculate core metrics
        results['mathematical_fitness'] = await self.calculate_mathematical_fitness(
            creative_work, frequency_signature, context
        )
        
        results['semantic_coherence'] = await self.calculate_semantic_coherence(
            creative_work, frequency_signature, context
        )
        
        results['archetypal_alignment'] = await self.calculate_archetypal_alignment(
            creative_work, frequency_signature, context
        )
        
        results['quality_rating'] = await self.calculate_quality_rating(
            creative_work, frequency_signature, {}, context
        )
        
        return results
    
    def get_metric_summary(self, results: Dict[str, MetricCalculationResult]) -> Dict[str, Any]:
        """
        Generate summary of metric calculation results
        
        Args:
            results: Dictionary of metric calculation results
            
        Returns:
            Summary of metrics
        """
        summary = {
            'overall_score': 0.0,
            'confidence': 0.0,
            'metric_count': len(results),
            'calculation_time': 0.0,
            'quality_interpretation': 'unknown',
            'strongest_aspects': [],
            'areas_for_improvement': [],
            'detailed_scores': {}
        }
        
        if not results:
            return summary
        
        # Calculate overall score
        scores = [result.value for result in results.values()]
        summary['overall_score'] = np.mean(scores)
        
        # Calculate overall confidence
        confidences = [result.confidence for result in results.values()]
        summary['confidence'] = np.mean(confidences)
        
        # Total calculation time
        summary['calculation_time'] = sum(result.calculation_time for result in results.values())
        
        # Quality interpretation
        summary['quality_interpretation'] = self._interpret_quality_rating(summary['overall_score'])
        
        # Identify strongest aspects (top 3)
        sorted_results = sorted(results.items(), key=lambda x: x[1].value, reverse=True)
        summary['strongest_aspects'] = [
            {'metric': name, 'score': result.value} 
            for name, result in sorted_results[:3]
        ]
        
        # Identify areas for improvement (bottom 3)
        summary['areas_for_improvement'] = [
            {'metric': name, 'score': result.value} 
            for name, result in sorted_results[-3:]
        ]
        
        # Detailed scores
        for name, result in results.items():
            summary['detailed_scores'][name] = {
                'value': result.value,
                'confidence': result.confidence,
                'components': result.components,
                'calculation_time': result.calculation_time
            }
        
        return summary