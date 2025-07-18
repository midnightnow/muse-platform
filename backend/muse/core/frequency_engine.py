"""
Frequency Engine for MUSE Platform

This module implements the archetypal frequency system used in MUSE's
Computational Platonism approach. It handles the 12 Muse archetypes,
hardware entropy integration, and frequency signature generation.
"""

import os
import json
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging


class MuseArchetype(Enum):
    """The 12 Muse archetypes used in MUSE frequency signatures"""
    CALLIOPE = "CALLIOPE"      # Epic poetry, eloquence
    CLIO = "CLIO"              # History, documentation
    ERATO = "ERATO"            # Lyric poetry, love
    EUTERPE = "EUTERPE"        # Music, harmony
    MELPOMENE = "MELPOMENE"    # Tragedy, drama
    POLYHYMNIA = "POLYHYMNIA"  # Sacred poetry, hymns
    TERPSICHORE = "TERPSICHORE" # Dance, movement
    THALIA = "THALIA"          # Comedy, joy
    URANIA = "URANIA"          # Astronomy, cosmos
    SOPHIA = "SOPHIA"          # Wisdom, philosophy
    TECHNE = "TECHNE"          # Craft, skill
    PSYCHE = "PSYCHE"          # Soul, psychology


@dataclass
class FrequencySignature:
    """Individual user's archetypal frequency signature"""
    id: str
    user_id: str
    harmonic_blend: Dict[str, float]  # Archetype -> strength (0.0-1.0)
    sacred_ratios: Dict[str, float]   # Mathematical constants affinity
    spiral_coordinates: Dict[str, float]  # 3D position in archetypal space
    entropy_seed: str
    primary_muse: str
    secondary_muse: str
    created_at: datetime
    last_tuned: Optional[datetime] = None


@dataclass
class CreativeConstraint:
    """Constraints for creative session based on frequency signature"""
    form_type: str
    syllable_pattern: List[int]
    rhyme_scheme: str
    emotional_palette: List[str]
    sacred_geometry_influence: Dict[str, float]
    entropy_level: float


class MuseFrequencyEngine:
    """
    Core frequency engine for MUSE platform
    
    Handles archetypal frequency generation, hardware entropy integration,
    and frequency signature management for Computational Platonism.
    """
    
    # Sacred mathematical constants
    PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
    PI = math.pi
    E = math.e
    
    # Archetypal frequency base values (in Hz conceptually)
    ARCHETYPAL_FREQUENCIES = {
        MuseArchetype.CALLIOPE: 432.0,      # Epic resonance
        MuseArchetype.CLIO: 528.0,          # Historical truth
        MuseArchetype.ERATO: 639.0,         # Love frequency
        MuseArchetype.EUTERPE: 741.0,       # Musical harmony
        MuseArchetype.MELPOMENE: 852.0,     # Tragic depth
        MuseArchetype.POLYHYMNIA: 963.0,    # Sacred resonance
        MuseArchetype.TERPSICHORE: 396.0,   # Movement rhythm
        MuseArchetype.THALIA: 285.0,        # Joyful lightness
        MuseArchetype.URANIA: 174.0,        # Cosmic vibration
        MuseArchetype.SOPHIA: 417.0,        # Wisdom frequency
        MuseArchetype.TECHNE: 693.0,        # Craftsmanship
        MuseArchetype.PSYCHE: 777.0,        # Soul resonance
    }
    
    def __init__(self, entropy_source: str = "/dev/hardcard"):
        """
        Initialize the frequency engine
        
        Args:
            entropy_source: Path to hardware entropy source
        """
        self.entropy_source = entropy_source
        self.logger = logging.getLogger(__name__)
        self._entropy_cache = []
        self._cache_size = 1024
        
        # Initialize entropy cache
        self._refill_entropy_cache()
        
    def read_hardware_entropy(self, bytes_needed: int = 32) -> bytes:
        """
        Read entropy from hardware source with fallback
        
        Args:
            bytes_needed: Number of random bytes needed
            
        Returns:
            Random bytes from hardware or fallback source
        """
        try:
            # First try the primary entropy source
            if os.path.exists(self.entropy_source):
                with open(self.entropy_source, 'rb') as f:
                    entropy_bytes = f.read(bytes_needed)
                    if len(entropy_bytes) == bytes_needed:
                        return entropy_bytes
                        
            # Fallback to system entropy
            if os.path.exists('/dev/urandom'):
                with open('/dev/urandom', 'rb') as f:
                    entropy_bytes = f.read(bytes_needed)
                    if len(entropy_bytes) == bytes_needed:
                        self.logger.info("Using /dev/urandom fallback for entropy")
                        return entropy_bytes
                        
        except Exception as e:
            self.logger.warning(f"Hardware entropy read failed: {e}")
            
        # Final fallback to Python's random
        self.logger.warning("Using Python random fallback for entropy")
        return bytes([random.randint(0, 255) for _ in range(bytes_needed)])
    
    def _refill_entropy_cache(self) -> None:
        """Refill the entropy cache for faster access"""
        entropy_bytes = self.read_hardware_entropy(self._cache_size)
        self._entropy_cache = list(entropy_bytes)
    
    def _get_entropy_float(self) -> float:
        """Get a float between 0.0 and 1.0 from entropy cache"""
        if len(self._entropy_cache) < 4:
            self._refill_entropy_cache()
            
        # Use 4 bytes to create a float
        bytes_for_float = self._entropy_cache[:4]
        self._entropy_cache = self._entropy_cache[4:]
        
        # Convert bytes to float in range [0.0, 1.0)
        int_value = sum(b * (256 ** i) for i, b in enumerate(bytes_for_float))
        return int_value / (256 ** 4)
    
    def generate_frequency_signature(self, assessment_data: Dict[str, Any]) -> FrequencySignature:
        """
        Generate a frequency signature from personality assessment
        
        Args:
            assessment_data: User's personality assessment results
            
        Returns:
            Complete frequency signature
        """
        # Generate unique signature ID
        signature_id = f"freq_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Extract user preferences and traits
        user_id = assessment_data.get('user_id', 'anonymous')
        creative_preferences = assessment_data.get('creative_preferences', {})
        personality_traits = assessment_data.get('personality_traits', {})
        mathematical_affinity = assessment_data.get('mathematical_affinity', {})
        
        # Generate entropy seed
        entropy_bytes = self.read_hardware_entropy(16)
        entropy_seed = entropy_bytes.hex()
        
        # Calculate harmonic blend based on assessment
        harmonic_blend = self._calculate_harmonic_blend(
            creative_preferences, 
            personality_traits, 
            entropy_seed
        )
        
        # Calculate sacred ratios affinity
        sacred_ratios = self._calculate_sacred_ratios(
            mathematical_affinity,
            harmonic_blend
        )
        
        # Generate spiral coordinates
        spiral_coordinates = self._calculate_spiral_coordinates(
            sacred_ratios.get('phi', 0.5),
            sacred_ratios.get('pi', 0.5),
            sacred_ratios.get('fibonacci', 0.5)
        )
        
        # Determine primary and secondary muses
        primary_muse = max(harmonic_blend.items(), key=lambda x: x[1])[0]
        secondary_candidates = sorted(harmonic_blend.items(), key=lambda x: x[1], reverse=True)
        secondary_muse = secondary_candidates[1][0] if len(secondary_candidates) > 1 else primary_muse
        
        return FrequencySignature(
            id=signature_id,
            user_id=user_id,
            harmonic_blend=harmonic_blend,
            sacred_ratios=sacred_ratios,
            spiral_coordinates=spiral_coordinates,
            entropy_seed=entropy_seed,
            primary_muse=primary_muse,
            secondary_muse=secondary_muse,
            created_at=datetime.now()
        )
    
    def _calculate_harmonic_blend(self, 
                                 creative_preferences: Dict[str, Any], 
                                 personality_traits: Dict[str, Any], 
                                 entropy_seed: str) -> Dict[str, float]:
        """
        Calculate the harmonic blend of archetypal frequencies
        
        Args:
            creative_preferences: User's creative preferences
            personality_traits: User's personality traits
            entropy_seed: Entropy seed for randomization
            
        Returns:
            Dictionary mapping archetype names to strength values
        """
        # Initialize base values with small random variations
        random.seed(entropy_seed)
        base_values = {archetype.value: 0.1 + self._get_entropy_float() * 0.2 
                       for archetype in MuseArchetype}
        
        # Adjust based on creative preferences
        if creative_preferences.get('poetry_style') == 'epic':
            base_values[MuseArchetype.CALLIOPE.value] += 0.3
        elif creative_preferences.get('poetry_style') == 'lyric':
            base_values[MuseArchetype.ERATO.value] += 0.3
        elif creative_preferences.get('poetry_style') == 'narrative':
            base_values[MuseArchetype.CLIO.value] += 0.3
            
        if creative_preferences.get('emotional_range') == 'tragic':
            base_values[MuseArchetype.MELPOMENE.value] += 0.25
        elif creative_preferences.get('emotional_range') == 'comedic':
            base_values[MuseArchetype.THALIA.value] += 0.25
        elif creative_preferences.get('emotional_range') == 'sacred':
            base_values[MuseArchetype.POLYHYMNIA.value] += 0.25
            
        # Adjust based on personality traits
        if personality_traits.get('analytical', False):
            base_values[MuseArchetype.SOPHIA.value] += 0.2
            base_values[MuseArchetype.URANIA.value] += 0.15
            
        if personality_traits.get('musical', False):
            base_values[MuseArchetype.EUTERPE.value] += 0.2
            base_values[MuseArchetype.TERPSICHORE.value] += 0.15
            
        if personality_traits.get('introspective', False):
            base_values[MuseArchetype.PSYCHE.value] += 0.25
            
        if personality_traits.get('technical', False):
            base_values[MuseArchetype.TECHNE.value] += 0.2
            
        # Normalize to ensure sum equals 1.0
        total = sum(base_values.values())
        if total > 0:
            normalized_values = {k: v / total for k, v in base_values.items()}
        else:
            # Fallback to equal distribution
            normalized_values = {k: 1.0 / len(base_values) for k in base_values.keys()}
            
        return normalized_values
    
    def _calculate_sacred_ratios(self, 
                                mathematical_affinity: Dict[str, Any], 
                                harmonic_blend: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate user's affinity for sacred mathematical constants
        
        Args:
            mathematical_affinity: User's mathematical preferences
            harmonic_blend: User's archetypal blend
            
        Returns:
            Dictionary of sacred ratio affinities
        """
        # Base affinities influenced by entropy
        affinities = {
            'phi': 0.3 + self._get_entropy_float() * 0.4,
            'pi': 0.3 + self._get_entropy_float() * 0.4,
            'e': 0.3 + self._get_entropy_float() * 0.4,
            'fibonacci': 0.3 + self._get_entropy_float() * 0.4,
            'sqrt_2': 0.3 + self._get_entropy_float() * 0.4,
            'sqrt_3': 0.3 + self._get_entropy_float() * 0.4
        }
        
        # Adjust based on mathematical affinity
        if mathematical_affinity.get('geometry_preference') == 'golden_ratio':
            affinities['phi'] += 0.3
            affinities['fibonacci'] += 0.2
            
        if mathematical_affinity.get('number_theory_interest', False):
            affinities['pi'] += 0.25
            affinities['e'] += 0.25
            
        # Adjust based on archetypal blend
        if harmonic_blend.get(MuseArchetype.URANIA.value, 0) > 0.2:
            affinities['pi'] += 0.2  # Cosmic/astronomical connection
            
        if harmonic_blend.get(MuseArchetype.SOPHIA.value, 0) > 0.2:
            affinities['phi'] += 0.2  # Wisdom/golden ratio connection
            
        if harmonic_blend.get(MuseArchetype.EUTERPE.value, 0) > 0.2:
            affinities['fibonacci'] += 0.2  # Musical harmony connection
            
        # Ensure values stay in [0.0, 1.0] range
        for key in affinities:
            affinities[key] = min(1.0, max(0.0, affinities[key]))
            
        return affinities
    
    def _calculate_spiral_coordinates(self, 
                                     phi_affinity: float, 
                                     pi_affinity: float, 
                                     fibonacci_affinity: float) -> Dict[str, float]:
        """
        Calculate 3D spiral coordinates in archetypal space
        
        Args:
            phi_affinity: Affinity for golden ratio (0.0-1.0)
            pi_affinity: Affinity for pi (0.0-1.0)
            fibonacci_affinity: Affinity for fibonacci (0.0-1.0)
            
        Returns:
            Dictionary with x, y, z coordinates
        """
        # Use sacred ratios to determine spiral parameters
        radius = phi_affinity * self.PHI
        angle = pi_affinity * 2 * self.PI
        height = fibonacci_affinity * math.sqrt(5)
        
        # Add entropy for uniqueness
        entropy_offset = self._get_entropy_float() * 0.1
        
        # Calculate spiral coordinates
        x = radius * math.cos(angle) + entropy_offset
        y = radius * math.sin(angle) + entropy_offset
        z = height + entropy_offset
        
        return {
            'x': x,
            'y': y,
            'z': z,
            'radius': radius,
            'angle': angle,
            'height': height
        }
    
    def tune_signature(self, 
                      signature: FrequencySignature, 
                      target_muses: List[str], 
                      blend_ratios: List[float]) -> FrequencySignature:
        """
        Tune a frequency signature toward specific muses
        
        Args:
            signature: Existing frequency signature
            target_muses: List of muse names to emphasize
            blend_ratios: Corresponding blend ratios for each muse
            
        Returns:
            Updated frequency signature
        """
        if len(target_muses) != len(blend_ratios):
            raise ValueError("Target muses and blend ratios must have same length")
            
        # Validate muse names
        valid_muses = [muse.value for muse in MuseArchetype]
        for muse in target_muses:
            if muse not in valid_muses:
                raise ValueError(f"Invalid muse: {muse}")
                
        # Create new harmonic blend
        new_blend = signature.harmonic_blend.copy()
        
        # Normalize blend ratios
        total_ratio = sum(blend_ratios)
        if total_ratio > 0:
            normalized_ratios = [ratio / total_ratio for ratio in blend_ratios]
        else:
            normalized_ratios = [1.0 / len(blend_ratios) for _ in blend_ratios]
        
        # Calculate adjustment factor
        adjustment_strength = 0.3  # How much to adjust toward targets
        
        # Adjust toward target muses
        for muse, ratio in zip(target_muses, normalized_ratios):
            current_strength = new_blend.get(muse, 0.1)
            target_strength = ratio * adjustment_strength
            new_blend[muse] = min(1.0, current_strength + target_strength)
            
        # Renormalize entire blend
        total = sum(new_blend.values())
        if total > 0:
            new_blend = {k: v / total for k, v in new_blend.items()}
            
        # Update primary and secondary muses
        sorted_muses = sorted(new_blend.items(), key=lambda x: x[1], reverse=True)
        primary_muse = sorted_muses[0][0]
        secondary_muse = sorted_muses[1][0] if len(sorted_muses) > 1 else primary_muse
        
        # Recalculate spiral coordinates
        phi_affinity = signature.sacred_ratios.get('phi', 0.5)
        pi_affinity = signature.sacred_ratios.get('pi', 0.5)
        fibonacci_affinity = signature.sacred_ratios.get('fibonacci', 0.5)
        
        new_spiral_coordinates = self._calculate_spiral_coordinates(
            phi_affinity, pi_affinity, fibonacci_affinity
        )
        
        # Create updated signature
        updated_signature = FrequencySignature(
            id=signature.id,
            user_id=signature.user_id,
            harmonic_blend=new_blend,
            sacred_ratios=signature.sacred_ratios,
            spiral_coordinates=new_spiral_coordinates,
            entropy_seed=signature.entropy_seed,
            primary_muse=primary_muse,
            secondary_muse=secondary_muse,
            created_at=signature.created_at,
            last_tuned=datetime.now()
        )
        
        return updated_signature
    
    def measure_resonance(self, sig1: FrequencySignature, sig2: FrequencySignature) -> float:
        """
        Measure resonance between two frequency signatures
        
        Args:
            sig1: First frequency signature
            sig2: Second frequency signature
            
        Returns:
            Resonance score between 0.0 and 1.0
        """
        # Calculate harmonic blend similarity (cosine similarity)
        blend1_values = [sig1.harmonic_blend.get(muse.value, 0.0) for muse in MuseArchetype]
        blend2_values = [sig2.harmonic_blend.get(muse.value, 0.0) for muse in MuseArchetype]
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(blend1_values, blend2_values))
        magnitude1 = math.sqrt(sum(a * a for a in blend1_values))
        magnitude2 = math.sqrt(sum(b * b for b in blend2_values))
        
        if magnitude1 == 0 or magnitude2 == 0:
            harmonic_similarity = 0.0
        else:
            harmonic_similarity = dot_product / (magnitude1 * magnitude2)
            
        # Calculate sacred ratios similarity
        sacred_keys = ['phi', 'pi', 'e', 'fibonacci', 'sqrt_2', 'sqrt_3']
        sacred1_values = [sig1.sacred_ratios.get(key, 0.0) for key in sacred_keys]
        sacred2_values = [sig2.sacred_ratios.get(key, 0.0) for key in sacred_keys]
        
        sacred_similarity = 1.0 - (sum(abs(a - b) for a, b in zip(sacred1_values, sacred2_values)) / len(sacred_keys))
        sacred_similarity = max(0.0, sacred_similarity)
        
        # Calculate spatial distance in archetypal space
        coord1 = sig1.spiral_coordinates
        coord2 = sig2.spiral_coordinates
        
        spatial_distance = math.sqrt(
            (coord1.get('x', 0) - coord2.get('x', 0)) ** 2 +
            (coord1.get('y', 0) - coord2.get('y', 0)) ** 2 +
            (coord1.get('z', 0) - coord2.get('z', 0)) ** 2
        )
        
        # Normalize spatial distance to similarity (smaller distance = higher similarity)
        max_distance = 10.0  # Approximate maximum distance in archetypal space
        spatial_similarity = 1.0 - min(spatial_distance / max_distance, 1.0)
        
        # Combine all similarity measures
        resonance_score = (
            harmonic_similarity * 0.5 +
            sacred_similarity * 0.3 +
            spatial_similarity * 0.2
        )
        
        return max(0.0, min(1.0, resonance_score))
    
    def generate_creative_constraints(self, signature: FrequencySignature) -> CreativeConstraint:
        """
        Generate creative constraints based on frequency signature
        
        Args:
            signature: User's frequency signature
            
        Returns:
            Creative constraints for poetry generation
        """
        primary_muse = signature.primary_muse
        secondary_muse = signature.secondary_muse
        
        # Determine form type based on primary muse
        form_mappings = {
            MuseArchetype.CALLIOPE.value: 'epic',
            MuseArchetype.CLIO.value: 'narrative',
            MuseArchetype.ERATO.value: 'sonnet',
            MuseArchetype.EUTERPE.value: 'lyric',
            MuseArchetype.MELPOMENE.value: 'dramatic',
            MuseArchetype.POLYHYMNIA.value: 'hymn',
            MuseArchetype.TERPSICHORE.value: 'rhythmic',
            MuseArchetype.THALIA.value: 'comedic',
            MuseArchetype.URANIA.value: 'cosmic',
            MuseArchetype.SOPHIA.value: 'philosophical',
            MuseArchetype.TECHNE.value: 'technical',
            MuseArchetype.PSYCHE.value: 'introspective'
        }
        
        form_type = form_mappings.get(primary_muse, 'free_verse')
        
        # Generate syllable pattern based on sacred ratios
        phi_affinity = signature.sacred_ratios.get('phi', 0.5)
        fibonacci_affinity = signature.sacred_ratios.get('fibonacci', 0.5)
        
        if fibonacci_affinity > 0.6:
            syllable_pattern = [5, 8, 13, 8, 5]  # Fibonacci sequence
        elif phi_affinity > 0.6:
            syllable_pattern = [8, 13, 8, 5, 8]  # Golden ratio inspired
        else:
            syllable_pattern = [10, 10, 10, 10]  # Traditional
            
        # Generate rhyme scheme
        rhyme_schemes = {
            'sonnet': 'ABAB CDCD EFEF GG',
            'epic': 'ABAB',
            'lyric': 'ABCB',
            'narrative': 'AABB',
            'hymn': 'ABCB',
            'free_verse': 'ABCD'
        }
        
        rhyme_scheme = rhyme_schemes.get(form_type, 'ABCD')
        
        # Generate emotional palette
        emotional_palettes = {
            MuseArchetype.CALLIOPE.value: ['heroic', 'noble', 'grand', 'elevated'],
            MuseArchetype.CLIO.value: ['historical', 'documentary', 'factual', 'memorial'],
            MuseArchetype.ERATO.value: ['romantic', 'passionate', 'tender', 'loving'],
            MuseArchetype.EUTERPE.value: ['harmonious', 'melodic', 'lyrical', 'flowing'],
            MuseArchetype.MELPOMENE.value: ['tragic', 'sorrowful', 'melancholy', 'profound'],
            MuseArchetype.POLYHYMNIA.value: ['sacred', 'reverent', 'spiritual', 'divine'],
            MuseArchetype.TERPSICHORE.value: ['rhythmic', 'dynamic', 'energetic', 'moving'],
            MuseArchetype.THALIA.value: ['comedic', 'light', 'joyful', 'playful'],
            MuseArchetype.URANIA.value: ['cosmic', 'astronomical', 'infinite', 'stellar'],
            MuseArchetype.SOPHIA.value: ['wise', 'philosophical', 'contemplative', 'deep'],
            MuseArchetype.TECHNE.value: ['skillful', 'crafted', 'precise', 'technical'],
            MuseArchetype.PSYCHE.value: ['introspective', 'psychological', 'emotional', 'inner']
        }
        
        emotional_palette = emotional_palettes.get(primary_muse, ['neutral', 'balanced', 'calm'])
        
        # Add secondary muse influence
        if secondary_muse != primary_muse:
            secondary_emotions = emotional_palettes.get(secondary_muse, [])
            emotional_palette.extend(secondary_emotions[:2])  # Add top 2 from secondary
            
        # Calculate sacred geometry influence
        sacred_geometry_influence = {
            'phi_weight': signature.sacred_ratios.get('phi', 0.5),
            'pi_weight': signature.sacred_ratios.get('pi', 0.5),
            'fibonacci_weight': signature.sacred_ratios.get('fibonacci', 0.5),
            'golden_spiral': phi_affinity > 0.7,
            'pentagonal_symmetry': phi_affinity > 0.8,
            'circular_imagery': signature.sacred_ratios.get('pi', 0.5) > 0.7
        }
        
        # Calculate entropy level
        entropy_level = min(1.0, signature.harmonic_blend.get(primary_muse, 0.5) + 
                           self._get_entropy_float() * 0.3)
        
        return CreativeConstraint(
            form_type=form_type,
            syllable_pattern=syllable_pattern,
            rhyme_scheme=rhyme_scheme,
            emotional_palette=emotional_palette,
            sacred_geometry_influence=sacred_geometry_influence,
            entropy_level=entropy_level
        )
    
    def serialize_signature(self, signature: FrequencySignature) -> str:
        """
        Serialize frequency signature to JSON string
        
        Args:
            signature: Frequency signature to serialize
            
        Returns:
            JSON string representation
        """
        signature_dict = asdict(signature)
        signature_dict['created_at'] = signature.created_at.isoformat()
        if signature.last_tuned:
            signature_dict['last_tuned'] = signature.last_tuned.isoformat()
        else:
            signature_dict['last_tuned'] = None
            
        return json.dumps(signature_dict, indent=2)
    
    def deserialize_signature(self, signature_json: str) -> FrequencySignature:
        """
        Deserialize frequency signature from JSON string
        
        Args:
            signature_json: JSON string representation
            
        Returns:
            FrequencySignature object
        """
        signature_dict = json.loads(signature_json)
        signature_dict['created_at'] = datetime.fromisoformat(signature_dict['created_at'])
        if signature_dict['last_tuned']:
            signature_dict['last_tuned'] = datetime.fromisoformat(signature_dict['last_tuned'])
        else:
            signature_dict['last_tuned'] = None
            
        return FrequencySignature(**signature_dict)
    
    def get_archetypal_info(self, archetype: MuseArchetype) -> Dict[str, Any]:
        """
        Get detailed information about a specific archetype
        
        Args:
            archetype: Muse archetype to get information about
            
        Returns:
            Dictionary with archetype information
        """
        archetype_info = {
            MuseArchetype.CALLIOPE: {
                'name': 'Calliope',
                'domain': 'Epic Poetry & Eloquence',
                'description': 'The eldest muse, presiding over epic poetry and grand narratives',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.CALLIOPE],
                'sacred_geometry': 'Golden spiral',
                'emotional_range': ['heroic', 'noble', 'grand', 'elevated'],
                'poetic_forms': ['epic', 'heroic verse', 'narrative poem']
            },
            MuseArchetype.CLIO: {
                'name': 'Clio',
                'domain': 'History & Documentation',
                'description': 'The muse of history, preserving truth and memory',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.CLIO],
                'sacred_geometry': 'Linear progression',
                'emotional_range': ['historical', 'documentary', 'factual', 'memorial'],
                'poetic_forms': ['narrative', 'historical verse', 'documentary poem']
            },
            MuseArchetype.ERATO: {
                'name': 'Erato',
                'domain': 'Lyric Poetry & Love',
                'description': 'The muse of love poetry and lyrical expression',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.ERATO],
                'sacred_geometry': 'Heart curve',
                'emotional_range': ['romantic', 'passionate', 'tender', 'loving'],
                'poetic_forms': ['sonnet', 'love song', 'lyric poem']
            },
            MuseArchetype.EUTERPE: {
                'name': 'Euterpe',
                'domain': 'Music & Harmony',
                'description': 'The muse of music and harmonic structure',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.EUTERPE],
                'sacred_geometry': 'Harmonic series',
                'emotional_range': ['harmonious', 'melodic', 'lyrical', 'flowing'],
                'poetic_forms': ['song', 'hymn', 'musical verse']
            },
            MuseArchetype.MELPOMENE: {
                'name': 'Melpomene',
                'domain': 'Tragedy & Drama',
                'description': 'The muse of tragedy and dramatic expression',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.MELPOMENE],
                'sacred_geometry': 'Descending spiral',
                'emotional_range': ['tragic', 'sorrowful', 'melancholy', 'profound'],
                'poetic_forms': ['tragedy', 'elegy', 'dramatic monologue']
            },
            MuseArchetype.POLYHYMNIA: {
                'name': 'Polyhymnia',
                'domain': 'Sacred Poetry & Hymns',
                'description': 'The muse of sacred poetry and divine praise',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.POLYHYMNIA],
                'sacred_geometry': 'Vesica piscis',
                'emotional_range': ['sacred', 'reverent', 'spiritual', 'divine'],
                'poetic_forms': ['hymn', 'prayer', 'sacred verse']
            },
            MuseArchetype.TERPSICHORE: {
                'name': 'Terpsichore',
                'domain': 'Dance & Movement',
                'description': 'The muse of dance and rhythmic movement',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.TERPSICHORE],
                'sacred_geometry': 'Rhythmic patterns',
                'emotional_range': ['rhythmic', 'dynamic', 'energetic', 'moving'],
                'poetic_forms': ['dance song', 'rhythmic verse', 'performance poem']
            },
            MuseArchetype.THALIA: {
                'name': 'Thalia',
                'domain': 'Comedy & Joy',
                'description': 'The muse of comedy and joyful expression',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.THALIA],
                'sacred_geometry': 'Playful curves',
                'emotional_range': ['comedic', 'light', 'joyful', 'playful'],
                'poetic_forms': ['comedy', 'light verse', 'satirical poem']
            },
            MuseArchetype.URANIA: {
                'name': 'Urania',
                'domain': 'Astronomy & Cosmos',
                'description': 'The muse of astronomy and cosmic understanding',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.URANIA],
                'sacred_geometry': 'Celestial mechanics',
                'emotional_range': ['cosmic', 'astronomical', 'infinite', 'stellar'],
                'poetic_forms': ['cosmic verse', 'astronomical poem', 'stellar meditation']
            },
            MuseArchetype.SOPHIA: {
                'name': 'Sophia',
                'domain': 'Wisdom & Philosophy',
                'description': 'The muse of wisdom and philosophical insight',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.SOPHIA],
                'sacred_geometry': 'Golden ratio',
                'emotional_range': ['wise', 'philosophical', 'contemplative', 'deep'],
                'poetic_forms': ['philosophical verse', 'wisdom poem', 'contemplative verse']
            },
            MuseArchetype.TECHNE: {
                'name': 'Techne',
                'domain': 'Craft & Skill',
                'description': 'The muse of technical skill and craftsmanship',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.TECHNE],
                'sacred_geometry': 'Precise structures',
                'emotional_range': ['skillful', 'crafted', 'precise', 'technical'],
                'poetic_forms': ['technical verse', 'craft poem', 'structured form']
            },
            MuseArchetype.PSYCHE: {
                'name': 'Psyche',
                'domain': 'Soul & Psychology',
                'description': 'The muse of the soul and psychological depth',
                'frequency': self.ARCHETYPAL_FREQUENCIES[MuseArchetype.PSYCHE],
                'sacred_geometry': 'Inner spirals',
                'emotional_range': ['introspective', 'psychological', 'emotional', 'inner'],
                'poetic_forms': ['psychological verse', 'introspective poem', 'soul song']
            }
        }
        
        return archetype_info.get(archetype, {})
    
    def get_frequency_analytics(self, signature: FrequencySignature) -> Dict[str, Any]:
        """
        Get detailed analytics about a frequency signature
        
        Args:
            signature: Frequency signature to analyze
            
        Returns:
            Dictionary with analytics data
        """
        analytics = {
            'signature_id': signature.id,
            'dominant_archetypes': [],
            'balance_metrics': {},
            'sacred_geometry_profile': {},
            'creative_potential': {},
            'resonance_patterns': {}
        }
        
        # Find dominant archetypes
        sorted_archetypes = sorted(signature.harmonic_blend.items(), 
                                 key=lambda x: x[1], reverse=True)
        analytics['dominant_archetypes'] = [
            {
                'archetype': archetype,
                'strength': strength,
                'info': self.get_archetypal_info(MuseArchetype(archetype))
            }
            for archetype, strength in sorted_archetypes[:3]
        ]
        
        # Calculate balance metrics
        blend_values = list(signature.harmonic_blend.values())
        analytics['balance_metrics'] = {
            'distribution_variance': np.var(blend_values),
            'entropy': -sum(v * math.log(v + 1e-10) for v in blend_values),
            'specialization_index': max(blend_values) / np.mean(blend_values),
            'diversity_index': 1 - sum(v * v for v in blend_values)
        }
        
        # Sacred geometry profile
        analytics['sacred_geometry_profile'] = {
            'phi_dominance': signature.sacred_ratios.get('phi', 0.5),
            'pi_affinity': signature.sacred_ratios.get('pi', 0.5),
            'fibonacci_resonance': signature.sacred_ratios.get('fibonacci', 0.5),
            'geometric_balance': sum(signature.sacred_ratios.values()) / len(signature.sacred_ratios),
            'spiral_position': signature.spiral_coordinates
        }
        
        # Creative potential analysis
        primary_strength = signature.harmonic_blend.get(signature.primary_muse, 0.5)
        secondary_strength = signature.harmonic_blend.get(signature.secondary_muse, 0.5)
        
        analytics['creative_potential'] = {
            'primary_focus': primary_strength,
            'secondary_support': secondary_strength,
            'creative_flexibility': 1 - primary_strength,  # Lower dominance = more flexibility
            'innovation_capacity': analytics['balance_metrics']['diversity_index'],
            'technical_precision': signature.sacred_ratios.get('phi', 0.5)
        }
        
        # Resonance patterns
        analytics['resonance_patterns'] = {
            'archetypal_clusters': self._identify_archetypal_clusters(signature),
            'sacred_ratio_clusters': self._identify_sacred_ratio_clusters(signature),
            'spatial_region': self._identify_spatial_region(signature.spiral_coordinates)
        }
        
        return analytics
    
    def _identify_archetypal_clusters(self, signature: FrequencySignature) -> List[str]:
        """Identify which archetypal clusters the signature belongs to"""
        clusters = []
        
        # Creative cluster (Calliope, Erato, Euterpe)
        creative_sum = (signature.harmonic_blend.get(MuseArchetype.CALLIOPE.value, 0) +
                       signature.harmonic_blend.get(MuseArchetype.ERATO.value, 0) +
                       signature.harmonic_blend.get(MuseArchetype.EUTERPE.value, 0))
        if creative_sum > 0.4:
            clusters.append('creative')
            
        # Intellectual cluster (Clio, Sophia, Urania)
        intellectual_sum = (signature.harmonic_blend.get(MuseArchetype.CLIO.value, 0) +
                           signature.harmonic_blend.get(MuseArchetype.SOPHIA.value, 0) +
                           signature.harmonic_blend.get(MuseArchetype.URANIA.value, 0))
        if intellectual_sum > 0.4:
            clusters.append('intellectual')
            
        # Emotional cluster (Melpomene, Thalia, Psyche)
        emotional_sum = (signature.harmonic_blend.get(MuseArchetype.MELPOMENE.value, 0) +
                        signature.harmonic_blend.get(MuseArchetype.THALIA.value, 0) +
                        signature.harmonic_blend.get(MuseArchetype.PSYCHE.value, 0))
        if emotional_sum > 0.4:
            clusters.append('emotional')
            
        # Spiritual cluster (Polyhymnia, Terpsichore, Techne)
        spiritual_sum = (signature.harmonic_blend.get(MuseArchetype.POLYHYMNIA.value, 0) +
                        signature.harmonic_blend.get(MuseArchetype.TERPSICHORE.value, 0) +
                        signature.harmonic_blend.get(MuseArchetype.TECHNE.value, 0))
        if spiritual_sum > 0.4:
            clusters.append('spiritual')
            
        return clusters
    
    def _identify_sacred_ratio_clusters(self, signature: FrequencySignature) -> List[str]:
        """Identify which sacred ratio clusters the signature belongs to"""
        clusters = []
        
        # Golden ratio cluster
        if signature.sacred_ratios.get('phi', 0.5) > 0.7:
            clusters.append('golden_ratio')
            
        # Transcendental cluster (pi, e)
        if (signature.sacred_ratios.get('pi', 0.5) > 0.6 and 
            signature.sacred_ratios.get('e', 0.5) > 0.6):
            clusters.append('transcendental')
            
        # Radical cluster (sqrt_2, sqrt_3)
        if (signature.sacred_ratios.get('sqrt_2', 0.5) > 0.6 and
            signature.sacred_ratios.get('sqrt_3', 0.5) > 0.6):
            clusters.append('radical')
            
        # Fibonacci cluster
        if signature.sacred_ratios.get('fibonacci', 0.5) > 0.7:
            clusters.append('fibonacci')
            
        return clusters
    
    def _identify_spatial_region(self, coordinates: Dict[str, float]) -> str:
        """Identify spatial region in archetypal space"""
        x, y, z = coordinates.get('x', 0), coordinates.get('y', 0), coordinates.get('z', 0)
        
        # Determine region based on coordinate ranges
        if x > 0 and y > 0 and z > 0:
            return 'creative_ascension'
        elif x < 0 and y > 0 and z > 0:
            return 'reflective_growth'
        elif x < 0 and y < 0 and z > 0:
            return 'introspective_wisdom'
        elif x > 0 and y < 0 and z > 0:
            return 'dynamic_expression'
        elif z < 0:
            return 'foundational_depths'
        else:
            return 'balanced_center'