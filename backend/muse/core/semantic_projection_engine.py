"""
Semantic Projection Engine for MUSE Platform

This module bridges meaning and mathematics by creating semantic-mathematical
mappings that align with the Computational Platonism approach. It provides
tools for projecting themes to geometry, calculating semantic fitness, and
ensuring emotional coherence through mathematical optimization.
"""

import re
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import json
from dataclasses import dataclass
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import syllapy
import pronouncing


class SemanticDimension(Enum):
    """Dimensions of semantic space for projection"""
    EMOTION = "emotion"
    THEME = "theme"
    IMAGERY = "imagery"
    RHYTHM = "rhythm"
    METAPHOR = "metaphor"
    NARRATIVE = "narrative"


@dataclass
class SemanticVector:
    """Represents a semantic vector in mathematical space"""
    dimensions: Dict[str, float]
    magnitude: float
    dominant_features: List[str]
    sacred_resonance: float
    archetypal_alignment: Dict[str, float]


@dataclass
class WordEmbedding:
    """Mathematical representation of a word"""
    word: str
    vector: np.ndarray
    semantic_cluster: int
    sacred_weight: float
    archetypal_affinities: Dict[str, float]
    phonetic_features: Dict[str, Any]


@dataclass
class ThemeProjection:
    """Projection of a theme onto sacred geometry"""
    theme: str
    geometric_coordinates: Dict[str, float]
    sacred_constant_alignment: Dict[str, float]
    emotional_resonance: float
    archetypal_mapping: Dict[str, float]


class SemanticProjectionEngine:
    """
    Core engine for semantic-mathematical bridging in MUSE
    
    This engine creates mathematical representations of meaning,
    ensuring that semantic content aligns with sacred geometry
    and archetypal frequencies.
    """
    
    # Sacred mathematical constants
    PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
    PI = math.pi
    E = math.e
    
    # Emotional base vectors (normalized)
    EMOTION_VECTORS = {
        'joy': np.array([0.8, 0.6, 0.9, 0.7, 0.5]),
        'love': np.array([0.9, 0.8, 0.7, 0.6, 0.8]),
        'peace': np.array([0.5, 0.9, 0.6, 0.8, 0.7]),
        'wonder': np.array([0.7, 0.5, 0.9, 0.8, 0.6]),
        'melancholy': np.array([0.3, 0.6, 0.4, 0.7, 0.5]),
        'passion': np.array([0.9, 0.7, 0.8, 0.5, 0.9]),
        'wisdom': np.array([0.6, 0.8, 0.7, 0.9, 0.6]),
        'mystery': np.array([0.4, 0.7, 0.6, 0.8, 0.9]),
        'transcendence': np.array([0.7, 0.9, 0.8, 0.6, 0.7]),
        'nostalgia': np.array([0.5, 0.4, 0.7, 0.8, 0.6])
    }
    
    # Theme-to-geometry mappings
    THEME_GEOMETRIES = {
        'nature': {'phi_weight': 0.9, 'pi_weight': 0.6, 'fibonacci_weight': 0.8},
        'love': {'phi_weight': 0.8, 'pi_weight': 0.7, 'fibonacci_weight': 0.6},
        'cosmos': {'phi_weight': 0.7, 'pi_weight': 0.9, 'fibonacci_weight': 0.5},
        'time': {'phi_weight': 0.6, 'pi_weight': 0.8, 'fibonacci_weight': 0.7},
        'memory': {'phi_weight': 0.5, 'pi_weight': 0.7, 'fibonacci_weight': 0.9},
        'journey': {'phi_weight': 0.7, 'pi_weight': 0.6, 'fibonacci_weight': 0.8},
        'transformation': {'phi_weight': 0.8, 'pi_weight': 0.5, 'fibonacci_weight': 0.7},
        'mystery': {'phi_weight': 0.6, 'pi_weight': 0.8, 'fibonacci_weight': 0.6},
        'wisdom': {'phi_weight': 0.9, 'pi_weight': 0.7, 'fibonacci_weight': 0.8},
        'beauty': {'phi_weight': 0.9, 'pi_weight': 0.6, 'fibonacci_weight': 0.7}
    }
    
    def __init__(self):
        """Initialize the semantic projection engine"""
        self.vocabulary = {}
        self.word_embeddings = {}
        self.theme_projections = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.semantic_clusters = {}
        self.archetypal_weights = self._initialize_archetypal_weights()
        
    def _initialize_archetypal_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize archetypal weight mappings for words"""
        return {
            'CALLIOPE': {
                'patterns': ['epic', 'heroic', 'grand', 'noble', 'virtue', 'courage', 'quest'],
                'weight': 0.9
            },
            'CLIO': {
                'patterns': ['history', 'ancient', 'time', 'memory', 'past', 'chronicle', 'legacy'],
                'weight': 0.8
            },
            'ERATO': {
                'patterns': ['love', 'heart', 'passion', 'romance', 'tender', 'beloved', 'desire'],
                'weight': 0.9
            },
            'EUTERPE': {
                'patterns': ['music', 'song', 'rhythm', 'melody', 'harmony', 'lyric', 'tune'],
                'weight': 0.8
            },
            'MELPOMENE': {
                'patterns': ['tragedy', 'sorrow', 'grief', 'loss', 'tears', 'mourning', 'pain'],
                'weight': 0.7
            },
            'POLYHYMNIA': {
                'patterns': ['sacred', 'divine', 'holy', 'prayer', 'worship', 'reverence', 'blessed'],
                'weight': 0.9
            },
            'TERPSICHORE': {
                'patterns': ['dance', 'movement', 'grace', 'flowing', 'rhythm', 'step', 'motion'],
                'weight': 0.8
            },
            'THALIA': {
                'patterns': ['joy', 'laughter', 'comedy', 'mirth', 'celebration', 'happiness', 'light'],
                'weight': 0.8
            },
            'URANIA': {
                'patterns': ['star', 'cosmos', 'universe', 'celestial', 'astronomy', 'infinite', 'space'],
                'weight': 0.9
            },
            'SOPHIA': {
                'patterns': ['wisdom', 'knowledge', 'understanding', 'insight', 'truth', 'enlightenment'],
                'weight': 0.9
            },
            'TECHNE': {
                'patterns': ['craft', 'skill', 'art', 'technique', 'mastery', 'creation', 'form'],
                'weight': 0.8
            },
            'PSYCHE': {
                'patterns': ['soul', 'mind', 'spirit', 'consciousness', 'inner', 'depth', 'psyche'],
                'weight': 0.9
            }
        }
    
    def project_theme_to_geometry(self, theme: str, sacred_constant: str = 'phi') -> ThemeProjection:
        """
        Map a theme to mathematical coordinates using sacred geometry
        
        Args:
            theme: The theme to project (e.g., 'love', 'nature', 'cosmos')
            sacred_constant: Primary sacred constant to use ('phi', 'pi', 'fibonacci')
            
        Returns:
            ThemeProjection with geometric coordinates and alignments
        """
        # Normalize theme
        theme_lower = theme.lower().strip()
        
        # Get base geometry for theme
        base_geometry = self.THEME_GEOMETRIES.get(theme_lower, {
            'phi_weight': 0.5,
            'pi_weight': 0.5,
            'fibonacci_weight': 0.5
        })
        
        # Calculate geometric coordinates
        phi_coord = base_geometry['phi_weight'] * self.PHI
        pi_coord = base_geometry['pi_weight'] * self.PI
        fibonacci_coord = base_geometry['fibonacci_weight'] * self._fibonacci_scaling(8)
        
        # Add sacred constant emphasis
        if sacred_constant == 'phi':
            phi_coord *= 1.3
        elif sacred_constant == 'pi':
            pi_coord *= 1.3
        elif sacred_constant == 'fibonacci':
            fibonacci_coord *= 1.3
        
        geometric_coordinates = {
            'x': phi_coord,
            'y': pi_coord,
            'z': fibonacci_coord,
            'magnitude': math.sqrt(phi_coord**2 + pi_coord**2 + fibonacci_coord**2)
        }
        
        # Calculate sacred constant alignment
        sacred_constant_alignment = {
            'phi': base_geometry['phi_weight'],
            'pi': base_geometry['pi_weight'],
            'fibonacci': base_geometry['fibonacci_weight'],
            'primary': sacred_constant
        }
        
        # Calculate emotional resonance
        emotional_resonance = self._calculate_theme_emotional_resonance(theme_lower)
        
        # Calculate archetypal mapping
        archetypal_mapping = self._calculate_theme_archetypal_mapping(theme_lower)
        
        projection = ThemeProjection(
            theme=theme,
            geometric_coordinates=geometric_coordinates,
            sacred_constant_alignment=sacred_constant_alignment,
            emotional_resonance=emotional_resonance,
            archetypal_mapping=archetypal_mapping
        )
        
        # Cache the projection
        self.theme_projections[theme_lower] = projection
        
        return projection
    
    def calculate_semantic_fitness(self, word_list: List[str], theme_vector: np.ndarray) -> float:
        """
        Measure semantic coherence using vector mathematics
        
        Args:
            word_list: List of words to evaluate
            theme_vector: Target semantic vector
            
        Returns:
            Semantic fitness score (0.0 to 1.0)
        """
        if not word_list or len(theme_vector) == 0:
            return 0.0
        
        # Generate word embeddings if not cached
        word_vectors = []
        for word in word_list:
            if word not in self.word_embeddings:
                self._generate_word_embedding(word)
            
            if word in self.word_embeddings:
                word_vectors.append(self.word_embeddings[word].vector)
        
        if not word_vectors:
            return 0.0
        
        # Calculate average word vector
        avg_word_vector = np.mean(word_vectors, axis=0)
        
        # Ensure vectors are the same length
        min_length = min(len(avg_word_vector), len(theme_vector))
        if min_length == 0:
            return 0.0
        
        avg_word_vector = avg_word_vector[:min_length]
        theme_vector = theme_vector[:min_length]
        
        # Calculate cosine similarity
        dot_product = np.dot(avg_word_vector, theme_vector)
        magnitude_product = np.linalg.norm(avg_word_vector) * np.linalg.norm(theme_vector)
        
        if magnitude_product == 0:
            return 0.0
        
        cosine_sim = dot_product / magnitude_product
        
        # Convert to 0-1 scale (cosine similarity ranges from -1 to 1)
        fitness = (cosine_sim + 1) / 2
        
        # Apply sacred geometry bonus
        sacred_bonus = self._calculate_sacred_geometry_bonus(word_list)
        fitness = min(1.0, fitness + sacred_bonus)
        
        return fitness
    
    def generate_word_embeddings(self, vocabulary: List[str]) -> Dict[str, WordEmbedding]:
        """
        Create mathematical representations of words
        
        Args:
            vocabulary: List of words to embed
            
        Returns:
            Dictionary mapping words to WordEmbedding objects
        """
        if not vocabulary:
            return {}
        
        embeddings = {}
        
        # Create TF-IDF vectors for context (if we have enough words)
        if len(vocabulary) > 10:
            try:
                # Create documents for TF-IDF (using word contexts)
                documents = []
                for word in vocabulary:
                    # Create a simple context using word associations
                    context = self._create_word_context(word)
                    documents.append(context)
                
                # Fit TF-IDF vectorizer
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
                
                # Perform clustering
                n_clusters = min(10, len(vocabulary) // 3)
                if n_clusters > 0:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(tfidf_matrix)
                else:
                    clusters = [0] * len(vocabulary)
                
            except Exception as e:
                # Fallback to simple embeddings
                tfidf_matrix = None
                clusters = [0] * len(vocabulary)
        else:
            tfidf_matrix = None
            clusters = [0] * len(vocabulary)
        
        # Generate embeddings for each word
        for i, word in enumerate(vocabulary):
            embedding = self._generate_word_embedding(word, tfidf_matrix, i, clusters[i])
            embeddings[word] = embedding
        
        self.word_embeddings.update(embeddings)
        return embeddings
    
    def _generate_word_embedding(self, word: str, tfidf_matrix: Optional[np.ndarray] = None, 
                                word_index: int = 0, cluster: int = 0) -> WordEmbedding:
        """Generate embedding for a single word"""
        
        # Base vector using word characteristics
        base_vector = self._create_base_word_vector(word)
        
        # Add TF-IDF information if available
        if tfidf_matrix is not None and word_index < tfidf_matrix.shape[0]:
            tfidf_vector = tfidf_matrix[word_index].toarray().flatten()
            # Combine with base vector (weighted combination)
            combined_vector = np.concatenate([base_vector, tfidf_vector[:min(10, len(tfidf_vector))]])
        else:
            combined_vector = base_vector
        
        # Calculate sacred weight
        sacred_weight = self._calculate_word_sacred_weight(word)
        
        # Calculate archetypal affinities
        archetypal_affinities = self._calculate_word_archetypal_affinities(word)
        
        # Generate phonetic features
        phonetic_features = self._extract_phonetic_features(word)
        
        return WordEmbedding(
            word=word,
            vector=combined_vector,
            semantic_cluster=cluster,
            sacred_weight=sacred_weight,
            archetypal_affinities=archetypal_affinities,
            phonetic_features=phonetic_features
        )
    
    def _create_base_word_vector(self, word: str) -> np.ndarray:
        """Create base semantic vector for a word"""
        # Initialize vector with word characteristics
        vector_components = []
        
        # Length-based features
        vector_components.append(len(word) / 10.0)  # Normalized length
        vector_components.append(self._count_syllables(word) / 5.0)  # Normalized syllable count
        
        # Vowel/consonant ratio
        vowels = sum(1 for char in word.lower() if char in 'aeiou')
        consonants = len(word) - vowels
        vector_components.append(vowels / len(word) if len(word) > 0 else 0)
        
        # Phonetic features
        vector_components.append(1.0 if word.lower().endswith('ing') else 0.0)  # Progressive
        vector_components.append(1.0 if word.lower().endswith('ed') else 0.0)   # Past tense
        vector_components.append(1.0 if word.lower().endswith('ly') else 0.0)   # Adverb
        vector_components.append(1.0 if word.lower().endswith('er') else 0.0)   # Comparative
        
        # Semantic category hints (based on common patterns)
        vector_components.append(1.0 if any(pattern in word.lower() for pattern in ['love', 'heart', 'soul']) else 0.0)
        vector_components.append(1.0 if any(pattern in word.lower() for pattern in ['light', 'bright', 'shine']) else 0.0)
        vector_components.append(1.0 if any(pattern in word.lower() for pattern in ['dark', 'shadow', 'night']) else 0.0)
        
        return np.array(vector_components)
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using syllapy"""
        try:
            return syllapy.count(word)
        except:
            # Fallback syllable counting
            word = word.lower().strip()
            vowels = 'aeiouy'
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            # Handle silent e
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            return max(1, syllable_count)
    
    def _calculate_word_sacred_weight(self, word: str) -> float:
        """Calculate sacred geometry weight for a word"""
        word_lower = word.lower()
        
        # Sacred number patterns
        sacred_weight = 0.0
        
        # Length-based sacred weights
        word_length = len(word)
        if word_length in [3, 5, 8, 13, 21]:  # Fibonacci numbers
            sacred_weight += 0.3
        
        # Syllable-based sacred weights
        syllables = self._count_syllables(word)
        if syllables in [1, 2, 3, 5, 8]:  # Fibonacci numbers
            sacred_weight += 0.2
        
        # Golden ratio approximation in letter patterns
        if word_length >= 3:
            first_part = word_length * 0.618
            if abs(first_part - round(first_part)) < 0.1:
                sacred_weight += 0.2
        
        # Sacred words and roots
        sacred_patterns = [
            'sacred', 'divine', 'holy', 'eternal', 'infinite', 'cosmic',
            'harmony', 'balance', 'golden', 'perfect', 'pure', 'light',
            'mystery', 'wisdom', 'truth', 'beauty', 'unity', 'spiral'
        ]
        
        for pattern in sacred_patterns:
            if pattern in word_lower:
                sacred_weight += 0.4
                break
        
        return min(1.0, sacred_weight)
    
    def _calculate_word_archetypal_affinities(self, word: str) -> Dict[str, float]:
        """Calculate archetypal affinities for a word"""
        word_lower = word.lower()
        affinities = {}
        
        for archetype, data in self.archetypal_weights.items():
            affinity = 0.0
            
            # Check for pattern matches
            for pattern in data['patterns']:
                if pattern in word_lower:
                    affinity += data['weight']
                    break
            
            # Partial matches (contains parts of patterns)
            for pattern in data['patterns']:
                if len(pattern) > 3:
                    # Check for partial matches
                    for i in range(len(pattern) - 2):
                        if pattern[i:i+3] in word_lower:
                            affinity += data['weight'] * 0.3
                            break
            
            affinities[archetype] = min(1.0, affinity)
        
        return affinities
    
    def _extract_phonetic_features(self, word: str) -> Dict[str, Any]:
        """Extract phonetic features using pronouncing library"""
        try:
            phones = pronouncing.phones_for_word(word)
            rhymes = pronouncing.rhymes(word)
            
            return {
                'phones': phones[0] if phones else '',
                'rhyme_count': len(rhymes),
                'stress_pattern': self._extract_stress_pattern(phones[0] if phones else ''),
                'has_rhymes': len(rhymes) > 0
            }
        except:
            return {
                'phones': '',
                'rhyme_count': 0,
                'stress_pattern': '',
                'has_rhymes': False
            }
    
    def _extract_stress_pattern(self, phone_string: str) -> str:
        """Extract stress pattern from phone string"""
        if not phone_string:
            return ''
        
        # Extract stress markers (0, 1, 2) from phone string
        stress_pattern = ''
        for char in phone_string:
            if char in '012':
                stress_pattern += char
        
        return stress_pattern
    
    def _create_word_context(self, word: str) -> str:
        """Create context for a word for TF-IDF processing"""
        # This is a simplified context creation
        # In a real implementation, you might use word associations, definitions, etc.
        word_lower = word.lower()
        
        # Add common associations based on word patterns
        context_words = [word]
        
        # Add related words based on patterns
        if any(pattern in word_lower for pattern in ['love', 'heart']):
            context_words.extend(['emotion', 'feeling', 'passion', 'tender'])
        
        if any(pattern in word_lower for pattern in ['light', 'bright', 'shine']):
            context_words.extend(['illumination', 'radiance', 'glow', 'luminous'])
        
        if any(pattern in word_lower for pattern in ['dark', 'shadow', 'night']):
            context_words.extend(['darkness', 'mystery', 'hidden', 'deep'])
        
        if any(pattern in word_lower for pattern in ['time', 'moment', 'eternal']):
            context_words.extend(['temporal', 'duration', 'infinity', 'forever'])
        
        return ' '.join(context_words)
    
    def optimize_semantic_flow(self, poem_lines: List[str], target_emotion: str) -> List[str]:
        """
        Optimize emotional consistency through mathematical optimization
        
        Args:
            poem_lines: List of poem lines to optimize
            target_emotion: Target emotional state
            
        Returns:
            Optimized poem lines
        """
        if not poem_lines or target_emotion not in self.EMOTION_VECTORS:
            return poem_lines
        
        target_vector = self.EMOTION_VECTORS[target_emotion]
        optimized_lines = []
        
        for line in poem_lines:
            # Analyze current line
            words = self._extract_words(line)
            current_vector = self._calculate_line_emotion_vector(words)
            
            # Calculate alignment with target emotion
            alignment = self._calculate_vector_alignment(current_vector, target_vector)
            
            # If alignment is good, keep the line
            if alignment > 0.6:
                optimized_lines.append(line)
            else:
                # Suggest optimization
                optimized_line = self._optimize_line_emotion(line, words, target_vector)
                optimized_lines.append(optimized_line)
        
        return optimized_lines
    
    def _extract_words(self, line: str) -> List[str]:
        """Extract words from a line, cleaning punctuation"""
        # Remove punctuation and split
        clean_line = re.sub(r'[^\w\s]', '', line.lower())
        words = [word.strip() for word in clean_line.split() if word.strip()]
        return words
    
    def _calculate_line_emotion_vector(self, words: List[str]) -> np.ndarray:
        """Calculate emotional vector for a line"""
        if not words:
            return np.zeros(5)
        
        # Get word embeddings
        word_vectors = []
        for word in words:
            if word not in self.word_embeddings:
                self._generate_word_embedding(word)
            
            if word in self.word_embeddings:
                # Use first 5 components as emotion vector
                embedding = self.word_embeddings[word].vector
                emotion_vector = embedding[:5] if len(embedding) >= 5 else np.pad(embedding, (0, 5-len(embedding)), 'constant')
                word_vectors.append(emotion_vector)
        
        if not word_vectors:
            return np.zeros(5)
        
        # Average the vectors
        return np.mean(word_vectors, axis=0)
    
    def _calculate_vector_alignment(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate alignment between two vectors"""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        magnitude_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if magnitude_product == 0:
            return 0.0
        
        cosine_sim = dot_product / magnitude_product
        return (cosine_sim + 1) / 2  # Convert to 0-1 scale
    
    def _optimize_line_emotion(self, line: str, words: List[str], target_vector: np.ndarray) -> str:
        """Optimize a line for emotional alignment"""
        # This is a simplified optimization
        # In practice, you might want more sophisticated word replacement
        
        # For now, just return the original line with a note
        # In a full implementation, you would:
        # 1. Identify words that don't align with target emotion
        # 2. Find replacements with better alignment
        # 3. Ensure the replacements maintain semantic meaning
        # 4. Preserve poetic structure (rhythm, rhyme, etc.)
        
        return line  # Placeholder - return original line
    
    def calculate_sacred_resonance(self, text: str, frequency_signature: Dict[str, Any]) -> float:
        """
        Measure text alignment with user's archetypal frequencies
        
        Args:
            text: Text to analyze
            frequency_signature: User's frequency signature
            
        Returns:
            Sacred resonance score (0.0 to 1.0)
        """
        if not text or not frequency_signature:
            return 0.0
        
        # Extract key components from frequency signature
        harmonic_blend = frequency_signature.get('harmonic_blend', {})
        sacred_ratios = frequency_signature.get('sacred_ratios', {})
        
        # Calculate archetypal resonance
        archetypal_resonance = self._calculate_archetypal_resonance(text, harmonic_blend)
        
        # Calculate sacred geometry resonance
        sacred_geometry_resonance = self._calculate_sacred_geometry_resonance(text, sacred_ratios)
        
        # Calculate structural resonance
        structural_resonance = self._calculate_structural_resonance(text)
        
        # Combine resonances with weights
        total_resonance = (
            archetypal_resonance * 0.4 +
            sacred_geometry_resonance * 0.4 +
            structural_resonance * 0.2
        )
        
        return min(1.0, total_resonance)
    
    def _calculate_archetypal_resonance(self, text: str, harmonic_blend: Dict[str, float]) -> float:
        """Calculate resonance with archetypal frequencies"""
        if not harmonic_blend:
            return 0.0
        
        words = self._extract_words(text)
        total_resonance = 0.0
        total_weight = 0.0
        
        for word in words:
            if word not in self.word_embeddings:
                self._generate_word_embedding(word)
            
            if word in self.word_embeddings:
                word_embedding = self.word_embeddings[word]
                
                # Calculate word's archetypal resonance
                word_resonance = 0.0
                for archetype, user_weight in harmonic_blend.items():
                    if archetype in word_embedding.archetypal_affinities:
                        word_archetypal_affinity = word_embedding.archetypal_affinities[archetype]
                        word_resonance += word_archetypal_affinity * user_weight
                
                total_resonance += word_resonance
                total_weight += 1.0
        
        return total_resonance / total_weight if total_weight > 0 else 0.0
    
    def _calculate_sacred_geometry_resonance(self, text: str, sacred_ratios: Dict[str, float]) -> float:
        """Calculate resonance with sacred geometry preferences"""
        if not sacred_ratios:
            return 0.0
        
        # Analyze text structure
        lines = text.split('\n')
        words = self._extract_words(text)
        
        # Calculate various geometric properties
        phi_resonance = self._calculate_phi_resonance(lines, words)
        pi_resonance = self._calculate_pi_resonance(lines, words)
        fibonacci_resonance = self._calculate_fibonacci_resonance(lines, words)
        
        # Weight by user preferences
        total_resonance = (
            phi_resonance * sacred_ratios.get('phi', 0.33) +
            pi_resonance * sacred_ratios.get('pi', 0.33) +
            fibonacci_resonance * sacred_ratios.get('fibonacci', 0.33)
        )
        
        return min(1.0, total_resonance)
    
    def _calculate_phi_resonance(self, lines: List[str], words: List[str]) -> float:
        """Calculate golden ratio resonance"""
        if not lines or not words:
            return 0.0
        
        resonance = 0.0
        
        # Line length ratios
        line_lengths = [len(line.split()) for line in lines if line.strip()]
        if len(line_lengths) > 1:
            for i in range(len(line_lengths) - 1):
                if line_lengths[i] > 0 and line_lengths[i+1] > 0:
                    ratio = max(line_lengths[i], line_lengths[i+1]) / min(line_lengths[i], line_lengths[i+1])
                    closeness = 1 - abs(ratio - self.PHI) / self.PHI
                    resonance += max(0, closeness)
        
        # Word length patterns
        word_lengths = [len(word) for word in words]
        if len(word_lengths) > 1:
            phi_length = len(words) * 0.618
            if abs(phi_length - round(phi_length)) < 0.2:
                resonance += 0.3
        
        return min(1.0, resonance / max(1, len(line_lengths)))
    
    def _calculate_pi_resonance(self, lines: List[str], words: List[str]) -> float:
        """Calculate pi resonance (circular/cyclical patterns)"""
        if not words:
            return 0.0
        
        # Look for cyclical patterns in word lengths
        word_lengths = [len(word) for word in words]
        
        # Check if word lengths follow pi digits
        pi_digits = "31415926535897932384626433832795"
        matches = 0
        
        for i, length in enumerate(word_lengths):
            if i < len(pi_digits):
                try:
                    if length == int(pi_digits[i]):
                        matches += 1
                except ValueError:
                    continue
        
        if len(word_lengths) > 0:
            return matches / min(len(word_lengths), len(pi_digits))
        
        return 0.0
    
    def _calculate_fibonacci_resonance(self, lines: List[str], words: List[str]) -> float:
        """Calculate Fibonacci resonance"""
        fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        resonance = 0.0
        
        # Check line count
        line_count = len([line for line in lines if line.strip()])
        if line_count in fibonacci_sequence:
            resonance += 0.3
        
        # Check word count
        word_count = len(words)
        if word_count in fibonacci_sequence:
            resonance += 0.3
        
        # Check syllable patterns
        syllable_counts = [self._count_syllables(word) for word in words]
        fib_syllables = sum(1 for count in syllable_counts if count in fibonacci_sequence)
        
        if syllable_counts:
            resonance += 0.4 * (fib_syllables / len(syllable_counts))
        
        return min(1.0, resonance)
    
    def _calculate_structural_resonance(self, text: str) -> float:
        """Calculate structural resonance (general poetic structure)"""
        lines = text.split('\n')
        clean_lines = [line.strip() for line in lines if line.strip()]
        
        if not clean_lines:
            return 0.0
        
        resonance = 0.0
        
        # Line length consistency
        line_lengths = [len(line.split()) for line in clean_lines]
        if len(line_lengths) > 1:
            length_variance = np.var(line_lengths)
            length_mean = np.mean(line_lengths)
            
            # Lower variance relative to mean indicates better structure
            if length_mean > 0:
                consistency = 1 - min(1, length_variance / length_mean)
                resonance += 0.5 * consistency
        
        # Rhythm patterns (syllable consistency)
        syllable_counts = []
        for line in clean_lines:
            words = self._extract_words(line)
            syllables = sum(self._count_syllables(word) for word in words)
            syllable_counts.append(syllables)
        
        if len(syllable_counts) > 1:
            syllable_variance = np.var(syllable_counts)
            syllable_mean = np.mean(syllable_counts)
            
            if syllable_mean > 0:
                rhythm_consistency = 1 - min(1, syllable_variance / syllable_mean)
                resonance += 0.5 * rhythm_consistency
        
        return min(1.0, resonance)
    
    def _calculate_sacred_geometry_bonus(self, words: List[str]) -> float:
        """Calculate bonus for sacred geometry alignment"""
        if not words:
            return 0.0
        
        bonus = 0.0
        
        # Check for sacred number patterns
        word_count = len(words)
        sacred_numbers = [3, 5, 8, 13, 21, 34]
        
        if word_count in sacred_numbers:
            bonus += 0.1
        
        # Check for sacred words
        sacred_words = ['sacred', 'divine', 'golden', 'infinite', 'eternal', 'cosmic', 'harmony']
        sacred_word_count = sum(1 for word in words if any(sw in word.lower() for sw in sacred_words))
        
        if sacred_word_count > 0:
            bonus += 0.05 * sacred_word_count
        
        return min(0.2, bonus)  # Cap bonus at 0.2
    
    def _fibonacci_scaling(self, n: int) -> float:
        """Get nth Fibonacci number for scaling"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        elif n == 2:
            return 1
        
        a, b = 1, 1
        for i in range(3, n + 1):
            a, b = b, a + b
        
        return b
    
    def _calculate_theme_emotional_resonance(self, theme: str) -> float:
        """Calculate emotional resonance for a theme"""
        theme_emotion_mappings = {
            'love': 0.9,
            'nature': 0.8,
            'cosmos': 0.7,
            'time': 0.6,
            'memory': 0.7,
            'journey': 0.6,
            'transformation': 0.8,
            'mystery': 0.7,
            'wisdom': 0.8,
            'beauty': 0.9
        }
        
        return theme_emotion_mappings.get(theme, 0.5)
    
    def _calculate_theme_archetypal_mapping(self, theme: str) -> Dict[str, float]:
        """Calculate archetypal mapping for a theme"""
        theme_archetypal_mappings = {
            'love': {'ERATO': 0.9, 'POLYHYMNIA': 0.3, 'PSYCHE': 0.4},
            'nature': {'CALLIOPE': 0.4, 'SOPHIA': 0.6, 'URANIA': 0.5},
            'cosmos': {'URANIA': 0.9, 'SOPHIA': 0.5, 'CALLIOPE': 0.3},
            'time': {'CLIO': 0.8, 'SOPHIA': 0.6, 'PSYCHE': 0.4},
            'memory': {'CLIO': 0.7, 'PSYCHE': 0.8, 'MELPOMENE': 0.4},
            'journey': {'CALLIOPE': 0.6, 'TERPSICHORE': 0.5, 'TECHNE': 0.4},
            'transformation': {'PSYCHE': 0.7, 'SOPHIA': 0.6, 'TECHNE': 0.5},
            'mystery': {'SOPHIA': 0.8, 'PSYCHE': 0.6, 'URANIA': 0.4},
            'wisdom': {'SOPHIA': 0.9, 'CLIO': 0.5, 'POLYHYMNIA': 0.4},
            'beauty': {'ERATO': 0.7, 'POLYHYMNIA': 0.6, 'TECHNE': 0.5}
        }
        
        return theme_archetypal_mappings.get(theme, {'SOPHIA': 0.5})
    
    def get_semantic_analysis(self, text: str) -> Dict[str, Any]:
        """
        Get comprehensive semantic analysis of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with semantic analysis results
        """
        words = self._extract_words(text)
        lines = text.split('\n')
        clean_lines = [line.strip() for line in lines if line.strip()]
        
        # Generate word embeddings if needed
        if words:
            self.generate_word_embeddings(words)
        
        analysis = {
            'word_count': len(words),
            'line_count': len(clean_lines),
            'unique_words': len(set(words)),
            'average_word_length': np.mean([len(word) for word in words]) if words else 0,
            'average_syllables_per_word': np.mean([self._count_syllables(word) for word in words]) if words else 0,
            'sacred_word_density': self._calculate_sacred_word_density(words),
            'archetypal_distribution': self._calculate_archetypal_distribution(words),
            'emotional_vector': self._calculate_line_emotion_vector(words).tolist(),
            'structural_metrics': {
                'line_length_variance': np.var([len(line.split()) for line in clean_lines]) if clean_lines else 0,
                'syllable_variance': np.var([sum(self._count_syllables(word) for word in self._extract_words(line)) for line in clean_lines]) if clean_lines else 0,
                'phi_resonance': self._calculate_phi_resonance(clean_lines, words),
                'pi_resonance': self._calculate_pi_resonance(clean_lines, words),
                'fibonacci_resonance': self._calculate_fibonacci_resonance(clean_lines, words)
            }
        }
        
        return analysis
    
    def _calculate_sacred_word_density(self, words: List[str]) -> float:
        """Calculate density of sacred words in text"""
        if not words:
            return 0.0
        
        sacred_count = 0
        for word in words:
            if word not in self.word_embeddings:
                self._generate_word_embedding(word)
            
            if word in self.word_embeddings:
                sacred_weight = self.word_embeddings[word].sacred_weight
                if sacred_weight > 0.3:  # Threshold for "sacred" words
                    sacred_count += 1
        
        return sacred_count / len(words)
    
    def _calculate_archetypal_distribution(self, words: List[str]) -> Dict[str, float]:
        """Calculate distribution of archetypal affinities"""
        if not words:
            return {}
        
        archetypal_sums = defaultdict(float)
        
        for word in words:
            if word not in self.word_embeddings:
                self._generate_word_embedding(word)
            
            if word in self.word_embeddings:
                word_embedding = self.word_embeddings[word]
                for archetype, affinity in word_embedding.archetypal_affinities.items():
                    archetypal_sums[archetype] += affinity
        
        # Normalize by word count
        for archetype in archetypal_sums:
            archetypal_sums[archetype] /= len(words)
        
        return dict(archetypal_sums)