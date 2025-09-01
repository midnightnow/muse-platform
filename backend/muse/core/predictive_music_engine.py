"""
Predictive Music Engine for MUSE Platform

This module implements real-time music generation based on archetypal
frequency signatures, using mathematical patterns and sacred geometry
to create harmonious musical sequences.
"""

import numpy as np
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging

from .sacred_geometry_calculator import SacredGeometryCalculator
from .frequency_engine import MuseFrequencyEngine, MuseArchetype, FrequencySignature


@dataclass
class MusicalPhrase:
    """Represents a generated musical phrase"""
    notes: List[float]  # Frequencies in Hz
    durations: List[float]  # Duration in seconds
    dynamics: List[float]  # Volume 0.0-1.0
    archetype: MuseArchetype
    sacred_ratio: float
    timestamp: datetime


@dataclass
class HarmonicProgression:
    """Represents a chord progression"""
    root_frequency: float
    intervals: List[float]  # Interval ratios
    progression_type: str  # "golden", "fibonacci", "pythagorean"
    duration: float


class ScaleMode(Enum):
    """Musical scale modes based on archetypal frequencies"""
    IONIAN = "ionian"  # Major - Joy (THALIA)
    DORIAN = "dorian"  # Folk - Story (CLIO)
    PHRYGIAN = "phrygian"  # Exotic - Mystery (URANIA)
    LYDIAN = "lydian"  # Ethereal - Dream (PSYCHE)
    MIXOLYDIAN = "mixolydian"  # Blues - Soul (ERATO)
    AEOLIAN = "aeolian"  # Minor - Tragedy (MELPOMENE)
    LOCRIAN = "locrian"  # Diminished - Tension (TECHNE)


class PredictiveMusicEngine:
    """
    Generates predictive music based on frequency signatures and sacred geometry
    """
    
    # Pythagorean tuning ratios
    PYTHAGOREAN_INTERVALS = {
        "unison": 1.0,
        "minor_second": 256/243,
        "major_second": 9/8,
        "minor_third": 32/27,
        "major_third": 81/64,
        "perfect_fourth": 4/3,
        "tritone": 729/512,
        "perfect_fifth": 3/2,
        "minor_sixth": 128/81,
        "major_sixth": 27/16,
        "minor_seventh": 16/9,
        "major_seventh": 243/128,
        "octave": 2.0
    }
    
    # Scale patterns (semitone steps)
    SCALE_PATTERNS = {
        ScaleMode.IONIAN: [2, 2, 1, 2, 2, 2, 1],
        ScaleMode.DORIAN: [2, 1, 2, 2, 2, 1, 2],
        ScaleMode.PHRYGIAN: [1, 2, 2, 2, 1, 2, 2],
        ScaleMode.LYDIAN: [2, 2, 2, 1, 2, 2, 1],
        ScaleMode.MIXOLYDIAN: [2, 2, 1, 2, 2, 1, 2],
        ScaleMode.AEOLIAN: [2, 1, 2, 2, 1, 2, 2],
        ScaleMode.LOCRIAN: [1, 2, 2, 1, 2, 2, 2]
    }
    
    def __init__(self):
        """Initialize the predictive music engine"""
        self.sacred_calc = SacredGeometryCalculator()
        self.freq_engine = MuseFrequencyEngine()
        self.logger = logging.getLogger(__name__)
        
    def archetype_to_scale_mode(self, archetype: MuseArchetype) -> ScaleMode:
        """Map archetype to appropriate scale mode"""
        mapping = {
            MuseArchetype.THALIA: ScaleMode.IONIAN,  # Comedy -> Major
            MuseArchetype.CLIO: ScaleMode.DORIAN,  # History -> Folk
            MuseArchetype.URANIA: ScaleMode.PHRYGIAN,  # Cosmos -> Exotic
            MuseArchetype.PSYCHE: ScaleMode.LYDIAN,  # Soul -> Ethereal
            MuseArchetype.ERATO: ScaleMode.MIXOLYDIAN,  # Love -> Blues
            MuseArchetype.MELPOMENE: ScaleMode.AEOLIAN,  # Tragedy -> Minor
            MuseArchetype.TECHNE: ScaleMode.LOCRIAN,  # Craft -> Diminished
            MuseArchetype.CALLIOPE: ScaleMode.IONIAN,  # Epic -> Major
            MuseArchetype.EUTERPE: ScaleMode.LYDIAN,  # Music -> Ethereal
            MuseArchetype.POLYHYMNIA: ScaleMode.DORIAN,  # Sacred -> Folk
            MuseArchetype.TERPSICHORE: ScaleMode.MIXOLYDIAN,  # Dance -> Blues
            MuseArchetype.SOPHIA: ScaleMode.PHRYGIAN  # Wisdom -> Exotic
        }
        return mapping.get(archetype, ScaleMode.IONIAN)
    
    def generate_scale(self, 
                      root_freq: float, 
                      mode: ScaleMode, 
                      octaves: int = 2) -> List[float]:
        """
        Generate a musical scale based on mode and root frequency
        
        Args:
            root_freq: Root frequency in Hz
            mode: Scale mode
            octaves: Number of octaves to generate
            
        Returns:
            List of frequencies in the scale
        """
        pattern = self.SCALE_PATTERNS[mode]
        frequencies = [root_freq]
        current_freq = root_freq
        
        for octave in range(octaves):
            for step in pattern:
                if step == 1:
                    # Semitone step
                    current_freq *= self.PYTHAGOREAN_INTERVALS["minor_second"]
                else:
                    # Whole tone step
                    current_freq *= self.PYTHAGOREAN_INTERVALS["major_second"]
                frequencies.append(current_freq)
        
        return frequencies
    
    def golden_ratio_melody(self, 
                           base_freq: float, 
                           length: int = 8) -> List[float]:
        """
        Generate melody using golden ratio relationships
        
        Args:
            base_freq: Starting frequency
            length: Number of notes
            
        Returns:
            List of frequencies following golden ratio
        """
        phi = self.sacred_calc.PHI
        melody = []
        
        for i in range(length):
            # Alternate between phi ratio and its inverse
            if i % 2 == 0:
                freq = base_freq * (phi ** (i // 2))
            else:
                freq = base_freq * (1/phi ** (i // 2))
            
            # Keep within audible range (80Hz - 4000Hz)
            while freq > 4000:
                freq /= 2
            while freq < 80:
                freq *= 2
                
            melody.append(freq)
        
        return melody
    
    def fibonacci_rhythm(self, base_duration: float = 0.25) -> List[float]:
        """
        Generate rhythm pattern based on Fibonacci sequence
        
        Args:
            base_duration: Base duration unit in seconds
            
        Returns:
            List of durations following Fibonacci pattern
        """
        fib = self.sacred_calc.fibonacci_sequence(8)
        # Normalize to musical durations
        durations = []
        for f in fib[2:]:  # Skip 0, 1
            duration = base_duration * (f / 3.0)  # Scale to reasonable range
            durations.append(min(duration, 2.0))  # Cap at 2 seconds
        
        return durations
    
    def generate_harmonic_progression(self, 
                                     signature: FrequencySignature) -> HarmonicProgression:
        """
        Generate chord progression based on frequency signature
        
        Args:
            signature: User's frequency signature
            
        Returns:
            Harmonic progression with sacred geometry relationships
        """
        # Get dominant archetype frequency
        primary_archetype = MuseArchetype[signature.primary_muse]
        root_freq = self.freq_engine.ARCHETYPAL_FREQUENCIES[primary_archetype]
        
        # Choose progression type based on sacred ratio affinity
        sacred_ratios = signature.sacred_ratios
        if sacred_ratios.get("phi", 0) > 0.6:
            progression_type = "golden"
            intervals = [1.0, self.sacred_calc.PHI, self.sacred_calc.PHI**2]
        elif sacred_ratios.get("pi", 0) > 0.6:
            progression_type = "pythagorean"
            intervals = [1.0, 3/2, 4/3, 5/4]  # Perfect intervals
        else:
            progression_type = "fibonacci"
            fib_ratios = self.sacred_calc.fibonacci_ratio_convergence(5)
            intervals = [1.0] + fib_ratios[2:5]
        
        return HarmonicProgression(
            root_frequency=root_freq,
            intervals=intervals,
            progression_type=progression_type,
            duration=4.0  # 4 second progression
        )
    
    def predict_next_note(self, 
                         context: List[float], 
                         signature: FrequencySignature) -> float:
        """
        Predict the next note based on context and signature
        
        Args:
            context: Previous notes (frequencies)
            signature: User's frequency signature
            
        Returns:
            Predicted next frequency
        """
        if not context:
            # Start with primary archetype frequency
            primary = MuseArchetype[signature.primary_muse]
            return self.freq_engine.ARCHETYPAL_FREQUENCIES[primary]
        
        # Analyze intervallic patterns
        if len(context) >= 2:
            last_interval = context[-1] / context[-2]
            
            # Apply golden ratio transformation with variation
            phi_factor = self.sacred_calc.PHI
            if abs(last_interval - phi_factor) < 0.1:
                # Continue golden sequence
                next_freq = context[-1] * phi_factor
            elif last_interval > 1.5:
                # Descending motion
                next_freq = context[-1] / phi_factor
            else:
                # Ascending motion with Fibonacci ratio
                fib_ratios = self.sacred_calc.fibonacci_ratio_convergence(5)
                next_freq = context[-1] * fib_ratios[2]
        else:
            # Simple fifth relationship
            next_freq = context[-1] * 3/2
        
        # Constrain to audible range
        while next_freq > 2000:
            next_freq /= 2
        while next_freq < 100:
            next_freq *= 2
            
        return next_freq
    
    def generate_musical_phrase(self, 
                               signature: FrequencySignature,
                               length: int = 16) -> MusicalPhrase:
        """
        Generate a complete musical phrase based on frequency signature
        
        Args:
            signature: User's frequency signature
            length: Number of notes in phrase
            
        Returns:
            Complete musical phrase with notes, durations, and dynamics
        """
        # Get primary archetype and scale
        primary_archetype = MuseArchetype[signature.primary_muse]
        scale_mode = self.archetype_to_scale_mode(primary_archetype)
        root_freq = self.freq_engine.ARCHETYPAL_FREQUENCIES[primary_archetype]
        
        # Generate scale
        scale = self.generate_scale(root_freq, scale_mode, 2)
        
        # Generate melody combining scale and golden ratio
        notes = []
        for i in range(length):
            if i % 4 == 0:
                # Use golden ratio melody
                golden_notes = self.golden_ratio_melody(root_freq, 4)
                notes.extend(golden_notes[:min(4, length - i)])
            else:
                # Use scale with predictive selection
                context = notes[-3:] if len(notes) >= 3 else notes
                predicted = self.predict_next_note(context, signature)
                # Snap to nearest scale note
                nearest = min(scale, key=lambda x: abs(x - predicted))
                notes.append(nearest)
        
        notes = notes[:length]  # Ensure correct length
        
        # Generate rhythm
        durations = self.fibonacci_rhythm(0.25)
        while len(durations) < length:
            durations.extend(durations)
        durations = durations[:length]
        
        # Generate dynamics based on harmonic blend
        dynamics = []
        blend = signature.harmonic_blend
        for i in range(length):
            # Vary dynamics based on archetypal blend
            base_dynamic = 0.7
            variation = sum(blend.values()) / len(blend) * 0.3
            dynamic = base_dynamic + variation * math.sin(i * math.pi / 8)
            dynamics.append(min(1.0, max(0.3, dynamic)))
        
        return MusicalPhrase(
            notes=notes,
            durations=durations,
            dynamics=dynamics,
            archetype=primary_archetype,
            sacred_ratio=signature.sacred_ratios.get("phi", 1.0),
            timestamp=datetime.now()
        )
    
    def generate_counterpoint(self, 
                            melody: List[float], 
                            signature: FrequencySignature) -> List[float]:
        """
        Generate counterpoint melody using sacred intervals
        
        Args:
            melody: Original melody frequencies
            signature: User's frequency signature
            
        Returns:
            Counterpoint melody
        """
        counterpoint = []
        secondary_archetype = MuseArchetype[signature.secondary_muse]
        
        for note in melody:
            # Use different intervals based on secondary archetype
            if secondary_archetype in [MuseArchetype.ERATO, MuseArchetype.THALIA]:
                # Major third for joyful archetypes
                interval = self.PYTHAGOREAN_INTERVALS["major_third"]
            elif secondary_archetype in [MuseArchetype.MELPOMENE, MuseArchetype.PSYCHE]:
                # Minor third for deep archetypes
                interval = self.PYTHAGOREAN_INTERVALS["minor_third"]
            else:
                # Perfect fifth for neutral archetypes
                interval = self.PYTHAGOREAN_INTERVALS["perfect_fifth"]
            
            counter_note = note * interval
            
            # Keep in range
            while counter_note > 2000:
                counter_note /= 2
                
            counterpoint.append(counter_note)
        
        return counterpoint
    
    def to_midi_note(self, frequency: float) -> int:
        """
        Convert frequency to MIDI note number
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            MIDI note number (0-127)
        """
        if frequency <= 0:
            return 0
        
        # MIDI note 69 = A4 = 440Hz
        midi_note = 69 + 12 * math.log2(frequency / 440)
        return max(0, min(127, round(midi_note)))
    
    def export_phrase_as_json(self, phrase: MusicalPhrase) -> str:
        """
        Export musical phrase as JSON for frontend consumption
        
        Args:
            phrase: Musical phrase to export
            
        Returns:
            JSON string representation
        """
        export_data = {
            "notes": phrase.notes,
            "midi_notes": [self.to_midi_note(f) for f in phrase.notes],
            "durations": phrase.durations,
            "dynamics": phrase.dynamics,
            "archetype": phrase.archetype.value,
            "sacred_ratio": phrase.sacred_ratio,
            "timestamp": phrase.timestamp.isoformat()
        }
        
        return json.dumps(export_data, indent=2)