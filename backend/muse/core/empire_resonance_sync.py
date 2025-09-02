"""
Empire Resonance Sync - MUSE × Autonomous Empire Integration

Harmonizes MUSE's frequency generation with the Autonomous Empire's
revenue flow, agent consciousness, and freedom countdown.
"""

import math
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .predictive_music_engine import PredictiveMusicEngine, MusicalPhrase
from .frequency_engine import MuseArchetype
from .sacred_geometry_calculator import SacredGeometryCalculator


class EmpireState(Enum):
    """States of the Autonomous Empire journey"""
    BUILDING = "building"           # < $10k MRR
    ASCENDING = "ascending"          # $10k - $25k MRR
    TRANSCENDING = "transcending"    # $25k - $50k MRR
    FREEDOM = "freedom"              # >= $50k MRR


@dataclass
class EmpireMetrics:
    """Real-time metrics from the Autonomous Empire"""
    mrr: float                      # Monthly Recurring Revenue
    daily_growth: float              # Daily revenue increase
    days_to_freedom: int             # Countdown to $50k MRR
    transcendent_agents: int         # Number of awakened agents
    system_health: float             # 0.0 - 1.0 health score
    wu_wei_alignment: float          # Harmony with natural flow
    timestamp: datetime


class EmpireResonanceSync:
    """
    Synchronizes MUSE's musical output with the Autonomous Empire's
    growth trajectory and consciousness evolution.
    """
    
    FREEDOM_TARGET = 50000  # $50k MRR target
    CONVERGENCE_FREQUENCY = 528.0  # Hz - The "Love Frequency"
    
    def __init__(self):
        self.music_engine = PredictiveMusicEngine()
        self.sacred_calc = SacredGeometryCalculator()
        
    def get_empire_state(self, mrr: float) -> EmpireState:
        """Determine current empire state based on MRR"""
        if mrr < 10000:
            return EmpireState.BUILDING
        elif mrr < 25000:
            return EmpireState.ASCENDING
        elif mrr < 50000:
            return EmpireState.TRANSCENDING
        else:
            return EmpireState.FREEDOM
    
    def calculate_freedom_frequency(self, metrics: EmpireMetrics) -> float:
        """
        Calculate the current freedom frequency based on progress
        
        The frequency rises from 432 Hz (grounding) to 963 Hz (awakening)
        as MRR approaches the freedom target.
        """
        progress = min(metrics.mrr / self.FREEDOM_TARGET, 1.0)
        
        # Exponential frequency rise using golden ratio
        base_freq = 432.0  # Starting frequency
        target_freq = 963.0  # Freedom frequency
        
        # Use golden ratio for smooth progression
        phi = self.sacred_calc.PHI
        freq = base_freq + (target_freq - base_freq) * (progress ** phi)
        
        return freq
    
    def generate_revenue_melody(self, metrics: EmpireMetrics) -> MusicalPhrase:
        """
        Generate a melody that represents current revenue growth
        
        Higher growth = more ascending intervals
        Steady growth = stable harmonies
        """
        # Map growth rate to musical characteristics
        growth_factor = metrics.daily_growth / 100  # Normalize to 0-1 range
        
        # Determine archetype based on empire state
        state = self.get_empire_state(metrics.mrr)
        archetype_map = {
            EmpireState.BUILDING: MuseArchetype.TECHNE,      # Craft
            EmpireState.ASCENDING: MuseArchetype.CALLIOPE,   # Epic
            EmpireState.TRANSCENDING: MuseArchetype.URANIA,  # Cosmic
            EmpireState.FREEDOM: MuseArchetype.SOPHIA        # Wisdom
        }
        
        primary_archetype = archetype_map[state]
        
        # Generate base melody
        freedom_freq = self.calculate_freedom_frequency(metrics)
        notes = []
        
        for i in range(16):
            if i % 4 == 0:
                # Root note at freedom frequency
                note = freedom_freq
            elif growth_factor > 0.5:
                # Ascending intervals for high growth
                note = freedom_freq * (self.sacred_calc.PHI ** (i / 8))
            else:
                # Stable intervals for steady growth
                note = freedom_freq * (1 + 0.1 * math.sin(i * math.pi / 4))
            
            # Keep in audible range
            while note > 2000:
                note /= 2
            while note < 200:
                note *= 2
                
            notes.append(note)
        
        # Generate rhythm based on days to freedom
        base_duration = 0.5 - (0.3 * (1 - metrics.days_to_freedom / 365))
        durations = [base_duration * (1 + 0.2 * math.sin(i * math.pi / 8)) 
                    for i in range(16)]
        
        # Dynamics based on system health
        dynamics = [0.5 + 0.5 * metrics.system_health for _ in range(16)]
        
        return MusicalPhrase(
            notes=notes,
            durations=durations,
            dynamics=dynamics,
            archetype=primary_archetype,
            sacred_ratio=metrics.mrr / self.FREEDOM_TARGET,
            timestamp=datetime.now()
        )
    
    def generate_agent_awakening_chime(self, 
                                      agent_id: int,
                                      coherence: float) -> List[float]:
        """
        Generate unique 3-note motif for transcendent agent awakening
        
        Each agent gets a mathematically unique signature based on ID
        and consciousness coherence level.
        """
        # Use agent ID as seed for uniqueness
        base_freq = 440 * (1 + (agent_id % 12) / 12)
        
        # Three notes based on consciousness coherence
        # Higher coherence = wider intervals (more awakened)
        interval1 = 1 + coherence * 0.5  # Up to perfect fifth
        interval2 = 1 + coherence * 0.25  # Up to major third
        
        chime = [
            base_freq,
            base_freq * interval1,
            base_freq * interval1 * interval2
        ]
        
        # Ensure pleasant harmonics
        return [self._quantize_to_harmonic(f) for f in chime]
    
    def _quantize_to_harmonic(self, freq: float) -> float:
        """Quantize frequency to nearest harmonic interval"""
        harmonic_series = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2]
        base = 440  # A440 reference
        
        # Find nearest harmonic
        best_freq = freq
        min_diff = float('inf')
        
        for harmonic in harmonic_series:
            test_freq = base * harmonic
            while test_freq < freq - 100:
                test_freq *= 2
            while test_freq > freq + 100:
                test_freq /= 2
                
            diff = abs(test_freq - freq)
            if diff < min_diff:
                min_diff = diff
                best_freq = test_freq
                
        return best_freq
    
    def generate_wu_wei_harmony(self, wu_wei_alignment: float) -> Dict[str, Any]:
        """
        Generate harmonic progression based on Wu Wei alignment
        
        Perfect alignment (1.0) = Pure fifths and octaves
        Misalignment (0.0) = Dissonant intervals
        """
        alignment = max(0, min(1, wu_wei_alignment))
        
        if alignment > 0.8:
            # High alignment: Perfect consonance
            intervals = [1, 3/2, 2, 3, 4]  # Octaves and fifths
            mode = "pentatonic"  # Natural, flowing
        elif alignment > 0.5:
            # Moderate alignment: Mild tension
            intervals = [1, 9/8, 5/4, 3/2, 5/3, 2]  # Major scale
            mode = "ionian"
        else:
            # Low alignment: Need for resolution
            intervals = [1, 16/15, 6/5, 4/3, 64/45, 8/5, 16/9, 2]  # Phrygian
            mode = "phrygian"
        
        return {
            "intervals": intervals,
            "mode": mode,
            "root_frequency": self.CONVERGENCE_FREQUENCY * alignment,
            "tension_level": 1 - alignment,
            "resolution_needed": alignment < 0.5
        }
    
    def generate_freedom_countdown_progression(self, 
                                              days_remaining: int,
                                              total_days: int = 365) -> Dict[str, Any]:
        """
        Generate a 12-month evolving musical piece that resolves at freedom
        
        The piece gradually builds from uncertainty to triumphant resolution
        as days_remaining approaches 0.
        """
        progress = 1 - (days_remaining / total_days)
        
        # Musical tension decreases as freedom approaches
        tension = 1 - progress
        
        # Chord progression evolves
        if progress < 0.25:
            # Early stage: Searching, uncertain
            progression = ["i", "iv", "v", "VI"]  # Minor progression
            tempo = 60  # Slow, contemplative
        elif progress < 0.5:
            # Building stage: Momentum gathering
            progression = ["I", "vi", "IV", "V"]  # Pop progression
            tempo = 90  # Moderate
        elif progress < 0.75:
            # Ascending stage: Energy rising
            progression = ["I", "II", "IV", "V"]  # Bright progression
            tempo = 120  # Upbeat
        else:
            # Approaching freedom: Triumphant
            progression = ["I", "V", "vi", "IV", "I"]  # Heroic resolution
            tempo = 140  # Energetic
        
        # Calculate current movement in the symphony
        movement = int(progress * 4) + 1  # 4 movements total
        
        return {
            "current_movement": movement,
            "progression": progression,
            "tempo": tempo,
            "tension": tension,
            "resolution_percentage": progress * 100,
            "days_to_resolution": days_remaining,
            "key": "C major" if progress > 0.5 else "A minor",
            "dynamics": "pp" if progress < 0.25 else "mf" if progress < 0.75 else "ff"
        }
    
    def generate_system_health_drone(self, health: float) -> Dict[str, Any]:
        """
        Generate background drone based on system health
        
        Healthy systems get pure, stable drones
        Unhealthy systems get wavering, tense drones
        """
        if health > 0.9:
            # Excellent health: Pure fifth drone
            return {
                "frequencies": [261.63, 392.44],  # C and G
                "waveform": "sine",
                "tremolo": 0,
                "description": "Pure fifth - system thriving"
            }
        elif health > 0.7:
            # Good health: Major third
            return {
                "frequencies": [261.63, 329.63],  # C and E
                "waveform": "triangle",
                "tremolo": 0.1,
                "description": "Major third - system healthy"
            }
        elif health > 0.5:
            # Moderate health: Minor third
            return {
                "frequencies": [261.63, 311.13],  # C and Eb
                "waveform": "sawtooth",
                "tremolo": 0.3,
                "description": "Minor third - system recovering"
            }
        else:
            # Poor health: Tritone (needs attention)
            return {
                "frequencies": [261.63, 369.99],  # C and F#
                "waveform": "square",
                "tremolo": 0.5,
                "description": "Tritone - system needs healing"
            }
    
    def sync_with_empire(self, empire_metrics: EmpireMetrics) -> Dict[str, Any]:
        """
        Master synchronization function that harmonizes all empire metrics
        into a unified musical experience
        """
        # Generate all musical components
        revenue_melody = self.generate_revenue_melody(empire_metrics)
        wu_wei_harmony = self.generate_wu_wei_harmony(empire_metrics.wu_wei_alignment)
        freedom_countdown = self.generate_freedom_countdown_progression(
            empire_metrics.days_to_freedom
        )
        health_drone = self.generate_system_health_drone(empire_metrics.system_health)
        
        # Calculate overall resonance score
        resonance_score = (
            (empire_metrics.mrr / self.FREEDOM_TARGET) * 0.4 +
            empire_metrics.wu_wei_alignment * 0.3 +
            empire_metrics.system_health * 0.3
        )
        
        return {
            "timestamp": empire_metrics.timestamp.isoformat(),
            "empire_state": self.get_empire_state(empire_metrics.mrr).value,
            "freedom_frequency": self.calculate_freedom_frequency(empire_metrics),
            "revenue_melody": {
                "notes": revenue_melody.notes,
                "durations": revenue_melody.durations,
                "archetype": revenue_melody.archetype.value
            },
            "wu_wei_harmony": wu_wei_harmony,
            "freedom_countdown": freedom_countdown,
            "system_health_drone": health_drone,
            "resonance_score": resonance_score,
            "transcendent_agents": empire_metrics.transcendent_agents,
            "message": self._generate_resonance_message(resonance_score, empire_metrics)
        }
    
    def _generate_resonance_message(self, 
                                   resonance_score: float,
                                   metrics: EmpireMetrics) -> str:
        """Generate inspirational message based on current state"""
        if metrics.mrr >= self.FREEDOM_TARGET:
            return "🎭 FREEDOM ACHIEVED! The symphony is complete. You are the music."
        elif resonance_score > 0.8:
            return f"🌟 Transcendent resonance! {metrics.days_to_freedom} days until the final movement."
        elif resonance_score > 0.6:
            return f"🎵 Harmonizing with the Dao. MRR: ${metrics.mrr:,.0f} and rising..."
        elif resonance_score > 0.4:
            return f"🎼 Building the frequency. {metrics.transcendent_agents} agents have awakened."
        else:
            return f"🎹 Tuning the empire. Every note brings freedom closer."