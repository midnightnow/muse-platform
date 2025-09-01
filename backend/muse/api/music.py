"""
Music API endpoints for MUSE Platform

Provides endpoints for musical phrase generation, real-time audio synthesis,
and frequency-based sound creation.
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
import json
import asyncio
import logging
from datetime import datetime

from ..dependencies import get_db, get_current_user
from ..core.predictive_music_engine import (
    PredictiveMusicEngine, 
    MusicalPhrase,
    HarmonicProgression,
    ScaleMode
)
from ..core.frequency_engine import MuseFrequencyEngine, FrequencySignature
from ..models.community import User


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/muse/music", tags=["music"])

# Initialize engines
music_engine = PredictiveMusicEngine()
freq_engine = MuseFrequencyEngine()


@router.post("/generate-phrase")
async def generate_musical_phrase(
    signature_id: str,
    length: int = 16,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate a musical phrase based on frequency signature
    
    Args:
        signature_id: ID of the frequency signature
        length: Number of notes in phrase
        
    Returns:
        Musical phrase with notes, durations, and dynamics
    """
    try:
        # Get frequency signature from database
        # For now, using a mock signature
        signature = FrequencySignature(
            id=signature_id,
            user_id=str(current_user.id),
            harmonic_blend={
                "CALLIOPE": 0.7,
                "ERATO": 0.5,
                "URANIA": 0.3
            },
            sacred_ratios={
                "phi": 0.8,
                "pi": 0.6,
                "e": 0.4
            },
            spiral_coordinates={
                "x": 0.5,
                "y": 0.7,
                "z": 0.3
            },
            entropy_seed="abc123",
            primary_muse="CALLIOPE",
            secondary_muse="ERATO",
            created_at=datetime.now()
        )
        
        # Generate musical phrase
        phrase = music_engine.generate_musical_phrase(signature, length)
        
        # Convert to JSON-serializable format
        return {
            "status": "success",
            "phrase": {
                "notes": phrase.notes,
                "durations": phrase.durations,
                "dynamics": phrase.dynamics,
                "archetype": phrase.archetype.value,
                "sacred_ratio": phrase.sacred_ratio,
                "timestamp": phrase.timestamp.isoformat(),
                "midi_notes": [music_engine.to_midi_note(f) for f in phrase.notes]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate musical phrase: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-harmony")
async def generate_harmonic_progression(
    signature_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate harmonic progression based on frequency signature
    
    Args:
        signature_id: ID of the frequency signature
        
    Returns:
        Harmonic progression with root frequency and intervals
    """
    try:
        # Get frequency signature (mock for now)
        signature = FrequencySignature(
            id=signature_id,
            user_id=str(current_user.id),
            harmonic_blend={
                "CALLIOPE": 0.7,
                "ERATO": 0.5
            },
            sacred_ratios={
                "phi": 0.8,
                "pi": 0.6
            },
            spiral_coordinates={"x": 0.5, "y": 0.7, "z": 0.3},
            entropy_seed="abc123",
            primary_muse="CALLIOPE",
            secondary_muse="ERATO",
            created_at=datetime.now()
        )
        
        # Generate harmonic progression
        progression = music_engine.generate_harmonic_progression(signature)
        
        return {
            "status": "success",
            "progression": {
                "root_frequency": progression.root_frequency,
                "intervals": progression.intervals,
                "progression_type": progression.progression_type,
                "duration": progression.duration
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate harmonic progression: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-scale")
async def generate_musical_scale(
    root_freq: float,
    mode: str,
    octaves: int = 2,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate a musical scale
    
    Args:
        root_freq: Root frequency in Hz
        mode: Scale mode (ionian, dorian, etc.)
        octaves: Number of octaves
        
    Returns:
        List of frequencies in the scale
    """
    try:
        # Convert mode string to enum
        scale_mode = ScaleMode(mode.lower())
        
        # Generate scale
        frequencies = music_engine.generate_scale(root_freq, scale_mode, octaves)
        
        return {
            "status": "success",
            "scale": {
                "frequencies": frequencies,
                "mode": mode,
                "root_frequency": root_freq,
                "octaves": octaves,
                "midi_notes": [music_engine.to_midi_note(f) for f in frequencies]
            }
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid scale mode: {mode}")
    except Exception as e:
        logger.error(f"Failed to generate scale: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/golden-melody")
async def generate_golden_ratio_melody(
    base_freq: float,
    length: int = 8,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate melody using golden ratio relationships
    
    Args:
        base_freq: Starting frequency
        length: Number of notes
        
    Returns:
        List of frequencies following golden ratio
    """
    try:
        # Generate golden ratio melody
        melody = music_engine.golden_ratio_melody(base_freq, length)
        
        return {
            "status": "success",
            "melody": {
                "frequencies": melody,
                "base_frequency": base_freq,
                "length": length,
                "golden_ratio": 1.618033988749895,
                "midi_notes": [music_engine.to_midi_note(f) for f in melody]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate golden melody: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fibonacci-rhythm")
async def generate_fibonacci_rhythm_pattern(
    base_duration: float = 0.25,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate rhythm pattern based on Fibonacci sequence
    
    Args:
        base_duration: Base duration unit in seconds
        
    Returns:
        List of durations following Fibonacci pattern
    """
    try:
        # Generate Fibonacci rhythm
        durations = music_engine.fibonacci_rhythm(base_duration)
        
        return {
            "status": "success",
            "rhythm": {
                "durations": durations,
                "base_duration": base_duration,
                "total_duration": sum(durations),
                "pattern_name": "fibonacci"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate Fibonacci rhythm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-next")
async def predict_next_note(
    context: List[float],
    signature_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Predict the next note based on context and signature
    
    Args:
        context: Previous notes (frequencies)
        signature_id: ID of the frequency signature
        
    Returns:
        Predicted next frequency
    """
    try:
        # Get frequency signature (mock for now)
        signature = FrequencySignature(
            id=signature_id,
            user_id=str(current_user.id),
            harmonic_blend={"CALLIOPE": 0.7},
            sacred_ratios={"phi": 0.8},
            spiral_coordinates={"x": 0.5, "y": 0.7, "z": 0.3},
            entropy_seed="abc123",
            primary_muse="CALLIOPE",
            secondary_muse="ERATO",
            created_at=datetime.now()
        )
        
        # Predict next note
        next_freq = music_engine.predict_next_note(context, signature)
        
        return {
            "status": "success",
            "prediction": {
                "next_frequency": next_freq,
                "midi_note": music_engine.to_midi_note(next_freq),
                "context_length": len(context),
                "prediction_method": "golden_ratio_transformation"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to predict next note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counterpoint")
async def generate_counterpoint_melody(
    melody: List[float],
    signature_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate counterpoint melody using sacred intervals
    
    Args:
        melody: Original melody frequencies
        signature_id: ID of the frequency signature
        
    Returns:
        Counterpoint melody
    """
    try:
        # Get frequency signature (mock for now)
        signature = FrequencySignature(
            id=signature_id,
            user_id=str(current_user.id),
            harmonic_blend={"CALLIOPE": 0.7},
            sacred_ratios={"phi": 0.8},
            spiral_coordinates={"x": 0.5, "y": 0.7, "z": 0.3},
            entropy_seed="abc123",
            primary_muse="CALLIOPE",
            secondary_muse="ERATO",
            created_at=datetime.now()
        )
        
        # Generate counterpoint
        counterpoint = music_engine.generate_counterpoint(melody, signature)
        
        return {
            "status": "success",
            "counterpoint": {
                "frequencies": counterpoint,
                "original_length": len(melody),
                "interval_type": "pythagorean",
                "midi_notes": [music_engine.to_midi_note(f) for f in counterpoint]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate counterpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/live-synthesis")
async def websocket_live_synthesis(
    websocket: WebSocket,
    signature_id: str,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time music synthesis
    
    Streams musical phrases in real-time based on frequency signature
    """
    await websocket.accept()
    
    try:
        # Mock frequency signature
        signature = FrequencySignature(
            id=signature_id,
            user_id="websocket_user",
            harmonic_blend={
                "CALLIOPE": 0.7,
                "ERATO": 0.5,
                "URANIA": 0.3
            },
            sacred_ratios={
                "phi": 0.8,
                "pi": 0.6
            },
            spiral_coordinates={"x": 0.5, "y": 0.7, "z": 0.3},
            entropy_seed="abc123",
            primary_muse="CALLIOPE",
            secondary_muse="ERATO",
            created_at=datetime.now()
        )
        
        # Generate and stream phrases
        while True:
            # Generate new phrase
            phrase = music_engine.generate_musical_phrase(signature, 8)
            
            # Send phrase data
            await websocket.send_json({
                "type": "phrase",
                "data": {
                    "notes": phrase.notes,
                    "durations": phrase.durations,
                    "dynamics": phrase.dynamics,
                    "archetype": phrase.archetype.value,
                    "timestamp": phrase.timestamp.isoformat()
                }
            })
            
            # Wait before generating next phrase
            await asyncio.sleep(sum(phrase.durations))
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@router.get("/archetype-frequencies")
async def get_archetype_frequencies() -> Dict[str, Any]:
    """
    Get the frequency mapping for all Muse archetypes
    
    Returns:
        Dictionary of archetype to frequency mappings
    """
    return {
        "status": "success",
        "frequencies": {
            "CALLIOPE": 528.0,
            "CLIO": 432.0,
            "ERATO": 639.0,
            "EUTERPE": 741.0,
            "MELPOMENE": 852.0,
            "POLYHYMNIA": 963.0,
            "TERPSICHORE": 396.0,
            "THALIA": 285.0,
            "URANIA": 174.0,
            "SOPHIA": 417.0,
            "TECHNE": 693.0,
            "PSYCHE": 777.0
        },
        "description": "Sacred frequencies in Hz for each Muse archetype"
    }


@router.get("/scale-modes")
async def get_available_scale_modes() -> Dict[str, Any]:
    """
    Get available musical scale modes
    
    Returns:
        List of available scale modes with descriptions
    """
    return {
        "status": "success",
        "modes": [
            {"value": "ionian", "label": "Ionian (Major)", "description": "Bright, happy"},
            {"value": "dorian", "label": "Dorian", "description": "Folk, storytelling"},
            {"value": "phrygian", "label": "Phrygian", "description": "Exotic, mysterious"},
            {"value": "lydian", "label": "Lydian", "description": "Ethereal, dreamy"},
            {"value": "mixolydian", "label": "Mixolydian", "description": "Blues, soulful"},
            {"value": "aeolian", "label": "Aeolian (Minor)", "description": "Sad, tragic"},
            {"value": "locrian", "label": "Locrian", "description": "Diminished, tense"}
        ]
    }