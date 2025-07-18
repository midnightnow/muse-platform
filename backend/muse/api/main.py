"""
Core API Router for MUSE Platform

This module implements the main API endpoints for MUSE's Computational Platonism
creative discovery system, including personality assessment, frequency signature
generation, and creative discovery sessions.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging

from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4

from database import get_db
from muse.core.frequency_engine import MuseFrequencyEngine
from muse.core.sacred_geometry_calculator import SacredGeometryCalculator
from muse.core.semantic_projection_engine import SemanticProjectionEngine
from muse.services.discovery_orchestrator import DiscoveryOrchestrator, DiscoveryMode
from muse.services.resonance_matcher import ResonanceMatcher
from muse.models.community import UserProfile, FrequencySignature, CommunityCreation


logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for request/response validation

class PersonalityAssessmentRequest(BaseModel):
    """Request model for personality assessment"""
    user_id: str = Field(..., description="User ID for the assessment")
    creative_preferences: Dict[str, Any] = Field(..., description="Creative preferences")
    personality_traits: Dict[str, Any] = Field(..., description="Personality traits")
    mathematical_affinity: Dict[str, Any] = Field(..., description="Mathematical preferences")
    preferred_forms: List[str] = Field(default_factory=list, description="Preferred poetic forms")
    favorite_themes: List[str] = Field(default_factory=list, description="Favorite themes")
    discovery_style: str = Field(default="balanced", description="Discovery style preference")
    
    @validator('discovery_style')
    def validate_discovery_style(cls, v):
        valid_styles = ['mathematical', 'intuitive', 'experimental', 'balanced']
        if v not in valid_styles:
            raise ValueError(f'Discovery style must be one of: {valid_styles}')
        return v


class FrequencySignatureResponse(BaseModel):
    """Response model for frequency signature"""
    id: str
    user_id: str
    harmonic_blend: Dict[str, float]
    sacred_ratios: Dict[str, float]
    spiral_coordinates: Dict[str, float]
    primary_muse: str
    secondary_muse: str
    entropy_seed: str
    characteristics: Dict[str, float]
    performance_metrics: Dict[str, float]
    created_at: datetime
    
    class Config:
        orm_mode = True


class SignatureTuningRequest(BaseModel):
    """Request model for signature tuning"""
    target_muses: List[str] = Field(..., description="Target muses to emphasize")
    blend_ratios: List[float] = Field(..., description="Corresponding blend ratios")
    
    @validator('target_muses')
    def validate_muses(cls, v):
        valid_muses = [
            'CALLIOPE', 'CLIO', 'ERATO', 'EUTERPE', 'MELPOMENE', 'POLYHYMNIA',
            'TERPSICHORE', 'THALIA', 'URANIA', 'SOPHIA', 'TECHNE', 'PSYCHE'
        ]
        for muse in v:
            if muse not in valid_muses:
                raise ValueError(f'Invalid muse: {muse}. Must be one of: {valid_muses}')
        return v
    
    @validator('blend_ratios')
    def validate_ratios(cls, v):
        if any(ratio < 0 or ratio > 1 for ratio in v):
            raise ValueError('Blend ratios must be between 0 and 1')
        return v


class DiscoverySessionRequest(BaseModel):
    """Request model for starting discovery session"""
    user_id: str = Field(..., description="User ID")
    theme: str = Field(..., description="Creative theme")
    form_type: str = Field(default="auto", description="Poetic form type")
    mode: str = Field(default="individual", description="Discovery mode")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Creative constraints")
    
    @validator('mode')
    def validate_mode(cls, v):
        valid_modes = ['individual', 'collaborative', 'guided', 'experimental']
        if v not in valid_modes:
            raise ValueError(f'Mode must be one of: {valid_modes}')
        return v


class SessionFeedbackRequest(BaseModel):
    """Request model for session feedback"""
    user_feedback: Dict[str, Any] = Field(..., description="User feedback")
    satisfaction_score: Optional[float] = Field(default=None, description="Satisfaction score (0-1)")
    
    @validator('satisfaction_score')
    def validate_satisfaction(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Satisfaction score must be between 0 and 1')
        return v


class DiscoverySessionResponse(BaseModel):
    """Response model for discovery session"""
    session_id: str
    user_id: str
    phase: str
    progress: float
    current_iteration: int
    max_iterations: int
    fitness_scores: Dict[str, float]
    current_discovery: Dict[str, Any]
    updated_at: datetime
    
    class Config:
        orm_mode = True


class DiscoveryResultResponse(BaseModel):
    """Response model for discovery result"""
    session_id: str
    user_id: str
    discovered_content: str
    form_type: str
    theme: str
    mathematical_fitness: float
    semantic_coherence: float
    archetypal_alignment: float
    sacred_geometry_score: float
    discovery_path: List[Dict[str, Any]]
    optimization_metrics: Dict[str, Any]
    user_satisfaction: Optional[float] = None
    
    class Config:
        orm_mode = True


# Import centralized dependencies
from muse.dependencies import (
    get_frequency_engine_dependency,
    get_geometry_calculator_dependency,
    get_semantic_engine_dependency,
    get_discovery_orchestrator_dependency,
    get_resonance_matcher_dependency
)


# Core API endpoints

@router.post("/assessment/complete", response_model=FrequencySignatureResponse)
async def complete_personality_assessment(
    request: PersonalityAssessmentRequest,
    db: Session = Depends(get_db),
    freq_engine: MuseFrequencyEngine = Depends(get_frequency_engine_dependency),
    background_tasks: BackgroundTasks = None
):
    """
    Complete personality assessment and generate frequency signature
    
    This endpoint implements the core of MUSE's Computational Platonism approach,
    transforming user personality data into a mathematical frequency signature
    that captures their archetypal creative essence.
    """
    try:
        # Validate user exists
        user = db.query(UserProfile).filter(UserProfile.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate frequency signature
        assessment_data = {
            'user_id': request.user_id,
            'creative_preferences': request.creative_preferences,
            'personality_traits': request.personality_traits,
            'mathematical_affinity': request.mathematical_affinity
        }
        
        frequency_signature = freq_engine.generate_frequency_signature(assessment_data)
        
        # Calculate additional characteristics
        characteristics = {
            'specialization_index': _calculate_specialization_index(frequency_signature.harmonic_blend),
            'diversity_index': _calculate_diversity_index(frequency_signature.harmonic_blend),
            'coherence_score': _calculate_coherence_score(frequency_signature),
            'uniqueness_score': _calculate_uniqueness_score(db, frequency_signature)
        }
        
        # Calculate performance metrics
        performance_metrics = {
            'discovery_success_rate': 0.0,  # Will be updated with usage
            'average_fitness_score': 0.0,
            'user_satisfaction_score': 0.0
        }
        
        # Store in database
        db_signature = FrequencySignature(
            user_id=request.user_id,
            harmonic_blend=frequency_signature.harmonic_blend,
            sacred_ratios=frequency_signature.sacred_ratios,
            spiral_coordinates=frequency_signature.spiral_coordinates,
            primary_muse=frequency_signature.primary_muse,
            secondary_muse=frequency_signature.secondary_muse,
            entropy_seed=frequency_signature.entropy_seed,
            specialization_index=characteristics['specialization_index'],
            diversity_index=characteristics['diversity_index'],
            coherence_score=characteristics['coherence_score'],
            uniqueness_score=characteristics['uniqueness_score'],
            discovery_success_rate=performance_metrics['discovery_success_rate'],
            average_fitness_score=performance_metrics['average_fitness_score'],
            user_satisfaction_score=performance_metrics['user_satisfaction_score']
        )
        
        db.add(db_signature)
        
        # Update user profile
        user.current_signature_id = db_signature.id
        user.primary_muse = frequency_signature.primary_muse
        user.secondary_muse = frequency_signature.secondary_muse
        user.harmonic_blend = frequency_signature.harmonic_blend
        user.sacred_ratios = frequency_signature.sacred_ratios
        user.spiral_coordinates = frequency_signature.spiral_coordinates
        user.preferred_forms = request.preferred_forms
        user.favorite_themes = request.favorite_themes
        user.discovery_style = request.discovery_style
        
        db.commit()
        
        # Schedule background analytics update
        if background_tasks:
            background_tasks.add_task(_update_signature_analytics, db_signature.id)
        
        return FrequencySignatureResponse(
            id=str(db_signature.id),
            user_id=request.user_id,
            harmonic_blend=frequency_signature.harmonic_blend,
            sacred_ratios=frequency_signature.sacred_ratios,
            spiral_coordinates=frequency_signature.spiral_coordinates,
            primary_muse=frequency_signature.primary_muse,
            secondary_muse=frequency_signature.secondary_muse,
            entropy_seed=frequency_signature.entropy_seed,
            characteristics=characteristics,
            performance_metrics=performance_metrics,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error completing assessment: {e}")
        raise HTTPException(status_code=500, detail="Assessment processing failed")


@router.get("/signatures/{signature_id}", response_model=FrequencySignatureResponse)
async def get_frequency_signature(
    signature_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve frequency signature details
    
    Returns complete frequency signature information including archetypal blend,
    sacred ratios, spiral coordinates, and performance metrics.
    """
    try:
        signature = db.query(FrequencySignature).filter(
            FrequencySignature.id == signature_id
        ).first()
        
        if not signature:
            raise HTTPException(status_code=404, detail="Frequency signature not found")
        
        return FrequencySignatureResponse(
            id=str(signature.id),
            user_id=str(signature.user_id),
            harmonic_blend=signature.harmonic_blend,
            sacred_ratios=signature.sacred_ratios,
            spiral_coordinates=signature.spiral_coordinates,
            primary_muse=signature.primary_muse,
            secondary_muse=signature.secondary_muse,
            entropy_seed=signature.entropy_seed,
            characteristics={
                'specialization_index': signature.specialization_index,
                'diversity_index': signature.diversity_index,
                'coherence_score': signature.coherence_score,
                'uniqueness_score': signature.uniqueness_score
            },
            performance_metrics={
                'discovery_success_rate': signature.discovery_success_rate,
                'average_fitness_score': signature.average_fitness_score,
                'user_satisfaction_score': signature.user_satisfaction_score
            },
            created_at=signature.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving signature: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signature")


@router.post("/signatures/{signature_id}/tune", response_model=FrequencySignatureResponse)
async def tune_frequency_signature(
    signature_id: str,
    request: SignatureTuningRequest,
    db: Session = Depends(get_db),
    freq_engine: MuseFrequencyEngine = Depends(get_frequency_engine_dependency)
):
    """
    Tune frequency signature toward specific archetypal blend
    
    This endpoint allows users to fine-tune their frequency signature by
    emphasizing specific muses while maintaining mathematical coherence.
    """
    try:
        # Get existing signature
        db_signature = db.query(FrequencySignature).filter(
            FrequencySignature.id == signature_id
        ).first()
        
        if not db_signature:
            raise HTTPException(status_code=404, detail="Frequency signature not found")
        
        # Validate target muses and ratios match
        if len(request.target_muses) != len(request.blend_ratios):
            raise HTTPException(
                status_code=400, 
                detail="Number of target muses must match number of blend ratios"
            )
        
        # Create frequency signature object
        current_signature = freq_engine.deserialize_signature(
            freq_engine.serialize_signature(
                freq_engine.FrequencySignature(
                    id=str(db_signature.id),
                    user_id=str(db_signature.user_id),
                    harmonic_blend=db_signature.harmonic_blend,
                    sacred_ratios=db_signature.sacred_ratios,
                    spiral_coordinates=db_signature.spiral_coordinates,
                    entropy_seed=db_signature.entropy_seed,
                    primary_muse=db_signature.primary_muse,
                    secondary_muse=db_signature.secondary_muse,
                    created_at=db_signature.created_at
                )
            )
        )
        
        # Apply tuning
        tuned_signature = freq_engine.tune_signature(
            current_signature, 
            request.target_muses, 
            request.blend_ratios
        )
        
        # Update database
        db_signature.harmonic_blend = tuned_signature.harmonic_blend
        db_signature.primary_muse = tuned_signature.primary_muse
        db_signature.secondary_muse = tuned_signature.secondary_muse
        db_signature.spiral_coordinates = tuned_signature.spiral_coordinates
        db_signature.tuning_count += 1
        db_signature.last_tuned_at = datetime.utcnow()
        
        # Update tuning history
        tuning_history = db_signature.tuning_history or []
        tuning_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'target_muses': request.target_muses,
            'blend_ratios': request.blend_ratios,
            'previous_primary': current_signature.primary_muse,
            'new_primary': tuned_signature.primary_muse
        })
        db_signature.tuning_history = tuning_history
        
        # Recalculate characteristics
        characteristics = {
            'specialization_index': _calculate_specialization_index(tuned_signature.harmonic_blend),
            'diversity_index': _calculate_diversity_index(tuned_signature.harmonic_blend),
            'coherence_score': _calculate_coherence_score(tuned_signature),
            'uniqueness_score': _calculate_uniqueness_score(db, tuned_signature)
        }
        
        db_signature.specialization_index = characteristics['specialization_index']
        db_signature.diversity_index = characteristics['diversity_index']
        db_signature.coherence_score = characteristics['coherence_score']
        db_signature.uniqueness_score = characteristics['uniqueness_score']
        
        db.commit()
        
        return FrequencySignatureResponse(
            id=str(db_signature.id),
            user_id=str(db_signature.user_id),
            harmonic_blend=tuned_signature.harmonic_blend,
            sacred_ratios=tuned_signature.sacred_ratios,
            spiral_coordinates=tuned_signature.spiral_coordinates,
            primary_muse=tuned_signature.primary_muse,
            secondary_muse=tuned_signature.secondary_muse,
            entropy_seed=tuned_signature.entropy_seed,
            characteristics=characteristics,
            performance_metrics={
                'discovery_success_rate': db_signature.discovery_success_rate,
                'average_fitness_score': db_signature.average_fitness_score,
                'user_satisfaction_score': db_signature.user_satisfaction_score
            },
            created_at=db_signature.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error tuning signature: {e}")
        raise HTTPException(status_code=500, detail="Signature tuning failed")


@router.post("/sessions/start", response_model=DiscoverySessionResponse)
async def start_discovery_session(
    request: DiscoverySessionRequest,
    db: Session = Depends(get_db),
    orchestrator: DiscoveryOrchestrator = Depends(get_discovery_orchestrator_dependency)
):
    """
    Start a new creative discovery session
    
    This endpoint initiates a discovery session that coordinates all three
    core engines (Frequency, Sacred Geometry, Semantic Projection) to guide
    the user through mathematical creative discovery.
    """
    try:
        # Validate user exists
        user = db.query(UserProfile).filter(UserProfile.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert mode string to enum
        mode_mapping = {
            'individual': DiscoveryMode.INDIVIDUAL,
            'collaborative': DiscoveryMode.COLLABORATIVE,
            'guided': DiscoveryMode.GUIDED,
            'experimental': DiscoveryMode.EXPERIMENTAL
        }
        
        discovery_mode = mode_mapping.get(request.mode, DiscoveryMode.INDIVIDUAL)
        
        # Start discovery session
        discovery_state = orchestrator.start_discovery_session(
            db=db,
            user_id=request.user_id,
            theme=request.theme,
            form_type=request.form_type,
            mode=discovery_mode,
            constraints=request.constraints
        )
        
        return DiscoverySessionResponse(
            session_id=discovery_state.session_id,
            user_id=discovery_state.user_id,
            phase=discovery_state.phase.value,
            progress=discovery_state.progress,
            current_iteration=discovery_state.current_iteration,
            max_iterations=discovery_state.max_iterations,
            fitness_scores=discovery_state.fitness_scores,
            current_discovery=discovery_state.current_discovery,
            updated_at=discovery_state.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting discovery session: {e}")
        raise HTTPException(status_code=500, detail="Failed to start discovery session")


@router.get("/sessions/{session_id}/status", response_model=DiscoverySessionResponse)
async def get_session_status(
    session_id: str,
    orchestrator: DiscoveryOrchestrator = Depends(get_discovery_orchestrator_dependency)
):
    """
    Get current status of a discovery session
    
    Returns the current phase, progress, fitness scores, and discovered content
    for an active discovery session.
    """
    try:
        session_status = orchestrator.get_session_status(session_id)
        
        if not session_status:
            raise HTTPException(status_code=404, detail="Discovery session not found")
        
        return DiscoverySessionResponse(
            session_id=session_status['session_id'],
            user_id=session_status.get('user_id', ''),
            phase=session_status['phase'],
            progress=session_status['progress'],
            current_iteration=session_status['current_iteration'],
            max_iterations=session_status['max_iterations'],
            fitness_scores=session_status['fitness_scores'],
            current_discovery=session_status['current_discovery'],
            updated_at=datetime.fromisoformat(session_status['updated_at'])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session status")


@router.post("/sessions/{session_id}/continue", response_model=DiscoverySessionResponse)
async def continue_discovery_session(
    session_id: str,
    request: Optional[SessionFeedbackRequest] = None,
    orchestrator: DiscoveryOrchestrator = Depends(get_discovery_orchestrator_dependency)
):
    """
    Continue an active discovery session
    
    Advances the discovery session to the next iteration, optionally incorporating
    user feedback to guide the optimization process.
    """
    try:
        user_feedback = request.user_feedback if request else None
        
        discovery_state = orchestrator.continue_discovery_session(
            session_id=session_id,
            user_feedback=user_feedback
        )
        
        return DiscoverySessionResponse(
            session_id=discovery_state.session_id,
            user_id=discovery_state.user_id,
            phase=discovery_state.phase.value,
            progress=discovery_state.progress,
            current_iteration=discovery_state.current_iteration,
            max_iterations=discovery_state.max_iterations,
            fitness_scores=discovery_state.fitness_scores,
            current_discovery=discovery_state.current_discovery,
            updated_at=discovery_state.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error continuing discovery session: {e}")
        raise HTTPException(status_code=500, detail="Failed to continue discovery session")


@router.post("/sessions/{session_id}/complete", response_model=DiscoveryResultResponse)
async def complete_discovery_session(
    session_id: str,
    request: Optional[SessionFeedbackRequest] = None,
    db: Session = Depends(get_db),
    orchestrator: DiscoveryOrchestrator = Depends(get_discovery_orchestrator_dependency)
):
    """
    Complete a discovery session and get final result
    
    Finalizes the discovery session, applies any final optimizations,
    and returns the completed creative discovery with all metrics.
    """
    try:
        user_satisfaction = request.satisfaction_score if request else None
        
        # Complete the session
        discovery_result = orchestrator.complete_discovery_session(
            session_id=session_id,
            user_satisfaction=user_satisfaction
        )
        
        # Store result in database if requested
        if request and request.user_feedback.get('save_to_community', False):
            try:
                # Create community creation
                community_creation = CommunityCreation(
                    creator_id=discovery_result.user_id,
                    title=f"Discovery: {discovery_result.theme}",
                    content=discovery_result.discovered_content,
                    content_preview=discovery_result.discovered_content[:500],
                    form_type=discovery_result.form_type,
                    primary_theme=discovery_result.theme,
                    mathematical_fitness=discovery_result.mathematical_fitness,
                    semantic_coherence=discovery_result.semantic_coherence,
                    archetypal_alignment=discovery_result.archetypal_alignment,
                    sacred_geometry_score=discovery_result.sacred_geometry_score,
                    discovery_coordinates=discovery_result.discovery_path,
                    discovery_time_seconds=discovery_result.optimization_metrics.get('session_duration', 0),
                    iteration_count=discovery_result.optimization_metrics.get('total_iterations', 0),
                    constraint_satisfaction=discovery_result.optimization_metrics.get('final_fitness', 0)
                )
                
                db.add(community_creation)
                db.commit()
                
            except Exception as e:
                logger.warning(f"Failed to save discovery to community: {e}")
                db.rollback()
        
        return DiscoveryResultResponse(
            session_id=discovery_result.session_id,
            user_id=discovery_result.user_id,
            discovered_content=discovery_result.discovered_content,
            form_type=discovery_result.form_type,
            theme=discovery_result.theme,
            mathematical_fitness=discovery_result.mathematical_fitness,
            semantic_coherence=discovery_result.semantic_coherence,
            archetypal_alignment=discovery_result.archetypal_alignment,
            sacred_geometry_score=discovery_result.sacred_geometry_score,
            discovery_path=discovery_result.discovery_path,
            optimization_metrics=discovery_result.optimization_metrics,
            user_satisfaction=discovery_result.user_satisfaction
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing discovery session: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete discovery session")


# Helper functions

def _calculate_specialization_index(harmonic_blend: Dict[str, float]) -> float:
    """Calculate specialization index (how specialized vs generalized)"""
    if not harmonic_blend:
        return 0.0
    
    values = list(harmonic_blend.values())
    max_value = max(values)
    mean_value = sum(values) / len(values)
    
    return max_value / mean_value if mean_value > 0 else 0.0


def _calculate_diversity_index(harmonic_blend: Dict[str, float]) -> float:
    """Calculate diversity index (how diverse across archetypes)"""
    if not harmonic_blend:
        return 0.0
    
    values = list(harmonic_blend.values())
    # Shannon diversity index
    diversity = -sum(v * math.log(v + 1e-10) for v in values)
    max_diversity = math.log(len(values))
    
    return diversity / max_diversity if max_diversity > 0 else 0.0


def _calculate_coherence_score(signature) -> float:
    """Calculate internal coherence score"""
    # Simplified coherence based on sacred ratios and harmonic blend consistency
    sacred_ratios = signature.sacred_ratios
    harmonic_blend = signature.harmonic_blend
    
    # Check if primary muse aligns with sacred ratios
    primary_strength = harmonic_blend.get(signature.primary_muse, 0.0)
    phi_affinity = sacred_ratios.get('phi', 0.5)
    
    # Simple coherence calculation
    coherence = (primary_strength + phi_affinity) / 2
    
    return min(1.0, coherence)


def _calculate_uniqueness_score(db: Session, signature) -> float:
    """Calculate uniqueness score compared to other signatures"""
    # Simplified uniqueness calculation
    # In a full implementation, this would compare against all signatures
    try:
        total_signatures = db.query(FrequencySignature).count()
        
        # For now, return a score based on primary muse rarity
        primary_muse_count = db.query(FrequencySignature).filter(
            FrequencySignature.primary_muse == signature.primary_muse
        ).count()
        
        if total_signatures == 0:
            return 1.0
        
        rarity = 1.0 - (primary_muse_count / total_signatures)
        return max(0.1, rarity)  # Minimum uniqueness of 0.1
        
    except Exception as e:
        logger.warning(f"Error calculating uniqueness: {e}")
        return 0.5  # Default uniqueness


def _update_signature_analytics(signature_id: str):
    """Background task to update signature analytics"""
    # This would implement analytics updates
    # For now, just log the update
    logger.info(f"Analytics update scheduled for signature {signature_id}")


import math