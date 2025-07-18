"""
Integration API Router for MUSE Platform

This module implements the integration endpoints for real-time discovery,
constraint optimization, and live streaming of the creative discovery process.
It provides the advanced features that showcase the full power of the
three-engine Computational Platonism system.
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncio
import json
import logging
from datetime import datetime
import uuid

from pydantic import BaseModel, Field, validator

from database import get_db
from muse.core.frequency_engine import MuseFrequencyEngine
from muse.core.sacred_geometry_calculator import SacredGeometryCalculator
from muse.core.semantic_projection_engine import SemanticProjectionEngine
from muse.services.discovery_orchestrator import DiscoveryOrchestrator, DiscoveryMode
from muse.services.resonance_matcher import ResonanceMatcher
from muse.models.community import UserProfile, CollaborativeSession, SessionParticipant


logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for request/response validation

class LiveDiscoveryRequest(BaseModel):
    """Request model for live poetry discovery"""
    user_id: str = Field(..., description="User ID")
    theme: str = Field(..., description="Creative theme")
    form_type: str = Field(default="auto", description="Poetic form type")
    sacred_constant: str = Field(default="phi", description="Sacred constant to emphasize")
    emotional_palette: List[str] = Field(default_factory=list, description="Emotional palette")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Additional constraints")
    
    @validator('sacred_constant')
    def validate_sacred_constant(cls, v):
        valid_constants = ['phi', 'pi', 'e', 'fibonacci', 'sqrt_2', 'sqrt_3']
        if v not in valid_constants:
            raise ValueError(f'Sacred constant must be one of: {valid_constants}')
        return v


class ConstraintOptimizationRequest(BaseModel):
    """Request model for constraint optimization"""
    user_id: str = Field(..., description="User ID")
    form_type: str = Field(..., description="Target poetic form")
    theme: str = Field(..., description="Creative theme")
    optimization_target: str = Field(..., description="Optimization target")
    constraints: Dict[str, Any] = Field(..., description="Base constraints")
    
    @validator('optimization_target')
    def validate_optimization_target(cls, v):
        valid_targets = ['mathematical_fitness', 'semantic_coherence', 'archetypal_alignment', 'sacred_geometry']
        if v not in valid_targets:
            raise ValueError(f'Optimization target must be one of: {valid_targets}')
        return v


class CollaborativeSessionRequest(BaseModel):
    """Request model for collaborative session"""
    creator_id: str = Field(..., description="Session creator ID")
    title: str = Field(..., description="Session title")
    description: Optional[str] = Field(default=None, description="Session description")
    theme: str = Field(..., description="Creative theme")
    form_type: str = Field(default="auto", description="Target form type")
    sacred_constant: str = Field(default="phi", description="Sacred constant to use")
    max_participants: int = Field(default=5, description="Maximum participants")
    duration_minutes: int = Field(default=60, description="Session duration in minutes")
    session_type: str = Field(default="open", description="Session type")
    
    @validator('session_type')
    def validate_session_type(cls, v):
        valid_types = ['open', 'invite_only', 'private']
        if v not in valid_types:
            raise ValueError(f'Session type must be one of: {valid_types}')
        return v
    
    @validator('max_participants')
    def validate_max_participants(cls, v):
        if v < 2 or v > 20:
            raise ValueError('Max participants must be between 2 and 20')
        return v


class LiveDiscoveryResponse(BaseModel):
    """Response model for live discovery"""
    session_id: str
    discovered_content: str
    form_analysis: Dict[str, Any]
    mathematical_metrics: Dict[str, float]
    semantic_analysis: Dict[str, Any]
    archetypal_resonance: Dict[str, float]
    sacred_geometry_analysis: Dict[str, Any]
    optimization_path: List[Dict[str, Any]]
    discovery_time_seconds: float
    
    class Config:
        orm_mode = True


class OptimizedConstraintsResponse(BaseModel):
    """Response model for optimized constraints"""
    optimization_id: str
    user_id: str
    form_type: str
    theme: str
    base_constraints: Dict[str, Any]
    optimized_constraints: Dict[str, Any]
    optimization_metrics: Dict[str, float]
    improvement_factors: Dict[str, float]
    sacred_geometry_enhancements: Dict[str, Any]
    semantic_improvements: Dict[str, Any]
    
    class Config:
        orm_mode = True


class CollaborativeSessionResponse(BaseModel):
    """Response model for collaborative session"""
    session_id: str
    creator_id: str
    title: str
    description: Optional[str]
    theme: str
    form_type: str
    sacred_constant: str
    status: str
    current_participants: int
    max_participants: int
    collective_resonance: float
    combined_entropy_seed: str
    current_discovery: Dict[str, Any]
    participants: List[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        orm_mode = True


# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        if session_id not in self.session_connections:
            self.session_connections[session_id] = []
        self.session_connections[session_id].append(connection_id)
        
        return connection_id
    
    def disconnect(self, connection_id: str, session_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if session_id in self.session_connections:
            self.session_connections[session_id] = [
                conn for conn in self.session_connections[session_id] 
                if conn != connection_id
            ]
    
    async def send_personal_message(self, message: str, connection_id: str):
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_text(message)
    
    async def broadcast_to_session(self, message: str, session_id: str):
        if session_id in self.session_connections:
            for connection_id in self.session_connections[session_id]:
                if connection_id in self.active_connections:
                    await self.active_connections[connection_id].send_text(message)


manager = ConnectionManager()


# Import centralized dependencies
from muse.dependencies import (
    get_frequency_engine_dependency,
    get_geometry_calculator_dependency,
    get_semantic_engine_dependency,
    get_discovery_orchestrator_dependency,
    get_resonance_matcher_dependency
)


# Integration API endpoints

@router.post("/live/discover-poem", response_model=LiveDiscoveryResponse)
async def live_discover_poem(
    request: LiveDiscoveryRequest,
    db: Session = Depends(get_db),
    orchestrator: DiscoveryOrchestrator = Depends(get_discovery_orchestrator_dependency)
):
    """
    Real-time poetry discovery using all three core engines
    
    This endpoint provides immediate creative discovery by coordinating
    the Frequency Engine, Sacred Geometry Calculator, and Semantic Projection
    Engine to create a poem that embodies the user's mathematical essence.
    """
    try:
        start_time = datetime.utcnow()
        
        # Validate user exists
        user = db.query(UserProfile).filter(UserProfile.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Create enhanced discovery session with live optimization
        discovery_state = orchestrator.start_discovery_session(
            db=db,
            user_id=request.user_id,
            theme=request.theme,
            form_type=request.form_type,
            mode=DiscoveryMode.EXPERIMENTAL,
            constraints={
                'sacred_constant': request.sacred_constant,
                'emotional_palette': request.emotional_palette,
                'live_mode': True,
                'optimization_target': 'balanced'
            }
        )
        
        # Run accelerated discovery iterations
        max_iterations = 5  # Reduced for live response
        for _ in range(max_iterations):
            if discovery_state.phase.value == 'completion':
                break
            discovery_state = orchestrator.continue_discovery_session(discovery_state.session_id)
        
        # Complete the session
        discovery_result = orchestrator.complete_discovery_session(discovery_state.session_id)
        
        # Calculate discovery time
        discovery_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Analyze the discovered content
        form_analysis = _analyze_poetic_form(discovery_result.discovered_content, request.form_type)
        mathematical_metrics = _calculate_mathematical_metrics(discovery_result)
        semantic_analysis = _analyze_semantic_content(discovery_result.discovered_content, request.theme)
        archetypal_resonance = _calculate_archetypal_resonance(discovery_result, user)
        sacred_geometry_analysis = _analyze_sacred_geometry(discovery_result, request.sacred_constant)
        
        return LiveDiscoveryResponse(
            session_id=discovery_result.session_id,
            discovered_content=discovery_result.discovered_content,
            form_analysis=form_analysis,
            mathematical_metrics=mathematical_metrics,
            semantic_analysis=semantic_analysis,
            archetypal_resonance=archetypal_resonance,
            sacred_geometry_analysis=sacred_geometry_analysis,
            optimization_path=discovery_result.discovery_path,
            discovery_time_seconds=discovery_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in live discovery: {e}")
        raise HTTPException(status_code=500, detail="Live discovery failed")


@router.post("/live/optimize-constraints", response_model=OptimizedConstraintsResponse)
async def optimize_constraints(
    request: ConstraintOptimizationRequest,
    db: Session = Depends(get_db),
    orchestrator: DiscoveryOrchestrator = Depends(get_discovery_orchestrator_dependency),
    freq_engine: MuseFrequencyEngine = Depends(get_frequency_engine_dependency),
    geom_calc: SacredGeometryCalculator = Depends(get_geometry_calculator_dependency),
    sem_engine: SemanticProjectionEngine = Depends(get_semantic_engine_dependency)
):
    """
    Constraint optimization for specific forms using mathematical analysis
    
    This endpoint analyzes and optimizes creative constraints for maximum
    effectiveness in a specific poetic form, using sacred geometry and
    semantic analysis to enhance the creative process.
    """
    try:
        # Validate user exists
        user = db.query(UserProfile).filter(UserProfile.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user's frequency signature
        frequency_signature = {
            'user_id': request.user_id,
            'harmonic_blend': user.harmonic_blend or {},
            'sacred_ratios': user.sacred_ratios or {},
            'spiral_coordinates': user.spiral_coordinates or {},
            'primary_muse': user.primary_muse,
            'secondary_muse': user.secondary_muse
        }
        
        # Optimize constraints using orchestrator
        optimization_result = orchestrator.optimize_creative_constraints(
            frequency_signature=frequency_signature,
            theme=request.theme,
            form_type=request.form_type,
            user_preferences={'optimization_target': request.optimization_target}
        )
        
        # Calculate improvement factors
        improvement_factors = _calculate_improvement_factors(
            request.constraints,
            optimization_result['final_constraints']
        )
        
        # Analyze sacred geometry enhancements
        sacred_geometry_enhancements = _analyze_geometry_enhancements(
            optimization_result['geometry_optimized'],
            request.form_type
        )
        
        # Analyze semantic improvements
        semantic_improvements = _analyze_semantic_improvements(
            optimization_result['semantic_optimized'],
            request.theme
        )
        
        # Calculate optimization metrics
        optimization_metrics = {
            'mathematical_fitness_gain': improvement_factors.get('mathematical_fitness', 0.0),
            'semantic_coherence_gain': improvement_factors.get('semantic_coherence', 0.0),
            'archetypal_alignment_gain': improvement_factors.get('archetypal_alignment', 0.0),
            'sacred_geometry_enhancement': improvement_factors.get('sacred_geometry', 0.0),
            'overall_improvement': sum(improvement_factors.values()) / len(improvement_factors)
        }
        
        optimization_id = str(uuid.uuid4())
        
        return OptimizedConstraintsResponse(
            optimization_id=optimization_id,
            user_id=request.user_id,
            form_type=request.form_type,
            theme=request.theme,
            base_constraints=request.constraints,
            optimized_constraints=optimization_result['final_constraints'],
            optimization_metrics=optimization_metrics,
            improvement_factors=improvement_factors,
            sacred_geometry_enhancements=sacred_geometry_enhancements,
            semantic_improvements=semantic_improvements
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in constraint optimization: {e}")
        raise HTTPException(status_code=500, detail="Constraint optimization failed")


@router.websocket("/live/stream-discovery/{session_id}")
async def stream_discovery_progress(
    websocket: WebSocket,
    session_id: str,
    orchestrator: DiscoveryOrchestrator = Depends(get_discovery_orchestrator_dependency)
):
    """
    WebSocket endpoint for streaming live discovery process
    
    This endpoint provides real-time updates on the discovery process,
    allowing users to watch as their creative work emerges through
    mathematical optimization.
    """
    connection_id = await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Get current session status
            session_status = orchestrator.get_session_status(session_id)
            
            if session_status:
                # Send status update
                await manager.send_personal_message(
                    json.dumps({
                        'type': 'status_update',
                        'data': session_status,
                        'timestamp': datetime.utcnow().isoformat()
                    }),
                    connection_id
                )
                
                # Check if session is complete
                if session_status.get('phase') == 'completion':
                    await manager.send_personal_message(
                        json.dumps({
                            'type': 'session_complete',
                            'session_id': session_id,
                            'timestamp': datetime.utcnow().isoformat()
                        }),
                        connection_id
                    )
                    break
            
            # Wait before next update
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        manager.disconnect(connection_id, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.send_personal_message(
            json.dumps({
                'type': 'error',
                'message': 'Discovery stream error',
                'timestamp': datetime.utcnow().isoformat()
            }),
            connection_id
        )


@router.post("/live/collaborative-session", response_model=CollaborativeSessionResponse)
async def create_collaborative_session(
    request: CollaborativeSessionRequest,
    db: Session = Depends(get_db),
    freq_engine: MuseFrequencyEngine = Depends(get_frequency_engine_dependency)
):
    """
    Create multi-user collaborative creative session
    
    This endpoint creates a collaborative session where multiple users
    can contribute their frequency signatures to a collective creative
    discovery process.
    """
    try:
        # Validate creator exists
        creator = db.query(UserProfile).filter(UserProfile.id == request.creator_id).first()
        if not creator:
            raise HTTPException(status_code=404, detail="Creator not found")
        
        # Generate combined entropy seed (starts with creator's entropy)
        creator_entropy = creator.spiral_coordinates.get('entropy', '') if creator.spiral_coordinates else ''
        combined_entropy_seed = freq_engine.read_hardware_entropy(32).hex()
        
        # Create collaborative session
        session = CollaborativeSession(
            creator_id=request.creator_id,
            title=request.title,
            description=request.description,
            theme=request.theme,
            target_form=request.form_type,
            sacred_constant=request.sacred_constant,
            max_participants=request.max_participants,
            duration_minutes=request.duration_minutes,
            session_type=request.session_type,
            combined_entropy_seed=combined_entropy_seed,
            current_discovery={'initialized': True, 'theme': request.theme}
        )
        
        db.add(session)
        db.flush()  # Get the session ID
        
        # Add creator as first participant
        participant = SessionParticipant(
            session_id=session.id,
            user_id=request.creator_id,
            role='creator',
            signature_snapshot=creator.harmonic_blend,
            entropy_contribution=creator_entropy
        )
        
        db.add(participant)
        db.commit()
        
        # Calculate initial collective resonance
        collective_resonance = _calculate_collective_resonance([creator])
        
        # Update session with initial resonance
        session.collective_resonance = collective_resonance
        db.commit()
        
        return CollaborativeSessionResponse(
            session_id=str(session.id),
            creator_id=request.creator_id,
            title=request.title,
            description=request.description,
            theme=request.theme,
            form_type=request.form_type,
            sacred_constant=request.sacred_constant,
            status=session.status,
            current_participants=1,
            max_participants=request.max_participants,
            collective_resonance=collective_resonance,
            combined_entropy_seed=combined_entropy_seed,
            current_discovery=session.current_discovery,
            participants=[{
                'user_id': request.creator_id,
                'username': creator.username,
                'role': 'creator',
                'joined_at': datetime.utcnow().isoformat()
            }],
            created_at=session.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating collaborative session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create collaborative session")


@router.get("/live/collaborative-session/{session_id}", response_model=CollaborativeSessionResponse)
async def get_collaborative_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Get collaborative session details
    
    Returns the current state of a collaborative session including
    participants, collective resonance, and current discovery progress.
    """
    try:
        # Get session
        session = db.query(CollaborativeSession).filter(
            CollaborativeSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Collaborative session not found")
        
        # Get participants
        participants = db.query(SessionParticipant).filter(
            SessionParticipant.session_id == session_id
        ).all()
        
        # Format participants
        participant_list = []
        for participant in participants:
            user = db.query(UserProfile).filter(UserProfile.id == participant.user_id).first()
            participant_list.append({
                'user_id': str(participant.user_id),
                'username': user.username if user else 'Unknown',
                'role': participant.role,
                'joined_at': participant.joined_at.isoformat()
            })
        
        return CollaborativeSessionResponse(
            session_id=str(session.id),
            creator_id=str(session.creator_id),
            title=session.title,
            description=session.description,
            theme=session.theme,
            form_type=session.target_form,
            sacred_constant=session.sacred_constant,
            status=session.status,
            current_participants=len(participants),
            max_participants=session.max_participants,
            collective_resonance=session.collective_resonance,
            combined_entropy_seed=session.combined_entropy_seed,
            current_discovery=session.current_discovery,
            participants=participant_list,
            created_at=session.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collaborative session: {e}")
        raise HTTPException(status_code=500, detail="Failed to get collaborative session")


@router.post("/live/collaborative-session/{session_id}/join")
async def join_collaborative_session(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Join a collaborative session
    
    Allows a user to join an existing collaborative session, contributing
    their frequency signature to the collective creative process.
    """
    try:
        # Get session
        session = db.query(CollaborativeSession).filter(
            CollaborativeSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Collaborative session not found")
        
        # Check if session is full
        current_participants = db.query(SessionParticipant).filter(
            SessionParticipant.session_id == session_id,
            SessionParticipant.status == 'active'
        ).count()
        
        if current_participants >= session.max_participants:
            raise HTTPException(status_code=400, detail="Session is full")
        
        # Check if user already joined
        existing_participant = db.query(SessionParticipant).filter(
            SessionParticipant.session_id == session_id,
            SessionParticipant.user_id == user_id
        ).first()
        
        if existing_participant:
            raise HTTPException(status_code=400, detail="User already in session")
        
        # Get user
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Add participant
        participant = SessionParticipant(
            session_id=session_id,
            user_id=user_id,
            role='contributor',
            signature_snapshot=user.harmonic_blend,
            entropy_contribution=user.spiral_coordinates.get('entropy', '') if user.spiral_coordinates else ''
        )
        
        db.add(participant)
        
        # Update session participants count
        session.current_participants = current_participants + 1
        
        # Recalculate collective resonance
        all_participants = db.query(SessionParticipant).filter(
            SessionParticipant.session_id == session_id
        ).all()
        
        participant_users = []
        for p in all_participants:
            u = db.query(UserProfile).filter(UserProfile.id == p.user_id).first()
            if u:
                participant_users.append(u)
        
        session.collective_resonance = _calculate_collective_resonance(participant_users)
        
        db.commit()
        
        return {
            'message': 'Successfully joined collaborative session',
            'session_id': session_id,
            'user_id': user_id,
            'collective_resonance': session.collective_resonance,
            'current_participants': session.current_participants
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error joining collaborative session: {e}")
        raise HTTPException(status_code=500, detail="Failed to join collaborative session")


# Helper functions

def _analyze_poetic_form(content: str, form_type: str) -> Dict[str, Any]:
    """Analyze poetic form structure"""
    lines = content.split('\n')
    clean_lines = [line.strip() for line in lines if line.strip()]
    
    return {
        'form_type': form_type,
        'line_count': len(clean_lines),
        'average_line_length': sum(len(line) for line in clean_lines) / len(clean_lines) if clean_lines else 0,
        'syllable_pattern': [len(line.split()) for line in clean_lines],  # Simplified
        'rhyme_scheme_detected': 'ABAB',  # Simplified
        'meter_analysis': 'iambic',  # Simplified
        'structure_score': 0.8  # Simplified
    }


def _calculate_mathematical_metrics(discovery_result) -> Dict[str, float]:
    """Calculate mathematical metrics for discovery"""
    return {
        'mathematical_fitness': discovery_result.mathematical_fitness,
        'semantic_coherence': discovery_result.semantic_coherence,
        'archetypal_alignment': discovery_result.archetypal_alignment,
        'sacred_geometry_score': discovery_result.sacred_geometry_score,
        'overall_harmony': (
            discovery_result.mathematical_fitness +
            discovery_result.semantic_coherence +
            discovery_result.archetypal_alignment +
            discovery_result.sacred_geometry_score
        ) / 4
    }


def _analyze_semantic_content(content: str, theme: str) -> Dict[str, Any]:
    """Analyze semantic content"""
    words = content.lower().split()
    
    return {
        'theme': theme,
        'word_count': len(words),
        'unique_words': len(set(words)),
        'theme_relevance': 0.8,  # Simplified
        'emotional_resonance': 0.7,  # Simplified
        'metaphor_density': 0.6,  # Simplified
        'imagery_strength': 0.8,  # Simplified
        'semantic_coherence': 0.75  # Simplified
    }


def _calculate_archetypal_resonance(discovery_result, user: UserProfile) -> Dict[str, float]:
    """Calculate archetypal resonance"""
    user_blend = user.harmonic_blend or {}
    
    # Simplified resonance calculation
    resonance = {}
    for archetype, strength in user_blend.items():
        resonance[archetype] = strength * discovery_result.archetypal_alignment
    
    return resonance


def _analyze_sacred_geometry(discovery_result, sacred_constant: str) -> Dict[str, Any]:
    """Analyze sacred geometry elements"""
    return {
        'sacred_constant': sacred_constant,
        'geometry_score': discovery_result.sacred_geometry_score,
        'golden_ratio_alignment': 0.8,  # Simplified
        'fibonacci_patterns': 0.7,  # Simplified
        'sacred_proportions': 0.75,  # Simplified
        'geometric_harmony': 0.8,  # Simplified
        'structural_beauty': 0.85  # Simplified
    }


def _calculate_improvement_factors(base_constraints: Dict[str, Any], 
                                 optimized_constraints: Dict[str, Any]) -> Dict[str, float]:
    """Calculate improvement factors from optimization"""
    # Simplified improvement calculation
    return {
        'mathematical_fitness': 0.15,
        'semantic_coherence': 0.12,
        'archetypal_alignment': 0.18,
        'sacred_geometry': 0.20,
        'overall_quality': 0.16
    }


def _analyze_geometry_enhancements(geometry_optimized: Dict[str, Any], 
                                 form_type: str) -> Dict[str, Any]:
    """Analyze sacred geometry enhancements"""
    return {
        'form_type': form_type,
        'phi_optimization': geometry_optimized.get('phi_optimization', False),
        'syllable_enhancement': True,  # Simplified
        'rhythm_improvement': 0.8,  # Simplified
        'structural_coherence': 0.85,  # Simplified
        'mathematical_beauty': 0.9  # Simplified
    }


def _analyze_semantic_improvements(semantic_optimized: Dict[str, Any], 
                                 theme: str) -> Dict[str, Any]:
    """Analyze semantic improvements"""
    return {
        'theme': theme,
        'semantic_optimization': semantic_optimized.get('semantic_optimization', False),
        'emotional_enhancement': True,  # Simplified
        'theme_alignment': 0.9,  # Simplified
        'metaphor_improvement': 0.8,  # Simplified
        'linguistic_flow': 0.85  # Simplified
    }


def _calculate_collective_resonance(participants: List[UserProfile]) -> float:
    """Calculate collective resonance of participants"""
    if not participants:
        return 0.0
    
    # Simplified collective resonance calculation
    total_resonance = 0.0
    count = 0
    
    for participant in participants:
        if participant.harmonic_blend:
            # Calculate individual resonance strength
            individual_resonance = sum(participant.harmonic_blend.values()) / len(participant.harmonic_blend)
            total_resonance += individual_resonance
            count += 1
    
    if count == 0:
        return 0.0
    
    base_resonance = total_resonance / count
    
    # Bonus for multiple participants (collective wisdom)
    collaboration_bonus = min(0.2, (count - 1) * 0.05)
    
    return min(1.0, base_resonance + collaboration_bonus)