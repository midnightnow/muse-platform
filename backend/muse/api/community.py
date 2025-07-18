"""
Community API Router for MUSE Platform

This module implements the community endpoints for the frequency-based social
network, including profiles, creations, social interactions, and content
curation based on archetypal resonance and sacred geometry principles.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import logging

from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4

from database import get_db
from muse.core.frequency_engine import MuseFrequencyEngine
from muse.core.sacred_geometry_calculator import SacredGeometryCalculator
from muse.core.semantic_projection_engine import SemanticProjectionEngine
from muse.services.resonance_matcher import ResonanceMatcher
from muse.services.community_curator import CommunityCurator, CurationMode, FeedType
from muse.models.community import (
    UserProfile, FrequencySignature, CommunityCreation, Comment, Like, Follow,
    CollaborativeSession, SessionParticipant, CommunityAnalytics
)


logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for request/response validation

class ProfileCreateRequest(BaseModel):
    """Request model for creating user profile"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    display_name: Optional[str] = Field(default=None, max_length=100, description="Display name")
    bio: Optional[str] = Field(default=None, max_length=500, description="Biography")
    avatar_url: Optional[str] = Field(default=None, description="Avatar URL")
    preferred_forms: List[str] = Field(default_factory=list, description="Preferred poetic forms")
    favorite_themes: List[str] = Field(default_factory=list, description="Favorite themes")
    discovery_style: str = Field(default="balanced", description="Discovery style")
    
    @validator('username')
    def validate_username(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError('Username must contain only letters, numbers, and underscores')
        return v
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v


class ProfileUpdateRequest(BaseModel):
    """Request model for updating user profile"""
    display_name: Optional[str] = Field(default=None, max_length=100)
    bio: Optional[str] = Field(default=None, max_length=500)
    avatar_url: Optional[str] = Field(default=None)
    preferred_forms: Optional[List[str]] = Field(default=None)
    favorite_themes: Optional[List[str]] = Field(default=None)
    discovery_style: Optional[str] = Field(default=None)
    profile_visibility: Optional[str] = Field(default=None)
    signature_sharing: Optional[bool] = Field(default=None)
    discovery_sharing: Optional[bool] = Field(default=None)


class CreationShareRequest(BaseModel):
    """Request model for sharing creation"""
    title: str = Field(..., min_length=1, max_length=200, description="Creation title")
    content: str = Field(..., min_length=1, description="Creation content")
    form_type: str = Field(..., description="Poetic form type")
    primary_theme: str = Field(..., description="Primary theme")
    secondary_themes: List[str] = Field(default_factory=list, description="Secondary themes")
    visibility: str = Field(default="public", description="Visibility setting")
    
    @validator('visibility')
    def validate_visibility(cls, v):
        valid_visibility = ['public', 'followers', 'private']
        if v not in valid_visibility:
            raise ValueError(f'Visibility must be one of: {valid_visibility}')
        return v


class CommentCreateRequest(BaseModel):
    """Request model for creating comment"""
    content: str = Field(..., min_length=1, max_length=1000, description="Comment content")
    parent_comment_id: Optional[str] = Field(default=None, description="Parent comment ID for threading")


class ProfileResponse(BaseModel):
    """Response model for user profile"""
    id: str
    username: str
    display_name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    primary_muse: Optional[str]
    secondary_muse: Optional[str]
    harmonic_blend: Optional[Dict[str, float]]
    sacred_ratios: Optional[Dict[str, float]]
    preferred_forms: Optional[List[str]]
    favorite_themes: Optional[List[str]]
    discovery_style: Optional[str]
    social_metrics: Dict[str, int]
    privacy_settings: Dict[str, Any]
    created_at: datetime
    last_active_at: datetime
    
    class Config:
        orm_mode = True


class CreationResponse(BaseModel):
    """Response model for community creation"""
    id: str
    creator_id: str
    title: str
    content: str
    content_preview: str
    form_type: str
    primary_theme: str
    secondary_themes: List[str]
    mathematical_analysis: Dict[str, float]
    sacred_geometry: Dict[str, Any]
    community_metrics: Dict[str, int]
    discovery_metrics: Dict[str, Any]
    visibility_data: Dict[str, Any]
    collaborative_data: Dict[str, Any]
    timestamps: Dict[str, str]
    
    class Config:
        orm_mode = True


class FeedResponse(BaseModel):
    """Response model for content feed"""
    items: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    curation_info: Dict[str, Any]
    
    class Config:
        orm_mode = True


class GalleryResponse(BaseModel):
    """Response model for gallery"""
    creations: List[Dict[str, Any]]
    filters: Dict[str, Any]
    pagination: Dict[str, Any]
    
    class Config:
        orm_mode = True


class CommunityMatch(BaseModel):
    """Response model for community matches"""
    user_id: str
    username: str
    display_name: str
    resonance_score: float
    compatibility_type: str
    shared_archetypes: List[str]
    complementary_traits: List[str]
    recommended_collaboration: str
    
    class Config:
        orm_mode = True


# Import centralized dependencies
from muse.dependencies import (
    get_semantic_engine_dependency,
    get_resonance_matcher_dependency,
    get_community_curator_dependency
)


# Community API endpoints

@router.post("/profiles/create", response_model=ProfileResponse)
async def create_user_profile(
    request: ProfileCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new user profile
    
    Creates a user profile in the MUSE community with basic information
    and preferences. The frequency signature will be generated separately
    through the assessment process.
    """
    try:
        # Check if username already exists
        existing_user = db.query(UserProfile).filter(
            UserProfile.username == request.username
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email already exists
        existing_email = db.query(UserProfile).filter(
            UserProfile.email == request.email
        ).first()
        
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create new user profile
        user_profile = UserProfile(
            username=request.username,
            email=request.email,
            display_name=request.display_name or request.username,
            bio=request.bio,
            avatar_url=request.avatar_url,
            preferred_forms=request.preferred_forms,
            favorite_themes=request.favorite_themes,
            discovery_style=request.discovery_style
        )
        
        db.add(user_profile)
        db.commit()
        db.refresh(user_profile)
        
        return ProfileResponse(
            id=str(user_profile.id),
            username=user_profile.username,
            display_name=user_profile.display_name,
            bio=user_profile.bio,
            avatar_url=user_profile.avatar_url,
            primary_muse=user_profile.primary_muse,
            secondary_muse=user_profile.secondary_muse,
            harmonic_blend=user_profile.harmonic_blend,
            sacred_ratios=user_profile.sacred_ratios,
            preferred_forms=user_profile.preferred_forms,
            favorite_themes=user_profile.favorite_themes,
            discovery_style=user_profile.discovery_style,
            social_metrics={
                'followers_count': user_profile.followers_count,
                'following_count': user_profile.following_count,
                'discoveries_count': user_profile.discoveries_count,
                'shared_creations_count': user_profile.shared_creations_count
            },
            privacy_settings={
                'profile_visibility': user_profile.profile_visibility,
                'signature_sharing': user_profile.signature_sharing,
                'discovery_sharing': user_profile.discovery_sharing
            },
            created_at=user_profile.created_at,
            last_active_at=user_profile.last_active_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user profile")


@router.get("/profiles/{user_id}", response_model=ProfileResponse)
async def get_user_profile(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Get user profile by ID
    
    Retrieves a user profile with their frequency signature data,
    social metrics, and community engagement information.
    """
    try:
        user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return ProfileResponse(
            id=str(user_profile.id),
            username=user_profile.username,
            display_name=user_profile.display_name,
            bio=user_profile.bio,
            avatar_url=user_profile.avatar_url,
            primary_muse=user_profile.primary_muse,
            secondary_muse=user_profile.secondary_muse,
            harmonic_blend=user_profile.harmonic_blend,
            sacred_ratios=user_profile.sacred_ratios,
            preferred_forms=user_profile.preferred_forms,
            favorite_themes=user_profile.favorite_themes,
            discovery_style=user_profile.discovery_style,
            social_metrics={
                'followers_count': user_profile.followers_count,
                'following_count': user_profile.following_count,
                'discoveries_count': user_profile.discoveries_count,
                'shared_creations_count': user_profile.shared_creations_count,
                'total_likes_received': user_profile.total_likes_received,
                'total_comments_received': user_profile.total_comments_received,
                'resonance_strength': user_profile.resonance_strength,
                'discovery_streak': user_profile.discovery_streak,
                'collaboration_count': user_profile.collaboration_count
            },
            privacy_settings={
                'profile_visibility': user_profile.profile_visibility,
                'signature_sharing': user_profile.signature_sharing,
                'discovery_sharing': user_profile.discovery_sharing
            },
            created_at=user_profile.created_at,
            last_active_at=user_profile.last_active_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user profile")


@router.put("/profiles/{user_id}", response_model=ProfileResponse)
async def update_user_profile(
    user_id: str,
    request: ProfileUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update user profile information
    
    Updates user profile with new information while preserving
    frequency signature data and social metrics.
    """
    try:
        user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Update fields if provided
        if request.display_name is not None:
            user_profile.display_name = request.display_name
        if request.bio is not None:
            user_profile.bio = request.bio
        if request.avatar_url is not None:
            user_profile.avatar_url = request.avatar_url
        if request.preferred_forms is not None:
            user_profile.preferred_forms = request.preferred_forms
        if request.favorite_themes is not None:
            user_profile.favorite_themes = request.favorite_themes
        if request.discovery_style is not None:
            user_profile.discovery_style = request.discovery_style
        if request.profile_visibility is not None:
            user_profile.profile_visibility = request.profile_visibility
        if request.signature_sharing is not None:
            user_profile.signature_sharing = request.signature_sharing
        if request.discovery_sharing is not None:
            user_profile.discovery_sharing = request.discovery_sharing
        
        # Update last active time
        user_profile.last_active_at = datetime.utcnow()
        
        db.commit()
        db.refresh(user_profile)
        
        return ProfileResponse(
            id=str(user_profile.id),
            username=user_profile.username,
            display_name=user_profile.display_name,
            bio=user_profile.bio,
            avatar_url=user_profile.avatar_url,
            primary_muse=user_profile.primary_muse,
            secondary_muse=user_profile.secondary_muse,
            harmonic_blend=user_profile.harmonic_blend,
            sacred_ratios=user_profile.sacred_ratios,
            preferred_forms=user_profile.preferred_forms,
            favorite_themes=user_profile.favorite_themes,
            discovery_style=user_profile.discovery_style,
            social_metrics={
                'followers_count': user_profile.followers_count,
                'following_count': user_profile.following_count,
                'discoveries_count': user_profile.discoveries_count,
                'shared_creations_count': user_profile.shared_creations_count,
                'total_likes_received': user_profile.total_likes_received,
                'total_comments_received': user_profile.total_comments_received,
                'resonance_strength': user_profile.resonance_strength,
                'discovery_streak': user_profile.discovery_streak,
                'collaboration_count': user_profile.collaboration_count
            },
            privacy_settings={
                'profile_visibility': user_profile.profile_visibility,
                'signature_sharing': user_profile.signature_sharing,
                'discovery_sharing': user_profile.discovery_sharing
            },
            created_at=user_profile.created_at,
            last_active_at=user_profile.last_active_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user profile")


@router.post("/creations/share", response_model=CreationResponse)
async def share_creation(
    request: CreationShareRequest,
    user_id: str,
    db: Session = Depends(get_db),
    sem_engine: SemanticProjectionEngine = Depends(get_semantic_engine_dependency)
):
    """
    Share a creation with the community
    
    Shares a creative work with the MUSE community, including analysis
    of its mathematical fitness, semantic coherence, and archetypal alignment.
    """
    try:
        # Validate user exists
        user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Analyze the creation
        semantic_analysis = sem_engine.get_semantic_analysis(request.content)
        
        # Calculate mathematical fitness (simplified)
        mathematical_fitness = _calculate_mathematical_fitness(request.content, semantic_analysis)
        
        # Calculate semantic coherence
        semantic_coherence = semantic_analysis.get('emotional_vector', [0.5])[0] if semantic_analysis.get('emotional_vector') else 0.5
        
        # Calculate archetypal alignment
        archetypal_alignment = _calculate_archetypal_alignment(user_profile, semantic_analysis)
        
        # Calculate sacred geometry score
        sacred_geometry_score = _calculate_sacred_geometry_score(semantic_analysis)
        
        # Create community creation
        creation = CommunityCreation(
            creator_id=user_id,
            title=request.title,
            content=request.content,
            content_preview=request.content[:500] + "..." if len(request.content) > 500 else request.content,
            form_type=request.form_type,
            primary_theme=request.primary_theme,
            secondary_themes=request.secondary_themes,
            mathematical_fitness=mathematical_fitness,
            semantic_coherence=semantic_coherence,
            archetypal_alignment=archetypal_alignment,
            sacred_geometry_score=sacred_geometry_score,
            visibility=request.visibility,
            creator_signature_snapshot=user_profile.harmonic_blend
        )
        
        db.add(creation)
        
        # Update user metrics
        user_profile.shared_creations_count += 1
        user_profile.discoveries_count += 1
        user_profile.last_active_at = datetime.utcnow()
        
        db.commit()
        db.refresh(creation)
        
        return CreationResponse(
            id=str(creation.id),
            creator_id=str(creation.creator_id),
            title=creation.title,
            content=creation.content,
            content_preview=creation.content_preview,
            form_type=creation.form_type,
            primary_theme=creation.primary_theme,
            secondary_themes=creation.secondary_themes,
            mathematical_analysis={
                'mathematical_fitness': creation.mathematical_fitness,
                'semantic_coherence': creation.semantic_coherence,
                'sacred_geometry_score': creation.sacred_geometry_score,
                'archetypal_alignment': creation.archetypal_alignment
            },
            sacred_geometry={
                'sacred_constant': creation.sacred_constant,
                'entropy_fingerprint': creation.entropy_fingerprint
            },
            community_metrics={
                'likes_count': creation.likes_count,
                'comments_count': creation.comments_count,
                'shares_count': creation.shares_count,
                'resonance_score': creation.resonance_score
            },
            discovery_metrics={
                'discovery_time_seconds': creation.discovery_time_seconds,
                'iteration_count': creation.iteration_count,
                'constraint_satisfaction': creation.constraint_satisfaction
            },
            visibility_data={
                'visibility': creation.visibility,
                'featured': creation.featured,
                'curator_highlighted': creation.curator_highlighted
            },
            collaborative_data={
                'is_collaborative': creation.is_collaborative,
                'collaboration_session_id': str(creation.collaboration_session_id) if creation.collaboration_session_id else None
            },
            timestamps={
                'created_at': creation.created_at.isoformat(),
                'updated_at': creation.updated_at.isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error sharing creation: {e}")
        raise HTTPException(status_code=500, detail="Failed to share creation")


@router.get("/gallery", response_model=GalleryResponse)
async def get_community_gallery(
    archetype: Optional[str] = Query(default=None, description="Filter by archetype"),
    form_type: Optional[str] = Query(default=None, description="Filter by form type"),
    theme: Optional[str] = Query(default=None, description="Filter by theme"),
    sacred_constant: Optional[str] = Query(default=None, description="Filter by sacred constant"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of items per page"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    Get community gallery with resonance-based filtering
    
    Returns a curated gallery of community creations with filters
    for archetypal alignment, sacred geometry, and thematic content.
    """
    try:
        # Build query with filters
        query = db.query(CommunityCreation).filter(
            CommunityCreation.visibility == 'public'
        )
        
        if archetype:
            # Filter by archetype in creator signature
            query = query.filter(
                CommunityCreation.creator_signature_snapshot.op('?')(archetype)
            )
        
        if form_type:
            query = query.filter(CommunityCreation.form_type == form_type)
        
        if theme:
            query = query.filter(CommunityCreation.primary_theme == theme)
        
        if sacred_constant:
            query = query.filter(CommunityCreation.sacred_constant == sacred_constant)
        
        # Get total count for pagination
        total_count = query.count()
        
        # Get creations with pagination
        creations = query.order_by(
            desc(CommunityCreation.resonance_score),
            desc(CommunityCreation.created_at)
        ).offset(offset).limit(limit).all()
        
        # Format creations
        formatted_creations = []
        for creation in creations:
            creator = db.query(UserProfile).filter(UserProfile.id == creation.creator_id).first()
            
            formatted_creations.append({
                'creation': creation.to_dict(),
                'creator': {
                    'id': str(creator.id),
                    'username': creator.username,
                    'display_name': creator.display_name
                } if creator else None,
                'engagement': {
                    'likes_count': creation.likes_count,
                    'comments_count': creation.comments_count,
                    'resonance_score': creation.resonance_score
                }
            })
        
        # Calculate pagination info
        has_next = offset + limit < total_count
        has_previous = offset > 0
        
        return GalleryResponse(
            creations=formatted_creations,
            filters={
                'archetype': archetype,
                'form_type': form_type,
                'theme': theme,
                'sacred_constant': sacred_constant
            },
            pagination={
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_next': has_next,
                'has_previous': has_previous,
                'next_offset': offset + limit if has_next else None,
                'previous_offset': max(0, offset - limit) if has_previous else None
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting community gallery: {e}")
        raise HTTPException(status_code=500, detail="Failed to get community gallery")


@router.get("/feed/{user_id}", response_model=FeedResponse)
async def get_personalized_feed(
    user_id: str,
    feed_type: str = Query(default="personalized", description="Type of feed"),
    limit: int = Query(default=20, ge=1, le=50, description="Number of items per page"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    curator: CommunityCurator = Depends(get_community_curator_dependency)
):
    """
    Get personalized content feed based on user's frequency signature
    
    Returns a curated feed of content based on archetypal resonance,
    sacred geometry alignment, and community interactions.
    """
    try:
        # Validate user exists
        user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Map feed type to curation mode
        curation_mode_map = {
            'personalized': CurationMode.RESONANCE_BASED,
            'resonant': CurationMode.RESONANCE_BASED,
            'archetypal': CurationMode.ARCHETYPAL_AFFINITY,
            'geometric': CurationMode.SACRED_GEOMETRY,
            'collaborative': CurationMode.COLLABORATIVE,
            'trending': CurationMode.TRENDING,
            'discovery': CurationMode.DISCOVERY
        }
        
        curation_mode = curation_mode_map.get(feed_type, CurationMode.RESONANCE_BASED)
        
        # Get curated feed
        if feed_type == "resonant":
            feed_items = curator.get_resonant_discoveries(db, user_id, limit=limit)
        else:
            feed_items = curator.curate_personalized_feed(
                db, user_id, limit=limit, offset=offset, mode=curation_mode
            )
        
        # Format feed items
        formatted_items = []
        for item in feed_items:
            formatted_items.append({
                'creation': item.creation,
                'creator': item.creator_profile,
                'curation': {
                    'relevance_score': item.curation_result.relevance_score,
                    'resonance_score': item.curation_result.resonance_score,
                    'curation_reason': item.curation_result.curation_reason,
                    'recommendation_strength': item.recommendation_strength
                },
                'engagement': item.engagement_metrics,
                'explanation': item.resonance_explanation
            })
        
        return FeedResponse(
            items=formatted_items,
            pagination={
                'limit': limit,
                'offset': offset,
                'has_next': len(formatted_items) == limit,
                'has_previous': offset > 0
            },
            curation_info={
                'feed_type': feed_type,
                'curation_mode': curation_mode.value,
                'personalization_strength': user_profile.resonance_strength,
                'archetypal_focus': user_profile.primary_muse
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting personalized feed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get personalized feed")


@router.get("/kindred/{user_id}", response_model=List[CommunityMatch])
async def find_kindred_spirits(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=20, description="Number of matches to return"),
    resonance_threshold: float = Query(default=0.7, ge=0.0, le=1.0, description="Resonance threshold"),
    db: Session = Depends(get_db),
    resonance_matcher: ResonanceMatcher = Depends(get_resonance_matcher_dependency)
):
    """
    Find kindred spirits with similar frequency signatures
    
    Finds users with exceptionally high archetypal resonance and
    compatibility for creative collaboration.
    """
    try:
        # Validate user exists
        user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Find kindred spirits
        kindred_spirits = resonance_matcher.find_kindred_spirits(
            db, user_id, limit=limit, resonance_threshold=resonance_threshold
        )
        
        # Format results
        formatted_matches = []
        for match in kindred_spirits:
            formatted_matches.append(CommunityMatch(
                user_id=match.user_id,
                username=match.username,
                display_name=match.display_name,
                resonance_score=match.resonance_score,
                compatibility_type=match.compatibility_type,
                shared_archetypes=match.shared_archetypes,
                complementary_traits=match.complementary_traits,
                recommended_collaboration=match.recommended_collaboration
            ))
        
        return formatted_matches
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding kindred spirits: {e}")
        raise HTTPException(status_code=500, detail="Failed to find kindred spirits")


@router.post("/follow/{target_user_id}")
async def follow_user(
    target_user_id: str,
    user_id: str,
    db: Session = Depends(get_db),
    resonance_matcher: ResonanceMatcher = Depends(get_resonance_matcher_dependency)
):
    """
    Follow another user
    
    Creates a following relationship based on archetypal resonance
    and shared creative interests.
    """
    try:
        # Validate both users exist
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        target_user = db.query(UserProfile).filter(UserProfile.id == target_user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not target_user:
            raise HTTPException(status_code=404, detail="Target user not found")
        
        if user_id == target_user_id:
            raise HTTPException(status_code=400, detail="Cannot follow yourself")
        
        # Check if already following
        existing_follow = db.query(Follow).filter(
            Follow.follower_id == user_id,
            Follow.followed_id == target_user_id
        ).first()
        
        if existing_follow:
            raise HTTPException(status_code=400, detail="Already following user")
        
        # Calculate resonance at follow time
        try:
            resonance_result = resonance_matcher.calculate_user_resonance(
                db, user_id, target_user_id
            )
            resonance_at_follow = resonance_result.overall_resonance
        except Exception as e:
            logger.warning(f"Error calculating resonance for follow: {e}")
            resonance_at_follow = 0.5  # Default resonance
        
        # Create follow relationship
        follow = Follow(
            follower_id=user_id,
            followed_id=target_user_id,
            resonance_at_follow=resonance_at_follow,
            follow_reason="manual"
        )
        
        db.add(follow)
        
        # Update user metrics
        user.following_count += 1
        target_user.followers_count += 1
        
        db.commit()
        
        return {
            'message': 'Successfully followed user',
            'user_id': user_id,
            'target_user_id': target_user_id,
            'resonance_at_follow': resonance_at_follow
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error following user: {e}")
        raise HTTPException(status_code=500, detail="Failed to follow user")


@router.post("/unfollow/{target_user_id}")
async def unfollow_user(
    target_user_id: str,
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Unfollow a user
    
    Removes a following relationship while preserving interaction history.
    """
    try:
        # Check if follow relationship exists
        follow = db.query(Follow).filter(
            Follow.follower_id == user_id,
            Follow.followed_id == target_user_id
        ).first()
        
        if not follow:
            raise HTTPException(status_code=404, detail="Not following user")
        
        # Remove follow relationship
        db.delete(follow)
        
        # Update user metrics
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        target_user = db.query(UserProfile).filter(UserProfile.id == target_user_id).first()
        
        if user:
            user.following_count = max(0, user.following_count - 1)
        if target_user:
            target_user.followers_count = max(0, target_user.followers_count - 1)
        
        db.commit()
        
        return {
            'message': 'Successfully unfollowed user',
            'user_id': user_id,
            'target_user_id': target_user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error unfollowing user: {e}")
        raise HTTPException(status_code=500, detail="Failed to unfollow user")


@router.post("/creations/{creation_id}/like")
async def like_creation(
    creation_id: str,
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Like a creation
    
    Adds a like to a creation, contributing to its resonance score
    and community engagement metrics.
    """
    try:
        # Validate creation exists
        creation = db.query(CommunityCreation).filter(
            CommunityCreation.id == creation_id
        ).first()
        
        if not creation:
            raise HTTPException(status_code=404, detail="Creation not found")
        
        # Check if already liked
        existing_like = db.query(Like).filter(
            Like.user_id == user_id,
            Like.creation_id == creation_id
        ).first()
        
        if existing_like:
            raise HTTPException(status_code=400, detail="Already liked creation")
        
        # Create like
        like = Like(
            user_id=user_id,
            creation_id=creation_id,
            resonance_strength=1.0
        )
        
        db.add(like)
        
        # Update creation metrics
        creation.likes_count += 1
        creation.resonance_score = _calculate_resonance_score(creation)
        
        # Update creator metrics
        creator = db.query(UserProfile).filter(UserProfile.id == creation.creator_id).first()
        if creator:
            creator.total_likes_received += 1
        
        db.commit()
        
        return {
            'message': 'Successfully liked creation',
            'creation_id': creation_id,
            'user_id': user_id,
            'new_likes_count': creation.likes_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error liking creation: {e}")
        raise HTTPException(status_code=500, detail="Failed to like creation")


@router.post("/creations/{creation_id}/comment")
async def comment_on_creation(
    creation_id: str,
    user_id: str,
    request: CommentCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Comment on a creation
    
    Adds a comment to a creation, fostering community discussion
    and creative dialogue.
    """
    try:
        # Validate creation exists
        creation = db.query(CommunityCreation).filter(
            CommunityCreation.id == creation_id
        ).first()
        
        if not creation:
            raise HTTPException(status_code=404, detail="Creation not found")
        
        # Validate parent comment if specified
        parent_comment = None
        thread_depth = 0
        
        if request.parent_comment_id:
            parent_comment = db.query(Comment).filter(
                Comment.id == request.parent_comment_id
            ).first()
            
            if not parent_comment:
                raise HTTPException(status_code=404, detail="Parent comment not found")
            
            thread_depth = parent_comment.thread_depth + 1
            
            # Limit threading depth
            if thread_depth > 3:
                raise HTTPException(status_code=400, detail="Maximum thread depth exceeded")
        
        # Create comment
        comment = Comment(
            creation_id=creation_id,
            author_id=user_id,
            content=request.content,
            parent_comment_id=request.parent_comment_id,
            thread_depth=thread_depth
        )
        
        db.add(comment)
        
        # Update creation metrics
        creation.comments_count += 1
        
        # Update creator metrics
        creator = db.query(UserProfile).filter(UserProfile.id == creation.creator_id).first()
        if creator:
            creator.total_comments_received += 1
        
        db.commit()
        db.refresh(comment)
        
        return {
            'message': 'Successfully commented on creation',
            'comment_id': str(comment.id),
            'creation_id': creation_id,
            'user_id': user_id,
            'new_comments_count': creation.comments_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error commenting on creation: {e}")
        raise HTTPException(status_code=500, detail="Failed to comment on creation")


# Helper functions

def _calculate_mathematical_fitness(content: str, semantic_analysis: Dict[str, Any]) -> float:
    """Calculate mathematical fitness score for content"""
    # Use semantic analysis metrics
    structural_metrics = semantic_analysis.get('structural_metrics', {})
    
    # Combine various mathematical measures
    phi_resonance = structural_metrics.get('phi_resonance', 0.0)
    pi_resonance = structural_metrics.get('pi_resonance', 0.0)
    fibonacci_resonance = structural_metrics.get('fibonacci_resonance', 0.0)
    
    # Calculate weighted average
    mathematical_fitness = (
        phi_resonance * 0.4 +
        pi_resonance * 0.3 +
        fibonacci_resonance * 0.3
    )
    
    return min(1.0, mathematical_fitness)


def _calculate_archetypal_alignment(user_profile: UserProfile, semantic_analysis: Dict[str, Any]) -> float:
    """Calculate archetypal alignment score"""
    if not user_profile.harmonic_blend:
        return 0.5
    
    # Get archetypal distribution from semantic analysis
    archetypal_distribution = semantic_analysis.get('archetypal_distribution', {})
    
    # Calculate alignment with user's harmonic blend
    alignment = 0.0
    total_weight = 0.0
    
    for archetype, user_strength in user_profile.harmonic_blend.items():
        content_strength = archetypal_distribution.get(archetype, 0.0)
        alignment += user_strength * content_strength
        total_weight += user_strength
    
    return alignment / total_weight if total_weight > 0 else 0.5


def _calculate_sacred_geometry_score(semantic_analysis: Dict[str, Any]) -> float:
    """Calculate sacred geometry score"""
    structural_metrics = semantic_analysis.get('structural_metrics', {})
    
    # Get sacred geometry metrics
    phi_resonance = structural_metrics.get('phi_resonance', 0.0)
    pi_resonance = structural_metrics.get('pi_resonance', 0.0)
    fibonacci_resonance = structural_metrics.get('fibonacci_resonance', 0.0)
    
    # Calculate sacred geometry score
    sacred_geometry_score = (phi_resonance + pi_resonance + fibonacci_resonance) / 3
    
    return min(1.0, sacred_geometry_score)


def _calculate_resonance_score(creation: CommunityCreation) -> float:
    """Calculate resonance score for a creation based on community engagement"""
    # Simple resonance calculation based on engagement
    base_score = creation.mathematical_fitness * 0.3 + creation.semantic_coherence * 0.3 + creation.archetypal_alignment * 0.4
    
    # Add engagement bonus
    engagement_factor = min(1.0, (creation.likes_count + creation.comments_count * 2) / 100)
    
    return min(1.0, base_score + engagement_factor * 0.2)