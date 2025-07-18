"""
Community Data Models for MUSE Platform

This module defines the SQLAlchemy models for the frequency-based social network
that connects users based on their archetypal resonance and mathematical 
creative discovery patterns.
"""

from sqlalchemy import Column, String, JSON, DateTime, ForeignKey, Integer, Float, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


Base = declarative_base()


class UserProfile(Base):
    """
    User profile with MUSE frequency signature and social metrics
    """
    __tablename__ = "user_profiles"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False)
    display_name = Column(String(100))
    bio = Column(Text)
    avatar_url = Column(String(255))
    
    # MUSE signature data (cached for performance)
    current_signature_id = Column(UUID(as_uuid=True), ForeignKey("frequency_signatures.id"))
    primary_muse = Column(String(30), index=True)  # For filtering and matching
    secondary_muse = Column(String(30))
    
    # Cached signature data for quick access
    harmonic_blend = Column(JSON)  # {"ERATO": 0.8, "CALLIOPE": 0.3, ...}
    sacred_ratios = Column(JSON)   # {"phi": 0.9, "pi": 0.4, "fibonacci": 0.7, ...}
    spiral_coordinates = Column(JSON)  # {"x": 1.2, "y": 0.8, "z": 0.3, "radius": 1.5, ...}
    
    # Creative preferences
    preferred_forms = Column(ARRAY(String))  # ["sonnet", "haiku", "villanelle"]
    favorite_themes = Column(ARRAY(String))  # ["love", "nature", "cosmos"]
    discovery_style = Column(String(50))     # "mathematical", "intuitive", "experimental"
    
    # Social metrics
    followers_count = Column(Integer, default=0)
    following_count = Column(Integer, default=0)
    discoveries_count = Column(Integer, default=0)
    shared_creations_count = Column(Integer, default=0)
    total_likes_received = Column(Integer, default=0)
    total_comments_received = Column(Integer, default=0)
    resonance_strength = Column(Float, default=0.0)  # Overall resonance with community
    
    # Engagement metrics
    last_active_at = Column(DateTime, default=datetime.utcnow)
    discovery_streak = Column(Integer, default=0)  # Consecutive days of discovery
    collaboration_count = Column(Integer, default=0)
    
    # Privacy settings
    profile_visibility = Column(String(20), default="public")  # public, followers, private
    signature_sharing = Column(Boolean, default=True)
    discovery_sharing = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    frequency_signatures = relationship("FrequencySignature", back_populates="user")
    creations = relationship("CommunityCreation", back_populates="creator")
    comments = relationship("Comment", back_populates="author")
    likes = relationship("Like", back_populates="user")
    
    # Following relationships
    followers = relationship("Follow", foreign_keys="Follow.followed_id", back_populates="followed")
    following = relationship("Follow", foreign_keys="Follow.follower_id", back_populates="follower")
    
    # Collaborative sessions
    collaborative_sessions = relationship("CollaborativeSession", back_populates="creator")
    session_participants = relationship("SessionParticipant", back_populates="user")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_primary_muse', primary_muse),
        Index('idx_user_resonance', resonance_strength),
        Index('idx_user_activity', last_active_at),
        Index('idx_user_discovery_count', discoveries_count),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user profile to dictionary"""
        return {
            'id': str(self.id),
            'username': self.username,
            'display_name': self.display_name,
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'primary_muse': self.primary_muse,
            'secondary_muse': self.secondary_muse,
            'harmonic_blend': self.harmonic_blend,
            'sacred_ratios': self.sacred_ratios,
            'spiral_coordinates': self.spiral_coordinates,
            'preferred_forms': self.preferred_forms,
            'favorite_themes': self.favorite_themes,
            'discovery_style': self.discovery_style,
            'social_metrics': {
                'followers_count': self.followers_count,
                'following_count': self.following_count,
                'discoveries_count': self.discoveries_count,
                'shared_creations_count': self.shared_creations_count,
                'total_likes_received': self.total_likes_received,
                'total_comments_received': self.total_comments_received,
                'resonance_strength': self.resonance_strength,
                'discovery_streak': self.discovery_streak,
                'collaboration_count': self.collaboration_count
            },
            'privacy_settings': {
                'profile_visibility': self.profile_visibility,
                'signature_sharing': self.signature_sharing,
                'discovery_sharing': self.discovery_sharing
            },
            'created_at': self.created_at.isoformat(),
            'last_active_at': self.last_active_at.isoformat()
        }


class FrequencySignature(Base):
    """
    Complete frequency signature data for mathematical creativity
    """
    __tablename__ = "frequency_signatures"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Signature metadata
    signature_version = Column(String(20), default="1.0")
    is_active = Column(Boolean, default=True)
    
    # Core archetypal data
    harmonic_blend = Column(JSON, nullable=False)  # Full 12-muse blend
    sacred_ratios = Column(JSON, nullable=False)   # Sacred constant affinities
    spiral_coordinates = Column(JSON, nullable=False)  # 3D archetypal space position
    
    # Derived metrics
    primary_muse = Column(String(30), nullable=False, index=True)
    secondary_muse = Column(String(30))
    entropy_seed = Column(String(64))  # Hardware entropy fingerprint
    
    # Signature characteristics
    specialization_index = Column(Float)  # How specialized vs generalized
    diversity_index = Column(Float)       # How diverse across archetypes
    coherence_score = Column(Float)       # Internal consistency
    uniqueness_score = Column(Float)      # Rarity in population
    
    # Tuning history
    tuning_count = Column(Integer, default=0)
    last_tuned_at = Column(DateTime)
    tuning_history = Column(JSON)  # History of adjustments
    
    # Performance metrics
    discovery_success_rate = Column(Float, default=0.0)
    average_fitness_score = Column(Float, default=0.0)
    user_satisfaction_score = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("UserProfile", back_populates="frequency_signatures")
    
    # Indexes
    __table_args__ = (
        Index('idx_signature_primary_muse', primary_muse),
        Index('idx_signature_user_active', user_id, is_active),
        Index('idx_signature_uniqueness', uniqueness_score),
        Index('idx_signature_success_rate', discovery_success_rate),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert frequency signature to dictionary"""
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'signature_version': self.signature_version,
            'is_active': self.is_active,
            'harmonic_blend': self.harmonic_blend,
            'sacred_ratios': self.sacred_ratios,
            'spiral_coordinates': self.spiral_coordinates,
            'primary_muse': self.primary_muse,
            'secondary_muse': self.secondary_muse,
            'entropy_seed': self.entropy_seed,
            'characteristics': {
                'specialization_index': self.specialization_index,
                'diversity_index': self.diversity_index,
                'coherence_score': self.coherence_score,
                'uniqueness_score': self.uniqueness_score
            },
            'tuning_info': {
                'tuning_count': self.tuning_count,
                'last_tuned_at': self.last_tuned_at.isoformat() if self.last_tuned_at else None,
                'tuning_history': self.tuning_history
            },
            'performance_metrics': {
                'discovery_success_rate': self.discovery_success_rate,
                'average_fitness_score': self.average_fitness_score,
                'user_satisfaction_score': self.user_satisfaction_score
            },
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class CommunityCreation(Base):
    """
    Shared creative discovery in the community
    """
    __tablename__ = "community_creations"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    creator_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Content data
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    content_preview = Column(String(500))  # First few lines for feed display
    
    # Discovery metadata
    discovery_coordinates = Column(JSON)  # Complete discovery path
    creator_signature_snapshot = Column(JSON)  # Signature at time of creation
    
    # Form and structure
    form_type = Column(String(50), index=True)  # sonnet, haiku, villanelle, etc.
    structure_metadata = Column(JSON)  # Syllable counts, rhyme scheme, etc.
    
    # Mathematical analysis
    mathematical_fitness = Column(Float, default=0.0)
    semantic_coherence = Column(Float, default=0.0)
    sacred_geometry_score = Column(Float, default=0.0)
    archetypal_alignment = Column(Float, default=0.0)
    
    # Thematic data
    primary_theme = Column(String(50), index=True)
    secondary_themes = Column(ARRAY(String))
    emotional_palette = Column(ARRAY(String))
    
    # Sacred geometry details
    sacred_constant = Column(String(20))  # phi, pi, fibonacci, etc.
    entropy_fingerprint = Column(String(64))  # Hardware entropy signature
    
    # Community engagement
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    shares_count = Column(Integer, default=0)
    resonance_score = Column(Float, default=0.0)  # Community resonance
    
    # Discovery metrics
    discovery_time_seconds = Column(Float)  # Time taken to discover
    iteration_count = Column(Integer, default=1)  # Discovery iterations
    constraint_satisfaction = Column(Float, default=0.0)
    
    # Visibility and sharing
    visibility = Column(String(20), default="public")  # public, followers, private
    featured = Column(Boolean, default=False)
    curator_highlighted = Column(Boolean, default=False)
    
    # Collaborative data
    is_collaborative = Column(Boolean, default=False)
    collaboration_session_id = Column(UUID(as_uuid=True), ForeignKey("collaborative_sessions.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    creator = relationship("UserProfile", back_populates="creations")
    comments = relationship("Comment", back_populates="creation", cascade="all, delete-orphan")
    likes = relationship("Like", back_populates="creation", cascade="all, delete-orphan")
    collaboration_session = relationship("CollaborativeSession", back_populates="creations")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_creation_creator', creator_id),
        Index('idx_creation_form_type', form_type),
        Index('idx_creation_theme', primary_theme),
        Index('idx_creation_resonance', resonance_score),
        Index('idx_creation_fitness', mathematical_fitness),
        Index('idx_creation_featured', featured),
        Index('idx_creation_created_at', created_at),
        Index('idx_creation_visibility', visibility),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert creation to dictionary"""
        return {
            'id': str(self.id),
            'creator_id': str(self.creator_id),
            'title': self.title,
            'content': self.content,
            'content_preview': self.content_preview,
            'discovery_coordinates': self.discovery_coordinates,
            'creator_signature_snapshot': self.creator_signature_snapshot,
            'form_data': {
                'form_type': self.form_type,
                'structure_metadata': self.structure_metadata
            },
            'mathematical_analysis': {
                'mathematical_fitness': self.mathematical_fitness,
                'semantic_coherence': self.semantic_coherence,
                'sacred_geometry_score': self.sacred_geometry_score,
                'archetypal_alignment': self.archetypal_alignment
            },
            'thematic_data': {
                'primary_theme': self.primary_theme,
                'secondary_themes': self.secondary_themes,
                'emotional_palette': self.emotional_palette
            },
            'sacred_geometry': {
                'sacred_constant': self.sacred_constant,
                'entropy_fingerprint': self.entropy_fingerprint
            },
            'community_metrics': {
                'likes_count': self.likes_count,
                'comments_count': self.comments_count,
                'shares_count': self.shares_count,
                'resonance_score': self.resonance_score
            },
            'discovery_metrics': {
                'discovery_time_seconds': self.discovery_time_seconds,
                'iteration_count': self.iteration_count,
                'constraint_satisfaction': self.constraint_satisfaction
            },
            'visibility_data': {
                'visibility': self.visibility,
                'featured': self.featured,
                'curator_highlighted': self.curator_highlighted
            },
            'collaborative_data': {
                'is_collaborative': self.is_collaborative,
                'collaboration_session_id': str(self.collaboration_session_id) if self.collaboration_session_id else None
            },
            'timestamps': {
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat()
            }
        }


class Comment(Base):
    """
    Comments on community creations
    """
    __tablename__ = "comments"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    creation_id = Column(UUID(as_uuid=True), ForeignKey("community_creations.id"), nullable=False)
    author_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Comment content
    content = Column(Text, nullable=False)
    
    # Threading support
    parent_comment_id = Column(UUID(as_uuid=True), ForeignKey("comments.id"))
    thread_depth = Column(Integer, default=0)
    
    # Engagement
    likes_count = Column(Integer, default=0)
    
    # Resonance analysis
    resonance_with_creation = Column(Float, default=0.0)
    archetypal_alignment = Column(Float, default=0.0)
    
    # Moderation
    is_flagged = Column(Boolean, default=False)
    is_hidden = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    creation = relationship("CommunityCreation", back_populates="comments")
    author = relationship("UserProfile", back_populates="comments")
    parent_comment = relationship("Comment", remote_side=[id])
    
    # Indexes
    __table_args__ = (
        Index('idx_comment_creation', creation_id),
        Index('idx_comment_author', author_id),
        Index('idx_comment_created_at', created_at),
        Index('idx_comment_parent', parent_comment_id),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert comment to dictionary"""
        return {
            'id': str(self.id),
            'creation_id': str(self.creation_id),
            'author_id': str(self.author_id),
            'content': self.content,
            'parent_comment_id': str(self.parent_comment_id) if self.parent_comment_id else None,
            'thread_depth': self.thread_depth,
            'likes_count': self.likes_count,
            'resonance_with_creation': self.resonance_with_creation,
            'archetypal_alignment': self.archetypal_alignment,
            'is_flagged': self.is_flagged,
            'is_hidden': self.is_hidden,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class Like(Base):
    """
    Likes on creations and comments
    """
    __tablename__ = "likes"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Target (creation or comment)
    creation_id = Column(UUID(as_uuid=True), ForeignKey("community_creations.id"))
    comment_id = Column(UUID(as_uuid=True), ForeignKey("comments.id"))
    
    # Resonance data
    resonance_strength = Column(Float, default=1.0)  # How much they resonated
    archetypal_match = Column(Float, default=0.0)    # Archetypal alignment
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("UserProfile", back_populates="likes")
    creation = relationship("CommunityCreation", back_populates="likes")
    comment = relationship("Comment")
    
    # Unique constraint: user can only like each item once
    __table_args__ = (
        Index('idx_like_user_creation', user_id, creation_id),
        Index('idx_like_user_comment', user_id, comment_id),
        Index('idx_like_created_at', created_at),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert like to dictionary"""
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'creation_id': str(self.creation_id) if self.creation_id else None,
            'comment_id': str(self.comment_id) if self.comment_id else None,
            'resonance_strength': self.resonance_strength,
            'archetypal_match': self.archetypal_match,
            'created_at': self.created_at.isoformat()
        }


class Follow(Base):
    """
    User following relationships
    """
    __tablename__ = "follows"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    follower_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    followed_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Relationship metadata
    resonance_at_follow = Column(Float, default=0.0)  # Resonance score when followed
    follow_reason = Column(String(50))  # "resonance", "discovery", "collaboration"
    
    # Engagement tracking
    interactions_count = Column(Integer, default=0)
    last_interaction_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    follower = relationship("UserProfile", foreign_keys=[follower_id], back_populates="following")
    followed = relationship("UserProfile", foreign_keys=[followed_id], back_populates="followers")
    
    # Unique constraint and indexes
    __table_args__ = (
        Index('idx_follow_follower', follower_id),
        Index('idx_follow_followed', followed_id),
        Index('idx_follow_resonance', resonance_at_follow),
        Index('idx_follow_created_at', created_at),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert follow to dictionary"""
        return {
            'id': str(self.id),
            'follower_id': str(self.follower_id),
            'followed_id': str(self.followed_id),
            'resonance_at_follow': self.resonance_at_follow,
            'follow_reason': self.follow_reason,
            'interactions_count': self.interactions_count,
            'last_interaction_at': self.last_interaction_at.isoformat() if self.last_interaction_at else None,
            'created_at': self.created_at.isoformat()
        }


class CollaborativeSession(Base):
    """
    Multi-user creative collaboration sessions
    """
    __tablename__ = "collaborative_sessions"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    creator_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Session metadata
    title = Column(String(200), nullable=False)
    description = Column(Text)
    session_type = Column(String(50), default="open")  # open, invite_only, private
    
    # Creative constraints
    target_form = Column(String(50))  # sonnet, haiku, etc.
    theme = Column(String(100))
    sacred_constant = Column(String(20))
    
    # Collaboration settings
    max_participants = Column(Integer, default=10)
    current_participants = Column(Integer, default=1)
    
    # Session state
    status = Column(String(30), default="active")  # active, paused, completed, cancelled
    
    # Combined entropy and resonance
    combined_entropy_seed = Column(String(128))  # Merged entropy from all participants
    collective_resonance = Column(Float, default=0.0)
    
    # Discovery tracking
    iterations_completed = Column(Integer, default=0)
    current_discovery = Column(JSON)  # Work in progress
    
    # Results
    final_creation_id = Column(UUID(as_uuid=True), ForeignKey("community_creations.id"))
    alternative_discoveries = Column(JSON)  # Other discoveries during session
    
    # Timing
    duration_minutes = Column(Integer, default=60)  # Planned duration
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    creator = relationship("UserProfile", back_populates="collaborative_sessions")
    participants = relationship("SessionParticipant", back_populates="session", cascade="all, delete-orphan")
    creations = relationship("CommunityCreation", back_populates="collaboration_session")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_creator', creator_id),
        Index('idx_session_status', status),
        Index('idx_session_created_at', created_at),
        Index('idx_session_type', session_type),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collaborative session to dictionary"""
        return {
            'id': str(self.id),
            'creator_id': str(self.creator_id),
            'title': self.title,
            'description': self.description,
            'session_type': self.session_type,
            'creative_constraints': {
                'target_form': self.target_form,
                'theme': self.theme,
                'sacred_constant': self.sacred_constant
            },
            'collaboration_settings': {
                'max_participants': self.max_participants,
                'current_participants': self.current_participants
            },
            'session_state': {
                'status': self.status,
                'combined_entropy_seed': self.combined_entropy_seed,
                'collective_resonance': self.collective_resonance
            },
            'discovery_tracking': {
                'iterations_completed': self.iterations_completed,
                'current_discovery': self.current_discovery
            },
            'results': {
                'final_creation_id': str(self.final_creation_id) if self.final_creation_id else None,
                'alternative_discoveries': self.alternative_discoveries
            },
            'timing': {
                'duration_minutes': self.duration_minutes,
                'started_at': self.started_at.isoformat(),
                'ended_at': self.ended_at.isoformat() if self.ended_at else None
            },
            'timestamps': {
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat()
            }
        }


class SessionParticipant(Base):
    """
    Participants in collaborative sessions
    """
    __tablename__ = "session_participants"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaborative_sessions.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Participation metadata
    role = Column(String(30), default="contributor")  # contributor, moderator, observer
    
    # Signature at time of joining
    signature_snapshot = Column(JSON)
    
    # Contribution tracking
    entropy_contribution = Column(String(64))  # Their entropy contribution
    ideas_contributed = Column(Integer, default=0)
    feedback_given = Column(Integer, default=0)
    
    # Resonance with session
    session_resonance = Column(Float, default=0.0)
    satisfaction_score = Column(Float, default=0.0)
    
    # Participation status
    status = Column(String(30), default="active")  # active, left, kicked, finished
    
    # Timestamps
    joined_at = Column(DateTime, default=datetime.utcnow)
    left_at = Column(DateTime)
    
    # Relationships
    session = relationship("CollaborativeSession", back_populates="participants")
    user = relationship("UserProfile", back_populates="session_participants")
    
    # Unique constraint and indexes
    __table_args__ = (
        Index('idx_participant_session', session_id),
        Index('idx_participant_user', user_id),
        Index('idx_participant_joined_at', joined_at),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session participant to dictionary"""
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'user_id': str(self.user_id),
            'role': self.role,
            'signature_snapshot': self.signature_snapshot,
            'contribution_tracking': {
                'entropy_contribution': self.entropy_contribution,
                'ideas_contributed': self.ideas_contributed,
                'feedback_given': self.feedback_given
            },
            'resonance_data': {
                'session_resonance': self.session_resonance,
                'satisfaction_score': self.satisfaction_score
            },
            'participation_status': {
                'status': self.status,
                'joined_at': self.joined_at.isoformat(),
                'left_at': self.left_at.isoformat() if self.left_at else None
            }
        }


# Additional helper tables for analytics and caching

class ResonanceCache(Base):
    """
    Cache for expensive resonance calculations
    """
    __tablename__ = "resonance_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user1_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    user2_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False)
    
    # Cached resonance data
    resonance_score = Column(Float, nullable=False)
    archetypal_similarity = Column(Float, nullable=False)
    sacred_ratios_similarity = Column(Float, nullable=False)
    spatial_similarity = Column(Float, nullable=False)
    
    # Cache metadata
    calculation_version = Column(String(20), default="1.0")
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # Cache expiration
    
    # Indexes
    __table_args__ = (
        Index('idx_resonance_cache_users', user1_id, user2_id),
        Index('idx_resonance_cache_score', resonance_score),
        Index('idx_resonance_cache_expires', expires_at),
    )


class CommunityAnalytics(Base):
    """
    Community-wide analytics and insights
    """
    __tablename__ = "community_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20), nullable=False)  # daily, weekly, monthly
    
    # User metrics
    total_users = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    new_users = Column(Integer, default=0)
    
    # Content metrics
    total_creations = Column(Integer, default=0)
    new_creations = Column(Integer, default=0)
    total_comments = Column(Integer, default=0)
    total_likes = Column(Integer, default=0)
    
    # Engagement metrics
    average_session_duration = Column(Float, default=0.0)
    discovery_success_rate = Column(Float, default=0.0)
    collaboration_sessions = Column(Integer, default=0)
    
    # Quality metrics
    average_mathematical_fitness = Column(Float, default=0.0)
    average_semantic_coherence = Column(Float, default=0.0)
    average_user_satisfaction = Column(Float, default=0.0)
    
    # Archetypal distribution
    archetypal_distribution = Column(JSON)  # Distribution of primary muses
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_analytics_period', period_start, period_end),
        Index('idx_analytics_type', period_type),
        Index('idx_analytics_created', created_at),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analytics to dictionary"""
        return {
            'id': str(self.id),
            'period': {
                'start': self.period_start.isoformat(),
                'end': self.period_end.isoformat(),
                'type': self.period_type
            },
            'user_metrics': {
                'total_users': self.total_users,
                'active_users': self.active_users,
                'new_users': self.new_users
            },
            'content_metrics': {
                'total_creations': self.total_creations,
                'new_creations': self.new_creations,
                'total_comments': self.total_comments,
                'total_likes': self.total_likes
            },
            'engagement_metrics': {
                'average_session_duration': self.average_session_duration,
                'discovery_success_rate': self.discovery_success_rate,
                'collaboration_sessions': self.collaboration_sessions
            },
            'quality_metrics': {
                'average_mathematical_fitness': self.average_mathematical_fitness,
                'average_semantic_coherence': self.average_semantic_coherence,
                'average_user_satisfaction': self.average_user_satisfaction
            },
            'archetypal_distribution': self.archetypal_distribution,
            'created_at': self.created_at.isoformat()
        }