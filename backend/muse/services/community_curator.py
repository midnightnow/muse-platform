"""
Community Curator Service for MUSE Platform

This service manages community content curation, recommendations, and social
interactions based on frequency-based resonance and archetypal compatibility.
It implements the social layer of Computational Platonism.
"""

import math
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
from collections import defaultdict, Counter
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from muse.models.community import (
    UserProfile, CommunityCreation, Comment, Like, Follow, 
    FrequencySignature, CommunityAnalytics
)
from muse.services.resonance_matcher import ResonanceMatcher, CommunityMatch


logger = logging.getLogger(__name__)


class CurationMode(Enum):
    """Content curation modes"""
    RESONANCE_BASED = "resonance_based"
    ARCHETYPAL_AFFINITY = "archetypal_affinity"
    SACRED_GEOMETRY = "sacred_geometry"
    COLLABORATIVE = "collaborative"
    TRENDING = "trending"
    DISCOVERY = "discovery"


class FeedType(Enum):
    """Types of content feeds"""
    PERSONALIZED = "personalized"
    RESONANT = "resonant"
    ARCHETYPAL = "archetypal"
    GEOMETRIC = "geometric"
    SOCIAL = "social"
    EXPLORE = "explore"


@dataclass
class CurationResult:
    """Result of content curation"""
    content_id: str
    relevance_score: float
    resonance_score: float
    curation_reason: str
    archetypal_match: Dict[str, float]
    geometric_alignment: float
    social_signals: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class FeedItem:
    """Individual item in a curated feed"""
    creation: Dict[str, Any]
    curation_result: CurationResult
    creator_profile: Dict[str, Any]
    engagement_metrics: Dict[str, Any]
    resonance_explanation: str
    recommendation_strength: float


@dataclass
class CommunityInsight:
    """Community insight based on archetypal analysis"""
    insight_type: str
    title: str
    description: str
    archetypal_patterns: Dict[str, float]
    sacred_geometry_elements: Dict[str, Any]
    community_resonance: float
    supporting_data: Dict[str, Any]


class CommunityCurator:
    """
    Core service for community content curation and recommendations
    
    This service implements frequency-based content curation using archetypal
    resonance and sacred geometry principles to create meaningful connections
    between creators and their audiences.
    """
    
    def __init__(self, resonance_matcher: ResonanceMatcher):
        """
        Initialize the community curator
        
        Args:
            resonance_matcher: ResonanceMatcher service instance
        """
        self.resonance_matcher = resonance_matcher
        self.curation_cache = {}
        self.cache_duration = timedelta(hours=1)
        
        # Curation weights
        self.curation_weights = {
            'resonance_score': 0.35,
            'archetypal_alignment': 0.25,
            'geometric_alignment': 0.15,
            'social_signals': 0.15,
            'recency': 0.10
        }
        
        # Social signal weights
        self.social_weights = {
            'likes': 0.4,
            'comments': 0.3,
            'shares': 0.2,
            'creator_resonance': 0.1
        }
    
    def curate_personalized_feed(self, db: Session, user_id: str, 
                               limit: int = 20, offset: int = 0,
                               mode: CurationMode = CurationMode.RESONANCE_BASED) -> List[FeedItem]:
        """
        Curate personalized content feed for a user
        
        Args:
            db: Database session
            user_id: User ID to curate feed for
            limit: Maximum number of items to return
            offset: Offset for pagination
            mode: Curation mode
            
        Returns:
            List of curated FeedItem objects
        """
        # Get user profile and frequency signature
        user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if not user_profile:
            raise ValueError(f"User profile not found: {user_id}")
        
        # Get recent community creations (excluding user's own)
        recent_creations = db.query(CommunityCreation).filter(
            CommunityCreation.creator_id != user_id,
            CommunityCreation.visibility == 'public',
            CommunityCreation.created_at >= datetime.utcnow() - timedelta(days=30)
        ).order_by(desc(CommunityCreation.created_at)).limit(limit * 3).all()
        
        # Curate content based on mode
        curated_items = []
        
        for creation in recent_creations:
            try:
                # Calculate curation score
                curation_result = self._calculate_curation_score(
                    db, user_profile, creation, mode
                )
                
                # Skip if score is too low
                if curation_result.relevance_score < 0.3:
                    continue
                
                # Create feed item
                feed_item = self._create_feed_item(db, creation, curation_result)
                curated_items.append(feed_item)
                
            except Exception as e:
                logger.warning(f"Error curating creation {creation.id}: {e}")
                continue
        
        # Sort by relevance score
        curated_items.sort(key=lambda x: x.curation_result.relevance_score, reverse=True)
        
        # Apply pagination
        return curated_items[offset:offset + limit]
    
    def get_resonant_discoveries(self, db: Session, user_id: str, 
                               resonance_threshold: float = 0.7,
                               limit: int = 10) -> List[FeedItem]:
        """
        Get discoveries that resonate with user's frequency signature
        
        Args:
            db: Database session
            user_id: User ID
            resonance_threshold: Minimum resonance threshold
            limit: Maximum number of discoveries
            
        Returns:
            List of resonant discoveries
        """
        user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if not user_profile:
            raise ValueError(f"User profile not found: {user_id}")
        
        # Get user's frequency signature
        user_signature = db.query(FrequencySignature).filter(
            FrequencySignature.user_id == user_id,
            FrequencySignature.is_active == True
        ).first()
        
        if not user_signature:
            # Fallback to profile-based curation
            return self.curate_personalized_feed(db, user_id, limit=limit)
        
        # Find creations with similar archetypal alignment
        similar_creations = db.query(CommunityCreation).filter(
            CommunityCreation.creator_id != user_id,
            CommunityCreation.visibility == 'public',
            CommunityCreation.archetypal_alignment >= resonance_threshold
        ).order_by(desc(CommunityCreation.archetypal_alignment)).limit(limit * 2).all()
        
        resonant_items = []
        
        for creation in similar_creations:
            try:
                # Calculate resonance with creator
                creator_resonance = self.resonance_matcher.calculate_user_resonance(
                    db, user_id, str(creation.creator_id)
                )
                
                # Check if meets resonance threshold
                if creator_resonance.overall_resonance >= resonance_threshold:
                    curation_result = CurationResult(
                        content_id=str(creation.id),
                        relevance_score=creator_resonance.overall_resonance,
                        resonance_score=creator_resonance.overall_resonance,
                        curation_reason="resonant_discovery",
                        archetypal_match=creator_resonance.detailed_analysis.get("archetypal_analysis", {}),
                        geometric_alignment=creator_resonance.spatial_similarity,
                        social_signals=self._calculate_social_signals(creation),
                        metadata={"resonance_type": creator_resonance.resonance_type}
                    )
                    
                    feed_item = self._create_feed_item(db, creation, curation_result)
                    resonant_items.append(feed_item)
                    
            except Exception as e:
                logger.warning(f"Error processing resonant creation {creation.id}: {e}")
                continue
        
        # Sort by resonance score
        resonant_items.sort(key=lambda x: x.curation_result.resonance_score, reverse=True)
        
        return resonant_items[:limit]
    
    def get_archetypal_gallery(self, db: Session, archetype: str, 
                             limit: int = 20, offset: int = 0,
                             user_id: Optional[str] = None) -> List[FeedItem]:
        """
        Get gallery of creations aligned with specific archetype
        
        Args:
            db: Database session
            archetype: Archetype to filter by
            limit: Maximum number of items
            offset: Offset for pagination
            user_id: Optional user ID for personalization
            
        Returns:
            List of archetypal gallery items
        """
        # Get creations with high archetypal alignment
        archetypal_creations = db.query(CommunityCreation).filter(
            CommunityCreation.visibility == 'public',
            CommunityCreation.archetypal_alignment >= 0.5
        ).order_by(desc(CommunityCreation.archetypal_alignment)).offset(offset).limit(limit).all()
        
        gallery_items = []
        
        for creation in archetypal_creations:
            try:
                # Check if creation aligns with requested archetype
                creator_signature = creation.creator_signature_snapshot
                if not creator_signature:
                    continue
                
                harmonic_blend = creator_signature.get('harmonic_blend', {})
                archetype_strength = harmonic_blend.get(archetype, 0.0)
                
                # Only include if strong alignment
                if archetype_strength < 0.2:
                    continue
                
                # Create curation result
                curation_result = CurationResult(
                    content_id=str(creation.id),
                    relevance_score=archetype_strength,
                    resonance_score=creation.resonance_score,
                    curation_reason=f"archetypal_alignment_{archetype}",
                    archetypal_match={archetype: archetype_strength},
                    geometric_alignment=creation.sacred_geometry_score,
                    social_signals=self._calculate_social_signals(creation),
                    metadata={"archetype": archetype, "strength": archetype_strength}
                )
                
                feed_item = self._create_feed_item(db, creation, curation_result)
                gallery_items.append(feed_item)
                
            except Exception as e:
                logger.warning(f"Error processing archetypal creation {creation.id}: {e}")
                continue
        
        return gallery_items
    
    def get_sacred_geometry_showcase(self, db: Session, sacred_constant: str,
                                   limit: int = 15, offset: int = 0) -> List[FeedItem]:
        """
        Get showcase of creations optimized for sacred geometry
        
        Args:
            db: Database session
            sacred_constant: Sacred constant to filter by ('phi', 'pi', 'fibonacci')
            limit: Maximum number of items
            offset: Offset for pagination
            
        Returns:
            List of sacred geometry showcase items
        """
        # Get creations with high sacred geometry scores
        geometric_creations = db.query(CommunityCreation).filter(
            CommunityCreation.visibility == 'public',
            CommunityCreation.sacred_geometry_score >= 0.6,
            CommunityCreation.sacred_constant == sacred_constant
        ).order_by(desc(CommunityCreation.sacred_geometry_score)).offset(offset).limit(limit).all()
        
        showcase_items = []
        
        for creation in geometric_creations:
            try:
                curation_result = CurationResult(
                    content_id=str(creation.id),
                    relevance_score=creation.sacred_geometry_score,
                    resonance_score=creation.resonance_score,
                    curation_reason=f"sacred_geometry_{sacred_constant}",
                    archetypal_match=creation.creator_signature_snapshot.get('harmonic_blend', {}) if creation.creator_signature_snapshot else {},
                    geometric_alignment=creation.sacred_geometry_score,
                    social_signals=self._calculate_social_signals(creation),
                    metadata={"sacred_constant": sacred_constant, "geometry_score": creation.sacred_geometry_score}
                )
                
                feed_item = self._create_feed_item(db, creation, curation_result)
                showcase_items.append(feed_item)
                
            except Exception as e:
                logger.warning(f"Error processing geometric creation {creation.id}: {e}")
                continue
        
        return showcase_items
    
    def get_collaborative_discoveries(self, db: Session, user_id: str,
                                    limit: int = 10) -> List[FeedItem]:
        """
        Get discoveries from collaborative sessions
        
        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of items
            
        Returns:
            List of collaborative discoveries
        """
        # Get creations from collaborative sessions
        collaborative_creations = db.query(CommunityCreation).filter(
            CommunityCreation.visibility == 'public',
            CommunityCreation.is_collaborative == True
        ).order_by(desc(CommunityCreation.created_at)).limit(limit * 2).all()
        
        discoveries = []
        
        for creation in collaborative_creations:
            try:
                # Calculate collaborative metrics
                collaborative_metrics = self._calculate_collaborative_metrics(db, creation)
                
                curation_result = CurationResult(
                    content_id=str(creation.id),
                    relevance_score=collaborative_metrics['relevance'],
                    resonance_score=creation.resonance_score,
                    curation_reason="collaborative_discovery",
                    archetypal_match=collaborative_metrics['archetypal_harmony'],
                    geometric_alignment=creation.sacred_geometry_score,
                    social_signals=self._calculate_social_signals(creation),
                    metadata={"collaborative_metrics": collaborative_metrics}
                )
                
                feed_item = self._create_feed_item(db, creation, curation_result)
                discoveries.append(feed_item)
                
            except Exception as e:
                logger.warning(f"Error processing collaborative creation {creation.id}: {e}")
                continue
        
        # Sort by relevance
        discoveries.sort(key=lambda x: x.curation_result.relevance_score, reverse=True)
        
        return discoveries[:limit]
    
    def generate_community_insights(self, db: Session, time_period: str = "week") -> List[CommunityInsight]:
        """
        Generate community insights based on archetypal patterns
        
        Args:
            db: Database session
            time_period: Time period for analysis ('day', 'week', 'month')
            
        Returns:
            List of community insights
        """
        # Calculate time range
        time_ranges = {
            'day': timedelta(days=1),
            'week': timedelta(weeks=1),
            'month': timedelta(days=30)
        }
        
        since_date = datetime.utcnow() - time_ranges.get(time_period, timedelta(weeks=1))
        
        # Get recent creations
        recent_creations = db.query(CommunityCreation).filter(
            CommunityCreation.created_at >= since_date,
            CommunityCreation.visibility == 'public'
        ).all()
        
        insights = []
        
        # Analyze archetypal patterns
        archetypal_insight = self._analyze_archetypal_patterns(recent_creations)
        if archetypal_insight:
            insights.append(archetypal_insight)
        
        # Analyze sacred geometry trends
        geometry_insight = self._analyze_geometry_trends(recent_creations)
        if geometry_insight:
            insights.append(geometry_insight)
        
        # Analyze collaborative patterns
        collaborative_insight = self._analyze_collaborative_patterns(db, recent_creations)
        if collaborative_insight:
            insights.append(collaborative_insight)
        
        # Analyze resonance patterns
        resonance_insight = self._analyze_resonance_patterns(recent_creations)
        if resonance_insight:
            insights.append(resonance_insight)
        
        return insights
    
    def recommend_creators(self, db: Session, user_id: str, 
                         limit: int = 10) -> List[CommunityMatch]:
        """
        Recommend creators based on archetypal compatibility
        
        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of recommendations
            
        Returns:
            List of creator recommendations
        """
        return self.resonance_matcher.find_community_matches(
            db, user_id, limit=limit, min_resonance=0.6
        )
    
    def _calculate_curation_score(self, db: Session, user_profile: UserProfile,
                                creation: CommunityCreation, mode: CurationMode) -> CurationResult:
        """Calculate curation score for a creation"""
        # Base scores
        scores = {
            'resonance_score': 0.0,
            'archetypal_alignment': 0.0,
            'geometric_alignment': 0.0,
            'social_signals': 0.0,
            'recency': 0.0
        }
        
        # Calculate resonance with creator
        try:
            creator_resonance = self.resonance_matcher.calculate_user_resonance(
                db, str(user_profile.id), str(creation.creator_id)
            )
            scores['resonance_score'] = creator_resonance.overall_resonance
            archetypal_match = creator_resonance.detailed_analysis.get("archetypal_analysis", {})
            scores['archetypal_alignment'] = creator_resonance.archetypal_similarity
        except Exception as e:
            logger.warning(f"Error calculating resonance: {e}")
            archetypal_match = {}
        
        # Calculate geometric alignment
        scores['geometric_alignment'] = creation.sacred_geometry_score
        
        # Calculate social signals
        social_signals = self._calculate_social_signals(creation)
        scores['social_signals'] = social_signals['normalized_score']
        
        # Calculate recency score
        age_hours = (datetime.utcnow() - creation.created_at).total_seconds() / 3600
        scores['recency'] = max(0.0, 1.0 - (age_hours / 168))  # Decay over week
        
        # Calculate weighted relevance score
        relevance_score = sum(
            scores[key] * weight 
            for key, weight in self.curation_weights.items()
        )
        
        # Determine curation reason
        curation_reason = self._determine_curation_reason(scores, mode)
        
        return CurationResult(
            content_id=str(creation.id),
            relevance_score=relevance_score,
            resonance_score=scores['resonance_score'],
            curation_reason=curation_reason,
            archetypal_match=archetypal_match,
            geometric_alignment=scores['geometric_alignment'],
            social_signals=social_signals,
            metadata={
                'mode': mode.value,
                'all_scores': scores,
                'creation_age_hours': age_hours
            }
        )
    
    def _calculate_social_signals(self, creation: CommunityCreation) -> Dict[str, Any]:
        """Calculate social signals for a creation"""
        signals = {
            'likes': creation.likes_count,
            'comments': creation.comments_count,
            'shares': creation.shares_count,
            'resonance': creation.resonance_score
        }
        
        # Normalize signals
        max_engagement = max(signals['likes'], signals['comments'], signals['shares'])
        normalized_score = 0.0
        
        if max_engagement > 0:
            normalized_score = (
                (signals['likes'] / max_engagement) * self.social_weights['likes'] +
                (signals['comments'] / max_engagement) * self.social_weights['comments'] +
                (signals['shares'] / max_engagement) * self.social_weights['shares'] +
                signals['resonance'] * self.social_weights['creator_resonance']
            )
        
        return {
            **signals,
            'normalized_score': normalized_score,
            'engagement_total': sum([signals['likes'], signals['comments'], signals['shares']])
        }
    
    def _determine_curation_reason(self, scores: Dict[str, float], mode: CurationMode) -> str:
        """Determine the primary reason for curation"""
        if mode == CurationMode.RESONANCE_BASED:
            if scores['resonance_score'] > 0.8:
                return "high_resonance"
            elif scores['archetypal_alignment'] > 0.7:
                return "archetypal_harmony"
            elif scores['geometric_alignment'] > 0.7:
                return "geometric_alignment"
            else:
                return "moderate_resonance"
        
        elif mode == CurationMode.ARCHETYPAL_AFFINITY:
            return "archetypal_affinity"
        
        elif mode == CurationMode.SACRED_GEOMETRY:
            return "sacred_geometry"
        
        elif mode == CurationMode.TRENDING:
            return "trending_content"
        
        else:
            return "curated_content"
    
    def _create_feed_item(self, db: Session, creation: CommunityCreation,
                         curation_result: CurationResult) -> FeedItem:
        """Create a feed item from creation and curation result"""
        # Get creator profile
        creator = db.query(UserProfile).filter(UserProfile.id == creation.creator_id).first()
        
        # Calculate engagement metrics
        engagement_metrics = {
            'likes': creation.likes_count,
            'comments': creation.comments_count,
            'shares': creation.shares_count,
            'resonance_score': creation.resonance_score,
            'mathematical_fitness': creation.mathematical_fitness,
            'semantic_coherence': creation.semantic_coherence
        }
        
        # Generate resonance explanation
        resonance_explanation = self._generate_resonance_explanation(curation_result)
        
        # Calculate recommendation strength
        recommendation_strength = min(1.0, curation_result.relevance_score * 1.2)
        
        return FeedItem(
            creation=creation.to_dict(),
            curation_result=curation_result,
            creator_profile=creator.to_dict() if creator else {},
            engagement_metrics=engagement_metrics,
            resonance_explanation=resonance_explanation,
            recommendation_strength=recommendation_strength
        )
    
    def _generate_resonance_explanation(self, curation_result: CurationResult) -> str:
        """Generate human-readable explanation of resonance"""
        reason = curation_result.curation_reason
        resonance_score = curation_result.resonance_score
        
        if reason == "high_resonance":
            return f"This creation strongly resonates with your frequency signature ({resonance_score:.1%} alignment)"
        elif reason == "archetypal_harmony":
            return "This creation aligns beautifully with your archetypal preferences"
        elif reason == "geometric_alignment":
            return "The sacred geometry in this creation matches your mathematical affinities"
        elif reason == "collaborative_discovery":
            return "This collaborative creation embodies the collective wisdom of multiple creators"
        else:
            return f"This creation resonates with your creative essence ({resonance_score:.1%} alignment)"
    
    def _calculate_collaborative_metrics(self, db: Session, creation: CommunityCreation) -> Dict[str, Any]:
        """Calculate metrics for collaborative creation"""
        if not creation.collaboration_session_id:
            return {'relevance': 0.5, 'archetypal_harmony': {}}
        
        # Get collaboration session
        session = db.query(CommunityCreation).filter(
            CommunityCreation.collaboration_session_id == creation.collaboration_session_id
        ).first()
        
        if not session:
            return {'relevance': 0.5, 'archetypal_harmony': {}}
        
        # Calculate collaborative harmony
        participants_count = 1  # Simplified - in real implementation, query session participants
        archetypal_diversity = len(creation.creator_signature_snapshot.get('harmonic_blend', {}))
        
        relevance = min(1.0, 0.5 + (participants_count * 0.2) + (archetypal_diversity * 0.1))
        
        return {
            'relevance': relevance,
            'archetypal_harmony': creation.creator_signature_snapshot.get('harmonic_blend', {}),
            'participants_count': participants_count,
            'archetypal_diversity': archetypal_diversity
        }
    
    def _analyze_archetypal_patterns(self, creations: List[CommunityCreation]) -> Optional[CommunityInsight]:
        """Analyze archetypal patterns in community creations"""
        if not creations:
            return None
        
        # Count archetypal patterns
        archetype_counts = defaultdict(int)
        archetype_strengths = defaultdict(float)
        
        for creation in creations:
            if creation.creator_signature_snapshot:
                harmonic_blend = creation.creator_signature_snapshot.get('harmonic_blend', {})
                for archetype, strength in harmonic_blend.items():
                    archetype_counts[archetype] += 1
                    archetype_strengths[archetype] += strength
        
        # Calculate averages
        archetype_patterns = {}
        for archetype in archetype_counts:
            archetype_patterns[archetype] = archetype_strengths[archetype] / archetype_counts[archetype]
        
        # Find dominant archetype
        dominant_archetype = max(archetype_patterns.items(), key=lambda x: x[1]) if archetype_patterns else None
        
        if not dominant_archetype:
            return None
        
        # Calculate community resonance
        resonance_scores = [c.resonance_score for c in creations if c.resonance_score]
        community_resonance = sum(resonance_scores) / len(resonance_scores) if resonance_scores else 0.0
        
        return CommunityInsight(
            insight_type="archetypal_pattern",
            title=f"Community Harmony: {dominant_archetype[0]} Energy Rising",
            description=f"The community is experiencing a surge of {dominant_archetype[0]} energy, "
                       f"with {archetype_counts[dominant_archetype[0]]} creators channeling this archetypal frequency.",
            archetypal_patterns=archetype_patterns,
            sacred_geometry_elements={},
            community_resonance=community_resonance,
            supporting_data={
                'total_creations': len(creations),
                'archetype_counts': dict(archetype_counts),
                'dominant_archetype': dominant_archetype[0],
                'dominant_strength': dominant_archetype[1]
            }
        )
    
    def _analyze_geometry_trends(self, creations: List[CommunityCreation]) -> Optional[CommunityInsight]:
        """Analyze sacred geometry trends"""
        if not creations:
            return None
        
        # Count sacred constants
        constant_counts = defaultdict(int)
        geometry_scores = defaultdict(list)
        
        for creation in creations:
            if creation.sacred_constant:
                constant_counts[creation.sacred_constant] += 1
                geometry_scores[creation.sacred_constant].append(creation.sacred_geometry_score)
        
        # Find trending constant
        trending_constant = max(constant_counts.items(), key=lambda x: x[1]) if constant_counts else None
        
        if not trending_constant:
            return None
        
        # Calculate average geometry score
        avg_score = sum(geometry_scores[trending_constant[0]]) / len(geometry_scores[trending_constant[0]])
        
        return CommunityInsight(
            insight_type="geometry_trend",
            title=f"Sacred Geometry Spotlight: {trending_constant[0].upper()} Resonance",
            description=f"Creators are gravitating toward {trending_constant[0]} sacred geometry, "
                       f"with {trending_constant[1]} creations achieving an average alignment of {avg_score:.1%}.",
            archetypal_patterns={},
            sacred_geometry_elements={
                'trending_constant': trending_constant[0],
                'average_score': avg_score,
                'creation_count': trending_constant[1]
            },
            community_resonance=avg_score,
            supporting_data={
                'constant_counts': dict(constant_counts),
                'geometry_scores': {k: sum(v)/len(v) for k, v in geometry_scores.items()}
            }
        )
    
    def _analyze_collaborative_patterns(self, db: Session, 
                                      creations: List[CommunityCreation]) -> Optional[CommunityInsight]:
        """Analyze collaborative creation patterns"""
        collaborative_creations = [c for c in creations if c.is_collaborative]
        
        if not collaborative_creations:
            return None
        
        # Calculate collaboration metrics
        total_collaborations = len(collaborative_creations)
        avg_resonance = sum(c.resonance_score for c in collaborative_creations) / total_collaborations
        
        # Analyze archetypal diversity in collaborations
        archetypal_diversity = []
        for creation in collaborative_creations:
            if creation.creator_signature_snapshot:
                harmonic_blend = creation.creator_signature_snapshot.get('harmonic_blend', {})
                diversity = len([v for v in harmonic_blend.values() if v > 0.1])
                archetypal_diversity.append(diversity)
        
        avg_diversity = sum(archetypal_diversity) / len(archetypal_diversity) if archetypal_diversity else 0
        
        return CommunityInsight(
            insight_type="collaborative_pattern",
            title="Collaborative Harmony: Collective Creativity Rising",
            description=f"Community collaboration is flourishing with {total_collaborations} joint creations, "
                       f"averaging {avg_diversity:.1f} archetypal voices per collaboration.",
            archetypal_patterns={},
            sacred_geometry_elements={},
            community_resonance=avg_resonance,
            supporting_data={
                'total_collaborations': total_collaborations,
                'average_resonance': avg_resonance,
                'average_diversity': avg_diversity,
                'collaboration_percentage': (total_collaborations / len(creations)) * 100
            }
        )
    
    def _analyze_resonance_patterns(self, creations: List[CommunityCreation]) -> Optional[CommunityInsight]:
        """Analyze community resonance patterns"""
        if not creations:
            return None
        
        # Calculate resonance statistics
        resonance_scores = [c.resonance_score for c in creations if c.resonance_score]
        
        if not resonance_scores:
            return None
        
        avg_resonance = sum(resonance_scores) / len(resonance_scores)
        high_resonance_count = len([s for s in resonance_scores if s > 0.8])
        
        # Analyze forms with high resonance
        form_resonance = defaultdict(list)
        for creation in creations:
            if creation.resonance_score and creation.form_type:
                form_resonance[creation.form_type].append(creation.resonance_score)
        
        # Find most resonant form
        best_form = None
        best_score = 0
        
        for form, scores in form_resonance.items():
            avg_form_score = sum(scores) / len(scores)
            if avg_form_score > best_score:
                best_score = avg_form_score
                best_form = form
        
        return CommunityInsight(
            insight_type="resonance_pattern",
            title="Community Resonance: Harmony in Diversity",
            description=f"The community is achieving {avg_resonance:.1%} average resonance, "
                       f"with {high_resonance_count} creations reaching exceptional harmony.",
            archetypal_patterns={},
            sacred_geometry_elements={},
            community_resonance=avg_resonance,
            supporting_data={
                'average_resonance': avg_resonance,
                'high_resonance_count': high_resonance_count,
                'best_form': best_form,
                'best_form_score': best_score,
                'total_creations': len(creations)
            }
        )