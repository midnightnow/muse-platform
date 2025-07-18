"""
Resonance Matcher Service for MUSE Platform

This service handles the calculation of archetypal similarity between users,
implementing the mathematical foundation for community connections based on
frequency signatures and sacred geometry resonance.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

from muse.core.frequency_engine import MuseFrequencyEngine, FrequencySignature
from muse.core.sacred_geometry_calculator import SacredGeometryCalculator
from muse.models.community import UserProfile, FrequencySignature as DBFrequencySignature, ResonanceCache


logger = logging.getLogger(__name__)


@dataclass
class ResonanceResult:
    """Result of resonance calculation between two users"""
    user1_id: str
    user2_id: str
    overall_resonance: float
    archetypal_similarity: float
    sacred_ratios_similarity: float
    spatial_similarity: float
    compatibility_score: float
    resonance_type: str
    detailed_analysis: Dict[str, Any]


@dataclass
class CommunityMatch:
    """Community match recommendation"""
    user_id: str
    username: str
    display_name: str
    resonance_score: float
    compatibility_type: str
    shared_archetypes: List[str]
    complementary_traits: List[str]
    recommended_collaboration: str


class ResonanceMatcher:
    """
    Core service for calculating archetypal resonance between users
    
    This service implements the mathematical foundation for community connections
    in MUSE's Computational Platonism approach, using frequency signatures and
    sacred geometry to determine compatibility.
    """
    
    def __init__(self, frequency_engine: MuseFrequencyEngine, geometry_calculator: SacredGeometryCalculator):
        """
        Initialize the resonance matcher
        
        Args:
            frequency_engine: MUSE frequency engine instance
            geometry_calculator: Sacred geometry calculator instance
        """
        self.frequency_engine = frequency_engine
        self.geometry_calculator = geometry_calculator
        self.cache_expiry_hours = 24  # Cache resonance calculations for 24 hours
        
    def calculate_user_resonance(self, db: Session, user1_id: str, user2_id: str, 
                                force_recalculate: bool = False) -> ResonanceResult:
        """
        Calculate resonance between two users with caching
        
        Args:
            db: Database session
            user1_id: First user ID
            user2_id: Second user ID
            force_recalculate: Force recalculation ignoring cache
            
        Returns:
            ResonanceResult with detailed analysis
        """
        # Check cache first unless forced recalculation
        if not force_recalculate:
            cached_result = self._get_cached_resonance(db, user1_id, user2_id)
            if cached_result:
                return cached_result
        
        # Get user profiles and signatures
        user1_profile = db.query(UserProfile).filter(UserProfile.id == user1_id).first()
        user2_profile = db.query(UserProfile).filter(UserProfile.id == user2_id).first()
        
        if not user1_profile or not user2_profile:
            raise ValueError(f"User profiles not found for {user1_id} or {user2_id}")
        
        # Get active frequency signatures
        user1_signature = self._get_user_frequency_signature(db, user1_id)
        user2_signature = self._get_user_frequency_signature(db, user2_id)
        
        if not user1_signature or not user2_signature:
            raise ValueError(f"Frequency signatures not found for {user1_id} or {user2_id}")
        
        # Calculate resonance components
        archetypal_similarity = self._calculate_archetypal_similarity(
            user1_signature.harmonic_blend, 
            user2_signature.harmonic_blend
        )
        
        sacred_ratios_similarity = self._calculate_sacred_ratios_similarity(
            user1_signature.sacred_ratios,
            user2_signature.sacred_ratios
        )
        
        spatial_similarity = self._calculate_spatial_similarity(
            user1_signature.spiral_coordinates,
            user2_signature.spiral_coordinates
        )
        
        # Calculate overall resonance with weighted combination
        overall_resonance = (
            archetypal_similarity * 0.45 +
            sacred_ratios_similarity * 0.35 +
            spatial_similarity * 0.20
        )
        
        # Calculate compatibility score (different from resonance)
        compatibility_score = self._calculate_compatibility_score(
            user1_profile, user2_profile, user1_signature, user2_signature
        )
        
        # Determine resonance type
        resonance_type = self._determine_resonance_type(
            archetypal_similarity, sacred_ratios_similarity, spatial_similarity
        )
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            user1_profile, user2_profile, user1_signature, user2_signature,
            archetypal_similarity, sacred_ratios_similarity, spatial_similarity
        )
        
        # Create result
        result = ResonanceResult(
            user1_id=user1_id,
            user2_id=user2_id,
            overall_resonance=overall_resonance,
            archetypal_similarity=archetypal_similarity,
            sacred_ratios_similarity=sacred_ratios_similarity,
            spatial_similarity=spatial_similarity,
            compatibility_score=compatibility_score,
            resonance_type=resonance_type,
            detailed_analysis=detailed_analysis
        )
        
        # Cache the result
        self._cache_resonance_result(db, result)
        
        return result
    
    def find_community_matches(self, db: Session, user_id: str, 
                             limit: int = 10, min_resonance: float = 0.6) -> List[CommunityMatch]:
        """
        Find community matches for a user based on resonance
        
        Args:
            db: Database session
            user_id: User ID to find matches for
            limit: Maximum number of matches to return
            min_resonance: Minimum resonance threshold
            
        Returns:
            List of CommunityMatch objects
        """
        # Get user's profile and signature
        user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if not user_profile:
            raise ValueError(f"User profile not found for {user_id}")
        
        user_signature = self._get_user_frequency_signature(db, user_id)
        if not user_signature:
            raise ValueError(f"Frequency signature not found for {user_id}")
        
        # Get all other users with similar primary muses (optimization)
        similar_users = db.query(UserProfile).filter(
            UserProfile.id != user_id,
            UserProfile.primary_muse.in_([user_profile.primary_muse, user_profile.secondary_muse])
        ).limit(limit * 3).all()  # Get more candidates for filtering
        
        matches = []
        
        for candidate in similar_users:
            try:
                # Calculate resonance
                resonance_result = self.calculate_user_resonance(db, user_id, str(candidate.id))
                
                # Check if meets minimum threshold
                if resonance_result.overall_resonance >= min_resonance:
                    match = self._create_community_match(
                        candidate, resonance_result, user_profile, user_signature
                    )
                    matches.append(match)
                    
            except Exception as e:
                logger.warning(f"Error calculating resonance for user {candidate.id}: {e}")
                continue
        
        # Sort by resonance score and return top matches
        matches.sort(key=lambda x: x.resonance_score, reverse=True)
        return matches[:limit]
    
    def find_kindred_spirits(self, db: Session, user_id: str, 
                           limit: int = 5, resonance_threshold: float = 0.8) -> List[CommunityMatch]:
        """
        Find kindred spirits - users with exceptionally high resonance
        
        Args:
            db: Database session
            user_id: User ID to find kindred spirits for
            limit: Maximum number of kindred spirits to return
            resonance_threshold: High resonance threshold for kindred spirits
            
        Returns:
            List of CommunityMatch objects for kindred spirits
        """
        # Use same logic as community matches but with higher threshold
        matches = self.find_community_matches(
            db, user_id, limit=limit * 2, min_resonance=resonance_threshold
        )
        
        # Filter for truly kindred spirits (high overall resonance and compatibility)
        kindred_spirits = [
            match for match in matches 
            if match.resonance_score >= resonance_threshold and 
            len(match.shared_archetypes) >= 2
        ]
        
        return kindred_spirits[:limit]
    
    def _get_cached_resonance(self, db: Session, user1_id: str, user2_id: str) -> Optional[ResonanceResult]:
        """Get cached resonance result if available and not expired"""
        # Ensure consistent ordering for cache lookup
        uid1, uid2 = sorted([user1_id, user2_id])
        
        cache_entry = db.query(ResonanceCache).filter(
            ResonanceCache.user1_id == uid1,
            ResonanceCache.user2_id == uid2,
            ResonanceCache.expires_at > datetime.utcnow()
        ).first()
        
        if cache_entry:
            return ResonanceResult(
                user1_id=user1_id,
                user2_id=user2_id,
                overall_resonance=cache_entry.resonance_score,
                archetypal_similarity=cache_entry.archetypal_similarity,
                sacred_ratios_similarity=cache_entry.sacred_ratios_similarity,
                spatial_similarity=cache_entry.spatial_similarity,
                compatibility_score=cache_entry.resonance_score,  # Simplified for cache
                resonance_type="cached",
                detailed_analysis={"cached": True}
            )
        
        return None
    
    def _cache_resonance_result(self, db: Session, result: ResonanceResult):
        """Cache resonance result for future use"""
        try:
            # Ensure consistent ordering
            uid1, uid2 = sorted([result.user1_id, result.user2_id])
            
            # Delete existing cache entry
            db.query(ResonanceCache).filter(
                ResonanceCache.user1_id == uid1,
                ResonanceCache.user2_id == uid2
            ).delete()
            
            # Create new cache entry
            cache_entry = ResonanceCache(
                user1_id=uid1,
                user2_id=uid2,
                resonance_score=result.overall_resonance,
                archetypal_similarity=result.archetypal_similarity,
                sacred_ratios_similarity=result.sacred_ratios_similarity,
                spatial_similarity=result.spatial_similarity,
                calculation_version="1.0",
                expires_at=datetime.utcnow() + timedelta(hours=self.cache_expiry_hours)
            )
            
            db.add(cache_entry)
            db.commit()
            
        except Exception as e:
            logger.error(f"Error caching resonance result: {e}")
            db.rollback()
    
    def _get_user_frequency_signature(self, db: Session, user_id: str) -> Optional[DBFrequencySignature]:
        """Get user's active frequency signature"""
        return db.query(DBFrequencySignature).filter(
            DBFrequencySignature.user_id == user_id,
            DBFrequencySignature.is_active == True
        ).first()
    
    def _calculate_archetypal_similarity(self, blend1: Dict[str, float], 
                                       blend2: Dict[str, float]) -> float:
        """Calculate archetypal similarity using cosine similarity"""
        if not blend1 or not blend2:
            return 0.0
        
        # Get all archetype keys
        all_archetypes = set(blend1.keys()) | set(blend2.keys())
        
        # Create vectors
        vec1 = np.array([blend1.get(archetype, 0.0) for archetype in all_archetypes])
        vec2 = np.array([blend2.get(archetype, 0.0) for archetype in all_archetypes])
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        return max(0.0, min(1.0, cosine_similarity))
    
    def _calculate_sacred_ratios_similarity(self, ratios1: Dict[str, float], 
                                          ratios2: Dict[str, float]) -> float:
        """Calculate sacred ratios similarity"""
        if not ratios1 or not ratios2:
            return 0.0
        
        # Get all ratio keys
        all_ratios = set(ratios1.keys()) | set(ratios2.keys())
        
        # Calculate average absolute difference
        total_diff = 0.0
        for ratio in all_ratios:
            val1 = ratios1.get(ratio, 0.5)  # Default to neutral
            val2 = ratios2.get(ratio, 0.5)
            total_diff += abs(val1 - val2)
        
        # Convert to similarity (lower difference = higher similarity)
        avg_diff = total_diff / len(all_ratios)
        similarity = 1.0 - avg_diff
        
        return max(0.0, min(1.0, similarity))
    
    def _calculate_spatial_similarity(self, coords1: Dict[str, float], 
                                    coords2: Dict[str, float]) -> float:
        """Calculate spatial similarity in archetypal space"""
        if not coords1 or not coords2:
            return 0.0
        
        # Extract coordinates
        x1, y1, z1 = coords1.get('x', 0), coords1.get('y', 0), coords1.get('z', 0)
        x2, y2, z2 = coords2.get('x', 0), coords2.get('y', 0), coords2.get('z', 0)
        
        # Calculate Euclidean distance
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        
        # Convert to similarity (smaller distance = higher similarity)
        max_distance = 10.0  # Approximate maximum distance in archetypal space
        similarity = 1.0 - min(distance / max_distance, 1.0)
        
        return max(0.0, similarity)
    
    def _calculate_compatibility_score(self, user1: UserProfile, user2: UserProfile,
                                     sig1: DBFrequencySignature, sig2: DBFrequencySignature) -> float:
        """Calculate compatibility score based on complementary traits"""
        compatibility = 0.0
        
        # Check for complementary archetypes
        if user1.primary_muse != user2.primary_muse:
            compatibility += 0.2  # Different primary muses can be complementary
        
        # Check discovery style compatibility
        if user1.discovery_style == user2.discovery_style:
            compatibility += 0.2  # Same discovery style
        elif (user1.discovery_style == "mathematical" and user2.discovery_style == "intuitive") or \
             (user1.discovery_style == "intuitive" and user2.discovery_style == "mathematical"):
            compatibility += 0.3  # Complementary styles
        
        # Check shared interests
        user1_themes = set(user1.favorite_themes or [])
        user2_themes = set(user2.favorite_themes or [])
        shared_themes = user1_themes & user2_themes
        
        if shared_themes:
            compatibility += 0.3 * min(len(shared_themes) / 3, 1.0)
        
        # Check creative form compatibility
        user1_forms = set(user1.preferred_forms or [])
        user2_forms = set(user2.preferred_forms or [])
        shared_forms = user1_forms & user2_forms
        
        if shared_forms:
            compatibility += 0.3 * min(len(shared_forms) / 3, 1.0)
        
        return min(1.0, compatibility)
    
    def _determine_resonance_type(self, archetypal_sim: float, sacred_sim: float, 
                                spatial_sim: float) -> str:
        """Determine the type of resonance based on component similarities"""
        if archetypal_sim > 0.8:
            return "archetypal_harmony"
        elif sacred_sim > 0.8:
            return "sacred_alignment"
        elif spatial_sim > 0.8:
            return "spatial_resonance"
        elif archetypal_sim > 0.6 and sacred_sim > 0.6:
            return "balanced_resonance"
        elif archetypal_sim > 0.7:
            return "archetypal_affinity"
        elif sacred_sim > 0.7:
            return "geometric_affinity"
        else:
            return "moderate_resonance"
    
    def _generate_detailed_analysis(self, user1: UserProfile, user2: UserProfile,
                                  sig1: DBFrequencySignature, sig2: DBFrequencySignature,
                                  archetypal_sim: float, sacred_sim: float, 
                                  spatial_sim: float) -> Dict[str, Any]:
        """Generate detailed analysis of resonance"""
        analysis = {
            "archetypal_analysis": {
                "similarity_score": archetypal_sim,
                "user1_primary": user1.primary_muse,
                "user2_primary": user2.primary_muse,
                "shared_archetypes": [],
                "complementary_archetypes": []
            },
            "sacred_geometry_analysis": {
                "similarity_score": sacred_sim,
                "phi_alignment": abs(sig1.sacred_ratios.get('phi', 0.5) - sig2.sacred_ratios.get('phi', 0.5)),
                "pi_alignment": abs(sig1.sacred_ratios.get('pi', 0.5) - sig2.sacred_ratios.get('pi', 0.5)),
                "fibonacci_alignment": abs(sig1.sacred_ratios.get('fibonacci', 0.5) - sig2.sacred_ratios.get('fibonacci', 0.5))
            },
            "spatial_analysis": {
                "similarity_score": spatial_sim,
                "spatial_distance": self._calculate_spatial_distance(sig1.spiral_coordinates, sig2.spiral_coordinates),
                "spatial_region_match": self._check_spatial_region_match(sig1.spiral_coordinates, sig2.spiral_coordinates)
            },
            "compatibility_factors": {
                "shared_themes": list(set(user1.favorite_themes or []) & set(user2.favorite_themes or [])),
                "shared_forms": list(set(user1.preferred_forms or []) & set(user2.preferred_forms or [])),
                "discovery_style_match": user1.discovery_style == user2.discovery_style
            }
        }
        
        # Find shared archetypes
        for archetype in sig1.harmonic_blend:
            if archetype in sig2.harmonic_blend:
                strength1 = sig1.harmonic_blend[archetype]
                strength2 = sig2.harmonic_blend[archetype]
                if strength1 > 0.2 and strength2 > 0.2:
                    analysis["archetypal_analysis"]["shared_archetypes"].append(archetype)
        
        return analysis
    
    def _calculate_spatial_distance(self, coords1: Dict[str, float], 
                                  coords2: Dict[str, float]) -> float:
        """Calculate spatial distance between coordinates"""
        x1, y1, z1 = coords1.get('x', 0), coords1.get('y', 0), coords1.get('z', 0)
        x2, y2, z2 = coords2.get('x', 0), coords2.get('y', 0), coords2.get('z', 0)
        
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    
    def _check_spatial_region_match(self, coords1: Dict[str, float], 
                                   coords2: Dict[str, float]) -> bool:
        """Check if coordinates are in the same spatial region"""
        x1, y1, z1 = coords1.get('x', 0), coords1.get('y', 0), coords1.get('z', 0)
        x2, y2, z2 = coords2.get('x', 0), coords2.get('y', 0), coords2.get('z', 0)
        
        # Simple region matching based on quadrants
        return (x1 > 0) == (x2 > 0) and (y1 > 0) == (y2 > 0) and (z1 > 0) == (z2 > 0)
    
    def _create_community_match(self, candidate: UserProfile, resonance_result: ResonanceResult,
                               user_profile: UserProfile, user_signature: DBFrequencySignature) -> CommunityMatch:
        """Create a CommunityMatch object from resonance result"""
        # Determine compatibility type
        compatibility_type = self._determine_compatibility_type(resonance_result)
        
        # Get shared archetypes
        shared_archetypes = resonance_result.detailed_analysis.get("archetypal_analysis", {}).get("shared_archetypes", [])
        
        # Get complementary traits
        complementary_traits = self._identify_complementary_traits(candidate, user_profile)
        
        # Recommend collaboration type
        collaboration_type = self._recommend_collaboration_type(resonance_result, candidate, user_profile)
        
        return CommunityMatch(
            user_id=str(candidate.id),
            username=candidate.username,
            display_name=candidate.display_name or candidate.username,
            resonance_score=resonance_result.overall_resonance,
            compatibility_type=compatibility_type,
            shared_archetypes=shared_archetypes,
            complementary_traits=complementary_traits,
            recommended_collaboration=collaboration_type
        )
    
    def _determine_compatibility_type(self, resonance_result: ResonanceResult) -> str:
        """Determine compatibility type based on resonance result"""
        if resonance_result.overall_resonance > 0.8:
            return "kindred_spirit"
        elif resonance_result.compatibility_score > 0.7:
            return "creative_partner"
        elif resonance_result.archetypal_similarity > 0.7:
            return "archetypal_twin"
        elif resonance_result.sacred_ratios_similarity > 0.7:
            return "geometric_ally"
        else:
            return "resonant_collaborator"
    
    def _identify_complementary_traits(self, candidate: UserProfile, user: UserProfile) -> List[str]:
        """Identify complementary traits between users"""
        traits = []
        
        # Different primary muses
        if candidate.primary_muse != user.primary_muse:
            traits.append(f"Complements {user.primary_muse} with {candidate.primary_muse}")
        
        # Different discovery styles
        if candidate.discovery_style != user.discovery_style:
            traits.append(f"Offers {candidate.discovery_style} perspective")
        
        # Different but compatible themes
        candidate_themes = set(candidate.favorite_themes or [])
        user_themes = set(user.favorite_themes or [])
        unique_themes = candidate_themes - user_themes
        
        if unique_themes:
            traits.append(f"Explores {', '.join(list(unique_themes)[:2])}")
        
        return traits
    
    def _recommend_collaboration_type(self, resonance_result: ResonanceResult, 
                                    candidate: UserProfile, user: UserProfile) -> str:
        """Recommend collaboration type based on resonance and profiles"""
        if resonance_result.overall_resonance > 0.8:
            return "joint_discovery_session"
        elif resonance_result.compatibility_score > 0.7:
            return "creative_duet"
        elif resonance_result.archetypal_similarity > 0.7:
            return "archetypal_harmony_creation"
        elif resonance_result.sacred_ratios_similarity > 0.7:
            return "geometric_poetry_collaboration"
        else:
            return "creative_exchange"