"""
Tests for MUSE Platform database models and relationships
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import json

from muse.models.community import (
    User, Community, CommunityMember, CreativeSession, 
    SharedCreation, CollaborativeProject, UserInteraction,
    CreativeWork, ValidationMetric, ResonanceScore
)
from database import Base


class TestUserModel:
    """Test suite for User model."""
    
    def test_user_creation(self, db_session):
        """Test basic user creation."""
        user = User(
            username="testuser",
            email="test@example.com",
            profile_data={
                "interests": ["mathematics", "art"],
                "creativity_level": 0.8,
                "preferred_themes": ["golden_ratio", "nature"]
            }
        )
        
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.profile_data["creativity_level"] == 0.8
        assert user.created_at is not None
        assert user.updated_at is not None
    
    def test_user_unique_constraints(self, db_session):
        """Test user unique constraints."""
        user1 = User(username="unique_user", email="unique@example.com")
        user2 = User(username="unique_user", email="different@example.com")
        
        db_session.add(user1)
        db_session.commit()
        
        # Should raise IntegrityError for duplicate username
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_user_email_unique_constraint(self, db_session):
        """Test user email unique constraint."""
        user1 = User(username="user1", email="same@example.com")
        user2 = User(username="user2", email="same@example.com")
        
        db_session.add(user1)
        db_session.commit()
        
        # Should raise IntegrityError for duplicate email
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_user_profile_data_json(self, db_session):
        """Test user profile data JSON storage."""
        complex_profile = {
            "interests": ["mathematics", "art", "music"],
            "creativity_scores": {
                "visual": 0.85,
                "musical": 0.72,
                "mathematical": 0.94
            },
            "preferences": {
                "collaboration_style": "complementary",
                "complexity_level": "advanced",
                "learning_style": "visual"
            },
            "achievements": [
                {"type": "creativity_milestone", "level": "expert", "date": "2024-01-15"},
                {"type": "collaboration_success", "count": 5, "date": "2024-01-20"}
            ]
        }
        
        user = User(
            username="profile_user",
            email="profile@example.com",
            profile_data=complex_profile
        )
        
        db_session.add(user)
        db_session.commit()
        
        # Retrieve and verify JSON data
        retrieved_user = db_session.query(User).filter_by(username="profile_user").first()
        assert retrieved_user.profile_data["creativity_scores"]["mathematical"] == 0.94
        assert len(retrieved_user.profile_data["achievements"]) == 2
    
    def test_user_relationships(self, db_session):
        """Test user relationships with other models."""
        user = User(username="rel_user", email="rel@example.com")
        db_session.add(user)
        db_session.commit()
        
        # Test creative sessions relationship
        session = CreativeSession(
            user_id=user.id,
            session_type="exploration",
            session_data={"prompt": "Test exploration"}
        )
        db_session.add(session)
        db_session.commit()
        
        assert len(user.creative_sessions) == 1
        assert user.creative_sessions[0].session_type == "exploration"
    
    def test_user_timestamps(self, db_session):
        """Test user timestamp behavior."""
        user = User(username="timestamp_user", email="timestamp@example.com")
        db_session.add(user)
        db_session.commit()
        
        original_created_at = user.created_at
        original_updated_at = user.updated_at
        
        # Update user
        user.profile_data = {"updated": True}
        db_session.commit()
        
        assert user.created_at == original_created_at
        assert user.updated_at > original_updated_at


class TestCommunityModel:
    """Test suite for Community model."""
    
    def test_community_creation(self, db_session):
        """Test basic community creation."""
        creator = User(username="creator", email="creator@example.com")
        db_session.add(creator)
        db_session.commit()
        
        community = Community(
            name="Test Community",
            description="A community for testing",
            creator_id=creator.id,
            community_type="open",
            settings={
                "privacy_level": "public",
                "moderation_level": "moderate",
                "collaboration_enabled": True
            }
        )
        
        db_session.add(community)
        db_session.commit()
        
        assert community.id is not None
        assert community.name == "Test Community"
        assert community.creator_id == creator.id
        assert community.community_type == "open"
        assert community.settings["collaboration_enabled"] is True
    
    def test_community_creator_relationship(self, db_session):
        """Test community creator relationship."""
        creator = User(username="community_creator", email="creator@example.com")
        db_session.add(creator)
        db_session.commit()
        
        community = Community(
            name="Creator's Community",
            description="Test community",
            creator_id=creator.id,
            community_type="open"
        )
        
        db_session.add(community)
        db_session.commit()
        
        assert community.creator.username == "community_creator"
        assert creator.created_communities[0].name == "Creator's Community"
    
    def test_community_settings_json(self, db_session):
        """Test community settings JSON storage."""
        creator = User(username="settings_creator", email="settings@example.com")
        db_session.add(creator)
        db_session.commit()
        
        complex_settings = {
            "privacy": {
                "visibility": "public",
                "join_approval": False,
                "content_visibility": "members_only"
            },
            "moderation": {
                "auto_moderation": True,
                "content_approval": False,
                "banned_words": ["spam", "inappropriate"]
            },
            "features": {
                "collaborative_creation": True,
                "real_time_chat": True,
                "file_sharing": True,
                "video_calls": False
            },
            "analytics": {
                "track_engagement": True,
                "generate_reports": True,
                "share_with_members": False
            }
        }
        
        community = Community(
            name="Settings Test Community",
            description="Testing settings",
            creator_id=creator.id,
            community_type="open",
            settings=complex_settings
        )
        
        db_session.add(community)
        db_session.commit()
        
        retrieved_community = db_session.query(Community).filter_by(name="Settings Test Community").first()
        assert retrieved_community.settings["features"]["collaborative_creation"] is True
        assert len(retrieved_community.settings["moderation"]["banned_words"]) == 2


class TestCommunityMemberModel:
    """Test suite for CommunityMember model."""
    
    def test_community_member_creation(self, db_session):
        """Test basic community member creation."""
        # Create user and community
        user = User(username="member_user", email="member@example.com")
        creator = User(username="creator", email="creator@example.com")
        db_session.add_all([user, creator])
        db_session.commit()
        
        community = Community(
            name="Member Test Community",
            description="Testing membership",
            creator_id=creator.id,
            community_type="open"
        )
        db_session.add(community)
        db_session.commit()
        
        # Create membership
        membership = CommunityMember(
            user_id=user.id,
            community_id=community.id,
            role="member",
            member_data={
                "join_reason": "interested in mathematics",
                "contribution_level": "active",
                "specialties": ["golden_ratio", "fractals"]
            }
        )
        
        db_session.add(membership)
        db_session.commit()
        
        assert membership.id is not None
        assert membership.user_id == user.id
        assert membership.community_id == community.id
        assert membership.role == "member"
        assert membership.member_data["contribution_level"] == "active"
    
    def test_community_member_relationships(self, db_session):
        """Test community member relationships."""
        # Create user and community
        user = User(username="rel_member", email="rel_member@example.com")
        creator = User(username="rel_creator", email="rel_creator@example.com")
        db_session.add_all([user, creator])
        db_session.commit()
        
        community = Community(
            name="Relationship Test Community",
            description="Testing relationships",
            creator_id=creator.id,
            community_type="open"
        )
        db_session.add(community)
        db_session.commit()
        
        # Create membership
        membership = CommunityMember(
            user_id=user.id,
            community_id=community.id,
            role="moderator"
        )
        db_session.add(membership)
        db_session.commit()
        
        # Test relationships
        assert membership.user.username == "rel_member"
        assert membership.community.name == "Relationship Test Community"
        assert user.community_memberships[0].role == "moderator"
        assert community.members[0].user.username == "rel_member"
    
    def test_community_member_unique_constraint(self, db_session):
        """Test unique constraint on user-community membership."""
        # Create user and community
        user = User(username="unique_member", email="unique@example.com")
        creator = User(username="unique_creator", email="unique_creator@example.com")
        db_session.add_all([user, creator])
        db_session.commit()
        
        community = Community(
            name="Unique Test Community",
            description="Testing uniqueness",
            creator_id=creator.id,
            community_type="open"
        )
        db_session.add(community)
        db_session.commit()
        
        # Create first membership
        membership1 = CommunityMember(
            user_id=user.id,
            community_id=community.id,
            role="member"
        )
        db_session.add(membership1)
        db_session.commit()
        
        # Try to create duplicate membership
        membership2 = CommunityMember(
            user_id=user.id,
            community_id=community.id,
            role="moderator"
        )
        db_session.add(membership2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestCreativeSessionModel:
    """Test suite for CreativeSession model."""
    
    def test_creative_session_creation(self, db_session):
        """Test basic creative session creation."""
        user = User(username="session_user", email="session@example.com")
        db_session.add(user)
        db_session.commit()
        
        session = CreativeSession(
            user_id=user.id,
            session_type="exploration",
            session_data={
                "initial_prompt": "Explore golden ratio in nature",
                "duration": 1800,
                "interactions": 12,
                "generated_content": {
                    "text": "The golden ratio appears in spiral galaxies",
                    "frequency_signature": [440.0, 554.37, 659.25],
                    "geometry_data": {"spiral_points": [[0, 0], [1, 1.618]]}
                }
            }
        )
        
        db_session.add(session)
        db_session.commit()
        
        assert session.id is not None
        assert session.user_id == user.id
        assert session.session_type == "exploration"
        assert session.session_data["duration"] == 1800
        assert len(session.session_data["generated_content"]["frequency_signature"]) == 3
    
    def test_creative_session_relationships(self, db_session):
        """Test creative session relationships."""
        user = User(username="session_rel_user", email="session_rel@example.com")
        db_session.add(user)
        db_session.commit()
        
        session = CreativeSession(
            user_id=user.id,
            session_type="collaboration",
            session_data={"collaborative_project": "fibonacci_art"}
        )
        db_session.add(session)
        db_session.commit()
        
        assert session.user.username == "session_rel_user"
        assert user.creative_sessions[0].session_type == "collaboration"
    
    def test_creative_session_complex_data(self, db_session):
        """Test creative session with complex data structures."""
        user = User(username="complex_session_user", email="complex@example.com")
        db_session.add(user)
        db_session.commit()
        
        complex_session_data = {
            "session_metadata": {
                "start_time": "2024-01-01T10:00:00Z",
                "end_time": "2024-01-01T11:30:00Z",
                "platform": "MUSE",
                "version": "1.0.0"
            },
            "user_inputs": [
                {"timestamp": "2024-01-01T10:00:00Z", "input": "Show me fibonacci in nature"},
                {"timestamp": "2024-01-01T10:15:00Z", "input": "Create a spiral pattern"},
                {"timestamp": "2024-01-01T10:30:00Z", "input": "Add golden ratio proportions"}
            ],
            "generated_outputs": [
                {
                    "timestamp": "2024-01-01T10:01:00Z",
                    "output_type": "text",
                    "content": "Fibonacci spirals appear in nautilus shells",
                    "validation_scores": {"creativity": 0.85, "coherence": 0.92}
                },
                {
                    "timestamp": "2024-01-01T10:16:00Z",
                    "output_type": "visualization",
                    "content": {"spiral_data": [[0, 0], [1, 1], [2, 3], [5, 8]]},
                    "validation_scores": {"creativity": 0.78, "coherence": 0.89}
                }
            ],
            "interaction_analysis": {
                "engagement_level": 0.92,
                "learning_progression": 0.76,
                "satisfaction_score": 4.3,
                "flow_state_indicators": ["sustained_attention", "creative_momentum"]
            }
        }
        
        session = CreativeSession(
            user_id=user.id,
            session_type="deep_exploration",
            session_data=complex_session_data
        )
        
        db_session.add(session)
        db_session.commit()
        
        retrieved_session = db_session.query(CreativeSession).filter_by(user_id=user.id).first()
        assert len(retrieved_session.session_data["user_inputs"]) == 3
        assert retrieved_session.session_data["interaction_analysis"]["engagement_level"] == 0.92
        assert "creative_momentum" in retrieved_session.session_data["interaction_analysis"]["flow_state_indicators"]


class TestSharedCreationModel:
    """Test suite for SharedCreation model."""
    
    def test_shared_creation_creation(self, db_session):
        """Test basic shared creation creation."""
        creator = User(username="creation_creator", email="creation@example.com")
        db_session.add(creator)
        db_session.commit()
        
        shared_creation = SharedCreation(
            creator_id=creator.id,
            title="Golden Spiral Art",
            description="Art piece inspired by golden ratio",
            content_type="visual_art",
            creation_data={
                "medium": "digital_art",
                "dimensions": {"width": 1920, "height": 1080},
                "color_palette": ["#FFD700", "#DAA520", "#B8860B"],
                "mathematical_elements": {
                    "golden_ratio_usage": 0.95,
                    "fibonacci_sequence": [1, 1, 2, 3, 5, 8, 13],
                    "spiral_equations": ["r = a * φ^θ"]
                }
            },
            tags=["golden_ratio", "spiral", "digital_art", "mathematics"]
        )
        
        db_session.add(shared_creation)
        db_session.commit()
        
        assert shared_creation.id is not None
        assert shared_creation.creator_id == creator.id
        assert shared_creation.title == "Golden Spiral Art"
        assert shared_creation.content_type == "visual_art"
        assert shared_creation.creation_data["mathematical_elements"]["golden_ratio_usage"] == 0.95
        assert "mathematics" in shared_creation.tags
    
    def test_shared_creation_relationships(self, db_session):
        """Test shared creation relationships."""
        creator = User(username="shared_creator", email="shared@example.com")
        db_session.add(creator)
        db_session.commit()
        
        creation = SharedCreation(
            creator_id=creator.id,
            title="Test Creation",
            description="Test description",
            content_type="text",
            creation_data={"content": "Test content"}
        )
        
        db_session.add(creation)
        db_session.commit()
        
        assert creation.creator.username == "shared_creator"
        assert creator.shared_creations[0].title == "Test Creation"
    
    def test_shared_creation_tags_array(self, db_session):
        """Test shared creation tags as array."""
        creator = User(username="tags_creator", email="tags@example.com")
        db_session.add(creator)
        db_session.commit()
        
        creation = SharedCreation(
            creator_id=creator.id,
            title="Tagged Creation",
            description="Testing tags",
            content_type="multimedia",
            creation_data={"type": "interactive"},
            tags=["fibonacci", "golden_ratio", "interactive", "mathematics", "art", "spiral"]
        )
        
        db_session.add(creation)
        db_session.commit()
        
        retrieved_creation = db_session.query(SharedCreation).filter_by(title="Tagged Creation").first()
        assert len(retrieved_creation.tags) == 6
        assert "fibonacci" in retrieved_creation.tags
        assert "spiral" in retrieved_creation.tags


class TestValidationMetricModel:
    """Test suite for ValidationMetric model."""
    
    def test_validation_metric_creation(self, db_session):
        """Test basic validation metric creation."""
        user = User(username="validation_user", email="validation@example.com")
        db_session.add(user)
        db_session.commit()
        
        session = CreativeSession(
            user_id=user.id,
            session_type="validation_test",
            session_data={"test": "data"}
        )
        db_session.add(session)
        db_session.commit()
        
        validation_metric = ValidationMetric(
            session_id=session.id,
            metric_type="creativity_assessment",
            metric_data={
                "creativity_score": 0.85,
                "originality_score": 0.78,
                "complexity_score": 0.92,
                "fluency_score": 0.81,
                "flexibility_score": 0.74,
                "elaboration_score": 0.88,
                "mathematical_coherence": 0.91,
                "semantic_relevance": 0.83
            }
        )
        
        db_session.add(validation_metric)
        db_session.commit()
        
        assert validation_metric.id is not None
        assert validation_metric.session_id == session.id
        assert validation_metric.metric_type == "creativity_assessment"
        assert validation_metric.metric_data["creativity_score"] == 0.85
        assert validation_metric.metric_data["mathematical_coherence"] == 0.91
    
    def test_validation_metric_relationships(self, db_session):
        """Test validation metric relationships."""
        user = User(username="val_rel_user", email="val_rel@example.com")
        db_session.add(user)
        db_session.commit()
        
        session = CreativeSession(
            user_id=user.id,
            session_type="validation_relationship_test",
            session_data={"relationship": "test"}
        )
        db_session.add(session)
        db_session.commit()
        
        validation_metric = ValidationMetric(
            session_id=session.id,
            metric_type="engagement_analysis",
            metric_data={
                "engagement_score": 0.87,
                "retention_score": 0.92,
                "satisfaction_score": 4.3
            }
        )
        
        db_session.add(validation_metric)
        db_session.commit()
        
        assert validation_metric.session.user.username == "val_rel_user"
        assert session.validation_metrics[0].metric_type == "engagement_analysis"
    
    def test_validation_metric_complex_data(self, db_session):
        """Test validation metric with complex data structures."""
        user = User(username="complex_val_user", email="complex_val@example.com")
        db_session.add(user)
        db_session.commit()
        
        session = CreativeSession(
            user_id=user.id,
            session_type="complex_validation",
            session_data={"complex": "test"}
        )
        db_session.add(session)
        db_session.commit()
        
        complex_metric_data = {
            "overall_scores": {
                "creativity": 0.85,
                "coherence": 0.92,
                "engagement": 0.78,
                "learning": 0.83
            },
            "detailed_analysis": {
                "creativity_components": {
                    "originality": 0.82,
                    "fluency": 0.79,
                    "flexibility": 0.88,
                    "elaboration": 0.85
                },
                "coherence_components": {
                    "logical_flow": 0.91,
                    "mathematical_consistency": 0.94,
                    "semantic_alignment": 0.89
                }
            },
            "statistical_analysis": {
                "confidence_intervals": {
                    "creativity": [0.78, 0.92],
                    "coherence": [0.88, 0.96]
                },
                "p_values": {
                    "creativity_improvement": 0.002,
                    "coherence_improvement": 0.001
                },
                "effect_sizes": {
                    "creativity_effect": 0.72,
                    "coherence_effect": 0.85
                }
            },
            "temporal_analysis": {
                "progression_rate": 0.15,
                "learning_velocity": 0.23,
                "skill_acquisition": [
                    {"skill": "pattern_recognition", "progress": 0.82},
                    {"skill": "creative_synthesis", "progress": 0.76},
                    {"skill": "mathematical_thinking", "progress": 0.91}
                ]
            }
        }
        
        validation_metric = ValidationMetric(
            session_id=session.id,
            metric_type="comprehensive_assessment",
            metric_data=complex_metric_data
        )
        
        db_session.add(validation_metric)
        db_session.commit()
        
        retrieved_metric = db_session.query(ValidationMetric).filter_by(session_id=session.id).first()
        assert retrieved_metric.metric_data["overall_scores"]["creativity"] == 0.85
        assert retrieved_metric.metric_data["statistical_analysis"]["p_values"]["creativity_improvement"] == 0.002
        assert len(retrieved_metric.metric_data["temporal_analysis"]["skill_acquisition"]) == 3


class TestResonanceScoreModel:
    """Test suite for ResonanceScore model."""
    
    def test_resonance_score_creation(self, db_session):
        """Test basic resonance score creation."""
        user1 = User(username="resonance_user1", email="res1@example.com")
        user2 = User(username="resonance_user2", email="res2@example.com")
        db_session.add_all([user1, user2])
        db_session.commit()
        
        creation1 = SharedCreation(
            creator_id=user1.id,
            title="Creation 1",
            description="First creation",
            content_type="text",
            creation_data={"content": "First content"}
        )
        
        creation2 = SharedCreation(
            creator_id=user2.id,
            title="Creation 2",
            description="Second creation",
            content_type="text",
            creation_data={"content": "Second content"}
        )
        
        db_session.add_all([creation1, creation2])
        db_session.commit()
        
        resonance_score = ResonanceScore(
            user_id=user1.id,
            target_id=creation2.id,
            target_type="shared_creation",
            score_data={
                "overall_resonance": 0.78,
                "frequency_resonance": 0.82,
                "semantic_resonance": 0.75,
                "geometry_resonance": 0.77,
                "emotional_resonance": 0.81,
                "calculated_at": "2024-01-01T12:00:00Z"
            }
        )
        
        db_session.add(resonance_score)
        db_session.commit()
        
        assert resonance_score.id is not None
        assert resonance_score.user_id == user1.id
        assert resonance_score.target_id == creation2.id
        assert resonance_score.target_type == "shared_creation"
        assert resonance_score.score_data["overall_resonance"] == 0.78
        assert resonance_score.score_data["frequency_resonance"] == 0.82
    
    def test_resonance_score_relationships(self, db_session):
        """Test resonance score relationships."""
        user = User(username="res_rel_user", email="res_rel@example.com")
        db_session.add(user)
        db_session.commit()
        
        creation = SharedCreation(
            creator_id=user.id,
            title="Resonance Test Creation",
            description="Testing resonance",
            content_type="art",
            creation_data={"art_type": "digital"}
        )
        db_session.add(creation)
        db_session.commit()
        
        resonance_score = ResonanceScore(
            user_id=user.id,
            target_id=creation.id,
            target_type="shared_creation",
            score_data={"overall_resonance": 0.85}
        )
        
        db_session.add(resonance_score)
        db_session.commit()
        
        assert resonance_score.user.username == "res_rel_user"
        assert user.resonance_scores[0].target_id == creation.id


class TestModelIntegration:
    """Integration tests for model relationships and constraints."""
    
    def test_complete_user_journey(self, db_session):
        """Test complete user journey through all models."""
        # 1. Create user
        user = User(
            username="journey_user",
            email="journey@example.com",
            profile_data={"interests": ["mathematics", "art"]}
        )
        db_session.add(user)
        db_session.commit()
        
        # 2. Create community
        community = Community(
            name="Journey Community",
            description="Testing user journey",
            creator_id=user.id,
            community_type="open"
        )
        db_session.add(community)
        db_session.commit()
        
        # 3. Create membership
        membership = CommunityMember(
            user_id=user.id,
            community_id=community.id,
            role="creator"
        )
        db_session.add(membership)
        db_session.commit()
        
        # 4. Create creative session
        session = CreativeSession(
            user_id=user.id,
            session_type="community_creation",
            session_data={"community_id": community.id}
        )
        db_session.add(session)
        db_session.commit()
        
        # 5. Create shared creation
        creation = SharedCreation(
            creator_id=user.id,
            title="Journey Creation",
            description="Created during journey",
            content_type="mixed",
            creation_data={"community_id": community.id}
        )
        db_session.add(creation)
        db_session.commit()
        
        # 6. Create validation metric
        validation = ValidationMetric(
            session_id=session.id,
            metric_type="journey_validation",
            metric_data={"journey_success": 0.95}
        )
        db_session.add(validation)
        db_session.commit()
        
        # 7. Create resonance score
        resonance = ResonanceScore(
            user_id=user.id,
            target_id=creation.id,
            target_type="shared_creation",
            score_data={"self_resonance": 0.92}
        )
        db_session.add(resonance)
        db_session.commit()
        
        # Verify all relationships work
        assert user.created_communities[0].name == "Journey Community"
        assert user.community_memberships[0].role == "creator"
        assert user.creative_sessions[0].session_type == "community_creation"
        assert user.shared_creations[0].title == "Journey Creation"
        assert user.creative_sessions[0].validation_metrics[0].metric_type == "journey_validation"
        assert user.resonance_scores[0].score_data["self_resonance"] == 0.92
    
    def test_cascade_deletion(self, db_session):
        """Test cascade deletion behavior."""
        # Create user with related objects
        user = User(username="cascade_user", email="cascade@example.com")
        db_session.add(user)
        db_session.commit()
        
        session = CreativeSession(
            user_id=user.id,
            session_type="cascade_test",
            session_data={"test": "cascade"}
        )
        db_session.add(session)
        db_session.commit()
        
        validation = ValidationMetric(
            session_id=session.id,
            metric_type="cascade_validation",
            metric_data={"test": "cascade"}
        )
        db_session.add(validation)
        db_session.commit()
        
        # Delete user should cascade to related objects
        db_session.delete(user)
        db_session.commit()
        
        # Verify cascade deletion
        assert db_session.query(CreativeSession).filter_by(id=session.id).first() is None
        assert db_session.query(ValidationMetric).filter_by(id=validation.id).first() is None
    
    def test_model_constraints_and_validations(self, db_session):
        """Test model constraints and validations."""
        # Test required fields
        user = User()  # Missing required fields
        db_session.add(user)
        
        with pytest.raises(IntegrityError):
            db_session.commit()
        
        db_session.rollback()
        
        # Test valid creation
        valid_user = User(username="valid_user", email="valid@example.com")
        db_session.add(valid_user)
        db_session.commit()
        
        assert valid_user.id is not None
    
    def test_json_field_operations(self, db_session):
        """Test JSON field operations and queries."""
        user = User(
            username="json_user",
            email="json@example.com",
            profile_data={
                "preferences": {
                    "theme": "dark",
                    "notifications": True,
                    "language": "en"
                },
                "scores": {
                    "creativity": 0.85,
                    "engagement": 0.92
                }
            }
        )
        db_session.add(user)
        db_session.commit()
        
        # Test JSON field access
        retrieved_user = db_session.query(User).filter_by(username="json_user").first()
        assert retrieved_user.profile_data["preferences"]["theme"] == "dark"
        assert retrieved_user.profile_data["scores"]["creativity"] == 0.85
        
        # Test JSON field update
        retrieved_user.profile_data["scores"]["creativity"] = 0.90
        db_session.commit()
        
        # Verify update
        updated_user = db_session.query(User).filter_by(username="json_user").first()
        assert updated_user.profile_data["scores"]["creativity"] == 0.90
    
    def test_timestamp_behavior(self, db_session):
        """Test timestamp behavior across models."""
        user = User(username="timestamp_test", email="timestamp@example.com")
        db_session.add(user)
        db_session.commit()
        
        creation_time = user.created_at
        
        # Update user
        user.profile_data = {"updated": True}
        db_session.commit()
        
        # created_at should remain unchanged, updated_at should change
        assert user.created_at == creation_time
        assert user.updated_at > creation_time
        
        # Test with other models
        session = CreativeSession(
            user_id=user.id,
            session_type="timestamp_test",
            session_data={"test": "timestamp"}
        )
        db_session.add(session)
        db_session.commit()
        
        session_creation_time = session.created_at
        
        # Update session
        session.session_data = {"test": "updated"}
        db_session.commit()
        
        assert session.created_at == session_creation_time
        assert session.updated_at > session_creation_time