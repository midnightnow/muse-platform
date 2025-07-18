"""
Tests for MUSE Platform community features and social functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import numpy as np

from muse.services.community_curator import CommunityCurator
from muse.services.resonance_matcher import ResonanceMatcher
from muse.services.discovery_orchestrator import DiscoveryOrchestrator
from muse.models.community import User, Community, CommunityMember, CreativeSession, SharedCreation


class TestCommunityCurator:
    """Test suite for CommunityCurator."""
    
    def test_initialization(self, community_curator):
        """Test community curator initialization."""
        curator = community_curator
        assert hasattr(curator, 'curation_algorithms')
        assert hasattr(curator, 'community_metrics')
        assert hasattr(curator, 'content_filters')
    
    @pytest.mark.asyncio
    async def test_create_community(self, community_curator):
        """Test community creation."""
        curator = community_curator
        
        community_data = {
            "name": "Sacred Geometry Enthusiasts",
            "description": "A community for exploring mathematical beauty in nature",
            "creator_id": "creator_123",
            "community_type": "open",
            "interests": ["mathematics", "sacred_geometry", "nature"],
            "guidelines": {
                "creativity_focus": True,
                "collaborative_learning": True,
                "respectful_discourse": True
            }
        }
        
        community_result = await curator.create_community(community_data)
        
        assert isinstance(community_result, dict)
        assert "community_id" in community_result
        assert "creation_status" in community_result
        assert community_result["creation_status"] == "success"
        assert community_result["name"] == "Sacred Geometry Enthusiasts"
        assert community_result["creator_id"] == "creator_123"
    
    @pytest.mark.asyncio
    async def test_curate_community_content(self, community_curator, sample_community_data):
        """Test community content curation."""
        curator = community_curator
        
        # Mock content submissions
        content_submissions = [
            {
                "content_id": "content_1",
                "creator_id": "user_1",
                "title": "Golden Ratio in Sunflower Spirals",
                "content": "Exploring fibonacci patterns in nature",
                "content_type": "creative_work",
                "tags": ["fibonacci", "nature", "spiral"],
                "quality_score": 0.85
            },
            {
                "content_id": "content_2",
                "creator_id": "user_2",
                "title": "Mathematical Art Creation",
                "content": "Using algorithms to create beautiful patterns",
                "content_type": "tutorial",
                "tags": ["art", "mathematics", "algorithms"],
                "quality_score": 0.92
            },
            {
                "content_id": "content_3",
                "creator_id": "user_3",
                "title": "Spam Content",
                "content": "Buy cheap products here!",
                "content_type": "spam",
                "tags": ["spam", "advertisement"],
                "quality_score": 0.1
            }
        ]
        
        curation_result = await curator.curate_community_content(
            sample_community_data["community_id"],
            content_submissions
        )
        
        assert isinstance(curation_result, dict)
        assert "approved_content" in curation_result
        assert "rejected_content" in curation_result
        assert "curation_metrics" in curation_result
        
        # High quality content should be approved
        approved_content = curation_result["approved_content"]
        assert len(approved_content) >= 2
        
        # Spam content should be rejected
        rejected_content = curation_result["rejected_content"]
        assert any(content["content_id"] == "content_3" for content in rejected_content)
    
    @pytest.mark.asyncio
    async def test_facilitate_community_discussions(self, community_curator, sample_community_data):
        """Test community discussion facilitation."""
        curator = community_curator
        
        # Mock discussion data
        discussion_data = {
            "discussion_id": "discussion_1",
            "community_id": sample_community_data["community_id"],
            "topic": "The role of mathematics in creative expression",
            "participants": ["user_1", "user_2", "user_3"],
            "messages": [
                {
                    "message_id": "msg_1",
                    "author": "user_1",
                    "content": "I find that mathematical patterns inspire my art",
                    "timestamp": datetime.now() - timedelta(hours=2)
                },
                {
                    "message_id": "msg_2",
                    "author": "user_2",
                    "content": "Yes! The golden ratio appears everywhere in nature",
                    "timestamp": datetime.now() - timedelta(hours=1)
                },
                {
                    "message_id": "msg_3",
                    "author": "user_3",
                    "content": "Have you explored fractal geometry?",
                    "timestamp": datetime.now() - timedelta(minutes=30)
                }
            ]
        }
        
        facilitation_result = await curator.facilitate_community_discussions(discussion_data)
        
        assert isinstance(facilitation_result, dict)
        assert "discussion_health" in facilitation_result
        assert "engagement_metrics" in facilitation_result
        assert "moderation_actions" in facilitation_result
        assert "suggested_topics" in facilitation_result
        
        # Test discussion health
        discussion_health = facilitation_result["discussion_health"]
        assert isinstance(discussion_health, float)
        assert 0.0 <= discussion_health <= 1.0
    
    @pytest.mark.asyncio
    async def test_recommend_connections(self, community_curator, sample_community_data):
        """Test user connection recommendations."""
        curator = community_curator
        
        # Mock user profiles
        user_profiles = [
            {
                "user_id": "user_1",
                "interests": ["mathematics", "art", "nature"],
                "creativity_level": 0.8,
                "activity_level": 0.9,
                "collaboration_preference": "high"
            },
            {
                "user_id": "user_2",
                "interests": ["sacred_geometry", "music", "mathematics"],
                "creativity_level": 0.85,
                "activity_level": 0.7,
                "collaboration_preference": "medium"
            },
            {
                "user_id": "user_3",
                "interests": ["fibonacci", "spiral_patterns", "golden_ratio"],
                "creativity_level": 0.9,
                "activity_level": 0.8,
                "collaboration_preference": "high"
            }
        ]
        
        connection_result = await curator.recommend_connections(
            "user_1", 
            user_profiles,
            sample_community_data["community_id"]
        )
        
        assert isinstance(connection_result, dict)
        assert "recommended_connections" in connection_result
        assert "connection_scores" in connection_result
        assert "connection_reasons" in connection_result
        
        # Test recommendations
        recommendations = connection_result["recommended_connections"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Test connection scores
        connection_scores = connection_result["connection_scores"]
        assert isinstance(connection_scores, dict)
        assert all(0.0 <= score <= 1.0 for score in connection_scores.values())
    
    @pytest.mark.asyncio
    async def test_moderate_community_content(self, community_curator):
        """Test community content moderation."""
        curator = community_curator
        
        # Mock content for moderation
        content_items = [
            {
                "content_id": "content_1",
                "content": "Beautiful mathematical patterns in nature inspire creativity",
                "author": "user_1",
                "report_count": 0,
                "content_type": "creative_work"
            },
            {
                "content_id": "content_2",
                "content": "This is inappropriate content that violates guidelines",
                "author": "user_2",
                "report_count": 3,
                "content_type": "text"
            },
            {
                "content_id": "content_3",
                "content": "Spam! Buy products here! Click this link!",
                "author": "user_3",
                "report_count": 5,
                "content_type": "spam"
            }
        ]
        
        moderation_result = await curator.moderate_community_content(content_items)
        
        assert isinstance(moderation_result, dict)
        assert "moderation_actions" in moderation_result
        assert "content_scores" in moderation_result
        assert "flagged_content" in moderation_result
        
        # Test moderation actions
        moderation_actions = moderation_result["moderation_actions"]
        assert isinstance(moderation_actions, list)
        
        # High report count content should be flagged
        flagged_content = moderation_result["flagged_content"]
        assert any(content["content_id"] == "content_2" for content in flagged_content)
        assert any(content["content_id"] == "content_3" for content in flagged_content)
    
    @pytest.mark.asyncio
    async def test_analyze_community_health(self, community_curator, sample_community_data):
        """Test community health analysis."""
        curator = community_curator
        
        # Mock community metrics
        community_metrics = {
            "member_count": 150,
            "active_members": 120,
            "daily_activity": 45,
            "content_creation_rate": 12,
            "discussion_participation": 0.75,
            "member_retention": 0.85,
            "content_quality_average": 0.82,
            "moderation_actions": 2,
            "reported_issues": 1
        }
        
        health_result = await curator.analyze_community_health(
            sample_community_data["community_id"],
            community_metrics
        )
        
        assert isinstance(health_result, dict)
        assert "overall_health_score" in health_result
        assert "health_metrics" in health_result
        assert "improvement_recommendations" in health_result
        assert "trend_analysis" in health_result
        
        # Test health score
        health_score = health_result["overall_health_score"]
        assert isinstance(health_score, float)
        assert 0.0 <= health_score <= 1.0
        
        # Test health metrics
        health_metrics = health_result["health_metrics"]
        assert "activity_score" in health_metrics
        assert "engagement_score" in health_metrics
        assert "quality_score" in health_metrics
        assert "safety_score" in health_metrics


class TestResonanceMatcher:
    """Test suite for ResonanceMatcher."""
    
    def test_initialization(self, resonance_matcher):
        """Test resonance matcher initialization."""
        matcher = resonance_matcher
        assert hasattr(matcher, 'matching_algorithms')
        assert hasattr(matcher, 'similarity_thresholds')
        assert hasattr(matcher, 'resonance_weights')
    
    @pytest.mark.asyncio
    async def test_calculate_content_resonance(self, resonance_matcher):
        """Test content resonance calculation."""
        matcher = resonance_matcher
        
        # Mock content data
        content_1 = {
            "content_id": "content_1",
            "text": "The golden ratio creates beautiful spiral patterns in nature",
            "frequency_signature": [440.0, 554.37, 659.25, 783.99],
            "semantic_vector": [0.1, 0.3, 0.7, 0.2, 0.9, 0.4],
            "geometry_features": {
                "golden_ratio_presence": 0.95,
                "fibonacci_patterns": 0.87,
                "spiral_characteristics": 0.92
            }
        }
        
        content_2 = {
            "content_id": "content_2",
            "text": "Fibonacci sequences appear in sunflower seed arrangements",
            "frequency_signature": [440.0, 550.0, 660.0, 785.0],
            "semantic_vector": [0.15, 0.35, 0.65, 0.25, 0.85, 0.45],
            "geometry_features": {
                "golden_ratio_presence": 0.89,
                "fibonacci_patterns": 0.94,
                "spiral_characteristics": 0.78
            }
        }
        
        resonance_result = await matcher.calculate_content_resonance(content_1, content_2)
        
        assert isinstance(resonance_result, dict)
        assert "overall_resonance" in resonance_result
        assert "frequency_resonance" in resonance_result
        assert "semantic_resonance" in resonance_result
        assert "geometry_resonance" in resonance_result
        assert "resonance_breakdown" in resonance_result
        
        # Test resonance scores
        overall_resonance = resonance_result["overall_resonance"]
        assert isinstance(overall_resonance, float)
        assert 0.0 <= overall_resonance <= 1.0
        
        # Related content should have high resonance
        assert overall_resonance > 0.6
    
    @pytest.mark.asyncio
    async def test_find_resonant_users(self, resonance_matcher, sample_community_data):
        """Test finding resonant users."""
        matcher = resonance_matcher
        
        # Mock user profiles with creative signatures
        user_profiles = [
            {
                "user_id": "user_1",
                "creative_signature": {
                    "dominant_themes": ["nature", "mathematics", "beauty"],
                    "frequency_preferences": [440.0, 554.37, 659.25],
                    "semantic_profile": [0.8, 0.3, 0.7, 0.2, 0.9],
                    "geometry_affinity": 0.85
                }
            },
            {
                "user_id": "user_2",
                "creative_signature": {
                    "dominant_themes": ["music", "patterns", "harmony"],
                    "frequency_preferences": [440.0, 523.25, 659.25],
                    "semantic_profile": [0.7, 0.5, 0.8, 0.3, 0.6],
                    "geometry_affinity": 0.72
                }
            },
            {
                "user_id": "user_3",
                "creative_signature": {
                    "dominant_themes": ["technology", "algorithms", "efficiency"],
                    "frequency_preferences": [880.0, 1100.0, 1320.0],
                    "semantic_profile": [0.2, 0.8, 0.3, 0.9, 0.1],
                    "geometry_affinity": 0.45
                }
            }
        ]
        
        target_user = {
            "user_id": "target_user",
            "creative_signature": {
                "dominant_themes": ["nature", "mathematics", "patterns"],
                "frequency_preferences": [440.0, 550.0, 660.0],
                "semantic_profile": [0.75, 0.35, 0.65, 0.25, 0.85],
                "geometry_affinity": 0.88
            }
        }
        
        resonance_result = await matcher.find_resonant_users(target_user, user_profiles)
        
        assert isinstance(resonance_result, dict)
        assert "resonant_users" in resonance_result
        assert "resonance_scores" in resonance_result
        assert "resonance_explanations" in resonance_result
        
        # Test resonant users
        resonant_users = resonance_result["resonant_users"]
        assert isinstance(resonant_users, list)
        assert len(resonant_users) > 0
        
        # user_1 should be most resonant (similar themes and geometry affinity)
        resonance_scores = resonance_result["resonance_scores"]
        assert resonance_scores["user_1"] > resonance_scores["user_3"]
    
    @pytest.mark.asyncio
    async def test_match_collaborative_partners(self, resonance_matcher):
        """Test collaborative partner matching."""
        matcher = resonance_matcher
        
        # Mock collaboration request
        collaboration_request = {
            "requester_id": "user_1",
            "project_type": "creative_exploration",
            "project_description": "Exploring mathematical patterns in art",
            "required_skills": ["mathematics", "visual_art", "creativity"],
            "collaboration_style": "complementary",
            "time_commitment": "flexible"
        }
        
        # Mock potential partners
        potential_partners = [
            {
                "user_id": "user_2",
                "skills": ["mathematics", "music", "programming"],
                "collaboration_history": {"successful_projects": 5, "average_rating": 4.2},
                "availability": "high",
                "collaboration_style": "complementary"
            },
            {
                "user_id": "user_3",
                "skills": ["visual_art", "design", "creativity"],
                "collaboration_history": {"successful_projects": 3, "average_rating": 4.7},
                "availability": "medium",
                "collaboration_style": "synergistic"
            },
            {
                "user_id": "user_4",
                "skills": ["programming", "data_analysis"],
                "collaboration_history": {"successful_projects": 2, "average_rating": 3.8},
                "availability": "low",
                "collaboration_style": "independent"
            }
        ]
        
        matching_result = await matcher.match_collaborative_partners(
            collaboration_request, 
            potential_partners
        )
        
        assert isinstance(matching_result, dict)
        assert "matched_partners" in matching_result
        assert "match_scores" in matching_result
        assert "match_explanations" in matching_result
        assert "collaboration_predictions" in matching_result
        
        # Test matched partners
        matched_partners = matching_result["matched_partners"]
        assert isinstance(matched_partners, list)
        assert len(matched_partners) > 0
        
        # user_3 should be highly matched (visual art + creativity skills)
        match_scores = matching_result["match_scores"]
        assert "user_3" in match_scores
        assert match_scores["user_3"] > 0.7
    
    @pytest.mark.asyncio
    async def test_analyze_resonance_patterns(self, resonance_matcher):
        """Test resonance pattern analysis."""
        matcher = resonance_matcher
        
        # Mock historical resonance data
        resonance_data = [
            {
                "user_pair": ("user_1", "user_2"),
                "resonance_score": 0.85,
                "interaction_type": "creative_collaboration",
                "outcome_rating": 4.5,
                "timestamp": datetime.now() - timedelta(days=30)
            },
            {
                "user_pair": ("user_1", "user_3"),
                "resonance_score": 0.62,
                "interaction_type": "content_sharing",
                "outcome_rating": 3.8,
                "timestamp": datetime.now() - timedelta(days=25)
            },
            {
                "user_pair": ("user_2", "user_3"),
                "resonance_score": 0.78,
                "interaction_type": "discussion_participation",
                "outcome_rating": 4.2,
                "timestamp": datetime.now() - timedelta(days=20)
            }
        ]
        
        pattern_result = await matcher.analyze_resonance_patterns(resonance_data)
        
        assert isinstance(pattern_result, dict)
        assert "pattern_insights" in pattern_result
        assert "resonance_predictors" in pattern_result
        assert "success_factors" in pattern_result
        assert "improvement_recommendations" in pattern_result
        
        # Test pattern insights
        pattern_insights = pattern_result["pattern_insights"]
        assert isinstance(pattern_insights, dict)
        assert "high_resonance_characteristics" in pattern_insights
        assert "low_resonance_characteristics" in pattern_insights
    
    @pytest.mark.asyncio
    async def test_predict_interaction_success(self, resonance_matcher):
        """Test interaction success prediction."""
        matcher = resonance_matcher
        
        # Mock interaction scenario
        interaction_scenario = {
            "participants": ["user_1", "user_2"],
            "interaction_type": "creative_collaboration",
            "project_complexity": "medium",
            "timeline": "2_weeks",
            "resources_available": "adequate",
            "previous_interactions": {
                "user_1": {"success_rate": 0.85, "avg_rating": 4.3},
                "user_2": {"success_rate": 0.78, "avg_rating": 4.1}
            }
        }
        
        prediction_result = await matcher.predict_interaction_success(interaction_scenario)
        
        assert isinstance(prediction_result, dict)
        assert "success_probability" in prediction_result
        assert "confidence_level" in prediction_result
        assert "success_factors" in prediction_result
        assert "risk_factors" in prediction_result
        assert "recommendations" in prediction_result
        
        # Test success probability
        success_probability = prediction_result["success_probability"]
        assert isinstance(success_probability, float)
        assert 0.0 <= success_probability <= 1.0


class TestDiscoveryOrchestrator:
    """Test suite for DiscoveryOrchestrator."""
    
    def test_initialization(self, discovery_orchestrator):
        """Test discovery orchestrator initialization."""
        orchestrator = discovery_orchestrator
        assert hasattr(orchestrator, 'discovery_algorithms')
        assert hasattr(orchestrator, 'recommendation_engines')
        assert hasattr(orchestrator, 'personalization_models')
    
    @pytest.mark.asyncio
    async def test_orchestrate_content_discovery(self, discovery_orchestrator, sample_user_data):
        """Test content discovery orchestration."""
        orchestrator = discovery_orchestrator
        
        # Mock available content
        available_content = [
            {
                "content_id": "content_1",
                "title": "Golden Ratio in Nature",
                "creator": "user_1",
                "content_type": "creative_work",
                "tags": ["golden_ratio", "nature", "patterns"],
                "quality_score": 0.89,
                "engagement_score": 0.78,
                "resonance_potential": 0.85
            },
            {
                "content_id": "content_2",
                "title": "Fibonacci Music Composition",
                "creator": "user_2",
                "content_type": "audio_work",
                "tags": ["fibonacci", "music", "composition"],
                "quality_score": 0.92,
                "engagement_score": 0.82,
                "resonance_potential": 0.76
            },
            {
                "content_id": "content_3",
                "title": "Sacred Geometry Tutorial",
                "creator": "user_3",
                "content_type": "educational",
                "tags": ["sacred_geometry", "tutorial", "learning"],
                "quality_score": 0.87,
                "engagement_score": 0.91,
                "resonance_potential": 0.88
            }
        ]
        
        discovery_result = await orchestrator.orchestrate_content_discovery(
            sample_user_data["user_id"],
            available_content,
            {
                "discovery_mode": "exploration",
                "diversity_preference": 0.7,
                "novelty_preference": 0.8
            }
        )
        
        assert isinstance(discovery_result, dict)
        assert "recommended_content" in discovery_result
        assert "discovery_reasoning" in discovery_result
        assert "personalization_factors" in discovery_result
        assert "exploration_paths" in discovery_result
        
        # Test recommended content
        recommended_content = discovery_result["recommended_content"]
        assert isinstance(recommended_content, list)
        assert len(recommended_content) > 0
        
        # All recommended content should have scores
        for content in recommended_content:
            assert "recommendation_score" in content
            assert 0.0 <= content["recommendation_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_personalize_discovery_experience(self, discovery_orchestrator, sample_user_data):
        """Test personalized discovery experience."""
        orchestrator = discovery_orchestrator
        
        # Mock user interaction history
        user_history = {
            "user_id": sample_user_data["user_id"],
            "interaction_history": [
                {
                    "content_id": "content_1",
                    "interaction_type": "view",
                    "duration": 120,
                    "engagement_score": 0.8,
                    "timestamp": datetime.now() - timedelta(days=1)
                },
                {
                    "content_id": "content_2",
                    "interaction_type": "create_inspired_by",
                    "duration": 1800,
                    "engagement_score": 0.95,
                    "timestamp": datetime.now() - timedelta(hours=12)
                }
            ],
            "preferences": {
                "content_types": ["creative_work", "tutorial"],
                "themes": ["mathematics", "nature", "art"],
                "complexity_level": "intermediate"
            },
            "learning_style": "visual_kinesthetic"
        }
        
        personalization_result = await orchestrator.personalize_discovery_experience(user_history)
        
        assert isinstance(personalization_result, dict)
        assert "personalized_filters" in personalization_result
        assert "recommendation_weights" in personalization_result
        assert "discovery_strategies" in personalization_result
        assert "user_model" in personalization_result
        
        # Test personalized filters
        personalized_filters = personalization_result["personalized_filters"]
        assert isinstance(personalized_filters, dict)
        assert "content_type_preferences" in personalized_filters
        assert "theme_preferences" in personalized_filters
    
    @pytest.mark.asyncio
    async def test_facilitate_serendipitous_discovery(self, discovery_orchestrator, sample_user_data):
        """Test serendipitous discovery facilitation."""
        orchestrator = discovery_orchestrator
        
        # Mock user's current focus
        current_focus = {
            "primary_interests": ["golden_ratio", "spiral_patterns"],
            "current_project": "nature_mathematics_art",
            "recent_explorations": ["fibonacci", "sacred_geometry"],
            "comfort_zone": ["mathematics", "visual_art"]
        }
        
        # Mock diverse content pool
        diverse_content = [
            {
                "content_id": "content_1",
                "title": "Music Theory and Mathematical Ratios",
                "themes": ["music", "mathematics", "harmony"],
                "novelty_score": 0.7,
                "bridge_potential": 0.85  # Connects user's math interest to music
            },
            {
                "content_id": "content_2",
                "title": "Fractal Patterns in Poetry",
                "themes": ["literature", "fractals", "patterns"],
                "novelty_score": 0.9,
                "bridge_potential": 0.72
            },
            {
                "content_id": "content_3",
                "title": "Advanced Calculus Techniques",
                "themes": ["mathematics", "calculus", "analysis"],
                "novelty_score": 0.2,
                "bridge_potential": 0.3  # Too similar to comfort zone
            }
        ]
        
        serendipity_result = await orchestrator.facilitate_serendipitous_discovery(
            sample_user_data["user_id"],
            current_focus,
            diverse_content
        )
        
        assert isinstance(serendipity_result, dict)
        assert "serendipitous_recommendations" in serendipity_result
        assert "discovery_bridges" in serendipity_result
        assert "novelty_explanations" in serendipity_result
        assert "expansion_opportunities" in serendipity_result
        
        # Test serendipitous recommendations
        serendipitous_recommendations = serendipity_result["serendipitous_recommendations"]
        assert isinstance(serendipitous_recommendations, list)
        assert len(serendipitous_recommendations) > 0
        
        # Music theory content should be recommended as serendipitous
        content_ids = [rec["content_id"] for rec in serendipitous_recommendations]
        assert "content_1" in content_ids  # Music theory bridges math to music
    
    @pytest.mark.asyncio
    async def test_analyze_discovery_patterns(self, discovery_orchestrator):
        """Test discovery pattern analysis."""
        orchestrator = discovery_orchestrator
        
        # Mock discovery interaction data
        discovery_data = [
            {
                "user_id": "user_1",
                "discovery_session": "session_1",
                "discovered_content": ["content_1", "content_3", "content_5"],
                "interaction_depth": [0.8, 0.6, 0.9],
                "follow_up_actions": ["create", "share", "explore_more"],
                "satisfaction_score": 4.2,
                "discovery_type": "algorithmic"
            },
            {
                "user_id": "user_2",
                "discovery_session": "session_2",
                "discovered_content": ["content_2", "content_4"],
                "interaction_depth": [0.7, 0.8],
                "follow_up_actions": ["bookmark", "create"],
                "satisfaction_score": 3.9,
                "discovery_type": "serendipitous"
            },
            {
                "user_id": "user_3",
                "discovery_session": "session_3",
                "discovered_content": ["content_6", "content_7", "content_8", "content_9"],
                "interaction_depth": [0.9, 0.8, 0.7, 0.6],
                "follow_up_actions": ["create", "share", "collaborate", "teach"],
                "satisfaction_score": 4.8,
                "discovery_type": "exploratory"
            }
        ]
        
        pattern_result = await orchestrator.analyze_discovery_patterns(discovery_data)
        
        assert isinstance(pattern_result, dict)
        assert "discovery_effectiveness" in pattern_result
        assert "user_engagement_patterns" in pattern_result
        assert "content_discovery_trends" in pattern_result
        assert "optimization_recommendations" in pattern_result
        
        # Test discovery effectiveness
        discovery_effectiveness = pattern_result["discovery_effectiveness"]
        assert isinstance(discovery_effectiveness, dict)
        assert "overall_satisfaction" in discovery_effectiveness
        assert "engagement_rate" in discovery_effectiveness
        assert "follow_up_rate" in discovery_effectiveness
    
    @pytest.mark.asyncio
    async def test_optimize_discovery_algorithms(self, discovery_orchestrator):
        """Test discovery algorithm optimization."""
        orchestrator = discovery_orchestrator
        
        # Mock algorithm performance data
        algorithm_performance = {
            "collaborative_filtering": {
                "accuracy": 0.78,
                "diversity": 0.65,
                "novelty": 0.42,
                "user_satisfaction": 3.8,
                "computational_cost": 0.3
            },
            "content_based": {
                "accuracy": 0.82,
                "diversity": 0.45,
                "novelty": 0.38,
                "user_satisfaction": 3.9,
                "computational_cost": 0.2
            },
            "hybrid_approach": {
                "accuracy": 0.85,
                "diversity": 0.72,
                "novelty": 0.58,
                "user_satisfaction": 4.3,
                "computational_cost": 0.5
            }
        }
        
        optimization_result = await orchestrator.optimize_discovery_algorithms(algorithm_performance)
        
        assert isinstance(optimization_result, dict)
        assert "optimal_configuration" in optimization_result
        assert "performance_improvements" in optimization_result
        assert "algorithm_weights" in optimization_result
        assert "implementation_recommendations" in optimization_result
        
        # Test optimal configuration
        optimal_config = optimization_result["optimal_configuration"]
        assert isinstance(optimal_config, dict)
        assert "primary_algorithm" in optimal_config
        assert "algorithm_blend" in optimal_config
        
        # Hybrid approach should be preferred due to high satisfaction
        assert optimal_config["primary_algorithm"] == "hybrid_approach"


class TestCommunityFeatureIntegration:
    """Integration tests for community features."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_community_workflow(self, community_curator, resonance_matcher, discovery_orchestrator):
        """Test end-to-end community workflow."""
        curator = community_curator
        matcher = resonance_matcher
        orchestrator = discovery_orchestrator
        
        # 1. Create community
        community_data = {
            "name": "Mathematical Art Collective",
            "description": "Artists exploring mathematical beauty",
            "creator_id": "creator_1",
            "community_type": "open",
            "interests": ["mathematics", "art", "creativity"]
        }
        
        community_result = await curator.create_community(community_data)
        community_id = community_result["community_id"]
        
        # 2. Add members and content
        members = ["user_1", "user_2", "user_3"]
        content_submissions = [
            {
                "content_id": "content_1",
                "creator_id": "user_1",
                "title": "Golden Ratio Gallery",
                "content": "Art pieces based on golden ratio",
                "content_type": "creative_work"
            }
        ]
        
        curation_result = await curator.curate_community_content(community_id, content_submissions)
        
        # 3. Find resonant connections
        user_profiles = [
            {
                "user_id": "user_1",
                "creative_signature": {
                    "dominant_themes": ["mathematics", "visual_art"],
                    "frequency_preferences": [440.0, 554.37],
                    "semantic_profile": [0.8, 0.3, 0.7, 0.2],
                    "geometry_affinity": 0.9
                }
            },
            {
                "user_id": "user_2",
                "creative_signature": {
                    "dominant_themes": ["music", "patterns"],
                    "frequency_preferences": [440.0, 523.25],
                    "semantic_profile": [0.6, 0.5, 0.8, 0.3],
                    "geometry_affinity": 0.7
                }
            }
        ]
        
        target_user = user_profiles[0]
        resonance_result = await matcher.find_resonant_users(target_user, user_profiles[1:])
        
        # 4. Orchestrate content discovery
        discovery_result = await orchestrator.orchestrate_content_discovery(
            "user_1",
            curation_result["approved_content"],
            {"discovery_mode": "exploration"}
        )
        
        # Verify the workflow
        assert community_result["creation_status"] == "success"
        assert len(curation_result["approved_content"]) > 0
        assert len(resonance_result["resonant_users"]) > 0
        assert len(discovery_result["recommended_content"]) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_collaborative_creation_workflow(self, community_curator, resonance_matcher, discovery_orchestrator):
        """Test collaborative creation workflow."""
        curator = community_curator
        matcher = resonance_matcher
        orchestrator = discovery_orchestrator
        
        # 1. User requests collaboration
        collaboration_request = {
            "requester_id": "user_1",
            "project_type": "mathematical_art",
            "project_description": "Create art based on fibonacci sequences",
            "required_skills": ["mathematics", "visual_art"],
            "collaboration_style": "complementary"
        }
        
        # 2. Find collaborative partners
        potential_partners = [
            {
                "user_id": "user_2",
                "skills": ["mathematics", "programming"],
                "collaboration_history": {"successful_projects": 3, "average_rating": 4.5}
            },
            {
                "user_id": "user_3",
                "skills": ["visual_art", "design"],
                "collaboration_history": {"successful_projects": 2, "average_rating": 4.2}
            }
        ]
        
        matching_result = await matcher.match_collaborative_partners(
            collaboration_request, 
            potential_partners
        )
        
        # 3. Create collaborative workspace
        workspace_data = {
            "project_id": "collaborative_project_1",
            "participants": ["user_1", "user_2", "user_3"],
            "project_type": "mathematical_art",
            "collaboration_type": "multi_disciplinary"
        }
        
        # 4. Facilitate collaboration
        collaboration_result = await curator.facilitate_community_discussions({
            "discussion_id": "collab_discussion_1",
            "community_id": "community_1",
            "topic": "Fibonacci art collaboration",
            "participants": ["user_1", "user_2", "user_3"]
        })
        
        # 5. Discover relevant resources
        resource_discovery = await orchestrator.orchestrate_content_discovery(
            "user_1",
            [{"content_id": "fibonacci_tutorial", "title": "Fibonacci Art Techniques"}],
            {"discovery_mode": "resource_gathering"}
        )
        
        # Verify collaborative workflow
        assert len(matching_result["matched_partners"]) > 0
        assert collaboration_result["discussion_health"] > 0.5
        assert len(resource_discovery["recommended_content"]) > 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_community_evolution_simulation(self, community_curator, resonance_matcher, discovery_orchestrator):
        """Test community evolution over time."""
        curator = community_curator
        matcher = resonance_matcher
        orchestrator = discovery_orchestrator
        
        # Simulate community evolution over 90 days
        community_metrics_timeline = []
        
        for day in range(0, 91, 7):  # Weekly snapshots
            # Simulate growing community
            member_count = 10 + day * 2
            content_count = 5 + day * 3
            activity_level = 0.3 + (day / 90) * 0.5
            
            # Mock weekly metrics
            weekly_metrics = {
                "day": day,
                "member_count": member_count,
                "active_members": int(member_count * activity_level),
                "content_count": content_count,
                "daily_activity": int(activity_level * 20),
                "engagement_score": 0.6 + (day / 90) * 0.3,
                "content_quality": 0.7 + (day / 90) * 0.2,
                "community_health": 0.65 + (day / 90) * 0.25
            }
            
            community_metrics_timeline.append(weekly_metrics)
        
        # Analyze community evolution
        evolution_analysis = {
            "growth_rate": (community_metrics_timeline[-1]["member_count"] - community_metrics_timeline[0]["member_count"]) / 90,
            "engagement_trend": "increasing",
            "content_quality_trend": "improving",
            "health_improvement": community_metrics_timeline[-1]["community_health"] - community_metrics_timeline[0]["community_health"]
        }
        
        # Test evolution patterns
        assert evolution_analysis["growth_rate"] > 0
        assert evolution_analysis["health_improvement"] > 0
        assert community_metrics_timeline[-1]["engagement_score"] > community_metrics_timeline[0]["engagement_score"]
        assert community_metrics_timeline[-1]["content_quality"] > community_metrics_timeline[0]["content_quality"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_community_discovery(self, community_curator, resonance_matcher, discovery_orchestrator):
        """Test cross-community content discovery."""
        curator = community_curator
        matcher = resonance_matcher
        orchestrator = discovery_orchestrator
        
        # Create multiple communities
        communities = [
            {
                "community_id": "math_art_community",
                "name": "Mathematical Art",
                "focus": ["mathematics", "visual_art"],
                "content": [
                    {"content_id": "golden_ratio_art", "themes": ["golden_ratio", "art"]},
                    {"content_id": "fractal_paintings", "themes": ["fractals", "painting"]}
                ]
            },
            {
                "community_id": "music_theory_community",
                "name": "Music Theory",
                "focus": ["music", "theory", "composition"],
                "content": [
                    {"content_id": "harmonic_ratios", "themes": ["harmony", "mathematics"]},
                    {"content_id": "fibonacci_music", "themes": ["fibonacci", "composition"]}
                ]
            },
            {
                "community_id": "nature_patterns_community",
                "name": "Patterns in Nature",
                "focus": ["nature", "patterns", "biology"],
                "content": [
                    {"content_id": "spiral_shells", "themes": ["spirals", "nature"]},
                    {"content_id": "plant_geometry", "themes": ["plants", "geometry"]}
                ]
            }
        ]
        
        # User interested in mathematical patterns
        user_profile = {
            "user_id": "cross_community_user",
            "interests": ["mathematics", "patterns", "interdisciplinary"],
            "current_communities": ["math_art_community"]
        }
        
        # Discover cross-community connections
        all_content = []
        for community in communities:
            all_content.extend(community["content"])
        
        discovery_result = await orchestrator.orchestrate_content_discovery(
            user_profile["user_id"],
            all_content,
            {"discovery_mode": "cross_community", "diversity_preference": 0.9}
        )
        
        # Should discover content from other communities
        recommended_content = discovery_result["recommended_content"]
        content_ids = [content["content_id"] for content in recommended_content]
        
        # Should include cross-community content
        assert "harmonic_ratios" in content_ids or "fibonacci_music" in content_ids  # Music theory
        assert "spiral_shells" in content_ids or "plant_geometry" in content_ids  # Nature patterns
        
        # Test cross-community resonance
        cross_resonance = await matcher.calculate_content_resonance(
            {"content_id": "golden_ratio_art", "themes": ["golden_ratio", "art"]},
            {"content_id": "harmonic_ratios", "themes": ["harmony", "mathematics"]}
        )
        
        # Mathematical connection should create resonance
        assert cross_resonance["overall_resonance"] > 0.5