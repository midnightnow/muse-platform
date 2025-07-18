"""
MUSE API Endpoints Integration Tests
Comprehensive testing for all API routes and functionality
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app


class TestCoreAPIEndpoints:
    """Test core MUSE API endpoints"""
    
    @pytest.mark.api
    def test_root_endpoint(self, client):
        """Test root endpoint returns platform information"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "services" in data
        assert "endpoints" in data
        assert "muse_companion" in data["services"]
        
    @pytest.mark.api
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["muse"] == "operational"
        
    @pytest.mark.api
    def test_assessment_start_endpoint(self, client):
        """Test personality assessment start"""
        response = client.post("/api/muse/assessment/start")
        assert response.status_code == 200
        
        data = response.json()
        assert "assessment_id" in data
        assert "questions" in data
        assert len(data["questions"]) > 0
        
    @pytest.mark.api
    def test_assessment_complete_endpoint(self, client, assessment_data):
        """Test personality assessment completion"""
        response = client.post("/api/muse/assessment/complete", json=assessment_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "signature_id" in data
        assert "frequency_signature" in data
        
        signature = data["frequency_signature"]
        assert "primary_muse" in signature
        assert "harmonic_blend" in signature
        assert "sacred_ratios" in signature
        assert "spiral_coordinates" in signature


class TestFrequencySignatureEndpoints:
    """Test frequency signature management endpoints"""
    
    @pytest.mark.api
    def test_get_signature_endpoint(self, client, sample_frequency_signature):
        """Test retrieving frequency signature"""
        # First create a signature (mock)
        with patch('muse.services.discovery_orchestrator.DiscoveryOrchestrator.get_signature') as mock_get:
            mock_get.return_value = sample_frequency_signature
            
            response = client.get("/api/muse/signatures/test_signature_id")
            assert response.status_code == 200
            
            data = response.json()
            assert "primary_muse" in data
            assert "harmonic_blend" in data
            
    @pytest.mark.api
    def test_tune_signature_endpoint(self, client):
        """Test frequency signature tuning"""
        tune_data = {
            "target_muses": ["ERATO", "SOPHIA"],
            "blend_ratios": [0.7, 0.3],
            "sacred_ratio_adjustments": {
                "phi": 0.1,
                "pi": -0.05
            }
        }
        
        response = client.post("/api/muse/signatures/test_id/tune", json=tune_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "tuned_signature" in data
        assert "changes_applied" in data


class TestDiscoveryEndpoints:
    """Test creative discovery endpoints"""
    
    @pytest.mark.api
    def test_start_session_endpoint(self, client):
        """Test starting a creative discovery session"""
        session_data = {
            "user_id": "test_user",
            "signature_id": "test_signature",
            "discovery_mode": "guided",
            "constraints": {
                "form_type": "sonnet",
                "theme": "nature",
                "sacred_constant": "phi"
            }
        }
        
        response = client.post("/api/muse/sessions/start", json=session_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "session_id" in data
        assert "initial_discovery" in data
        assert "status" in data
        
    @pytest.mark.api
    def test_session_status_endpoint(self, client):
        """Test getting session status"""
        with patch('muse.services.discovery_orchestrator.DiscoveryOrchestrator.get_session_status') as mock_status:
            mock_status.return_value = {
                "session_id": "test_session",
                "status": "active",
                "progress": 0.3,
                "current_iteration": 5,
                "fitness_scores": [0.6, 0.7, 0.75, 0.78, 0.82]
            }
            
            response = client.get("/api/muse/sessions/test_session/status")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "active"
            assert "progress" in data
            assert "fitness_scores" in data


class TestLiveDiscoveryEndpoints:
    """Test live discovery integration endpoints"""
    
    @pytest.mark.api
    def test_discover_poem_endpoint(self, client):
        """Test real-time poem discovery"""
        discovery_request = {
            "signature_id": "test_signature",
            "form_constraints": {
                "form_type": "haiku",
                "syllable_pattern": [5, 7, 5]
            },
            "semantic_constraints": {
                "theme": "nature",
                "emotion": "tranquil"
            },
            "sacred_constant": "phi",
            "max_iterations": 10
        }
        
        response = client.post("/api/muse/live/discover-poem", json=discovery_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "discovered_poem" in data
        assert "mathematical_fitness" in data
        assert "semantic_coherence" in data
        assert "discovery_path" in data
        
    @pytest.mark.api
    def test_optimize_constraints_endpoint(self, client):
        """Test constraint optimization"""
        optimization_request = {
            "current_constraints": {
                "form_type": "sonnet",
                "theme": "love",
                "sacred_constant": "pi"
            },
            "optimization_goals": {
                "mathematical_fitness": 0.9,
                "semantic_coherence": 0.85,
                "archetypal_alignment": 0.8
            },
            "signature_id": "test_signature"
        }
        
        response = client.post("/api/muse/live/optimize-constraints", json=optimization_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "optimized_constraints" in data
        assert "fitness_improvement" in data
        assert "optimization_steps" in data


class TestCommunityEndpoints:
    """Test community features endpoints"""
    
    @pytest.mark.api
    def test_create_profile_endpoint(self, client, sample_user_profile):
        """Test creating user profile"""
        response = client.post("/api/muse/community/profiles/create", json=sample_user_profile)
        assert response.status_code == 201
        
        data = response.json()
        assert "profile_id" in data
        assert "message" in data
        assert data["username"] == sample_user_profile["username"]
        
    @pytest.mark.api
    def test_get_profile_endpoint(self, client):
        """Test retrieving user profile"""
        with patch('muse.api.community.get_user_profile_from_db') as mock_get:
            mock_profile = {
                "id": "test_id",
                "username": "test_user",
                "primary_muse": "CALLIOPE",
                "resonance_strength": 0.85
            }
            mock_get.return_value = mock_profile
            
            response = client.get("/api/muse/community/profiles/test_id")
            assert response.status_code == 200
            
            data = response.json()
            assert data["username"] == "test_user"
            assert data["primary_muse"] == "CALLIOPE"
            
    @pytest.mark.api
    def test_share_creation_endpoint(self, client):
        """Test sharing a creative discovery"""
        creation_data = {
            "creator_id": "test_user",
            "title": "Golden Spiral Sonnet",
            "content_preview": "In golden spirals mathematics dance...",
            "form_type": "sonnet",
            "mathematical_fitness": 0.92,
            "semantic_coherence": 0.88,
            "discovery_coordinates": {
                "entropy_seed": 0.618,
                "sacred_constant": "phi",
                "theme": "mathematics"
            },
            "tags": ["golden_ratio", "mathematics", "beauty"]
        }
        
        response = client.post("/api/muse/community/creations/share", json=creation_data)
        assert response.status_code == 201
        
        data = response.json()
        assert "creation_id" in data
        assert "message" in data
        
    @pytest.mark.api
    def test_gallery_endpoint(self, client):
        """Test community gallery with filtering"""
        # Test basic gallery
        response = client.get("/api/muse/community/gallery")
        assert response.status_code == 200
        
        data = response.json()
        assert "creations" in data
        assert "total_count" in data
        assert "page" in data
        
        # Test with filters
        response = client.get("/api/muse/community/gallery?form_type=sonnet&min_fitness=0.8")
        assert response.status_code == 200
        
        data = response.json()
        assert "creations" in data
        
    @pytest.mark.api
    def test_kindred_spirits_endpoint(self, client):
        """Test finding kindred spirits"""
        with patch('muse.services.resonance_matcher.ResonanceMatcher.find_kindred_spirits') as mock_find:
            mock_find.return_value = [
                {
                    "user_id": "kindred_1",
                    "username": "math_poet",
                    "resonance_score": 0.89,
                    "shared_muses": ["CALLIOPE", "URANIA"],
                    "compatibility_reasons": ["shared_golden_ratio_affinity", "similar_cosmic_themes"]
                }
            ]
            
            response = client.get("/api/muse/community/kindred/test_user")
            assert response.status_code == 200
            
            data = response.json()
            assert "kindred_spirits" in data
            assert len(data["kindred_spirits"]) > 0
            assert data["kindred_spirits"][0]["resonance_score"] > 0.8


class TestValidationEndpoints:
    """Test validation framework endpoints"""
    
    @pytest.mark.api
    @pytest.mark.validation
    def test_validation_summary_endpoint(self, client):
        """Test validation dashboard summary"""
        response = client.get("/api/muse/validation/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_experiments" in data
        assert "active_experiments" in data
        assert "recent_insights" in data
        assert "platform_health" in data
        
    @pytest.mark.api
    @pytest.mark.validation
    def test_create_experiment_endpoint(self, client):
        """Test creating validation experiment"""
        experiment_data = {
            "hypothesis": "Sacred geometry constraints improve creative output quality",
            "control_description": "Standard poetic forms without optimization",
            "experimental_description": "Forms optimized with golden ratio and fibonacci",
            "variables": ["mathematical_fitness", "user_satisfaction"],
            "sample_size": 100,
            "duration_days": 30
        }
        
        response = client.post("/api/muse/validation/experiment", json=experiment_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "experiment_id" in data
        assert "message" in data
        
    @pytest.mark.api
    @pytest.mark.validation
    def test_experiment_details_endpoint(self, client):
        """Test getting experiment details"""
        with patch('muse.validation.validation_dashboard.ValidationDashboard.get_experiment_details') as mock_details:
            mock_details.return_value = {
                "experiment_id": "test_exp",
                "hypothesis": "Test hypothesis",
                "status": "running",
                "progress": 45.2,
                "participants_recruited": 67
            }
            
            response = client.get("/api/muse/validation/experiment/test_exp")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "running"
            assert "progress" in data


class TestErrorHandling:
    """Test API error handling and edge cases"""
    
    @pytest.mark.api
    def test_invalid_signature_id(self, client):
        """Test handling of invalid signature IDs"""
        response = client.get("/api/muse/signatures/invalid_id")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data or "detail" in data
        
    @pytest.mark.api
    def test_malformed_json_request(self, client):
        """Test handling of malformed JSON"""
        response = client.post(
            "/api/muse/assessment/complete",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
    @pytest.mark.api
    def test_missing_required_fields(self, client):
        """Test validation of required fields"""
        incomplete_data = {
            "username": "test_user"
            # Missing required fields like email
        }
        
        response = client.post("/api/muse/community/profiles/create", json=incomplete_data)
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        
    @pytest.mark.api
    def test_rate_limiting(self, client):
        """Test API rate limiting (if implemented)"""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.get("/api/muse/signatures/test_id")
            responses.append(response.status_code)
            
        # Check if any requests were rate limited (429 status)
        # This test may pass if rate limiting is not implemented
        rate_limited = any(status == 429 for status in responses)
        # We don't assert this must be true since rate limiting may not be implemented
        
    @pytest.mark.api
    def test_large_payload_handling(self, client):
        """Test handling of large request payloads"""
        large_data = {
            "content": "x" * 10000,  # Very large content
            "metadata": {"key": "value"} * 1000  # Large metadata
        }
        
        response = client.post("/api/muse/community/creations/share", json=large_data)
        # Should either accept or reject gracefully, not crash
        assert response.status_code in [200, 201, 413, 422]


class TestWebSocketEndpoints:
    """Test WebSocket endpoints for real-time features"""
    
    @pytest.mark.api
    @pytest.mark.slow
    def test_websocket_discovery_stream(self, client):
        """Test WebSocket streaming for live discovery"""
        # Note: Testing WebSocket with TestClient requires special handling
        # This is a placeholder for WebSocket testing
        
        # In a real implementation, you would use something like:
        # with client.websocket_connect("/api/muse/live/stream-discovery/test_session") as websocket:
        #     data = websocket.receive_json()
        #     assert "discovery_update" in data
        
        # For now, just test that the endpoint exists
        response = client.get("/api/muse/live/stream-discovery/test_session")
        # WebSocket endpoints return different status codes
        assert response.status_code in [426, 400]  # Upgrade Required or Bad Request


class TestAPIPerformance:
    """Test API performance characteristics"""
    
    @pytest.mark.api
    @pytest.mark.performance
    def test_response_times(self, client):
        """Test API response times"""
        import time
        
        endpoints = [
            "/health",
            "/api/muse/assessment/start",
            "/api/muse/community/gallery"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            response_time = time.time() - start_time
            
            # Assert response time is reasonable (adjust thresholds as needed)
            assert response_time < 2.0, f"Endpoint {endpoint} took {response_time:.2f}s"
            assert response.status_code in [200, 201, 404, 422]  # Valid status codes
            
    @pytest.mark.api
    @pytest.mark.performance
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/health")
            results.append(response.status_code)
            
        # Create multiple threads to simulate concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        duration = time.time() - start_time
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        # Should handle concurrent requests reasonably quickly
        assert duration < 5.0