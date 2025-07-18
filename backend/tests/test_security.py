"""
MUSE Platform Security Tests
Comprehensive security testing for authentication, validation, and injection attacks
"""

import pytest
import jwt
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from muse.core.authentication import (
    AuthenticationManager, 
    validate_password_strength,
    sanitize_input,
    validate_email,
    validate_username,
    rate_limiter
)
from muse.core.input_validation import (
    InputValidator,
    SecurityValidator,
    ValidationError,
    UserProfileValidation,
    CreationValidation,
    AssessmentValidation
)
from muse.core.error_handling import (
    MuseError,
    ValidationError as ErrorHandlingValidationError,
    AuthenticationError,
    AuthorizationError
)


class TestAuthentication:
    """Test authentication and token management"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        auth_manager = AuthenticationManager()
        password = "TestPassword123!"
        
        # Hash password
        hashed = auth_manager.get_password_hash(password)
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long
        
        # Verify correct password
        assert auth_manager.verify_password(password, hashed)
        
        # Verify incorrect password
        assert not auth_manager.verify_password("WrongPassword", hashed)
        
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification"""
        auth_manager = AuthenticationManager()
        
        token_data = {
            "sub": "user123",
            "username": "testuser",
            "email": "test@example.com",
            "scopes": ["user"]
        }
        
        # Create access token
        access_token = auth_manager.create_access_token(token_data)
        assert isinstance(access_token, str)
        assert len(access_token) > 100  # JWT tokens are long
        
        # Verify token
        verified_data = auth_manager.verify_token(access_token)
        assert verified_data.user_id == "user123"
        assert verified_data.username == "testuser"
        assert verified_data.email == "test@example.com"
        assert verified_data.scopes == ["user"]
        
    def test_token_expiration(self):
        """Test token expiration handling"""
        auth_manager = AuthenticationManager()
        
        # Create expired token
        token_data = {"sub": "user123"}
        expired_delta = timedelta(seconds=-1)  # Already expired
        
        expired_token = auth_manager.create_access_token(token_data, expired_delta)
        
        # Verify expired token raises exception
        with pytest.raises(Exception):  # Should raise HTTPException but we'll catch any
            auth_manager.verify_token(expired_token)
            
    def test_invalid_token(self):
        """Test handling of invalid tokens"""
        auth_manager = AuthenticationManager()
        
        # Test malformed token
        with pytest.raises(Exception):
            auth_manager.verify_token("invalid.token.here")
        
        # Test token with wrong signature
        fake_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.invalid_signature"
        with pytest.raises(Exception):
            auth_manager.verify_token(fake_token)
            
    def test_refresh_token_flow(self):
        """Test refresh token functionality"""
        auth_manager = AuthenticationManager()
        
        token_data = {
            "sub": "user123",
            "username": "testuser",
            "email": "test@example.com"
        }
        
        # Create refresh token
        refresh_token = auth_manager.create_refresh_token(token_data)
        
        # Use refresh token to create new access token
        new_access_token = auth_manager.refresh_access_token(refresh_token)
        
        # Verify new access token
        verified_data = auth_manager.verify_token(new_access_token)
        assert verified_data.user_id == "user123"
        
    def test_password_strength_validation(self):
        """Test password strength requirements"""
        # Valid passwords
        assert validate_password_strength("StrongPass123!")
        assert validate_password_strength("MyP@ssw0rd")
        assert validate_password_strength("C0mplex!ty")
        
        # Invalid passwords
        assert not validate_password_strength("weak")  # Too short
        assert not validate_password_strength("nouppercase123!")  # No uppercase
        assert not validate_password_strength("NOLOWERCASE123!")  # No lowercase
        assert not validate_password_strength("NoDigits!")  # No digits
        assert not validate_password_strength("NoSpecial123")  # No special chars
        
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        rate_limiter.requests.clear()  # Clear previous requests
        
        # Test within limits
        for i in range(5):
            assert not rate_limiter.is_rate_limited("test_key", max_requests=10, window_seconds=60)
        
        # Test exceeding limits
        for i in range(10):
            rate_limiter.is_rate_limited("test_key2", max_requests=5, window_seconds=60)
        
        # Should be rate limited now
        assert rate_limiter.is_rate_limited("test_key2", max_requests=5, window_seconds=60)


class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_string_validation(self):
        """Test string validation with various parameters"""
        # Valid strings
        assert InputValidator.validate_string("Hello World", max_length=20) == "Hello World"
        assert InputValidator.validate_string("Test", min_length=2, max_length=10) == "Test"
        
        # Invalid strings - too short
        with pytest.raises(ValidationError):
            InputValidator.validate_string("Hi", min_length=5)
        
        # Invalid strings - too long
        with pytest.raises(ValidationError):
            InputValidator.validate_string("Very long string", max_length=5)
        
        # Empty strings
        with pytest.raises(ValidationError):
            InputValidator.validate_string("", allow_empty=False)
        
        assert InputValidator.validate_string("", allow_empty=True) == ""
        
    def test_email_validation(self):
        """Test email format validation"""
        # Valid emails
        assert InputValidator.validate_email("test@example.com") == "test@example.com"
        assert InputValidator.validate_email("user.name+tag@domain.co.uk") == "user.name+tag@domain.co.uk"
        
        # Invalid emails
        with pytest.raises(ValidationError):
            InputValidator.validate_email("invalid-email")
        
        with pytest.raises(ValidationError):
            InputValidator.validate_email("@domain.com")
            
        with pytest.raises(ValidationError):
            InputValidator.validate_email("user@")
            
    def test_username_validation(self):
        """Test username format validation"""
        # Valid usernames
        assert InputValidator.validate_username("validuser") == "validuser"
        assert InputValidator.validate_username("user_123") == "user_123"
        assert InputValidator.validate_username("TestUser") == "testuser"  # Should be lowercased
        
        # Invalid usernames
        with pytest.raises(ValidationError):
            InputValidator.validate_username("ab")  # Too short
            
        with pytest.raises(ValidationError):
            InputValidator.validate_username("user-name")  # Invalid character
            
        with pytest.raises(ValidationError):
            InputValidator.validate_username("user@domain")  # Invalid character
            
    def test_numeric_validation(self):
        """Test numeric range validation"""
        # Valid numbers
        assert InputValidator.validate_numeric_range(5, min_val=0, max_val=10) == 5
        assert InputValidator.validate_numeric_range(3.14, min_val=0) == 3.14
        
        # Invalid numbers
        with pytest.raises(ValidationError):
            InputValidator.validate_numeric_range(-5, min_val=0)
            
        with pytest.raises(ValidationError):
            InputValidator.validate_numeric_range(15, max_val=10)
            
        with pytest.raises(ValidationError):
            InputValidator.validate_numeric_range("not_a_number")
            
    def test_probability_validation(self):
        """Test probability value validation"""
        # Valid probabilities
        assert InputValidator.validate_probability(0.0) == 0.0
        assert InputValidator.validate_probability(0.5) == 0.5
        assert InputValidator.validate_probability(1.0) == 1.0
        
        # Invalid probabilities
        with pytest.raises(ValidationError):
            InputValidator.validate_probability(-0.1)
            
        with pytest.raises(ValidationError):
            InputValidator.validate_probability(1.1)
            
    def test_poetry_content_validation(self):
        """Test poetry content validation"""
        # Valid poetry
        valid_poem = "Roses are red,\nViolets are blue,\nPoetry is art,\nAnd so are you."
        assert InputValidator.validate_poetry_content(valid_poem) == valid_poem
        
        # Empty content
        with pytest.raises(ValidationError):
            InputValidator.validate_poetry_content("")
            
        with pytest.raises(ValidationError):
            InputValidator.validate_poetry_content("   \n  \t  ")
            
        # Too long content
        long_content = "A" * 20000
        with pytest.raises(ValidationError):
            InputValidator.validate_poetry_content(long_content)
            
    def test_json_validation(self):
        """Test JSON object validation"""
        # Valid JSON
        valid_json = {"key": "value", "number": 42}
        assert InputValidator.validate_json_object(valid_json) == valid_json
        
        # Invalid JSON - not a dict
        with pytest.raises(ValidationError):
            InputValidator.validate_json_object("not a dict")
            
        # Too many keys
        large_json = {f"key_{i}": i for i in range(200)}
        with pytest.raises(ValidationError):
            InputValidator.validate_json_object(large_json, max_keys=100)
            
        # Too deep nesting
        deep_json = {"level1": {"level2": {"level3": {"level4": {"level5": "deep"}}}}}
        with pytest.raises(ValidationError):
            InputValidator.validate_json_object(deep_json, max_depth=3)
            
    def test_frequency_signature_validation(self):
        """Test frequency signature validation"""
        # Valid signature
        valid_signature = {
            "primary_muse": "CALLIOPE",
            "harmonic_blend": {
                "CALLIOPE": 0.8,
                "ERATO": 0.6
            },
            "sacred_ratios": {
                "phi": 0.9,
                "pi": 0.7
            },
            "spiral_coordinates": {
                "x": 1.2,
                "y": 0.8,
                "z": 0.3
            }
        }
        
        result = InputValidator.validate_frequency_signature(valid_signature)
        assert result == valid_signature
        
        # Missing required field
        invalid_signature = valid_signature.copy()
        del invalid_signature["primary_muse"]
        
        with pytest.raises(ValidationError):
            InputValidator.validate_frequency_signature(invalid_signature)
            
        # Invalid muse archetype
        invalid_signature = valid_signature.copy()
        invalid_signature["primary_muse"] = "INVALID_MUSE"
        
        with pytest.raises(ValidationError):
            InputValidator.validate_frequency_signature(invalid_signature)
            
        # Invalid probability in harmonic blend
        invalid_signature = valid_signature.copy()
        invalid_signature["harmonic_blend"]["CALLIOPE"] = 1.5
        
        with pytest.raises(ValidationError):
            InputValidator.validate_frequency_signature(invalid_signature)


class TestSecurityValidation:
    """Test security-specific validation"""
    
    def test_injection_detection(self):
        """Test detection of potential injection attacks"""
        # Safe inputs
        assert not SecurityValidator.check_for_injection_attempts("Hello World")
        assert not SecurityValidator.check_for_injection_attempts("user@example.com")
        assert not SecurityValidator.check_for_injection_attempts("Simple text content")
        
        # Potential XSS
        assert SecurityValidator.check_for_injection_attempts("<script>alert('xss')</script>")
        assert SecurityValidator.check_for_injection_attempts("javascript:alert('xss')")
        assert SecurityValidator.check_for_injection_attempts("<img onload='alert(1)'>")
        
        # Potential SQL injection
        assert SecurityValidator.check_for_injection_attempts("'; DROP TABLE users; --")
        assert SecurityValidator.check_for_injection_attempts("UNION SELECT * FROM passwords")
        assert SecurityValidator.check_for_injection_attempts("1' OR '1'='1")
        
        # Path traversal
        assert SecurityValidator.check_for_injection_attempts("../../../etc/passwd")
        assert SecurityValidator.check_for_injection_attempts("..\\..\\windows\\system32")
        
        # Encoded attacks
        assert SecurityValidator.check_for_injection_attempts("%3Cscript%3E")
        assert SecurityValidator.check_for_injection_attempts("\\x3cscript\\x3e")
        
    def test_safe_input_validation(self):
        """Test safe input validation"""
        # Safe inputs should pass
        safe_input = "This is a safe input string"
        assert SecurityValidator.validate_safe_input(safe_input) == safe_input
        
        # Dangerous inputs should raise exceptions
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "javascript:alert(1)"
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(ValidationError):
                SecurityValidator.validate_safe_input(dangerous_input)


class TestPydanticValidation:
    """Test Pydantic model validation"""
    
    def test_user_profile_validation(self):
        """Test user profile validation model"""
        # Valid profile
        valid_data = {
            "username": "testuser",
            "email": "test@example.com",
            "display_name": "Test User",
            "bio": "A test user profile"
        }
        
        profile = UserProfileValidation(**valid_data)
        assert profile.username == "testuser"
        assert profile.email == "test@example.com"
        
        # Invalid email
        invalid_data = valid_data.copy()
        invalid_data["email"] = "invalid-email"
        
        with pytest.raises(ValueError):
            UserProfileValidation(**invalid_data)
            
        # Invalid username
        invalid_data = valid_data.copy()
        invalid_data["username"] = "ab"  # Too short
        
        with pytest.raises(ValueError):
            UserProfileValidation(**invalid_data)
            
    def test_creation_validation(self):
        """Test creation validation model"""
        # Valid creation
        valid_data = {
            "title": "Test Poem",
            "content": "Roses are red,\nViolets are blue",
            "form_type": "free_verse",
            "theme": "nature",
            "tags": ["poetry", "test"]
        }
        
        creation = CreationValidation(**valid_data)
        assert creation.title == "Test Poem"
        assert creation.form_type == "free_verse"
        
        # Invalid form type
        invalid_data = valid_data.copy()
        invalid_data["form_type"] = "invalid_form"
        
        with pytest.raises(ValueError):
            CreationValidation(**invalid_data)
            
        # Too many tags
        invalid_data = valid_data.copy()
        invalid_data["tags"] = [f"tag_{i}" for i in range(15)]
        
        with pytest.raises(ValueError):
            CreationValidation(**invalid_data)
            
    def test_assessment_validation(self):
        """Test assessment validation model"""
        # Valid assessment
        valid_data = {
            "creative_preferences": {"creativity": 8, "structure": 6},
            "personality_traits": {"openness": 9, "conscientiousness": 7},
            "sacred_geometry_affinity": {"phi": 8, "pi": 7},
            "thematic_preferences": {"nature": 9, "love": 6}
        }
        
        assessment = AssessmentValidation(**valid_data)
        assert assessment.creative_preferences["creativity"] == 8
        
        # Invalid rating value
        invalid_data = valid_data.copy()
        invalid_data["creative_preferences"]["creativity"] = 15  # Out of range
        
        with pytest.raises(ValueError):
            AssessmentValidation(**invalid_data)
            
        # Invalid rating type
        invalid_data = valid_data.copy()
        invalid_data["personality_traits"]["openness"] = "high"  # Should be int
        
        with pytest.raises(ValueError):
            AssessmentValidation(**invalid_data)


class TestErrorHandling:
    """Test error handling and logging"""
    
    def test_muse_error_creation(self):
        """Test MUSE error class"""
        error = MuseError(
            message="Test error",
            category="validation",
            severity="medium",
            details={"field": "username"}
        )
        
        assert error.message == "Test error"
        assert error.category == "validation"
        assert error.severity == "medium"
        assert error.details["field"] == "username"
        assert error.trace_id is not None
        assert error.timestamp is not None
        
    def test_specific_error_types(self):
        """Test specific error type classes"""
        # Authentication error
        auth_error = AuthenticationError("Invalid credentials")
        assert auth_error.category == "authentication"
        assert auth_error.user_message == "Authentication failed"
        
        # Authorization error
        authz_error = AuthorizationError("Insufficient permissions", required_permission="admin")
        assert authz_error.category == "authorization"
        assert authz_error.details["required_permission"] == "admin"
        
        # Validation error
        val_error = ErrorHandlingValidationError("Invalid input", field="email")
        assert val_error.category == "validation"
        assert val_error.details["field"] == "email"
        
    @patch('muse.core.error_handling.logger')
    def test_error_logging(self, mock_logger):
        """Test error logging functionality"""
        from muse.core.error_handling import ErrorHandler
        
        handler = ErrorHandler()
        error = MuseError("Test error", severity="high")
        
        trace_id = handler.log_error(error)
        
        assert trace_id is not None
        assert len(trace_id) > 10  # UUID should be long
        mock_logger.error.assert_called_once()
        
    def test_error_response_creation(self):
        """Test error response creation"""
        from muse.core.error_handling import ErrorHandler
        
        handler = ErrorHandler()
        error = MuseError("Test error", category="validation")
        trace_id = "test-trace-id"
        
        response = handler.create_error_response(error, trace_id)
        
        assert response.error is True
        assert response.message == error.user_message
        assert response.category == error.category.value
        assert response.trace_id == trace_id


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components"""
    
    def test_authentication_flow(self):
        """Test complete authentication flow"""
        auth_manager = AuthenticationManager()
        
        # Create user credentials
        password = "SecurePass123!"
        hashed_password = auth_manager.get_password_hash(password)
        
        # Simulate user data
        user_data = {
            "id": "user123",
            "username": "testuser",
            "email": "test@example.com",
            "display_name": "Test User",
            "is_active": True
        }
        
        # Verify password
        assert auth_manager.verify_password(password, hashed_password)
        
        # Create tokens
        from muse.core.authentication import AuthenticatedUser
        user = AuthenticatedUser(**user_data)
        tokens = auth_manager.create_token_pair(user)
        
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert tokens.token_type == "bearer"
        
        # Verify access token
        token_data = auth_manager.verify_token(tokens.access_token)
        assert token_data.user_id == "user123"
        assert token_data.username == "testuser"
        
    def test_validation_pipeline(self):
        """Test complete validation pipeline"""
        # Test user profile validation
        profile_data = {
            "username": "testuser",
            "email": "test@example.com",
            "display_name": "Test User",
            "bio": "A test user bio"
        }
        
        # Validate profile
        profile = UserProfileValidation(**profile_data)
        assert profile.username == "testuser"
        
        # Test creation validation
        creation_data = {
            "title": "Test Creation",
            "content": "This is test poetry content\nWith multiple lines",
            "form_type": "free_verse",
            "theme": "nature",
            "tags": ["test", "poetry"]
        }
        
        # Validate creation
        creation = CreationValidation(**creation_data)
        assert creation.form_type == "free_verse"
        
        # Test security validation
        safe_content = SecurityValidator.validate_safe_input(creation.content)
        assert safe_content == creation.content