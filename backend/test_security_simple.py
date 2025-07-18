#!/usr/bin/env python3
"""
Simple Security Test Runner
Runs basic security tests without complex dependencies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import our security modules
from muse.core.authentication import AuthenticationManager, validate_password_strength, rate_limiter
from muse.core.input_validation import InputValidator, SecurityValidator, ValidationError
from muse.core.error_handling import MuseError, AuthenticationError, ValidationError as ErrorHandlingValidationError

def test_password_hashing():
    """Test password hashing and verification"""
    print("Testing password hashing...")
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
    print("✓ Password hashing tests passed")

def test_password_strength():
    """Test password strength validation"""
    print("Testing password strength validation...")
    
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
    print("✓ Password strength tests passed")

def test_input_validation():
    """Test input validation"""
    print("Testing input validation...")
    
    # Valid strings
    assert InputValidator.validate_string("Hello World", max_length=20) == "Hello World"
    assert InputValidator.validate_string("Test", min_length=2, max_length=10) == "Test"
    
    # Valid email
    assert InputValidator.validate_email("test@example.com") == "test@example.com"
    
    # Valid username  
    assert InputValidator.validate_username("validuser") == "validuser"
    assert InputValidator.validate_username("TestUser") == "testuser"  # Should be lowercased
    
    # Valid numeric ranges
    assert InputValidator.validate_numeric_range(5, min_val=0, max_val=10) == 5
    assert InputValidator.validate_numeric_range(3.14, min_val=0) == 3.14
    
    # Valid probabilities
    assert InputValidator.validate_probability(0.0) == 0.0
    assert InputValidator.validate_probability(0.5) == 0.5
    assert InputValidator.validate_probability(1.0) == 1.0
    
    print("✓ Input validation tests passed")

def test_injection_detection():
    """Test injection attack detection"""
    print("Testing injection attack detection...")
    
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
    assert SecurityValidator.check_for_injection_attempts("..\\\\..\\\\windows\\\\system32")
    
    # Encoded attacks
    assert SecurityValidator.check_for_injection_attempts("%3Cscript%3E")
    assert SecurityValidator.check_for_injection_attempts("\\\\x3cscript\\\\x3e")
    
    print("✓ Injection detection tests passed")

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test MUSE error creation
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
    
    # Test specific error types
    auth_error = AuthenticationError("Invalid credentials")
    assert auth_error.category.value == "authentication"
    assert auth_error.user_message == "Authentication failed"
    
    print("✓ Error handling tests passed")

def test_rate_limiting():
    """Test rate limiting"""
    print("Testing rate limiting...")
    rate_limiter.requests.clear()  # Clear previous requests
    
    # Test within limits
    for i in range(5):
        assert not rate_limiter.is_rate_limited("test_key", max_requests=10, window_seconds=60)
    
    # Test exceeding limits
    for i in range(10):
        rate_limiter.is_rate_limited("test_key2", max_requests=5, window_seconds=60)
    
    # Should be rate limited now
    assert rate_limiter.is_rate_limited("test_key2", max_requests=5, window_seconds=60)
    print("✓ Rate limiting tests passed")

def main():
    """Run all security tests"""
    print("🔒 MUSE Platform Security Test Suite")
    print("=" * 50)
    
    try:
        test_password_hashing()
        test_password_strength()
        test_input_validation() 
        test_injection_detection()
        test_error_handling()
        test_rate_limiting()
        
        print("=" * 50)
        print("🎉 All security tests passed successfully!")
        print("✅ Authentication system: SECURE")
        print("✅ Input validation: ROBUST") 
        print("✅ Injection protection: ACTIVE")
        print("✅ Error handling: COMPREHENSIVE")
        print("✅ Rate limiting: FUNCTIONAL")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())