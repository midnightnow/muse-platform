"""
MUSE Platform Input Validation and Sanitization
Comprehensive validation for all user inputs with security focus
"""

import re
import html
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, validator
from enum import Enum
import json

class ValidationError(Exception):
    """Custom validation error"""
    pass

class MuseArchetype(str, Enum):
    """Valid Muse archetypes for validation"""
    CALLIOPE = "CALLIOPE"
    CLIO = "CLIO"
    ERATO = "ERATO"
    EUTERPE = "EUTERPE"
    MELPOMENE = "MELPOMENE"
    POLYHYMNIA = "POLYHYMNIA"
    TERPSICHORE = "TERPSICHORE"
    THALIA = "THALIA"
    URANIA = "URANIA"
    SOPHIA = "SOPHIA"
    TECHNE = "TECHNE"
    PSYCHE = "PSYCHE"

class PoeticForm(str, Enum):
    """Valid poetic forms"""
    SONNET = "sonnet"
    HAIKU = "haiku"
    VILLANELLE = "villanelle"
    FREE_VERSE = "free_verse"
    QUATRAIN = "quatrain"
    COUPLET = "couplet"
    BALLAD = "ballad"
    GHAZAL = "ghazal"
    ODE = "ode"
    SESTINA = "sestina"
    PANTOUM = "pantoum"
    RONDEAU = "rondeau"

class SacredConstant(str, Enum):
    """Valid sacred constants"""
    PHI = "phi"
    PI = "pi"
    E = "e"
    SQRT_2 = "sqrt_2"
    SQRT_3 = "sqrt_3"
    SQRT_5 = "sqrt_5"

class Theme(str, Enum):
    """Valid creative themes"""
    NATURE = "nature"
    LOVE = "love"
    COSMOS = "cosmos"
    TIME = "time"
    MEMORY = "memory"
    MATHEMATICS = "mathematics"
    BEAUTY = "beauty"
    MYSTERY = "mystery"
    HARMONY = "harmony"
    TRANSFORMATION = "transformation"

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    # Regex patterns for common validations
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]{3,30}$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    
    # Content validation patterns
    SAFE_TEXT_PATTERN = re.compile(r'^[a-zA-Z0-9\s\.\,\!\?\;\:\-\'\"\(\)\[\]\{\}\/\\\n\r]*$')
    POETRY_CONTENT_PATTERN = re.compile(r'^[a-zA-Z0-9\s\.\,\!\?\;\:\-\'\"\(\)\[\]\{\}\/\\\n\r\u00C0-\u017F]*$')
    
    @staticmethod
    def validate_string(
        value: str, 
        min_length: int = 0, 
        max_length: int = 1000,
        pattern: Optional[re.Pattern] = None,
        allow_empty: bool = False
    ) -> str:
        """Validate and sanitize string input"""
        if not isinstance(value, str):
            raise ValidationError("Value must be a string")
        
        if not allow_empty and not value.strip():
            raise ValidationError("Value cannot be empty")
        
        # Remove null bytes and control characters except newlines, tabs, carriage returns
        sanitized = ''.join(
            char for char in value 
            if ord(char) >= 32 or char in '\n\r\t'
        )
        
        # HTML escape
        sanitized = html.escape(sanitized, quote=True)
        
        # Length validation
        if len(sanitized) < min_length:
            raise ValidationError(f"Value must be at least {min_length} characters")
        
        if len(sanitized) > max_length:
            raise ValidationError(f"Value must be at most {max_length} characters")
        
        # Pattern validation
        if pattern and not pattern.match(sanitized):
            raise ValidationError("Value contains invalid characters")
        
        return sanitized.strip()
    
    @staticmethod
    def validate_username(username: str) -> str:
        """Validate username format"""
        if not InputValidator.USERNAME_PATTERN.match(username):
            raise ValidationError(
                "Username must be 3-30 characters, alphanumeric and underscore only"
            )
        return username.lower()
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format"""
        email = email.lower().strip()
        if not InputValidator.EMAIL_PATTERN.match(email):
            raise ValidationError("Invalid email format")
        return email
    
    @staticmethod
    def validate_uuid(uuid_str: str) -> str:
        """Validate UUID format"""
        if not InputValidator.UUID_PATTERN.match(uuid_str):
            raise ValidationError("Invalid UUID format")
        return uuid_str.lower()
    
    @staticmethod
    def validate_numeric_range(
        value: Union[int, float], 
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        allow_none: bool = False
    ) -> Union[int, float]:
        """Validate numeric value within range"""
        if value is None and allow_none:
            return None
        
        if not isinstance(value, (int, float)):
            raise ValidationError("Value must be numeric")
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"Value must be at least {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"Value must be at most {max_val}")
        
        return value
    
    @staticmethod
    def validate_probability(value: float) -> float:
        """Validate probability value (0.0 to 1.0)"""
        return InputValidator.validate_numeric_range(value, 0.0, 1.0)
    
    @staticmethod
    def validate_percentage(value: float) -> float:
        """Validate percentage value (0.0 to 100.0)"""
        return InputValidator.validate_numeric_range(value, 0.0, 100.0)
    
    @staticmethod
    def validate_poetry_content(content: str) -> str:
        """Validate poetry content with enhanced character set"""
        if not content.strip():
            raise ValidationError("Poetry content cannot be empty")
        
        # Allow broader character set for international poetry
        if len(content) > 10000:  # Reasonable limit for poetry
            raise ValidationError("Poetry content too long (max 10,000 characters)")
        
        # Remove dangerous characters but preserve poetic ones
        sanitized = re.sub(r'[<>]', '', content)  # Remove HTML tags
        sanitized = html.escape(sanitized, quote=False)  # Escape but preserve quotes
        
        return sanitized.strip()
    
    @staticmethod
    def validate_json_object(obj: Any, max_depth: int = 10, max_keys: int = 100) -> Dict:
        """Validate JSON object structure and size"""
        if not isinstance(obj, dict):
            raise ValidationError("Value must be a JSON object")
        
        def check_depth(data, current_depth=0):
            if current_depth > max_depth:
                raise ValidationError(f"JSON depth exceeds maximum of {max_depth}")
            
            if isinstance(data, dict):
                if len(data) > max_keys:
                    raise ValidationError(f"JSON object has too many keys (max {max_keys})")
                for value in data.values():
                    check_depth(value, current_depth + 1)
            elif isinstance(data, list):
                for item in data:
                    check_depth(item, current_depth + 1)
        
        check_depth(obj)
        return obj
    
    @staticmethod
    def validate_frequency_signature(signature: Dict[str, Any]) -> Dict[str, Any]:
        """Validate frequency signature structure"""
        required_fields = ["primary_muse", "harmonic_blend", "sacred_ratios", "spiral_coordinates"]
        
        for field in required_fields:
            if field not in signature:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate primary muse
        try:
            MuseArchetype(signature["primary_muse"])
        except ValueError:
            raise ValidationError("Invalid primary muse archetype")
        
        # Validate harmonic blend
        harmonic_blend = signature["harmonic_blend"]
        if not isinstance(harmonic_blend, dict):
            raise ValidationError("Harmonic blend must be an object")
        
        for muse, value in harmonic_blend.items():
            try:
                MuseArchetype(muse)
            except ValueError:
                raise ValidationError(f"Invalid muse in harmonic blend: {muse}")
            
            InputValidator.validate_probability(value)
        
        # Validate sacred ratios
        sacred_ratios = signature["sacred_ratios"]
        if not isinstance(sacred_ratios, dict):
            raise ValidationError("Sacred ratios must be an object")
        
        for constant, value in sacred_ratios.items():
            try:
                SacredConstant(constant)
            except ValueError:
                raise ValidationError(f"Invalid sacred constant: {constant}")
            
            InputValidator.validate_probability(value)
        
        # Validate spiral coordinates
        coords = signature["spiral_coordinates"]
        if not isinstance(coords, dict):
            raise ValidationError("Spiral coordinates must be an object")
        
        required_coords = ["x", "y", "z"]
        for coord in required_coords:
            if coord not in coords:
                raise ValidationError(f"Missing coordinate: {coord}")
            
            if not isinstance(coords[coord], (int, float)):
                raise ValidationError(f"Coordinate {coord} must be numeric")
        
        return signature
    
    @staticmethod
    def validate_discovery_constraints(constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate creative discovery constraints"""
        validated = {}
        
        # Form type validation
        if "form_type" in constraints:
            try:
                PoeticForm(constraints["form_type"])
                validated["form_type"] = constraints["form_type"]
            except ValueError:
                raise ValidationError("Invalid poetic form")
        
        # Sacred constant validation
        if "sacred_constant" in constraints:
            try:
                SacredConstant(constraints["sacred_constant"])
                validated["sacred_constant"] = constraints["sacred_constant"]
            except ValueError:
                raise ValidationError("Invalid sacred constant")
        
        # Theme validation
        if "theme" in constraints:
            try:
                Theme(constraints["theme"])
                validated["theme"] = constraints["theme"]
            except ValueError:
                raise ValidationError("Invalid theme")
        
        # Numeric constraints
        numeric_fields = ["max_iterations", "target_fitness", "syllable_count", "line_count"]
        for field in numeric_fields:
            if field in constraints:
                validated[field] = InputValidator.validate_numeric_range(
                    constraints[field], 
                    min_val=1 if field in ["max_iterations", "syllable_count", "line_count"] else 0,
                    max_val=1000 if field == "max_iterations" else 1 if field == "target_fitness" else 100
                )
        
        return validated

class SecurityValidator:
    """Additional security-focused validation"""
    
    # Suspicious patterns that might indicate attacks
    SUSPICIOUS_PATTERNS = [
        re.compile(r'<script.*?>', re.IGNORECASE),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
        re.compile(r'(union|select|insert|update|delete|drop|create|alter)\s+', re.IGNORECASE),
        re.compile(r'\.\./'),  # Path traversal
        re.compile(r'\\x[0-9a-f]{2}', re.IGNORECASE),  # Hex encoded
        re.compile(r'%[0-9a-f]{2}', re.IGNORECASE),  # URL encoded
    ]
    
    @staticmethod
    def check_for_injection_attempts(value: str) -> bool:
        """Check if input contains potential injection attempts"""
        for pattern in SecurityValidator.SUSPICIOUS_PATTERNS:
            if pattern.search(value):
                return True
        return False
    
    @staticmethod
    def validate_safe_input(value: str) -> str:
        """Validate input is safe from common attacks"""
        if SecurityValidator.check_for_injection_attempts(value):
            raise ValidationError("Input contains potentially dangerous content")
        
        return value

# Pydantic models with validation
class UserProfileValidation(BaseModel):
    """Validated user profile data"""
    username: str
    email: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        return InputValidator.validate_username(v)
    
    @validator('email')
    def validate_email(cls, v):
        return InputValidator.validate_email(v)
    
    @validator('display_name')
    def validate_display_name(cls, v):
        if v is not None:
            return InputValidator.validate_string(v, max_length=100)
        return v
    
    @validator('bio')
    def validate_bio(cls, v):
        if v is not None:
            return InputValidator.validate_string(v, max_length=500)
        return v

class CreationValidation(BaseModel):
    """Validated creation data"""
    title: Optional[str] = None
    content: str
    form_type: PoeticForm
    theme: Optional[Theme] = None
    tags: Optional[List[str]] = None
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None:
            return InputValidator.validate_string(v, max_length=200)
        return v
    
    @validator('content')
    def validate_content(cls, v):
        return InputValidator.validate_poetry_content(v)
    
    @validator('tags')
    def validate_tags(cls, v):
        if v is not None:
            if len(v) > 10:
                raise ValueError("Too many tags (max 10)")
            
            validated_tags = []
            for tag in v:
                validated_tag = InputValidator.validate_string(
                    tag, 
                    min_length=1, 
                    max_length=50,
                    pattern=re.compile(r'^[a-zA-Z0-9_]+$')
                )
                validated_tags.append(validated_tag.lower())
            
            return validated_tags
        return v

class AssessmentValidation(BaseModel):
    """Validated assessment data"""
    creative_preferences: Dict[str, int]
    personality_traits: Dict[str, int]
    sacred_geometry_affinity: Dict[str, int]
    thematic_preferences: Dict[str, int]
    
    @validator('creative_preferences', 'personality_traits', 'sacred_geometry_affinity', 'thematic_preferences')
    def validate_rating_dict(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Must be a dictionary")
        
        for key, value in v.items():
            if not isinstance(key, str) or len(key) > 50:
                raise ValueError("Invalid key format")
            
            if not isinstance(value, int) or not 1 <= value <= 10:
                raise ValueError("Rating must be integer between 1 and 10")
        
        return v