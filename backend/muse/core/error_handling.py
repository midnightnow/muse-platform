"""
MUSE Platform Error Handling and Logging
Comprehensive error management with structured logging
"""

import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("muse.error_handling")

class ErrorCategory(str, Enum):
    """Error categories for classification"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    MATHEMATICAL = "mathematical"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MuseError(Exception):
    """Base exception class for MUSE platform"""
    
    def __init__(
        self, 
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.user_message = user_message or message
        self.error_code = error_code or f"{category.value}_{int(datetime.utcnow().timestamp())}"
        self.timestamp = datetime.utcnow()
        self.trace_id = str(uuid.uuid4())
        
        super().__init__(self.message)

class ValidationError(MuseError):
    """Validation-related errors"""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details={"field": field} if field else {},
            **kwargs
        )

class AuthenticationError(MuseError):
    """Authentication-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="Authentication failed",
            **kwargs
        )

class AuthorizationError(MuseError):
    """Authorization-related errors"""
    def __init__(self, message: str, required_permission: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.MEDIUM,
            details={"required_permission": required_permission} if required_permission else {},
            user_message="Access denied",
            **kwargs
        )

class NotFoundError(MuseError):
    """Resource not found errors"""
    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        super().__init__(
            message=f"{resource_type} with ID {resource_id} not found",
            category=ErrorCategory.NOT_FOUND,
            severity=ErrorSeverity.LOW,
            details={"resource_type": resource_type, "resource_id": resource_id},
            user_message=f"{resource_type} not found",
            **kwargs
        )

class RateLimitError(MuseError):
    """Rate limiting errors"""
    def __init__(self, limit: int, window: int, **kwargs):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window} seconds",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.LOW,
            details={"limit": limit, "window": window},
            user_message="Too many requests, please try again later",
            **kwargs
        )

class MathematicalError(MuseError):
    """Mathematical computation errors"""
    def __init__(self, operation: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(
            message=f"Mathematical error in {operation}",
            category=ErrorCategory.MATHEMATICAL,
            severity=ErrorSeverity.HIGH,
            details=details or {},
            user_message="Mathematical computation failed",
            **kwargs
        )

class DatabaseError(MuseError):
    """Database operation errors"""
    def __init__(self, operation: str, table: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Database error during {operation}" + (f" on {table}" if table else ""),
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            details={"operation": operation, "table": table},
            user_message="Database operation failed",
            **kwargs
        )

class ExternalAPIError(MuseError):
    """External API errors"""
    def __init__(self, service: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message=f"External API error from {service}" + (f" (status: {status_code})" if status_code else ""),
            category=ErrorCategory.EXTERNAL_API,
            severity=ErrorSeverity.MEDIUM,
            details={"service": service, "status_code": status_code},
            user_message="External service temporarily unavailable",
            **kwargs
        )

class SystemError(MuseError):
    """System-level errors"""
    def __init__(self, component: str, **kwargs):
        super().__init__(
            message=f"System error in {component}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            details={"component": component},
            user_message="System error occurred",
            **kwargs
        )

class ErrorResponse(BaseModel):
    """Standardized error response model"""
    error: bool = True
    message: str
    category: str
    error_code: str
    trace_id: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        self.logger = logger
    
    def log_error(self, error: Exception, request: Optional[Request] = None) -> str:
        """Log error with context and return trace ID"""
        trace_id = str(uuid.uuid4())
        
        # Extract error information
        if isinstance(error, MuseError):
            error_info = {
                "trace_id": trace_id,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "user_message": error.user_message,
                "error_code": error.error_code,
                "details": error.details,
                "timestamp": error.timestamp.isoformat()
            }
        else:
            error_info = {
                "trace_id": trace_id,
                "category": ErrorCategory.UNKNOWN.value,
                "severity": ErrorSeverity.MEDIUM.value,
                "message": str(error),
                "error_type": type(error).__name__,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Add request context if available
        if request:
            error_info.update({
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent"),
                "ip_address": request.client.host if request.client else None
            })
        
        # Add stack trace for debugging
        error_info["stack_trace"] = traceback.format_exc()
        
        # Log based on severity
        if isinstance(error, MuseError):
            if error.severity == ErrorSeverity.CRITICAL:
                self.logger.critical("Critical error occurred", **error_info)
            elif error.severity == ErrorSeverity.HIGH:
                self.logger.error("High severity error occurred", **error_info)
            elif error.severity == ErrorSeverity.MEDIUM:
                self.logger.warning("Medium severity error occurred", **error_info)
            else:
                self.logger.info("Low severity error occurred", **error_info)
        else:
            self.logger.error("Unhandled exception occurred", **error_info)
        
        return trace_id
    
    def create_error_response(self, error: Exception, trace_id: str) -> ErrorResponse:
        """Create standardized error response"""
        if isinstance(error, MuseError):
            return ErrorResponse(
                message=error.user_message,
                category=error.category.value,
                error_code=error.error_code,
                trace_id=trace_id,
                timestamp=error.timestamp.isoformat(),
                details=error.details if error.severity != ErrorSeverity.CRITICAL else None
            )
        else:
            return ErrorResponse(
                message="An unexpected error occurred",
                category=ErrorCategory.UNKNOWN.value,
                error_code=f"unknown_{int(datetime.utcnow().timestamp())}",
                trace_id=trace_id,
                timestamp=datetime.utcnow().isoformat()
            )
    
    def handle_error(self, error: Exception, request: Optional[Request] = None) -> JSONResponse:
        """Handle error and return appropriate HTTP response"""
        trace_id = self.log_error(error, request)
        error_response = self.create_error_response(error, trace_id)
        
        # Map error types to HTTP status codes
        status_code = self._get_status_code(error)
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict()
        )
    
    def _get_status_code(self, error: Exception) -> int:
        """Map error types to appropriate HTTP status codes"""
        if isinstance(error, ValidationError):
            return status.HTTP_422_UNPROCESSABLE_ENTITY
        elif isinstance(error, AuthenticationError):
            return status.HTTP_401_UNAUTHORIZED
        elif isinstance(error, AuthorizationError):
            return status.HTTP_403_FORBIDDEN
        elif isinstance(error, NotFoundError):
            return status.HTTP_404_NOT_FOUND
        elif isinstance(error, RateLimitError):
            return status.HTTP_429_TOO_MANY_REQUESTS
        elif isinstance(error, (MathematicalError, DatabaseError)):
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        elif isinstance(error, ExternalAPIError):
            return status.HTTP_502_BAD_GATEWAY
        elif isinstance(error, SystemError):
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        else:
            return status.HTTP_500_INTERNAL_SERVER_ERROR

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions for common error scenarios
def raise_validation_error(message: str, field: Optional[str] = None):
    """Raise a validation error"""
    raise ValidationError(message, field=field)

def raise_not_found(resource_type: str, resource_id: str):
    """Raise a not found error"""
    raise NotFoundError(resource_type, resource_id)

def raise_authentication_error(message: str = "Authentication failed"):
    """Raise an authentication error"""
    raise AuthenticationError(message)

def raise_authorization_error(message: str = "Access denied", required_permission: Optional[str] = None):
    """Raise an authorization error"""
    raise AuthorizationError(message, required_permission=required_permission)

def raise_mathematical_error(operation: str, details: Optional[Dict] = None):
    """Raise a mathematical computation error"""
    raise MathematicalError(operation, details=details)

def raise_database_error(operation: str, table: Optional[str] = None):
    """Raise a database error"""
    raise DatabaseError(operation, table=table)

# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in a specific context"""
    
    def __init__(self, operation: str, logger_context: Optional[Dict] = None):
        self.operation = operation
        self.logger_context = logger_context or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        logger.info(f"Starting {self.operation}", **self.logger_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            logger.info(
                f"Completed {self.operation}",
                duration=duration,
                **self.logger_context
            )
        else:
            logger.error(
                f"Failed {self.operation}",
                duration=duration,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.logger_context
            )
        
        # Don't suppress exceptions
        return False