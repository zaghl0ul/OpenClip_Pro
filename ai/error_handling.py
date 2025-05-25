"""Centralized error handling utilities for the AI module."""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorHandler:
    """Centralized error handling with consistent logging and formatting."""
    
    @staticmethod
    def format_error_message(
        error: Exception,
        context: Optional[str] = None,
        include_type: bool = True
    ) -> str:
        """Format an error message consistently."""
        parts = []
        
        if context:
            parts.append(context)
        
        if include_type:
            parts.append(f"{type(error).__name__}")
        
        parts.append(str(error))
        
        return ": ".join(parts)
    
    @staticmethod
    def log_error(
        error: Exception,
        context: Optional[str] = None,
        level: int = logging.ERROR,
        exc_info: bool = True
    ) -> None:
        """Log an error with consistent formatting."""
        message = ErrorHandler.format_error_message(error, context)
        logger.log(level, message, exc_info=exc_info)
    
    @staticmethod
    def is_retryable_error(error: Exception) -> bool:
        """Check if an error is retryable."""
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            OSError,
        )
        
        # Check for specific error messages
        error_str = str(error).lower()
        retryable_messages = [
            "timeout",
            "connection",
            "rate limit",
            "temporary",
            "unavailable",
        ]
        
        return (
            isinstance(error, retryable_errors) or
            any(msg in error_str for msg in retryable_messages)
        )


def handle_errors(
    default_return: Any = None,
    error_prefix: str = "",
    log_level: int = logging.ERROR,
    reraise: bool = False,
    retryable: bool = False,
    max_retries: int = 3
):
    """
    Decorator for consistent error handling.
    
    Args:
        default_return: Value to return on error
        error_prefix: Prefix for error messages
        log_level: Logging level for errors
        reraise: Whether to re-raise the exception
        retryable: Whether to retry on retryable errors
        max_retries: Maximum number of retries
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            retries = 0
            last_error = None
            
            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    context = f"{error_prefix} in {func.__name__}" if error_prefix else f"Error in {func.__name__}"
                    
                    if retryable and retries < max_retries and ErrorHandler.is_retryable_error(e):
                        retries += 1
                        logger.warning(f"{context}: {e}. Retrying ({retries}/{max_retries})...")
                        continue
                    
                    ErrorHandler.log_error(e, context, log_level)
                    
                    if reraise:
                        raise
                    
                    return default_return
            
            # If we've exhausted retries
            if last_error:
                context = f"{error_prefix} in {func.__name__} after {max_retries} retries"
                ErrorHandler.log_error(last_error, context, log_level)
                
                if reraise:
                    raise last_error
            
            return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            retries = 0
            last_error = None
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    context = f"{error_prefix} in {func.__name__}" if error_prefix else f"Error in {func.__name__}"
                    
                    if retryable and retries < max_retries and ErrorHandler.is_retryable_error(e):
                        retries += 1
                        logger.warning(f"{context}: {e}. Retrying ({retries}/{max_retries})...")
                        continue
                    
                    ErrorHandler.log_error(e, context, log_level)
                    
                    if reraise:
                        raise
                    
                    return default_return
            
            # If we've exhausted retries
            if last_error:
                context = f"{error_prefix} in {func.__name__} after {max_retries} retries"
                ErrorHandler.log_error(last_error, context, log_level)
                
                if reraise:
                    raise last_error
            
            return default_return
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def error_context(
    context: str,
    default_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = True
):
    """
    Context manager for consistent error handling.
    
    Usage:
        with error_context("Processing data", default_return=[]):
            # code that might raise exceptions
    """
    try:
        yield
    except Exception as e:
        ErrorHandler.log_error(e, context, log_level)
        
        if reraise:
            raise
        
        return default_return


def validate_input(
    value: Any,
    expected_type: Type,
    name: str,
    allow_none: bool = False,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    choices: Optional[list] = None
) -> Any:
    """
    Validate input with consistent error messages.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Parameter name for error messages
        allow_none: Whether None is allowed
        min_value: Minimum numeric value
        max_value: Maximum numeric value
        min_length: Minimum length for sequences
        max_length: Maximum length for sequences
        choices: Valid choices for the value
    
    Returns:
        The validated value
    
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        if allow_none:
            return value
        raise ValueError(f"{name} cannot be None")
    
    if not isinstance(value, expected_type):
        raise ValueError(
            f"{name} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    
    # Numeric validation
    if isinstance(value, (int, float)):
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value}")
        
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value}, got {value}")
    
    # Sequence validation
    if hasattr(value, '__len__'):
        length = len(value)
        
        if min_length is not None and length < min_length:
            raise ValueError(f"{name} must have at least {min_length} items, got {length}")
        
        if max_length is not None and length > max_length:
            raise ValueError(f"{name} must have at most {max_length} items, got {length}")
    
    # Choice validation
    if choices is not None and value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value}")
    
    return value


class APIErrorClassifier:
    """Classify API errors for better handling."""
    
    @staticmethod
    def classify_error(error: Exception) -> str:
        """
        Classify an error into categories.
        
        Returns:
            Error category: 'auth', 'rate_limit', 'invalid_request', 
            'server_error', 'network', 'timeout', 'unknown'
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Authentication errors
        if any(term in error_str for term in ['auth', 'api key', 'unauthorized', '401']):
            return 'auth'
        
        # Rate limiting
        if any(term in error_str for term in ['rate limit', '429', 'too many requests']):
            return 'rate_limit'
        
        # Invalid request
        if any(term in error_str for term in ['invalid', 'bad request', '400', 'validation']):
            return 'invalid_request'
        
        # Server errors
        if any(term in error_str for term in ['500', '502', '503', 'server error', 'internal']):
            return 'server_error'
        
        # Network errors
        if any(term in error_type for term in ['connection', 'network']) or \
           any(term in error_str for term in ['connection', 'network', 'dns']):
            return 'network'
        
        # Timeout errors
        if 'timeout' in error_type or 'timeout' in error_str:
            return 'timeout'
        
        return 'unknown'
    
    @staticmethod
    def get_user_friendly_message(error: Exception) -> str:
        """Get a user-friendly error message."""
        category = APIErrorClassifier.classify_error(error)
        
        messages = {
            'auth': "Authentication failed. Please check your API key.",
            'rate_limit': "Rate limit exceeded. Please try again later.",
            'invalid_request': "Invalid request. Please check your input.",
            'server_error': "Server error. The service may be temporarily unavailable.",
            'network': "Network error. Please check your internet connection.",
            'timeout': "Request timed out. Please try again.",
            'unknown': "An unexpected error occurred. Please try again."
        }
        
        return messages.get(category, messages['unknown'])


# Import asyncio at module level
import asyncio 