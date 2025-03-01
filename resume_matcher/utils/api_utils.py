"""
Utilities for API interactions in the Resume Matcher system.
"""
import time
import logging
import functools
from typing import Callable, Any, TypeVar, Dict
import requests

logger = logging.getLogger(__name__)

# TypeVar for generic function return type
T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 30.0,
    retryable_exceptions: tuple = (requests.exceptions.RequestException,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor to multiply backoff by after each failure
        max_backoff: Maximum backoff time in seconds
        retryable_exceptions: Tuple of exceptions that should trigger a retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            backoff = initial_backoff
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    # Last attempt - re-raise the exception
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}: {str(e)}")
                        raise
                    
                    # Calculate backoff time, don't exceed max_backoff
                    backoff = min(backoff * backoff_factor, max_backoff)
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after error: {str(e)}. Backing off for {backoff:.2f}s"
                    )
                    
                    # Wait before retrying
                    time.sleep(backoff)
        
        return wrapper
    
    return decorator


def create_api_client(
    base_url: str,
    headers: Dict[str, str] = None,
    timeout: int = 30
) -> requests.Session:
    """
    Create a requests Session with common configuration.
    
    Args:
        base_url: Base URL for API requests
        headers: Headers to include in all requests
        timeout: Request timeout in seconds
        
    Returns:
        Configured requests Session
    """
    session = requests.Session()
    
    # Set default headers
    default_headers = {
        'User-Agent': 'ResumeMatcherClient/1.0',
        'Accept': 'application/json'
    }
    
    # Update with custom headers
    if headers:
        default_headers.update(headers)
    
    session.headers.update(default_headers)
    
    # Configure the session with a base URL and timeout
    session.base_url = base_url
    session.timeout = timeout
    
    # Add a hook to log all requests
    def log_request(response, *args, **kwargs):
        request = response.request
        logger.debug(f"API Request: {request.method} {request.url}")
        logger.debug(f"API Response: {response.status_code}")
        return response
    
    session.hooks['response'] = [log_request]
    
    return session


def safe_api_request(
    method: str, 
    url: str, 
    session: requests.Session = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Make a safe API request with error handling.
    
    Args:
        method: HTTP method (get, post, etc.)
        url: URL to request
        session: Optional requests Session to use
        **kwargs: Additional arguments to pass to the request
        
    Returns:
        Response data as dictionary or None
    """
    try:
        # Use provided session or create a new one
        client = session or requests
        
        # Make the request
        response = client.request(method, url, **kwargs)
        response.raise_for_status()
        
        # Return JSON response if possible
        try:
            return response.json()
        except ValueError:
            # Not JSON, return the text
            return {"text": response.text}
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error in API request: {str(e)}")
        # Try to get error details from response
        try:
            error_data = e.response.json()
            logger.error(f"API error details: {error_data}")
        except ValueError:
            logger.error(f"API error response (non-JSON): {e.response.text}")
        raise
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in API request: {str(e)}")
        raise
