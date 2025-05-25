# ai/base.py
"""Base classes and interfaces for AI provider implementations."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

logger = logging.getLogger(__name__)


# --- Enums ---
class ModelCapability(Enum):
    """Supported model capabilities."""
    TEXT = "text"
    IMAGE = "image"
    JSON = "json"
    AUDIO = "audio"
    VIDEO = "video"


class ModelQuality(Enum):
    """Model quality tiers."""
    FAST = "fast"
    STANDARD = "standard"
    PREMIUM = "premium"


class ProviderType(Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


# --- Data Classes ---
@dataclass
class ModelInfo:
    """Information about an AI model."""
    name: str
    provider: ProviderType
    capabilities: List[ModelCapability] = field(default_factory=list)
    quality: ModelQuality = ModelQuality.STANDARD
    requires_api_key: bool = True
    is_vision_model: bool = False
    is_local: bool = False
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    
    @property
    def supports_images(self) -> bool:
        """Check if model supports image input."""
        return ModelCapability.IMAGE in self.capabilities
    
    @property
    def supports_json(self) -> bool:
        """Check if model supports JSON output mode."""
        return ModelCapability.JSON in self.capabilities


@dataclass
class AnalysisRequest:
    """Request for AI analysis."""
    prompt: str
    model: ModelInfo
    images: List[str] = field(default_factory=list)
    temperature: float = 0.5
    max_tokens: int = 1000
    timeout: int = 120
    format_type: str = "text"  # "text" or "json"
    
    def validate(self) -> None:
        """Validate the request parameters."""
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        
        if self.images and not self.model.supports_images:
            raise ValueError(f"Model {self.model.name} does not support images")
        
        if self.format_type == "json" and not self.model.supports_json:
            logger.warning(f"Model {self.model.name} does not have native JSON support")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temperature}")
        
        if self.max_tokens <= 0:
            raise ValueError(f"Max tokens must be positive, got {self.max_tokens}")


@dataclass
class AnalysisResponse:
    """Response from AI analysis."""
    content: str
    model: ModelInfo
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    
    @property
    def is_error(self) -> bool:
        """Check if response is an error."""
        return self.content.startswith("Error:")
    
    def get_error_message(self) -> Optional[str]:
        """Extract error message if response is an error."""
        if self.is_error:
            return self.content[6:].strip()  # Remove "Error:" prefix
        return None


# --- Protocols ---
class ModelRegistryProtocol(Protocol):
    """Protocol for model registry implementations."""
    
    def get_model_info(self, provider: str, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        ...
    
    def list_models_for_provider(self, provider: str) -> List[str]:
        """List all models for a provider."""
        ...
    
    def list_vision_models(self) -> List[ModelInfo]:
        """List all models that support vision."""
        ...


class APIKeyManagerProtocol(Protocol):
    """Protocol for API key management."""
    
    def get_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        ...
    
    def save_key(self, provider: str, key: str) -> bool:
        """Save API key for a provider."""
        ...
    
    def delete_key(self, provider: str) -> bool:
        """Delete API key for a provider."""
        ...


# --- Abstract Base Classes ---
class AIProvider(ABC):
    """Abstract base class for AI provider implementations."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client = None
        self._initialized = False
    
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider client."""
        pass
    
    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Perform analysis with the AI model."""
        pass
    
    @abstractmethod
    async def list_available_models(self) -> List[ModelInfo]:
        """List available models from this provider."""
        pass
    
    @abstractmethod
    async def validate_model(self, model_name: str) -> bool:
        """Check if a model is available and valid."""
        pass
    
    async def close(self) -> None:
        """Clean up resources."""
        self._initialized = False
        if hasattr(self._client, 'close'):
            await self._client.close()
    
    def _format_error_response(self, error: Exception, context: str = "") -> AnalysisResponse:
        """Format an error as an AnalysisResponse."""
        error_msg = f"Error: {context}: {type(error).__name__}: {str(error)}" if context else f"Error: {type(error).__name__}: {str(error)}"
        return AnalysisResponse(
            content=error_msg,
            model=ModelInfo(name="unknown", provider=self.provider_type)
        )


class BaseAIProvider(AIProvider):
    """Base implementation with common functionality."""
    
    async def initialize(self) -> None:
        """Default initialization."""
        if self._initialized:
            return
        
        if self.requires_api_key and not self.api_key:
            raise ValueError(f"API key required for {self.provider_type.value}")
        
        await self._create_client()
        self._initialized = True
    
    @property
    def requires_api_key(self) -> bool:
        """Check if this provider requires an API key."""
        return self.provider_type != ProviderType.OLLAMA
    
    @abstractmethod
    async def _create_client(self) -> None:
        """Create the provider-specific client."""
        pass
    
    async def _prepare_images(self, images: List[str]) -> List[Any]:
        """Prepare images for the specific provider format."""
        # Default implementation, override in subclasses
        return images
    
    def _validate_response(self, response: Any) -> None:
        """Validate provider response."""
        # Default implementation, override in subclasses
        if response is None:
            raise ValueError("Received null response from provider")


# --- Error Handling ---
class AIProviderError(Exception):
    """Base exception for AI provider errors."""
    pass


class APIKeyError(AIProviderError):
    """Exception for API key related errors."""
    pass


class ModelNotFoundError(AIProviderError):
    """Exception when a model is not found."""
    pass


class AnalysisError(AIProviderError):
    """Exception during analysis."""
    pass


# --- Utilities ---
def create_error_response(error: Exception, model_info: Optional[ModelInfo] = None) -> AnalysisResponse:
    """Create a standardized error response."""
    error_message = f"Error: {type(error).__name__}: {str(error)}"
    
    if model_info is None:
        model_info = ModelInfo(name="unknown", provider=ProviderType.OPENAI)
    
    return AnalysisResponse(
        content=error_message,
        model=model_info,
        finish_reason="error"
    )


async def with_timeout(coro, timeout: int, error_message: str = "Operation timed out"):
    """Execute a coroutine with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise AnalysisError(f"{error_message} after {timeout} seconds") 